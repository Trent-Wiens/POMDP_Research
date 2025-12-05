using POMDPs
using POMDPTools
using POMDPModels
using RockSample
using NativeSARSOP
using Random
using LinearAlgebra
using Statistics
using DataFrames
using CSV
using StaticArrays

# --- GLOBAL STATE FOR RHC INJECTION ---
# We use this to "pass" the calculated belief into the solver
const RHC_BELIEF_REF = Ref{Any}(nothing)

# --- HELPER FUNCTIONS ---

function global2local(pos, sub_map)
    return (pos[1] - sub_map[1] + 1, pos[2] - sub_map[2] + 1)
end

function make_sub_POMDP_dynamic(pos, map_size, rock_pos, rock_probs, global_pomdp, horizon)
    # [Identical logic to previous turn...]
    # Define Window
    maxx = min(pos[1] + horizon, map_size[1])
    minx = max(pos[1] - horizon, 1)
    maxy = min(pos[2] + horizon, map_size[2])
    miny = max(pos[2] - horizon, 1)
    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

    sub_rocks_global = [r for r in rock_pos if sub_map[1] ≤ r[1] ≤ sub_map[3] && sub_map[2] ≤ r[2] ≤ sub_map[4]]
    
    if isempty(sub_rocks_global)
        dists = [sum(abs.(r .- pos)) for r in rock_pos]
        nearest_idx = argmin(dists)
        target_rock = rock_pos[nearest_idx]
        push!(sub_rocks_global, target_rock)
        lx = clamp(target_rock[1] - sub_map[1] + 1, 1, sub_map[3]-sub_map[1]+1)
        ly = clamp(target_rock[2] - sub_map[2] + 1, 1, sub_map[4]-sub_map[2]+1)
        local_sub_rocks = [(lx, ly)]
    else
        local_sub_rocks = [global2local(r, sub_map) for r in sub_rocks_global]
    end

    sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)
    loc_pos = global2local(pos, sub_map)

    sub_pomdp = RockSamplePOMDP(
        map_size = sub_map_size,
        rocks_positions = local_sub_rocks,
        init_pos = loc_pos,
        sensor_efficiency = global_pomdp.sensor_efficiency,
        discount_factor = global_pomdp.discount_factor,
        good_rock_reward = global_pomdp.good_rock_reward,
        bad_rock_penalty = global_pomdp.bad_rock_penalty,
        step_penalty = 0.0, 
        exit_reward = global_pomdp.exit_reward
    )

    # Belief Construction
    global_indices = [findfirst(==(r), rock_pos) for r in sub_rocks_global]
    local_probs = [rock_probs[i] for i in global_indices]
    
    states = ordered_states(sub_pomdp)
    probs = zeros(length(states))
    
    ax = clamp(loc_pos[1], 1, sub_map_size[1])
    ay = clamp(loc_pos[2], 1, sub_map_size[2])
    clamped_loc_pos = (ax, ay)

    relevant_indices = findall(s -> s.pos == clamped_loc_pos, states)
    
    if isempty(relevant_indices)
        probs .= 1.0/length(states)
    else
        for idx in relevant_indices
            s = states[idx]
            p = 1.0
            for r in 1:length(local_sub_rocks)
                p *= s.rocks[r] ? local_probs[r] : (1.0 - local_probs[r])
            end
            probs[idx] = p
        end
        total_p = sum(probs)
        if total_p > 0
            probs ./= total_p
        else
            probs[relevant_indices] .= 1.0 / length(relevant_indices)
        end
    end
    
    init_dist = SparseCat(states, probs)
    return sub_pomdp, init_dist, sub_map, global_indices
end

# --- RUNNER: PHASE 1 (GLOBAL) ---
# This runs with the STANDARD initialstate function

function run_all_globals(test_rocks, num_sims, cutoff)
    println(">>> PHASE 1: Running Global Benchmarks (Standard POMDP)...")
    results = DataFrame(Rocks=Int[], Method=String[], Reward=Float64[], ExpValue=Float64[], TotalTime=Float64[])

    for num_rocks in test_rocks
        if num_rocks > cutoff
            continue
        end

        map_size = (10, 10)
        rock_pos = shuffle([(x,y) for x in 1:10, y in 1:10])[1:num_rocks]
        pomdp = RockSamplePOMDP(
            map_size=map_size, rocks_positions=rock_pos, init_pos=(1,1),
            sensor_efficiency=20.0, good_rock_reward=20.0
        )

        println("   Processing $num_rocks rocks...")
        t_start = time()
        try
            solver = SARSOPSolver(precision=1e-3, max_time=60.0, verbose=false)
            policy = solve(solver, pomdp)
            off_time = time() - t_start
            
            b0 = initialstate(pomdp)
            val = value(policy, b0)

            for i in 1:num_sims
                sim = RolloutSimulator(max_steps=50, rng=MersenneTwister(i))
                t_sim = time()
                r = simulate(sim, pomdp, policy)
                push!(results, (num_rocks, "Global", r, val, off_time + (time() - t_sim)))
            end
        catch e
            println("   Failed: $e")
        end
    end
    return results
end

# --- RUNNER: PHASE 2 (RHC) ---
# This OVERWRITES initialstate and runs RHC

function run_all_rhc(test_rocks, num_sims, horizon)
    println("\n>>> PHASE 2: Running RHC Benchmarks (Custom Belief)...")
    
    # === THE OVERWRITE ===
    # We redefine initialstate to return whatever is in RHC_BELIEF_REF
    # This affects ALL RockSamplePOMDPs from this point forward.
    function POMDPs.initialstate(p::RockSamplePOMDP)
        return RHC_BELIEF_REF[]
    end

    results = DataFrame(Rocks=Int[], Method=String[], Reward=Float64[], ExpValue=Float64[], TotalTime=Float64[])

    for num_rocks in test_rocks
        println("   Processing $num_rocks rocks...")
        map_size = (10, 10)
        rock_pos = shuffle([(x,y) for x in 1:10, y in 1:10])[1:num_rocks]

        pomdp = RockSamplePOMDP(
            map_size=map_size, rocks_positions=rock_pos, init_pos=(1,1),
            sensor_efficiency=20.0, good_rock_reward=20.0
        )

        for i in 1:num_sims
            rng = MersenneTwister(i)
            s = rand(rng, POMDPs.initialstate(pomdp)) # Just gets random start pos
            # Reset rock beliefs manually for the simulation tracking
            current_rock_probs = fill(0.5, num_rocks)
            
            acc_reward = 0.0
            disc = 1.0
            total_exp_val = 0.0
            step_times = Float64[]
            
            for t in 1:50
                t_step = time()
                
                sub_pomdp, b0, sub_map, g_indices = make_sub_POMDP_dynamic(
                    s.pos, map_size, rock_pos, current_rock_probs, pomdp, horizon
                )
                
                # === INJECT BELIEF ===
                # We put our calculated belief into the global ref
                RHC_BELIEF_REF[] = b0
                
                # Now solve calls initialstate(), which grabs RHC_BELIEF_REF[]
                solver = SARSOPSolver(precision=0.1, max_time=0.5, verbose=false)
                policy = solve(solver, sub_pomdp)
                
                total_exp_val += value(policy, b0)
                a_local = action(policy, b0)
                
                # Map Actions
                a_global = a_local
                if a_local > 5
                   idx = a_local - 5
                   a_global = idx <= length(g_indices) ? g_indices[idx] + 5 : 1
                end
                
                push!(step_times, time() - t_step)
                
                sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a_global, rng)
                
                # Update Tracker
                if a_global > 5
                    ridx = a_global - 5
                    eff = pomdp.sensor_efficiency
                    d = norm(Float64[s.pos[1] - rock_pos[ridx][1], s.pos[2] - rock_pos[ridx][2]])
                    eta = (1 + 2^(-d / eff)) * 0.5
                    prior = current_rock_probs[ridx]
                    is_good = (o == 1)
                    pg = is_good ? eta : (1.0 - eta)
                    pb = is_good ? (1.0 - eta) : eta
                    denom = pg * prior + pb * (1.0 - prior)
                    current_rock_probs[ridx] = denom > 0 ? (pg * prior)/denom : prior
                elseif a_global == 1 && s.pos in rock_pos
                    current_rock_probs[findfirst(==(s.pos), rock_pos)] = 0.0
                end

                acc_reward += r * disc
                disc *= pomdp.discount_factor
                s = sp
                if isterminal(pomdp, s) break end
            end
            push!(results, (num_rocks, "RHC", acc_reward, total_exp_val/50, sum(step_times)))
        end
    end
    return results
end

# --- MAIN ---

function main()
    TEST_ROCKS = [5, 8, 12, 15]
    NUM_SIMS = 5
    GLOBAL_CUTOFF = 8
    HORIZON = 4
    
    # 1. Run All Globals
    df_global = run_all_globals(TEST_ROCKS, NUM_SIMS, GLOBAL_CUTOFF)
    
    # 2. Run All RHC (This modifies initialstate!)
    df_rhc = run_all_rhc(TEST_ROCKS, NUM_SIMS, HORIZON)
    
    # 3. Merge and Save
    final_df = vcat(df_global, df_rhc)
    
    println("\n=== FINAL RESULTS ===")
    display(final_df)
    CSV.write("rocksample_comparison_final.csv", final_df)
end

main()