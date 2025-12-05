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

# --- 1. HELPER FUNCTIONS ---

function global2local(pos, sub_map)
    local_x = pos[1] - sub_map[1] + 1
    local_y = pos[2] - sub_map[2] + 1
    return (local_x, local_y)
end

function make_sub_POMDP_dynamic(pos, map_size, rock_pos, rock_probs, global_pomdp, horizon)
    # Define Window
    maxx = min(pos[1] + horizon, map_size[1])
    minx = max(pos[1] - horizon, 1)
    maxy = min(pos[2] + horizon, map_size[2])
    miny = max(pos[2] - horizon, 1)
    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

    # Filter Rocks inside window
    sub_rocks_global = [r for r in rock_pos if sub_map[1] ≤ r[1] ≤ sub_map[3] && sub_map[2] ≤ r[2] ≤ sub_map[4]]
    
    # Heuristic: If no rocks in window, add the nearest one to encourage movement
    if isempty(sub_rocks_global)
        dists = [sum(abs.(r .- pos)) for r in rock_pos]
        nearest_idx = argmin(dists)
        push!(sub_rocks_global, rock_pos[nearest_idx])
    end

    local_sub_rocks = [global2local(r, sub_map) for r in sub_rocks_global]
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
        step_penalty = global_pomdp.step_penalty,
        exit_reward = global_pomdp.exit_reward
    )

    # Construct Initial Belief for Sub-POMDP
    # Map local rocks -> global indices -> get probability from current belief
    numLocal = length(local_sub_rocks)
    global_indices = [findfirst(==(r), rock_pos) for r in sub_rocks_global]
    local_probs = [rock_probs[i] for i in global_indices]
    
    states = ordered_states(sub_pomdp)
    probs = zeros(length(states))
    
    # Filter states to only match current agent position
    relevant_indices = findall(s -> s.pos == loc_pos, states)
    
    for idx in relevant_indices
        s = states[idx]
        p = 1.0
        for r in 1:numLocal
            # Probability of this specific rock configuration
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
    
    init_dist = SparseCat(states, probs)
    return sub_pomdp, init_dist, sub_map, global_indices
end

# --- 2. EXPERIMENT RUNNERS ---

function run_global_batch(num_rocks, num_sims, rock_cutoff)
    results = DataFrame(
        Rocks=Int[], Method=String[], Reward=Float64[], 
        TotalTime=Float64[], AvgStepTime=Float64[]
    )
    
    if num_rocks > rock_cutoff
        return results
    end

    map_size = (10, 10)
    all_pos = [(x,y) for x in 1:10, y in 1:10]
    rock_pos = shuffle(all_pos)[1:num_rocks]
    
    pomdp = RockSamplePOMDP(
        map_size=map_size, 
        rocks_positions=rock_pos,
        init_pos=(1,1),
        sensor_efficiency=20.0,
        good_rock_reward=20.0
    )

    println("  > Solving Global (Rocks: $num_rocks)...")
    
    # MEASURE OFFLINE TIME
    t_solve_start = time()
    try
        solver = SARSOPSolver(precision=1e-3, max_time=60.0, verbose=false)
        policy = solve(solver, pomdp)
        offline_time = time() - t_solve_start
        
        for i in 1:num_sims
            sim = RolloutSimulator(max_steps=50, rng=MersenneTwister(i))
            
            # MEASURE ONLINE TIME
            t_sim_start = time()
            r = simulate(sim, pomdp, policy)
            online_time = time() - t_sim_start
            
            # Total Time = Offline Setup + Online Execution
            total_time = offline_time + online_time
            avg_step_time = online_time / 50.0 # Approximation
            
            push!(results, (num_rocks, "Global", r, total_time, avg_step_time))
        end
    catch e
        println("    Global Failed: $e")
    end
    return results
end

function run_rhc_batch(num_rocks, num_sims, horizon)
    results = DataFrame(
        Rocks=Int[], Method=String[], Reward=Float64[], 
        TotalTime=Float64[], AvgStepTime=Float64[]
    )
    
    map_size = (10, 10)
    all_pos = [(x,y) for x in 1:10, y in 1:10]
    rock_pos = shuffle(all_pos)[1:num_rocks]

    pomdp = RockSamplePOMDP(
        map_size=map_size, 
        rocks_positions=rock_pos,
        init_pos=(1,1),
        sensor_efficiency=20.0,
        good_rock_reward=20.0
    )

    for i in 1:num_sims
        rng = MersenneTwister(i)
        
        s = rand(rng, initialstate(pomdp))
        current_rock_probs = fill(0.5, num_rocks)
        
        acc_reward = 0.0
        disc = 1.0
        
        step_times = Float64[]
        
        for t in 1:50
            # --- START STEP TIMER ---
            t_step_start = time()
            
            # 1. Plan
            sub_pomdp, b0, sub_map, g_indices = make_sub_POMDP_dynamic(
                s.pos, map_size, rock_pos, current_rock_probs, pomdp, horizon
            )
            
            # Lower precision for speed
            solver = SARSOPSolver(precision=0.1, max_time=0.5, verbose=false)
            policy = solve(solver, sub_pomdp)
            a_local = action(policy, b0)

            println(a_local)
            
            # Map Action to Global
            a_global = a_local
            if a_local > 5
                local_rock_idx = a_local - 5
                global_rock_idx = g_indices[local_rock_idx]
                a_global = global_rock_idx + 5
            end
            
            # --- STOP STEP TIMER ---
            push!(step_times, time() - t_step_start)
            
            # 2. Execute
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a_global, rng)
            
            # 3. Update Belief
            if a_global > 5 # Sense
                rock_idx = a_global - 5
                efficiency = pomdp.sensor_efficiency
                d = norm(Float64[s.pos[1] - rock_pos[rock_idx][1], s.pos[2] - rock_pos[rock_idx][2]])
                eta = (1 + 2^(-d / efficiency)) * 0.5
                prior = current_rock_probs[rock_idx]
                is_good_obs = (o == 1) # 1=Good, 2=Bad
                p_good = is_good_obs ? eta : (1.0 - eta)
                p_bad = is_good_obs ? (1.0 - eta) : eta
                current_rock_probs[rock_idx] = (p_good * prior) / (p_good * prior + p_bad * (1.0 - prior))
            elseif a_global == 1 # Sample
                 # If we sample, the rock is effectively consumed/known. 
                 # In RockSample logic, we usually treat it as Bad (value 0) after sampling to prevent re-sampling.
                 # We need to know WHICH rock was sampled. RockSample assumes you sample the rock at your location.
                 if s.pos in rock_pos
                     idx = findfirst(==(s.pos), rock_pos)
                     current_rock_probs[idx] = 0.0 # Mark as bad/collected
                 end
            end

            acc_reward += r * disc
            disc *= pomdp.discount_factor
            s = sp
            
            if isterminal(pomdp, s)
                break
            end
        end
        
        # Calculate Times
        total_time = sum(step_times)
        avg_step_time = mean(step_times)
        
        push!(results, (num_rocks, "RHC", acc_reward, total_time, avg_step_time))
    end
    
    return results
end

# --- 3. MAIN EXECUTION ---

function main()
    TEST_ROCKS = [5, 8, 12, 15, 20]
    NUM_SIMS = 10
    GLOBAL_CUTOFF = 12 # Global will fail after 8 rocks
    HORIZON = 3       # 7x7 Window
    
    FILENAME = "rocksample_timing_results.csv"
    
    all_data = DataFrame(
        Rocks=Int[], Method=String[], Reward=Float64[], 
        TotalTime=Float64[], AvgStepTime=Float64[]
    )

    println("Starting RockSample Timing Test...")
    println("-"^80)
    println("Rocks | Global(Rew) | Global(Total Time) | RHC(Rew) | RHC(Total) | RHC(Step)")
    println("-"^80)

    for N in TEST_ROCKS
        # 1. Global
        df_g = run_global_batch(N, NUM_SIMS, GLOBAL_CUTOFF)
        append!(all_data, df_g)
        
        g_rew = isempty(df_g) ? missing : mean(df_g.Reward)
        g_time = isempty(df_g) ? missing : mean(df_g.TotalTime)

        # 2. RHC
        df_r = run_rhc_batch(N, NUM_SIMS, HORIZON)
        append!(all_data, df_r)
        
        r_rew = mean(df_r.Reward)
        r_tot = mean(df_r.TotalTime)
        r_step = mean(df_r.AvgStepTime)
        
        # Formatting
        g_rew_s = ismissing(g_rew) ? "--- " : rpad(string(round(g_rew, digits=2)), 5)
        g_time_s = ismissing(g_time) ? "--- " : rpad(string(round(g_time, digits=2)), 6)
        
        r_rew_s = rpad(string(round(r_rew, digits=2)), 5)
        r_tot_s = rpad(string(round(r_tot, digits=2)), 5)
        r_step_s = string(round(r_step, digits=4))

        println(" $N    | $g_rew_s       | $g_time_s s            | $r_rew_s    | $r_tot_s s   | $r_step_s s")
    end
    
    CSV.write(FILENAME, all_data)
    println("\nData saved to $FILENAME")
end

main()