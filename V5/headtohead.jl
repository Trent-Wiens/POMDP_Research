using POMDPs
using POMDPModels
using POMDPTools
using NativeSARSOP
using RockSample
using DataFrames
using CSV
using Dates
using Random
using LinearAlgebra

# =============================================================================
# 1. RECEDING HORIZON HELPER FUNCTIONS
# =============================================================================

function global2local(pos, sub_map)
    local_x = pos[1] - sub_map[1] + 1
    local_y = pos[2] - sub_map[2] + 1
    return [local_x, local_y]
end

function local2global(pos, sub_map)
    global_x = pos[1] + sub_map[1] - 1
    global_y = pos[2] + sub_map[2] - 1
    return [global_x, global_y]
end

function make_sub_POMDP(pos, map_size, rock_pos, rock_probs, pomdp)
    horizon = 3 # SHORT Horizon for speed

    # Define boundaries
    maxx = min(pos[1] + horizon, map_size[1])
    minx = max(pos[1] - horizon, 1)
    maxy = min(pos[2] + horizon, map_size[2])
    miny = max(pos[2] - horizon, 1)
    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

    # Find rocks inside the horizon
    sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

    # Heuristic: If no rocks in horizon, add the nearest one to pull agent towards it
    if isempty(sub_rocks)
        dists = [sum(abs.(r .- pos)) for r in rock_pos]
        nearest_idx = argmin(dists)
        push!(sub_rocks, rock_pos[nearest_idx])
        
        # Expand submap to include this rock if needed
        r = rock_pos[nearest_idx]
        sub_map = (min(sub_map[1], r[1]), min(sub_map[2], r[2]), max(sub_map[3], r[1]), max(sub_map[4], r[2]))
    end

    # Dimensions
    sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)
    local_sub_rocks = Tuple.(global2local.(sub_rocks, Ref(sub_map)))
    local_init_pos = global2local(pos, sub_map)

    # Sub-POMDP
    sub_pomdp = RockSamplePOMDP(
        map_size = sub_map_size,
        rocks_positions = collect(local_sub_rocks),
        init_pos = local_init_pos,
        sensor_efficiency = pomdp.sensor_efficiency,
        discount_factor = pomdp.discount_factor,
        good_rock_reward = pomdp.good_rock_reward,
        bad_rock_penalty = pomdp.bad_rock_penalty,
        step_penalty = pomdp.step_penalty
    )

    # Calculate Local Beliefs from Global Beliefs
    numRock = length(sub_pomdp.rocks_positions)
    global_idx_map = Vector{Union{Int, Nothing}}(undef, numRock)
    rockings = zeros(numRock)
    
    # rock_probs is a SparseCat; vals are positions, probs are probabilities
    # We need to find the probability for each rock in sub_rocks
    for i in 1:numRock
        g_rock = sub_rocks[i]
        idx = findfirst(==(g_rock), rock_probs.vals)
        global_idx_map[i] = idx
        if idx !== nothing
            rockings[i] = rock_probs.probs[idx]
        else
            rockings[i] = 0.5 # Default if lost
        end
    end

    notRockings = ones(numRock) .- rockings

    # Create Initial State Distribution for Sub-POMDP
    states = ordered_states(sub_pomdp)
    indc = findall(s -> s.pos == local_init_pos, states)
    init_states = states[indc]
    init_probs = zeros(length(init_states))

    j = 1
    for s in init_states
        mask = s.rocks
        # Calculate prob of this specific rock configuration
        p = 1.0
        for i in 1:numRock
            p *= (mask[i] ? rockings[i] : notRockings[i])
        end
        init_probs[j] = p
        j += 1
    end
    
    # Normalize
    if sum(init_probs) > 0
        init_probs ./= sum(init_probs)
    else
        init_probs .= 1.0 / length(init_probs)
    end

    init_state = SparseCat(init_states, init_probs)

    return sub_pomdp, init_state, sub_map, global_idx_map
end

# =============================================================================
# 2. SIMULATION RUNNERS
# =============================================================================

# --- RUNNER A: STANDARD SARSOP (Offline) ---
function run_standard(pomdp, true_state, rng_seed, max_steps)
    # 1. SOLVE (Offline)
    solver = SARSOPSolver(precision=1e-3, max_time=60.0, verbose=false)
    
    # Standard SARSOP might crash on large maps, so we wrap in try-catch
    policy = nothing
    solve_time = @elapsed begin
        try
            policy = solve(solver, pomdp)
        catch e
            return (NaN, NaN, "Failed: Solver Crash")
        end
    end

    # 2. SIMULATE (Online)
    rng = MersenneTwister(rng_seed)
    
    # Current Real State
    curr_state = true_state
    # Current Belief (Start with uniform/prior)
    updater = DiscreteUpdater(pomdp)
    belief = initialize_belief(updater, initialstate(pomdp))
    
    total_reward = 0.0
    discount = 1.0
    
    steps = 0
    for i in 1:max_steps
        if isterminal(pomdp, curr_state)
            break
        end

        # Action
        a = action(policy, belief)
        
        # Transition (Real World)
        sp, o, r = gen(pomdp, curr_state, a, rng)
        
        # Accumulate Reward
        total_reward += r * discount
        discount *= pomdp.discount_factor
        
        # Update Belief
        belief = update(updater, belief, a, o)
        curr_state = sp
        steps += 1
    end

    return (total_reward, solve_time, "Success")
end

# --- RUNNER B: RECEDING HORIZON (Online) ---
function run_receding_horizon(pomdp, true_state, rng_seed, max_steps)
    
    rng = MersenneTwister(rng_seed)
    
    # Global Belief (Just tracking probability of each rock being Good)
    # vals = rock positions, probs = probability of being Good
    rock_probs = SparseCat(pomdp.rocks_positions, fill(0.5, length(pomdp.rocks_positions)))
    
    curr_state = true_state # True Global State
    curr_pos = pomdp.init_pos
    
    total_reward = 0.0
    discount = 1.0
    
    sim_time = @elapsed begin
        for i in 1:max_steps
            if isterminal(pomdp, curr_state)
                break
            end

            # 1. Make Sub-POMDP
            sub_pomdp, sub_init_belief, sub_map, global_idx_map = make_sub_POMDP(curr_pos, pomdp.map_size, pomdp.rocks_positions, rock_probs, pomdp)
            
            # 2. Solve Sub-POMDP (Online)
            # We set a very short time limit because we do this EVERY step
            POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = sub_init_belief
            solver = SARSOPSolver(precision=1e-3, max_time=1.0, verbose=false)
            policy = solve(solver, sub_pomdp)
            
            # 3. Get Action from Sub-Policy
            # We take the best action for the current sub-belief
            a_sub = action(policy, sub_init_belief)
            
            # 4. Convert Sub-Action to Global Action
            a_global = 0
            if a_sub <= 5
                a_global = a_sub # Move actions map 1:1
            else
                # Sensing action
                local_rock_idx = a_sub - 5
                # Find which global rock this corresponds to
                # We need to map the local rock index back to the global rock index
                # sub_pomdp.rocks_positions[local_rock_idx] is the LOCAL position
                local_pos = sub_pomdp.rocks_positions[local_rock_idx]
                global_pos = local2global(local_pos, sub_map)
                
                # Find index in main pomdp
                global_idx = findfirst(==(global_pos), pomdp.rocks_positions)
                a_global = 5 + global_idx
            end

            # 5. Execute in REAL WORLD
            sp, o, r = gen(pomdp, curr_state, a_global, rng)
            
            # 6. Accumulate Reward
            total_reward += r * discount
            discount *= pomdp.discount_factor
            
            # 7. Update Global Belief (Manually)
            # If we sensed, update that specific rock's probability
            if a_global > 5
                rock_idx = a_global - 5
                # Simple Bayesian update for independent rocks
                p = rock_probs.probs[rock_idx]
                # Efficiency 
                eta = pomdp.sensor_efficiency
                dist = norm(Float64.(pomdp.rocks_positions[rock_idx] .- curr_state.pos))
                efficiency = (1 + 2^(-dist / eta)) * 0.5
                
                if o == 1 # Good
                    p_new = (efficiency * p) / (efficiency * p + (1-efficiency) * (1-p))
                elseif o == 2 # Bad
                    p_new = ((1-efficiency) * p) / ((1-efficiency) * p + efficiency * (1-p))
                else
                    p_new = p
                end
                rock_probs.probs[rock_idx] = p_new
            end
            
            # If we sampled, the rock effectively becomes "Bad" (collected) or stays "Bad"
            if a_global == 1
                # If we are at a rock location, we sampled it. 
                # In standard RS, sampled rocks effectively disappear or reward 0. 
                # We'll assume the belief drops to 0 (Bad) to stop resampling.
                if curr_state.pos in pomdp.rocks_positions
                    r_idx = findfirst(==(curr_state.pos), pomdp.rocks_positions)
                    rock_probs.probs[r_idx] = 0.0
                end
            end

            curr_state = sp
            curr_pos = curr_state.pos
        end
    end

    return (total_reward, sim_time, "Success")
end

# =============================================================================
# 3. MAIN EXPERIMENT LOOP
# =============================================================================

timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
filename = "Benchmark_Standard_vs_RH_$(timestamp).csv"

df = DataFrame(
    MapSize = String[],
    NumRocks = Int[],
    Method = String[],
    TotalReward = Float64[],
    TimeTaken = Float64[],
    Status = String[],
    Seed = Int[]
)
CSV.write(filename, df)

mapsizes = [(5,5)]
# Test 3 (Easy), 10 (Medium), 25 (Hard)
numrocks_list = [3, 10, 25] 
MAX_STEPS = 100

for size in mapsizes
    for numrocks in numrocks_list
        
        # 1. SETUP THE COMMON ARENA
        current_seed = Int(time_ns())
        Random.seed!(current_seed)
        
        # Generate Map
        possible_coords = [(x, y) for x in 1:size[1], y in 1:size[2] if (x,y) != (1,1)]
        rock_locs = first(shuffle(possible_coords), numrocks)
        
        pomdp = RockSamplePOMDP(
            map_size = size,
            rocks_positions = rock_locs,
            init_pos = (1,1),
            sensor_efficiency = 20.0,
            discount_factor = 0.95
        )

        # GENERATE TRUE STATE (The Ground Truth)
        # We must use a separate RNG for state generation to keep it consistent
        state_rng = MersenneTwister(current_seed)
        true_initial_state = rand(state_rng, initialstate(pomdp))

        println("\n--- BENCHMARK: Size=$size, Rocks=$numrocks, Seed=$current_seed ---")

        # ---------------------------------------------------------------------
        # RUN STANDARD SARSOP
        # ---------------------------------------------------------------------
        # Only run standard if rocks < 12 (otherwise it crashes/hangs)
        if numrocks < 12
            print("  > Standard... ")
            r_std, t_std, s_std = run_standard(pomdp, true_initial_state, current_seed, MAX_STEPS)
            println("Reward: $(round(r_std, digits=2)), Time: $(round(t_std, digits=2))s")
            
            push!(df, ("$size", numrocks, "Standard", r_std, t_std, s_std, current_seed))
        else
            println("  > Standard... SKIPPED (Too Large)")
            push!(df, ("$size", numrocks, "Standard", NaN, NaN, "Skipped", current_seed))
        end

        # ---------------------------------------------------------------------
        # RUN RECEDING HORIZON
        # ---------------------------------------------------------------------
        print("  > Receding... ")
        # RH should run on everything
        r_rh, t_rh, s_rh = run_receding_horizon(pomdp, true_initial_state, current_seed, MAX_STEPS)
        println("Reward: $(round(r_rh, digits=2)), Time: $(round(t_rh, digits=2))s")

        push!(df, ("$size", numrocks, "RH", r_rh, t_rh, s_rh, current_seed))
        
        # Save progress
        CSV.write(filename, df)
    end
end

println("\nBenchmark Complete. Results saved to $filename")