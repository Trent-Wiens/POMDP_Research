using POMDPs
using POMDPTools
using POMDPModels
using DroneSurveillance
using NativeSARSOP
using Random
using LinearAlgebra
using Statistics
using DataFrames
using CSV

# --- 1. RHC HELPER FUNCTIONS ---

function get_clamped_local(global_p, bounds, local_size)
    lx = global_p[1] - bounds[1] + 1
    ly = global_p[2] - bounds[2] + 1
    cx = clamp(lx, 1, local_size[1])
    cy = clamp(ly, 1, local_size[2])
    return [cx, cy]
end

function global2local(pos, sub_map_bounds)
    return [pos[1] - sub_map_bounds[1] + 1, pos[2] - sub_map_bounds[2] + 1]
end

function make_sub_POMDP_dynamic(quad_pos, agent_pos, global_pomdp, horizon)
    g_size = global_pomdp.size
    
    minx = max(quad_pos[1] - horizon, 1)
    maxx = min(quad_pos[1] + horizon, g_size[1])
    miny = max(quad_pos[2] - horizon, 1)
    maxy = min(quad_pos[2] + horizon, g_size[2])

    sub_bounds = (minx, miny, maxx, maxy)
    sub_size = (maxx - minx + 1, maxy - miny + 1)

    local_region_A = get_clamped_local(global_pomdp.region_A, sub_bounds, sub_size)
    local_region_B = get_clamped_local(global_pomdp.region_B, sub_bounds, sub_size)

    sub_pomdp = DroneSurveillancePOMDP(
        size = sub_size,
        region_A = local_region_A,
        region_B = local_region_B,
        fov = global_pomdp.fov,
        agent_policy = global_pomdp.agent_policy,
        camera = global_pomdp.camera,
        discount_factor = global_pomdp.discount_factor
    )

    local_quad = global2local(quad_pos, sub_bounds)
    local_agent = get_clamped_local(agent_pos, sub_bounds, sub_size)

    init_belief = Deterministic(DSState(local_quad, local_agent))

    return sub_pomdp, init_belief, sub_bounds
end

# --- 2. EXPERIMENT RUNNERS (Returning DataFrames) ---

function run_global_batch(map_size, num_sims, global_cutoff)
    results = DataFrame(
        MapSize = Int[], 
        RunID = Int[], 
        Method = String[], 
        Reward = Float64[], 
        OfflineTime = Float64[], 
        AvgStepTime = Float64[]
    )
    
    # Skip if too big
    if map_size[1] > global_cutoff
        return results
    end

    pomdp = DroneSurveillancePOMDP(
        size = map_size,
        region_A = [1, 1],
        region_B = [map_size[1], map_size[2]],
        fov = (3, 3)
    )

    println("  > Solving Global (Size $(map_size))...")
    t_start = time()
    try
        # Solver setup
        solver = SARSOPSolver(precision=1e-3, max_time=60.0, verbose=false) 
        policy = solve(solver, pomdp)
        solve_time = time() - t_start
        
        # Run Simulations
        for i in 1:num_sims
            seed = Random.rand(1:1000)
            sim = RolloutSimulator(max_steps=50, rng=MersenneTwister(seed))
            # Measure simulation time just for kicks, though for Global it's negligible
            t_sim_start = time()
            r = simulate(sim, pomdp, policy)
            total_sim_time = time() - t_sim_start
            avg_step_time = total_sim_time / 50.0 # Approx
            
            push!(results, (map_size[1], i, "Global", r, solve_time, avg_step_time))
        end
    catch e
        println("    Global Solver Failed: $e")
    end

    return results
end

function run_rhc_batch(map_size, num_sims, horizon)
    results = DataFrame(
        MapSize = Int[], 
        RunID = Int[], 
        Method = String[], 
        Reward = Float64[], 
        OfflineTime = Float64[], 
        AvgStepTime = Float64[]
    )

    pomdp = DroneSurveillancePOMDP(
        size = map_size,
        region_A = [1, 1],
        region_B = [map_size[1], map_size[2]],
        fov = (3, 3)
    )

    for i in 1:num_sims
        rng = MersenneTwister(i)
        
        init_state = rand(rng, initialstate(pomdp))
        quad_pos = copy(init_state.quad)
        agent_pos = copy(init_state.agent)
        
        accumulated_reward = 0.0
        discount = 1.0
        step_times = Float64[]
        
        for t in 1:50
            t_step_start = time()
            
            # Plan
            sub_pomdp, sub_b0, bounds = make_sub_POMDP_dynamic(quad_pos, agent_pos, pomdp, horizon)
            solver = SARSOPSolver(precision=0.1, max_time=1.0, verbose=false) 
            policy = solve(solver, sub_pomdp)
            a = action(policy, sub_b0)
            
            dt = time() - t_step_start
            push!(step_times, dt)
            
            # Execute
            s = DSState(quad_pos, agent_pos)
            sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
            
            accumulated_reward += r * discount
            discount *= pomdp.discount_factor
            
            quad_pos = sp.quad
            agent_pos = sp.agent
            
            if isterminal(pomdp, sp)
                break
            end
        end
        
        # Offline time is 0 for RHC
        push!(results, (map_size[1], i, "RHC", accumulated_reward, 0.0, mean(step_times)))
    end

    return results
end

# --- 3. MAIN EXECUTION ---

function main()
    # Configuration
    TEST_SIZES = [5, 7, 10, 15, 20]
    NUM_SIMS = 10     # How many times to run each map
    GLOBAL_CUTOFF = 7 # Stop Global after 7x7
    FILENAME = "simulation_results_2.csv"
    
    # Master DataFrame
    all_data = DataFrame(
        MapSize = Int[], 
        RunID = Int[], 
        Method = String[], 
        Reward = Float64[], 
        OfflineTime = Float64[], 
        AvgStepTime = Float64[]
    )

    println("Starting Simulations...")
    println("Saving results to: $FILENAME")

    for N in TEST_SIZES
        map_dim = (N, N)
        println("Processing Map Size: $N x $N")
        
        # 1. Run Global
        df_global = run_global_batch(map_dim, NUM_SIMS, GLOBAL_CUTOFF)
        append!(all_data, df_global)
        
        # 2. Run RHC
        df_rhc = run_rhc_batch(map_dim, NUM_SIMS, 2) # Horizon 2
        append!(all_data, df_rhc)
    end

    # Save to CSV
    CSV.write(FILENAME, all_data)
    println("\nDone! Data saved to $FILENAME")
    
    # Show a preview
    println("\nFirst 5 rows:")
    display(first(all_data, 5))
    println("\nLast 5 rows:")
    display(last(all_data, 5))
end

main()