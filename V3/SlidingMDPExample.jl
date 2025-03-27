# Complete example showing how to use Sliding MDP for DroneRockSample
using POMDPs
using POMDPTools
using Random
using Plots

# Include our modules
include("DroneRockSample.jl")
include("SlidingDroneRockSample.jl")
include("VisualizeSlidingMDP.jl")

using .DroneRockSample
using .SlidingDroneRockSample

function compare_sliding_vs_regular()
    println("=== Comparing Sliding MDP vs Regular MDP ===")
    
    # Create a test problem
    map_size = (8, 8)
    rocks_positions = [(2, 6), (4, 3), (6, 5)]
    init_pos = (1, 1)
    
    # Parameters
    sensor_efficiency = 20.0
    discount_factor = 0.95
    good_rock_reward = 20.0
    fly_penalty = -0.2
    
    # Random seed for reproducibility
    rng = MersenneTwister(42)
    
    println("\n1. Creating and solving regular DroneRockSample POMDP...")
    start_time = time()
    
    # Create regular POMDP
    regular_pomdp = DroneRockSamplePOMDP(
        map_size = map_size,
        rocks_positions = rocks_positions,
        init_pos = RSPos(init_pos...),
        sensor_efficiency = sensor_efficiency,
        discount_factor = discount_factor,
        good_rock_reward = good_rock_reward,
        fly_penalty = fly_penalty
    )
    
    # Solve with Value Iteration
    solver = ValueIterationSolver(max_iterations=1000, belres=1e-6)
    regular_policy = solve(solver, UnderlyingMDP(regular_pomdp))
    
    regular_time = time() - start_time
    println("  Done in $(round(regular_time, digits=2)) seconds")
    
    println("\n2. Creating Sliding DroneRockSample POMDP...")
    start_time = time()
    
    # Create sliding POMDP
    smdp = create_sliding_drone_rock_sample(
        map_size,
        rocks_positions,
        init_pos = init_pos,
        sensor_efficiency = sensor_efficiency,
        discount_factor = discount_factor,
        good_rock_reward = good_rock_reward,
        fly_penalty = fly_penalty,
        horizon_limit = 0.7,
        splitting_param = 2.0
    )
    
    sliding_time = time() - start_time
    println("  Done in $(round(sliding_time, digits=2)) seconds")
    
    # Run simulations for both
    println("\n3. Running simulation with regular POMDP...")
    
    # Initialize the state
    init_state = RSState(RSPos(init_pos...), SVector{length(rocks_positions),Bool}([true, true, true]))
    
    # Run simulation with regular POMDP
    reg_rewards = []
    reg_steps = []
    
    # Run multiple trials
    n_trials = 5
    
    for trial in 1:n_trials
        println("\n--- Regular POMDP Trial $trial ---")
        s = init_state
        total_reward = 0.0
        discount = 1.0
        steps = 0
        
        while !isterminal(regular_pomdp, s) && steps < 30
            # Get action from policy
            a = action(regular_policy, s)
            println("State: $(s.pos), Action: $(action_to_string(regular_pomdp, a))")
            
            # Get next state and reward
            sp = rand(transition(regular_pomdp, s, a))
            r = reward(regular_pomdp, s, a)
            
            # Update
            total_reward += discount * r
            discount *= discount_factor
            s = sp
            steps += 1
        end
        
        println("Completed in $steps steps with reward $total_reward")
        push!(reg_rewards, total_reward)
        push!(reg_steps, steps)
    end
    
    println("\n4. Running simulation with Sliding POMDP...")
    sliding_results, sliding_rewards = [], []
    
    for trial in 1:n_trials
        println("\n--- Sliding POMDP Trial $trial ---")
        # Reset SMDP
        smdp = create_sliding_drone_rock_sample(
            map_size,
            rocks_positions,
            init_pos = init_pos,
            sensor_efficiency = sensor_efficiency,
            discount_factor = discount_factor,
            good_rock_reward = good_rock_reward,
            fly_penalty = fly_penalty,
            horizon_limit = 0.7,
            splitting_param = 2.0
        )
        
        # Run sliding simulation
        results, reward = run_sliding_pomdp_simulation(smdp, 30)
        push!(sliding_results, results)
        push!(sliding_rewards, reward)
    end
    
    # Comparison
    println("\n=== Results Summary ===")
    println("Regular POMDP:")
    println("  Avg reward: $(mean(reg_rewards))")
    println("  Avg steps: $(mean(reg_steps))")
    println("  Solve time: $(round(regular_time, digits=2)) seconds")
    println("  State space size: $(length(states(regular_pomdp)))")
    
    println("\nSliding POMDP:")
    println("  Avg reward: $(mean(sliding_rewards))")
    println("  Avg steps: $(mean(map(r -> length(r), sliding_results)))")
    println("  Initial setup time: $(round(sliding_time, digits=2)) seconds")
    println("  Final state space size: $(length(states(smdp.current_pomdp)))")
    
    # Visualization
    best_sliding_idx = argmax(sliding_rewards)
    best_sliding_result = sliding_results[best_sliding_idx]
    
    # Run visualization for the best sliding result
    run_visualization(smdp, best_sliding_result)
    
    return smdp, best_sliding_result, regular_pomdp, regular_policy
end

# Run the comparison
println("Starting comparison...")
smdp, best_result, reg_pomdp, reg_policy = compare_sliding_vs_regular()
println("Comparison complete!")