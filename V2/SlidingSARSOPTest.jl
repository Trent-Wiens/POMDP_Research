using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using Cairo

# Include the DroneRockSample module first
include("DroneRockSample.jl")
# Save a reference to the module
const DRS = DroneRockSample

# Include our SlidingSARSOP implementation
include("SlidingSARSOP.jl")
# Set the DroneRockSample module in SlidingSARSOP
SlidingSARSOP.set_dronerocksample(DRS)

# Comparison function to evaluate performance
function compare_policies(pomdp, standard_policy, sliding_policy, num_trials=10, max_steps=20)
    standard_rewards = []
    sliding_rewards = []
    
    for i in 1:num_trials
        println("Running trial $i of $num_trials...")
        rng = MersenneTwister(i)  # Use deterministic but different seeds
        
        # Run with standard policy
        standard_hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        standard_history = simulate(standard_hr, pomdp, standard_policy)
        push!(standard_rewards, discounted_reward(standard_history))
        
        # Run with sliding policy
        sliding_hr = HistoryRecorder(max_steps=max_steps, rng=rng)
        sliding_history = simulate(sliding_hr, pomdp, sliding_policy)
        push!(sliding_rewards, discounted_reward(sliding_history))
    end
    
    # Print results
    println("\nResults over $num_trials trials:")
    println("Standard SARSOP: Mean reward = $(mean(standard_rewards)), Std = $(std(standard_rewards))")
    println("Sliding SARSOP: Mean reward = $(mean(sliding_rewards)), Std = $(std(sliding_rewards))")
    
    return standard_rewards, sliding_rewards
end

function run_comparison()
    # Create a DroneRockSample POMDP
    pomdp = DroneRockSamplePOMDP(
        map_size = (10, 10),
        rocks_positions = [(2, 8), (4, 5), (7, 2), (8, 9)],
        sensor_efficiency = 20.0,
        discount_factor = 0.95,
        good_rock_reward = 20.0,
        fly_penalty = -0.2
    )
    
    println("POMDP created with $(length(pomdp.rocks_positions)) rocks")
    
    # Solve with standard SARSOP (full state space)
    println("Solving with standard SARSOP...")
    start_time = time()
    standard_solver = SARSOPSolver(precision=1e-3, max_time=10.0)
    standard_policy = POMDPs.solve(standard_solver, pomdp)
    standard_time = time() - start_time
    println("Standard SARSOP completed in $standard_time seconds")
    
    # Solve with Sliding SARSOP
    println("Initializing SlidingSARSOP...")
    start_time = time()
    sliding_solver = SlidingSARSOPSolver(
        horizon_distance = 6,  # Adjust based on map size
        precision = 1e-3,
        timeout = 2.0,
        belief_points = 1000,
        include_goal_state = true
    )
    sliding_policy = POMDPs.solve(sliding_solver, pomdp)
    sliding_time = time() - start_time
    println("SlidingSARSOP initialization completed in $sliding_time seconds")
    
    # Compare the policies
    standard_rewards, sliding_rewards = compare_policies(pomdp, standard_policy, sliding_policy, 5, 30)
    
    # Create a GIF of both policies
    println("Creating simulation GIFs...")
    
    # Standard policy simulation
    sim1 = GifSimulator(
        filename = "StandardSARSOP.gif", 
        max_steps = 30,
        rng = MersenneTwister(42), 
        show_progress = true
    )
    saved_gif1 = simulate(sim1, pomdp, standard_policy)
    
    # Sliding policy simulation
    sim2 = GifSimulator(
        filename = "SlidingSARSOP.gif", 
        max_steps = 30,
        rng = MersenneTwister(42), 
        show_progress = true
    )
    saved_gif2 = simulate(sim2, pomdp, sliding_policy)
    
    println("GIFs saved to: $(saved_gif1.filename) and $(saved_gif2.filename)")
    
    return pomdp, standard_policy, sliding_policy, standard_rewards, sliding_rewards
end

# Run the comparison
pomdp, standard_policy, sliding_policy, standard_rewards, sliding_rewards = run_comparison()