using POMDPs
using POMDPTools
using NativeSARSOP
using Random
using Plots
using Statistics
using LinearAlgebra
using Dates
using JLD2  # For saving results

# Include the modules
include("DroneRockSample.jl")
using .DroneRockSample

include("SlidingSARSOP.jl")
using .SlidingSARSOP

"""
    evaluate_computational_efficiency(map_sizes, rock_densities)

Compare the computational efficiency of standard SARSOP vs SlidingSARSOP
across different problem sizes.
"""
function evaluate_computational_efficiency(map_sizes=[5, 10, 15], rock_densities=[0.1, 0.2])
    results = Dict()
    
    for map_size in map_sizes
        for rock_density in rock_densities
            println("Evaluating map size $map_size with rock density $rock_density...")
            
            # Calculate number of rocks
            num_rocks = max(1, round(Int, map_size * map_size * rock_density))
            
            # Create POMDP
            rng = MersenneTwister(123)  # Fixed seed for reproducibility
            pomdp = DroneRockSamplePOMDP(map_size, num_rocks, rng)
            
            # Time standard SARSOP
            println("  Running standard SARSOP...")
            standard_times = []
            max_time = min(30.0, 2.0 * map_size)  # Scale max time with problem size
            
            try
                start_time = time()
                solver = SARSOPSolver(precision=1e-2, max_time=max_time)
                policy = POMDPs.solve(solver, pomdp)
                solve_time = time() - start_time
                push!(standard_times, solve_time)
                println("  Standard SARSOP completed in $solve_time seconds")
            catch e
                println("  Standard SARSOP failed: $e")
                push!(standard_times, NaN)
            end
            
            # Time SlidingSARSOP initialization
            println("  Running SlidingSARSOP...")
            sliding_init_times = []
            
            try
                start_time = time()
                sliding_solver = SlidingSARSOPSolver(
                    horizon_distance = 2 * DroneRockSample.MAX_FLIGHT_DISTANCE,
                    precision = 1e-2,
                    timeout = 2.0
                )
                sliding_policy = POMDPs.solve(sliding_solver, pomdp)
                init_time = time() - start_time
                push!(sliding_init_times, init_time)
                println("  SlidingSARSOP initialization completed in $init_time seconds")
                
                # Measure the average time for local policy computation
                local_policy_times = []
                for i in 1:5
                    # Create a random initial belief
                    b0 = initialstate(pomdp)
                    
                    # Time local policy computation
                    start_time = time()
                    a = action(sliding_policy, b0)
                    local_time = time() - start_time
                    push!(local_policy_times, local_time)
                end
                
                avg_local_time = mean(local_policy_times)
                println("  Average time for local policy computation: $avg_local_time seconds")
                
                results[(map_size, rock_density)] = (
                    standard_time = mean(standard_times),
                    sliding_init_time = mean(sliding_init_times),
                    sliding_local_time = avg_local_time,
                    num_rocks = num_rocks
                )
            catch e
                println("  SlidingSARSOP failed: $e")
                results[(map_size, rock_density)] = (
                    standard_time = mean(standard_times),
                    sliding_init_time = NaN,
                    sliding_local_time = NaN,
                    num_rocks = num_rocks
                )
            end
        end
    end
    
    # Plot results
    plot_efficiency_results(results, map_sizes, rock_densities)
    
    return results
end

function plot_efficiency_results(results, map_sizes, rock_densities)
    # Plot computation time vs problem size
    p1 = plot(title="Computation Time vs Problem Size", 
              xlabel="Map Size (Length)", 
              ylabel="Time (seconds)",
              yscale=:log10,
              legend=:topleft)
    
    for rock_density in rock_densities
        # Extract data for this density
        sizes = []
        std_times = []
        slide_init_times = []
        slide_local_times = []
        
        for map_size in map_sizes
            if haskey(results, (map_size, rock_density))
                r = results[(map_size, rock_density)]
                push!(sizes, map_size)
                push!(std_times, r.standard_time)
                push!(slide_init_times, r.sliding_init_time)
                push!(slide_local_times, r.sliding_local_time)
            end
        end
        
        if !isempty(sizes)
            plot!(p1, sizes, std_times, marker=:circle, 
                 label="Standard SARSOP ($(rock_density*100)% rocks)")
            plot!(p1, sizes, slide_init_times, marker=:square, 
                 label="SlidingSARSOP Init ($(rock_density*100)% rocks)")
            plot!(p1, sizes, slide_local_times, marker=:diamond, 
                 label="SlidingSARSOP Local ($(rock_density*100)% rocks)")
        end
    end
    
    savefig(p1, "computation_time_vs_problem_size.png")
    display(p1)
    
    # Plot computation time vs number of rocks
    p2 = plot(title="Computation Time vs Number of Rocks", 
              xlabel="Number of Rocks", 
              ylabel="Time (seconds)",
              yscale=:log10,
              legend=:topleft)
    
    # Extract data
    rocks = []
    std_times = []
    slide_init_times = []
    slide_local_times = []
    
    for (key, r) in results
        push!(rocks, r.num_rocks)
        push!(std_times, r.standard_time)
        push!(slide_init_times, r.sliding_init_time)
        push!(slide_local_times, r.sliding_local_time)
    end
    
    # Sort by number of rocks
    perm = sortperm(rocks)
    rocks = rocks[perm]
    std_times = std_times[perm]
    slide_init_times = slide_init_times[perm]
    slide_local_times = slide_local_times[perm]
    
    plot!(p2, rocks, std_times, marker=:circle, label="Standard SARSOP")
    plot!(p2, rocks, slide_init_times, marker=:square, label="SlidingSARSOP Init")
    plot!(p2, rocks, slide_local_times, marker=:diamond, label="SlidingSARSOP Local")
    
    savefig(p2, "computation_time_vs_num_rocks.png")
    display(p2)
end

"""
    evaluate_solution_quality(map_size=10, num_rocks=4, num_trials=10)

Compare the solution quality of standard SARSOP vs SlidingSARSOP.
"""
function evaluate_solution_quality(map_size=10, num_rocks=4, num_trials=10, max_steps=30)
    println("Evaluating solution quality on map size $map_size with $num_rocks rocks...")
    
    # Create POMDP
    rng = MersenneTwister(123)  # Fixed seed for reproducibility
    pomdp = DroneRockSamplePOMDP(map_size, num_rocks, rng)
    
    # Solve with standard SARSOP
    println("  Solving with standard SARSOP...")
    standard_solver = SARSOPSolver(precision=1e-3, max_time=30.0)
    standard_policy = POMDPs.solve(standard_solver, pomdp)
    
    # Initialize SlidingSARSOP
    println("  Initializing SlidingSARSOP...")
    sliding_solver = SlidingSARSOPSolver(
        horizon_distance = 2 * DroneRockSample.MAX_FLIGHT_DISTANCE,
        precision = 1e-3,
        timeout = 2.0
    )
    sliding_policy = POMDPs.solve(sliding_solver, pomdp)
    
    # Compare policies
    println("  Running $num_trials trials for comparison...")
    standard_rewards = []
    sliding_rewards = []
    standard_steps = []
    sliding_steps = []
    
    for i in 1:num_trials
        # Set different seed for each trial
        trial_rng = MersenneTwister(i)
        
        # Run standard policy
        standard_hr = HistoryRecorder(max_steps=max_steps, rng=copy(trial_rng))
        standard_history = simulate(standard_hr, pomdp, standard_policy)
        push!(standard_rewards, discounted_reward(standard_history))
        push!(standard_steps, length(standard_history))
        
        # Run sliding policy
        sliding_hr = HistoryRecorder(max_steps=max_steps, rng=copy(trial_rng))
        sliding_history = simulate(sliding_hr, pomdp, sliding_policy)
        push!(sliding_rewards, discounted_reward(sliding_history))
        push!(sliding_steps, length(sliding_history))
        
        println("  Trial $i: Standard reward = $(standard_rewards[end]), Sliding reward = $(sliding_rewards[end])")
    end
    
    # Calculate statistics
    results = (
        standard_mean_reward = mean(standard_rewards),
        standard_std_reward = std(standard_rewards),
        standard_mean_steps = mean(standard_steps),
        standard_std_steps = std(standard_steps),
        sliding_mean_reward = mean(sliding_rewards),
        sliding_std_reward = std(sliding_rewards),
        sliding_mean_steps = mean(sliding_steps),
        sliding_std_steps = std(sliding_steps)
    )
    
    # Print results
    println("\nResults over $num_trials trials:")
    println("  Standard SARSOP: Mean reward = $(results.standard_mean_reward), Std = $(results.standard_std_reward)")
    println("  Standard SARSOP: Mean steps = $(results.standard_mean_steps), Std = $(results.standard_std_steps)")
    println("  Sliding SARSOP: Mean reward = $(results.sliding_mean_reward), Std = $(results.sliding_std_reward)")
    println("  Sliding SARSOP: Mean steps = $(results.sliding_mean_steps), Std = $(results.sliding_std_steps)")
    
    # Plot results
    plot_quality_results(standard_rewards, sliding_rewards, standard_steps, sliding_steps)
    
    return results, standard_rewards, sliding_rewards, standard_steps, sliding_steps
end

function plot_quality_results(standard_rewards, sliding_rewards, standard_steps, sliding_steps)
    # Plot rewards
    p1 = plot(title="Rewards Comparison", 
              xlabel="Trial", 
              ylabel="Discounted Reward",
              legend=:bottomright)
    
    plot!(p1, 1:length(standard_rewards), standard_rewards, marker=:circle, label="Standard SARSOP")
    plot!(p1, 1:length(sliding_rewards), sliding_rewards, marker=:square, label="SlidingSARSOP")
    
    # Add mean lines
    hline!(p1, [mean(standard_rewards)], linestyle=:dash, label="Standard Mean")
    hline!(p1, [mean(sliding_rewards)], linestyle=:dash, label="Sliding Mean")
    
    savefig(p1, "rewards_comparison.png")
    display(p1)
    
    # Plot steps
    p2 = plot(title="Steps to Completion", 
              xlabel="Trial", 
              ylabel="Number of Steps",
              legend=:topright)
    
    plot!(p2, 1:length(standard_steps), standard_steps, marker=:circle, label="Standard SARSOP")
    plot!(p2, 1:length(sliding_steps), sliding_steps, marker=:square, label="SlidingSARSOP")
    
    # Add mean lines
    hline!(p2, [mean(standard_steps)], linestyle=:dash, label="Standard Mean")
    hline!(p2, [mean(sliding_steps)], linestyle=:dash, label="Sliding Mean")
    
    savefig(p2, "steps_comparison.png")
    display(p2)
end

"""
    evaluate_adaptability()

Evaluate how well SlidingSARSOP adapts to changes in the environment.
"""
function evaluate_adaptability(map_size=10, num_trials=5)
    println("Evaluating adaptability...")
    
    # Initial POMDP with rocks only on the left side of the map
    left_rocks = [(2, 3), (3, 5), (4, 2)]
    initial_pomdp = DroneRockSamplePOMDP(
        map_size = (map_size, map_size),
        rocks_positions = left_rocks,
        sensor_efficiency = 20.0,
        discount_factor = 0.95,
        good_rock_reward = 20.0,
        fly_penalty = -0.2
    )
    
    # Modified POMDP with some rocks on the right side
    # This simulates a scenario where new information becomes available
    right_rocks = [(7, 7), (8, 3), (9, 8)]
    all_rocks = vcat(left_rocks, right_rocks)
    modified_pomdp = DroneRockSamplePOMDP(
        map_size = (map_size, map_size),
        rocks_positions = all_rocks,
        sensor_efficiency = 20.0,
        discount_factor = 0.95,
        good_rock_reward = 20.0,
        fly_penalty = -0.2
    )
    
    # Pre-compute standard SARSOP policies for both POMDPs
    println("  Solving standard SARSOP for initial POMDP...")
    standard_initial_solver = SARSOPSolver(precision=1e-3, max_time=30.0)
    standard_initial_policy = POMDPs.solve(standard_initial_solver, initial_pomdp)
    
    println("  Solving standard SARSOP for modified POMDP...")
    standard_modified_solver = SARSOPSolver(precision=1e-3, max_time=30.0)
    standard_modified_policy = POMDPs.solve(standard_modified_solver, modified_pomdp)
    
    # Initialize SlidingSARSOP policy (will be updated during simulation)
    println("  Initializing SlidingSARSOP...")
    sliding_solver = SlidingSARSOPSolver(
        horizon_distance = 2 * DroneRockSample.MAX_FLIGHT_DISTANCE,
        precision = 1e-3,
        timeout = 2.0
    )
    
    # Run trials
    results = []
    
    for trial in 1:num_trials
        println("  Running trial $trial/$num_trials...")
        rng = MersenneTwister(100 + trial)
        
        # Scenario 1: Standard SARSOP with initial policy only
        standard_initial_hr = HistoryRecorder(max_steps=40, rng=copy(rng))
        standard_initial_history = simulate(standard_initial_hr, modified_pomdp, standard_initial_policy)
        initial_reward = discounted_reward(standard_initial_history)
        
        # Scenario 2: Standard SARSOP with updated policy
        standard_modified_hr = HistoryRecorder(max_steps=40, rng=copy(rng))
        standard_modified_history = simulate(standard_modified_hr, modified_pomdp, standard_modified_policy)
        modified_reward = discounted_reward(standard_modified_history)
        
        # Scenario 3: SlidingSARSOP (adapts automatically)
        sliding_policy = POMDPs.solve(sliding_solver, modified_pomdp)
        sliding_hr = HistoryRecorder(max_steps=40, rng=copy(rng))
        sliding_history = simulate(sliding_hr, modified_pomdp, sliding_policy)
        sliding_reward = discounted_reward(sliding_history)
        
        println("    Initial: $initial_reward, Modified: $modified_reward, Sliding: $sliding_reward")
        
        push!(results, (
            initial_policy_reward = initial_reward,
            modified_policy_reward = modified_reward,
            sliding_policy_reward = sliding_reward,
            initial_steps = length(standard_initial_history),
            modified_steps = length(standard_modified_history),
            sliding_steps = length(sliding_history)
        ))
    end
    
    # Calculate statistics
    mean_initial_reward = mean([r.initial_policy_reward for r in results])
    mean_modified_reward = mean([r.modified_policy_reward for r in results])
    mean_sliding_reward = mean([r.sliding_policy_reward for r in results])
    
    mean_initial_steps = mean([r.initial_steps for r in results])
    mean_modified_steps = mean([r.modified_steps for r in results])
    mean_sliding_steps = mean([r.sliding_steps for r in results])
    
    println("\nAdaptability Results:")
    println("  Mean reward using initial policy: $mean_initial_reward")
    println("  Mean reward using updated policy: $mean_modified_reward")
    println("  Mean reward using SlidingSARSOP: $mean_sliding_reward")
    println("  Mean steps using initial policy: $mean_initial_steps")
    println("  Mean steps using updated policy: $mean_modified_steps")
    println("  Mean steps using SlidingSARSOP: $mean_sliding_steps")
    
    # Plot results
    plot_adaptability_results(results)
    
    return results
end

function plot_adaptability_results(results)
    # Extract data
    initial_rewards = [r.initial_policy_reward for r in results]
    modified_rewards = [r.modified_policy_reward for r in results]
    sliding_rewards = [r.sliding_policy_reward for r in results]
    
    initial_steps = [r.initial_steps for r in results]
    modified_steps = [r.modified_steps for r in results]
    sliding_steps = [r.sliding_steps for r in results]
    
    # Plot rewards
    p1 = plot(title="Adaptability: Rewards Comparison", 
              xlabel="Trial", 
              ylabel="Discounted Reward",
              legend=:bottomright)
    
    plot!(p1, 1:length(initial_rewards), initial_rewards, marker=:circle, 
          label="Initial Policy (outdated)")
    plot!(p1, 1:length(modified_rewards), modified_rewards, marker=:square, 
          label="Updated Policy (recomputed)")
    plot!(p1, 1:length(sliding_rewards), sliding_rewards, marker=:diamond, 
          label="SlidingSARSOP (adaptive)")
    
    savefig(p1, "adaptability_rewards.png")
    display(p1)
    
    # Plot steps
    p2 = plot(title="Adaptability: Steps Comparison", 
              xlabel="Trial", 
              ylabel="Number of Steps",
              legend=:topright)
    
    plot!(p2, 1:length(initial_steps), initial_steps, marker=:circle, 
          label="Initial Policy (outdated)")
    plot!(p2, 1:length(modified_steps), modified_steps, marker=:square, 
          label="Updated Policy (recomputed)")
    plot!(p2, 1:length(sliding_steps), sliding_steps, marker=:diamond, 
          label="SlidingSARSOP (adaptive)")
    
    savefig(p2, "adaptability_steps.png")
    display(p2)
    
    # Bar chart comparing means
    p3 = bar(["Initial Policy", "Updated Policy", "SlidingSARSOP"],
             [mean(initial_rewards), mean(modified_rewards), mean(sliding_rewards)],
             title="Mean Rewards by Method",
             ylabel="Mean Discounted Reward",
             legend=false)
    
    savefig(p3, "mean_rewards_by_method.png")
    display(p3)
end

# Main evaluation runner
function run_all_evaluations()
    timestamp = Dates.format(now(), "yyyy-mm-dd_HH-MM-SS")
    results_file = "sarsop_evaluation_results_$timestamp.jld2"
    
    all_results = Dict()
    
    # Evaluate computational efficiency
    println("\n=== COMPUTATIONAL EFFICIENCY EVALUATION ===")
    efficiency_results = evaluate_computational_efficiency()
    all_results["efficiency"] = efficiency_results
    
    # Evaluate solution quality
    println("\n=== SOLUTION QUALITY EVALUATION ===")
    quality_results, standard_rewards, sliding_rewards, standard_steps, sliding_steps = 
        evaluate_solution_quality()
    all_results["quality"] = quality_results
    all_results["standard_rewards"] = standard_rewards
    all_results["sliding_rewards"] = sliding_rewards
    all_results["standard_steps"] = standard_steps
    all_results["sliding_steps"] = sliding_steps
    
    # Evaluate adaptability
    println("\n=== ADAPTABILITY EVALUATION ===")
    adaptability_results = evaluate_adaptability()
    all_results["adaptability"] = adaptability_results
    
    # Save results
    save(results_file, all_results)
    println("\nResults saved to $results_file")
    
    return all_results
end

# Run a specific evaluation only
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting evaluations...")
    results = run_all_evaluations()
    println("All evaluations complete!")
end