using POMDPs
using POMDPTools
using Random
using Plots

# Include our Sliding MDP implementation
include("SlidingDroneRockSample.jl")
using .SlidingDroneRockSample

# Set random seed for reproducibility
Random.seed!(123)

function run_test()
    println("Creating Sliding Drone Rock Sample POMDP...")
    
    # Create a small test problem
    # Make sure to use the fully qualified function name
    smdp = SlidingDroneRockSample.create_sliding_drone_rock_sample(
        (5, 5),               # Map size
        [(2, 4), (4, 2)],     # Rock positions
        init_pos=(1, 1),      # Initial drone position
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=20.0,
        fly_penalty=-0.2,
        horizon_limit=0.7,    # Probability threshold for horizon
        splitting_param=2.0   # Value difference threshold for splitting
    )
    
    println("Initial POMDP created with map size: $(smdp.current_pomdp.map_size)")
    println("Initial grid refinement:")
    display(smdp.grid_refinement)
    
    # Run the simulation
    results, total_reward = SlidingDroneRockSample.run_sliding_pomdp_simulation(smdp, 15)
    
    # Print summary
    println("\n----- Simulation Summary -----")
    println("Total reward: $total_reward")
    println("Steps taken: $(length(results))")
    println("Final map size: $(results[end].map_size)")
    
    # Visualize the grid refinement over time
    visualize_grid_refinement(results)
    
    return results, smdp
end

function visualize_grid_refinement(results)
    n_steps = length(results)
    
    # Create a plot for each step
    for (i, step) in enumerate(results)
        # Create a heatmap of the grid refinement
        refinement = step.grid_refinement
        p = heatmap(refinement, 
                    title="Grid Refinement - Step $i",
                    c=:viridis, 
                    colorbar_title="Refinement Level",
                    axis=false,
                    aspect_ratio=:equal)
        
        # Mark the drone position
        drone_pos = step.state.pos
        scatter!([drone_pos[2]], [drone_pos[1]], 
                 marker=:circle, 
                 markersize=10,
                 color=:red,
                 label="Drone")
        
        # Mark rock positions
        for (j, rock_pos) in enumerate(results[1].state.rocks)
            if rock_pos
                annotate!(j*2, j*2, text("R$j", :white, :center, 8))
            end
        end
        
        # Save the plot
        savefig(p, "grid_refinement_step_$i.png")
    end
    
    println("Visualizations saved to files: grid_refinement_step_*.png")
end

# Run the test
results, smdp = run_test()