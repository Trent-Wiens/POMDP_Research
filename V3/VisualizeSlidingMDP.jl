using POMDPs
using POMDPTools
using Plots
using Compose

# Include our Sliding MDP implementation
include("SlidingDroneRockSample.jl")
using .SlidingDroneRockSample
import .DroneRockSample: RSPos, RSState, DroneRockSamplePOMDP

"""
    visualize_smdp_state(smdp, step_results)

Create a visualization of the SMDP state at each step.
"""
function visualize_smdp_state(smdp, step_results)
    for (i, step) in enumerate(step_results)
        # Draw the grid
        plot_grid = plot_smdp_grid(step.grid_refinement, step.state, smdp.current_pomdp)
        
        # Save plot
        savefig(plot_grid, "smdp_grid_step_$i.png")
    end
    
    println("Saved grid visualizations to smdp_grid_step_*.png")
end

"""
    plot_smdp_grid(grid_refinement, current_state, pomdp)

Plot the grid with refinement levels and drone position.
"""
function plot_smdp_grid(grid_refinement, current_state, pomdp)
    rows, cols = size(grid_refinement)
    
    # Create a heatmap of refinement levels
    p = heatmap(
        1:cols, 1:rows, 
        grid_refinement', 
        c=:viridis,
        colorbar_title="Refinement Level",
        axis=true,
        framestyle=:box,
        xticks=1:cols,
        yticks=1:rows,
        title="Grid Resolution",
        xlabel="X",
        ylabel="Y",
        aspect_ratio=:equal,
        legend=true
    )
    
    # Mark the drone position
    drone_x, drone_y = current_state.pos
    scatter!(
        [drone_x], [drone_y],
        marker=:star,
        markersize=12,
        color=:red,
        label="Drone"
    )
    
    # Mark rock positions
    for (i, rock_pos) in enumerate(pomdp.rocks_positions)
        rx, ry = rock_pos
        rock_label = (i == 1) ? "Rocks" : ""
        rock_color = current_state.rocks[i] ? :green : :darkred
        scatter!(
            [rx], [ry],
            marker=:diamond,
            markersize=10,
            color=rock_color,
            label=rock_label
        )
        annotate!(rx, ry, text("R$i", 8, :white))
    end
    
    # Mark the exit area
    exit_x = cols + 1
    exit_y = rows ÷ 2
    annotate!(cols + 0.5, exit_y, text("EXIT", 10, :red))
    
    return p
end

"""
    create_animation(smdp, step_results)

Create an animation of the drone's journey through the environment.
"""
function create_animation(smdp, step_results)
    # Create animation
    anim = @animate for (i, step) in enumerate(step_results)
        plot_smdp_grid(step.grid_refinement, step.state, smdp.current_pomdp)
    end
    
    # Save animation
    gif(anim, "smdp_simulation.gif", fps=2)
    println("Saved animation to smdp_simulation.gif")
end

"""
    visualize_value_function(smdp)

Visualize the value function for each rock configuration.
"""
function visualize_value_function(smdp)
    # Get all possible rock configurations
    rocks_count = length(smdp.current_pomdp.rocks_positions)
    rock_configs = collect(Iterators.product(ntuple(x->[false, true], rocks_count)...))
    
    for rock_config in rock_configs
        # Create a matrix to hold values
        rows, cols = smdp.current_pomdp.map_size
        value_grid = zeros(rows, cols)
        
        # Fill in values
        for x in 1:rows
            for y in 1:cols
                state = RSState(RSPos(x, y), SVector{rocks_count, Bool}(rock_config))
                if !haskey(smdp.policy.stateindexes, state)
                    # Skip if state is not in the policy
                    continue
                end
                state_idx = smdp.policy.stateindexes[state]
                value_grid[x, y] = smdp.value_function[state_idx]
            end
        end
        
        # Create the plot
        p = heatmap(
            1:cols, 1:rows,
            value_grid',
            c=:plasma,
            colorbar_title="Value",
            axis=true,
            framestyle=:box,
            title="Value Function - Rocks: $rock_config",
            xlabel="X",
            ylabel="Y",
            aspect_ratio=:equal
        )
        
        # Save the plot
        rock_str = join(Int.(rock_config), "")
        savefig(p, "value_function_rocks_$rock_str.png")
    end
    
    println("Saved value function visualizations to value_function_rocks_*.png")
end

"""
    run_visualization(results_file)

Run all visualization functions on a saved results file.
"""
function run_visualization(smdp, step_results)
    # Visualize individual steps
    visualize_smdp_state(smdp, step_results)
    
    # Create animation
    create_animation(smdp, step_results)
    
    # Visualize value function
    visualize_value_function(smdp)
end