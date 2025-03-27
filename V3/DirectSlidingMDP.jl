# DirectSlidingMDP.jl
# A direct, all-in-one implementation of the Sliding MDP concept
# This avoids module export issues by keeping everything in a single file

using POMDPs
using POMDPTools
using DiscreteValueIteration
using StaticArrays
using Random
using LinearAlgebra
using Plots
using Statistics

# Include DroneRockSample module
include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: RSPos, RSState, DroneRockSamplePOMDP

# Set random seed for reproducibility
Random.seed!(42)

"""
A struct to hold the Sliding MDP state
"""
mutable struct SlidingMDPState
    # The current POMDP 
    pomdp::DroneRockSamplePOMDP
    
    # Current drone state
    current_state::RSState
    
    # Value function
    value_function::Vector{Float64}
    
    # Policy
    policy::ValueIterationPolicy
    
    # Parameters
    splitting_param::Float64
    
    # History of grid refinements
    refinement_history::Vector{Matrix{Int}}
    
    # Simulation steps history
    steps::Vector{NamedTuple}
end

"""
Create and initialize a Sliding MDP
"""
function initialize_sliding_mdp(
    map_size::Tuple{Int,Int}, 
    rocks_positions::Vector{Tuple{Int,Int}};
    init_pos::Tuple{Int,Int}=(1,1),
    sensor_efficiency::Float64=20.0,
    discount_factor::Float64=0.95,
    good_rock_reward::Float64=10.0,
    fly_penalty::Float64=-0.2,
    splitting_param::Float64=2.0
)
    # Create the POMDP
    pomdp = DroneRockSamplePOMDP(
        map_size=map_size,
        rocks_positions=rocks_positions,
        init_pos=RSPos(init_pos...),
        sensor_efficiency=sensor_efficiency,
        discount_factor=discount_factor,
        good_rock_reward=good_rock_reward,
        fly_penalty=fly_penalty
    )
    
    # Create the initial state (with all rocks being good)
    rock_count = length(rocks_positions)
    current_state = RSState(RSPos(init_pos...), SVector{rock_count, Bool}(fill(true, rock_count)))
    
    # Solve the initial POMDP
    solver = ValueIterationSolver(max_iterations=1000, belres=1e-6)
    policy = solve(solver, UnderlyingMDP(pomdp))
    
    # Create the refinement history
    refinement_history = [zeros(Int, map_size)]
    
    # Return the Sliding MDP state
    return SlidingMDPState(
        pomdp,
        current_state,
        policy.util,
        policy,
        splitting_param,
        refinement_history,
        []
    )
end

"""
Find neighboring states to the current state
"""
function find_neighbors(smdp::SlidingMDPState)
    neighbors = []
    
    # Check all adjacent grid cells
    for dx in -1:1
        for dy in -1:1
            if dx == 0 && dy == 0
                continue  # Skip self
            end
            
            # Check if neighbor is valid
            new_x = smdp.current_state.pos[1] + dx
            new_y = smdp.current_state.pos[2] + dy
            
            if new_x >= 1 && new_x <= smdp.pomdp.map_size[1] &&
               new_y >= 1 && new_y <= smdp.pomdp.map_size[2]
                # Create the neighbor state with same rock configuration
                neighbor_state = RSState(RSPos(new_x, new_y), smdp.current_state.rocks)
                
                # Get its value
                neighbor_idx = POMDPs.stateindex(smdp.pomdp, neighbor_state)
                neighbor_value = smdp.value_function[neighbor_idx]
                
                push!(neighbors, (state=neighbor_state, value=neighbor_value))
            end
        end
    end
    
    return neighbors
end

"""
Check for large value differences and perform grid refinement
"""
function refine_grid(smdp::SlidingMDPState)
    # Get the current state's value
    current_idx = POMDPs.stateindex(smdp.pomdp, smdp.current_state)
    current_value = smdp.value_function[current_idx]
    
    # Find neighbors
    neighbors = find_neighbors(smdp)
    
    # Check for large value differences
    large_diffs = []
    for neighbor in neighbors
        diff = abs(current_value - neighbor.value)
        if diff > smdp.splitting_param
            push!(large_diffs, (state=neighbor.state, diff=diff))
        end
    end
    
    # If large differences found, refine the grid
    if !isempty(large_diffs)
        println("Found $(length(large_diffs)) neighbors with large value differences")
        
        # In a full implementation, we would:
        # 1. Create a new, finer grid
        # 2. Add new states between the current state and neighbors with large diffs
        # 3. Recompute the transition, reward, and value functions
        
        # For this simplified version, we'll just mark the refinement
        new_refinement = copy(smdp.refinement_history[end])
        
        # Mark current position and positions with large diffs as refined
        x, y = smdp.current_state.pos
        new_refinement[x, y] += 1
        
        for neighbor in large_diffs
            nx, ny = neighbor.state.pos
            new_refinement[nx, ny] += 1
        end
        
        # Add to history
        push!(smdp.refinement_history, new_refinement)
        
        return true
    else
        # No refinement needed, just copy the last refinement
        push!(smdp.refinement_history, copy(smdp.refinement_history[end]))
        return false
    end
end

"""
Take a step in the simulation
"""
function take_step(smdp::SlidingMDPState)
    # 1. Check for grid refinement
    println("\n--- Step $(length(smdp.steps) + 1) ---")
    println("Current state: Position=$(smdp.current_state.pos), Rocks=$(smdp.current_state.rocks)")
    
    refined = refine_grid(smdp)
    
    # 2. Get best action from policy
    best_action = action(smdp.policy, smdp.current_state)
    action_str = action_to_string(smdp.pomdp, best_action)
    println("Taking action: $action_str")
    
    # 3. Execute action
    next_state = rand(transition(smdp.pomdp, smdp.current_state, best_action))
    r = reward(smdp.pomdp, smdp.current_state, best_action)
    
    # 4. Record step
    push!(smdp.steps, (
        state = smdp.current_state,
        action = best_action,
        action_str = action_str,
        next_state = next_state,
        reward = r,
        refined = refined
    ))
    
    # 5. Update current state
    smdp.current_state = next_state
    
    # Check if terminal
    if isterminal(smdp.pomdp, smdp.current_state)
        println("Reached terminal state!")
        return true
    end
    
    return false
end

"""
Run a full simulation
"""
function run_simulation(smdp::SlidingMDPState, max_steps::Int=15)
    total_reward = 0.0
    discount = 1.0
    
    for step in 1:max_steps
        # Take a step
        terminal = take_step(smdp)
        
        # Update total reward
        r = smdp.steps[end].reward
        total_reward += discount * r
        discount *= smdp.pomdp.discount_factor
        
        # Stop if terminal
        if terminal
            break
        end
    end
    
    println("\nSimulation complete!")
    println("Total reward: $total_reward")
    println("Total steps: $(length(smdp.steps))")
    
    return total_reward
end

"""
Visualize grid refinement
"""
function visualize_refinement(smdp::SlidingMDPState)
    for (i, grid) in enumerate(smdp.refinement_history)
        # Create the plot
        p = heatmap(grid, 
                title="Grid Refinement - Step $i",
                c=:viridis, 
                colorbar_title="Refinement Level",
                axis=true,
                xticks=1:smdp.pomdp.map_size[1],
                yticks=1:smdp.pomdp.map_size[2],
                aspect_ratio=:equal)
        
        # Mark the drone position if we have step data
        if i <= length(smdp.steps)
            drone_pos = smdp.steps[i].state.pos
            scatter!([drone_pos[1]], [drone_pos[2]], 
                     marker=:star, 
                     markersize=10,
                     color=:red,
                     label="Drone")
            
            # Add action annotation
            action_str = smdp.steps[i].action_str
            annotate!(drone_pos[1], drone_pos[2] + 0.3, text(action_str, 6, :red))
        end
        
        # Mark rock positions
        for (j, rock_pos) in enumerate(smdp.pomdp.rocks_positions)
            rx, ry = rock_pos
            rock_color = i <= length(smdp.steps) && smdp.steps[i].state.rocks[j] ? :green : :darkred
            scatter!([rx], [ry],
                    marker=:diamond,
                    markersize=8,
                    color=rock_color,
                    label=(j==1 ? "Rocks" : ""))
            annotate!(rx, ry, text("R$j", 8, :white))
        end
        
        # Save the plot
        savefig(p, "direct_grid_refinement_$i.png")
    end
    
    println("Saved $(length(smdp.refinement_history)) visualization images")
    
    # Create animation
    anim = @animate for i in 1:length(smdp.refinement_history)
        grid = smdp.refinement_history[i]
        p = heatmap(grid, 
                title="Grid Refinement Animation",
                c=:viridis, 
                colorbar_title="Refinement Level",
                axis=true,
                xticks=1:smdp.pomdp.map_size[1],
                yticks=1:smdp.pomdp.map_size[2],
                aspect_ratio=:equal)
        
        # Mark the drone position
        if i <= length(smdp.steps)
            drone_pos = smdp.steps[i].state.pos
            scatter!([drone_pos[1]], [drone_pos[2]], 
                     marker=:star, 
                     markersize=10,
                     color=:red,
                     label="Drone")
        end
        
        # Mark rock positions
        for (j, rock_pos) in enumerate(smdp.pomdp.rocks_positions)
            rx, ry = rock_pos
            rock_color = i <= length(smdp.steps) && smdp.steps[i].state.rocks[j] ? :green : :darkred
            scatter!([rx], [ry],
                    marker=:diamond,
                    markersize=8,
                    color=rock_color,
                    label=(j==1 ? "Rocks" : ""))
        end
    end
    
    gif(anim, "grid_refinement_animation.gif", fps=2)
    println("Saved animation to grid_refinement_animation.gif")
end

"""
Run the complete Sliding MDP example
"""
function main()
    # Create the Sliding MDP
    println("Initializing Sliding MDP...")
    smdp = initialize_sliding_mdp(
        (8, 8),                  # Map size
        [(2, 6), (4, 3), (6, 5)], # Rock positions
        init_pos=(1, 1),
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=20.0,
        fly_penalty=-0.2,
        splitting_param=2.0
    )
    
    # Run simulation
    println("Running simulation...")
    total_reward = run_simulation(smdp, 20)
    
    # Visualize results
    println("Creating visualizations...")
    visualize_refinement(smdp)
    
    return smdp
end

# Run the main function
smdp = main()
println("Simulation completed successfully!")