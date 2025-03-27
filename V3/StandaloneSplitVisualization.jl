# CompleteSlidingMDP.jl
# A complete standalone implementation with all required functions defined locally

using POMDPs
using POMDPTools
using DiscreteValueIteration
using StaticArrays
using Random
using LinearAlgebra
using Plots
using Statistics

# Set random seed for reproducibility
Random.seed!(42)

# Include DroneRockSample module just for types, not functions
include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: RSPos, RSState, DroneRockSamplePOMDP

# Constants and functions needed for action handling
# These are copied from drone-actions.jl to avoid module dependencies
const MAX_FLIGHT_DISTANCE = 3
const ACTION_SAMPLE = 1
const ACTION_FLY_START = 2

# Calculate number of possible fly actions
function count_fly_actions(max_distance)
    count = 0
    for dx in -max_distance:max_distance
        for dy in -max_distance:max_distance
            if dx == 0 && dy == 0
                continue  # Skip staying in place
            end
            if abs(dx) + abs(dy) <= max_distance  # Manhattan distance
                count += 1
            end
        end
    end
    return count
end

const N_FLY_ACTIONS = count_fly_actions(MAX_FLIGHT_DISTANCE)
const N_BASIC_ACTIONS = 1 + N_FLY_ACTIONS  # Sample + Fly actions

# Generate all possible flight directions
function generate_flight_dirs()
    dirs = RSPos[]
    for dx in -MAX_FLIGHT_DISTANCE:MAX_FLIGHT_DISTANCE
        for dy in -MAX_FLIGHT_DISTANCE:MAX_FLIGHT_DISTANCE
            if dx == 0 && dy == 0
                continue  # Skip staying in place
            end
            if abs(dx) + abs(dy) <= MAX_FLIGHT_DISTANCE  # Manhattan distance
                push!(dirs, RSPos(dx, dy))
            end
        end
    end
    return dirs
end

const FLIGHT_DIRS = generate_flight_dirs()

# Action helper functions
function action_to_direction(a::Int)
    if a == ACTION_SAMPLE
        return RSPos(0, 0)  # No movement for sampling
    elseif a >= ACTION_FLY_START && a < ACTION_FLY_START + N_FLY_ACTIONS
        return FLIGHT_DIRS[a - ACTION_FLY_START + 1]
    else
        return RSPos(0, 0)  # No movement for sensing
    end
end

function is_fly_action(a::Int)
    return a >= ACTION_FLY_START && a < ACTION_FLY_START + N_FLY_ACTIONS
end

function is_sense_action(a::Int, K::Int)
    return a >= N_BASIC_ACTIONS+1 && a <= N_BASIC_ACTIONS+K
end

function action_to_string(pomdp::DroneRockSamplePOMDP, a::Int)
    if a == ACTION_SAMPLE
        return "Sample"
    elseif is_fly_action(a)
        dir = action_to_direction(a)
        return "Fly($(dir[1]),$(dir[2]))"
    elseif is_sense_action(a, length(pomdp.rocks_positions))
        rock_ind = a - N_BASIC_ACTIONS
        return "Sense Rock $rock_ind"
    else
        return "Unknown($a)"
    end
end

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
    create_split_grid(refinement_matrix)

Convert a refinement level matrix into an actual split grid structure.
This creates a representation where we can see the actual cell divisions.
"""
function create_split_grid(refinement_matrix)
    rows, cols = size(refinement_matrix)
    
    # Start with a basic grid structure
    # Each entry contains: [x_start, y_start, width, height]
    grid_cells = []
    
    # Initialize with the base grid cells
    for r in 1:rows
        for c in 1:cols
            # Base cell coordinates (x, y, width, height)
            push!(grid_cells, [c, r, 1, 1, refinement_matrix[r, c]])
        end
    end
    
    # Now refine the grid based on refinement levels
    refined_cells = []
    
    for cell in grid_cells
        x, y, w, h, refine_level = cell
        
        if refine_level == 0
            # No refinement, keep the cell as is
            push!(refined_cells, cell)
        else
            # Split the cell based on refinement level
            new_cells = split_cell(x, y, w, h, refine_level)
            append!(refined_cells, new_cells)
        end
    end
    
    return refined_cells
end

"""
    split_cell(x, y, width, height, level)

Split a single cell into subcells based on refinement level.
"""
function split_cell(x, y, width, height, level)
    if level == 0
        # Base case: no more splitting
        return [[x, y, width, height, 0]]
    end
    
    # Otherwise, split the cell into 4 equal parts
    subcells = []
    
    # Halve the width and height
    new_width = width / 2
    new_height = height / 2
    
    # Top-left subcell
    push!(subcells, [x, y, new_width, new_height, level-1])
    
    # Top-right subcell
    push!(subcells, [x + new_width, y, new_width, new_height, level-1])
    
    # Bottom-left subcell
    push!(subcells, [x, y + new_height, new_width, new_height, level-1])
    
    # Bottom-right subcell
    push!(subcells, [x + new_width, y + new_height, new_width, new_height, level-1])
    
    # Recursively split each subcell
    result = []
    for subcell in subcells
        append!(result, split_cell(subcell[1], subcell[2], subcell[3], subcell[4], subcell[5]))
    end
    
    return result
end

"""
    draw_split_grid(cells, map_size, drone_pos, rocks_positions, rocks_status)

Draw a grid with split cells to visualize refinement.
"""
function draw_split_grid(cells, map_size, drone_pos, rocks_positions, rocks_status)
    rows, cols = map_size
    
    # Create a blank plot with the right dimensions
    p = plot(
        xlim=(0.5, cols+0.5),
        ylim=(0.5, rows+0.5),
        aspect_ratio=:equal,
        legend=true,
        title="Grid with Cell Splitting",
        xlabel="X",
        ylabel="Y",
        size=(800, 800),
        dpi=150
    )
    
    # Draw the split cells
    for cell in cells
        x, y, w, h, _ = cell
        
        # Draw cell borders
        plot!(
            [x, x+w, x+w, x, x],
            [y, y, y+h, y+h, y],
            color=:black,
            linewidth=1,
            label=false
        )
    end
    
    # Mark the drone position
    scatter!(
        [drone_pos[1]],
        [drone_pos[2]],
        marker=:star,
        markersize=10,
        color=:red,
        label="Drone"
    )
    
    # Mark rock positions
    for (i, (rx, ry)) in enumerate(rocks_positions)
        rock_color = rocks_status[i] ? :green : :darkred
        rock_label = i == 1 ? "Rocks" : false
        
        scatter!(
            [rx],
            [ry],
            marker=:diamond,
            markersize=8,
            color=rock_color,
            label=rock_label
        )
        
        annotate!(rx, ry, text("R$i", 8, :white))
    end
    
    return p
end

"""
    visualize_split_grid(smdp)

Create visualizations showing explicit cell splitting.
"""
function visualize_split_grid(smdp)
    for (i, grid) in enumerate(smdp.refinement_history)
        # Convert refinement levels to actual split cells
        cells = create_split_grid(grid)
        
        # Get drone and rock information for this step
        drone_pos = i <= length(smdp.steps) ? smdp.steps[i].state.pos : RSPos(1, 1)
        rocks_status = i <= length(smdp.steps) ? smdp.steps[i].state.rocks : fill(true, length(smdp.pomdp.rocks_positions))
        
        # Draw the grid
        p = draw_split_grid(
            cells,
            smdp.pomdp.map_size,
            drone_pos,
            smdp.pomdp.rocks_positions,
            rocks_status
        )
        
        # Add step information
        if i <= length(smdp.steps)
            action_str = smdp.steps[i].action_str
            
            # Highlight sensing actions more prominently
            if startswith(action_str, "Sense")
                # Draw a line from drone to sensed rock
                rock_ind = parse(Int, split(action_str, " ")[3])
                rock_pos = smdp.pomdp.rocks_positions[rock_ind]
                
                plot!(
                    [drone_pos[1], rock_pos[1]],
                    [drone_pos[2], rock_pos[2]],
                    linestyle=:dash,
                    linewidth=2,
                    color=:orange,
                    label=false
                )
                
                # Add action annotation with different color
                annotate!(drone_pos[1], drone_pos[2] + 0.3, text(action_str, 8, :orange))
            else
                # Regular action annotation
                annotate!(drone_pos[1], drone_pos[2] + 0.3, text(action_str, 8, :red))
            end
        end
        
        # Save the plot
        savefig(p, "grid_split_step_$i.png")
    end
    
    println("Saved split grid visualizations to grid_split_step_*.png")
    
    # Create animation
    anim = @animate for i in 1:length(smdp.refinement_history)
        grid = smdp.refinement_history[i]
        cells = create_split_grid(grid)
        
        # Get drone and rock information for this step
        drone_pos = i <= length(smdp.steps) ? smdp.steps[i].state.pos : RSPos(1, 1)
        rocks_status = i <= length(smdp.steps) ? smdp.steps[i].state.rocks : fill(true, length(smdp.pomdp.rocks_positions))
        
        # Create plot
        p = draw_split_grid(
            cells,
            smdp.pomdp.map_size,
            drone_pos,
            smdp.pomdp.rocks_positions,
            rocks_status
        )
        
        # Add step information
        if i <= length(smdp.steps)
            action_str = smdp.steps[i].action_str
            
            # Highlight sensing actions more prominently
            if startswith(action_str, "Sense")
                # Draw a line from drone to sensed rock
                rock_ind = parse(Int, split(action_str, " ")[3])
                rock_pos = smdp.pomdp.rocks_positions[rock_ind]
                
                plot!(
                    [drone_pos[1], rock_pos[1]],
                    [drone_pos[2], rock_pos[2]],
                    linestyle=:dash,
                    linewidth=2,
                    color=:orange,
                    label=false
                )
            end
            
            # Add step number and action text
            title!("Step $i: $action_str")
        end
    end
    
    gif(anim, "grid_split_animation.gif", fps=2)
    println("Saved animation to grid_split_animation.gif")
end

"""
Run the main simulation and visualization
"""
function main()
    # Create the Sliding MDP
    println("Initializing Sliding MDP...")
    smdp = initialize_sliding_mdp(
        (6, 6),                # Map size (smaller for clearer visualization)
        [(2, 4), (4, 2)],      # Rock positions
        init_pos=(1, 1),
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=20.0,
        fly_penalty=-0.2,
        splitting_param=2.0
    )
    
    # Run simulation
    println("Running simulation...")
    run_simulation(smdp, 15)
    
    # Visualize the results
    println("Creating split grid visualizations...")
    visualize_split_grid(smdp)
    
    return smdp
end

# Run the main function
smdp = main()
println("Simulation completed successfully!")