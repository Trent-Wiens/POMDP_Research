using POMDPs
using POMDPTools
using StaticArrays
using LinearAlgebra  # For norm() function
using Printf
using DiscreteValueIteration
using Statistics     # For mean() function
using NativeSARSOP

# Include the DroneRockSample module
include("DroneRockSample.jl")
using .DroneRockSample

function iterative_sliding_resolution(pomdp, center_state, horizon, values, threshold, max_iterations=5)
    current_pomdp = pomdp
    current_values = values
    current_state = center_state
    
    println("\n=== Beginning Iterative Sliding Resolution Process ===")
    println("Maximum iterations: $max_iterations")
    println("Value difference threshold: $threshold")
    
    # Keep track of expansion history
    expansion_history = []
    
    # Iteratively apply sliding resolution until no large differences remain or max iterations reached
    for iteration in 1:max_iterations
        println("\n--- Iteration $iteration ---")
        
        # Apply sliding resolution
        expanded_pomdp, expanded_values, rev_col_map, rev_row_map, column_mapping, row_mapping, 
            columns_to_expand, rows_to_expand, max_diff = expand_grid_with_sliding_resolution(
                current_pomdp, current_state, horizon, current_values, threshold)
        
        # Record expansion information
        push!(expansion_history, (
            iteration = iteration,
            original_size = length(current_pomdp),
            expanded_size = length(expanded_pomdp),
            columns_expanded = length(columns_to_expand),
            rows_expanded = length(rows_to_expand),
            max_difference = max_diff
        ))
        
        # Check if we're done - no more large differences found
        if isempty(columns_to_expand) && isempty(rows_to_expand)
            println("✓ No more differences above threshold found. Stopping iteration.")
            break
        end
        
        # Update for next iteration
        current_pomdp = expanded_pomdp
        current_values = expanded_values
        
        # Update center state to match the expanded grid
        center_pos = current_state.pos
        new_center_pos = RSPos(column_mapping[center_pos[1]], row_mapping[center_pos[2]])
        current_state = RSState(new_center_pos, current_state.rocks)
        
        # Stop if we've reached the last iteration
        if iteration == max_iterations
            println("⚠ Reached maximum iterations. Stopping.")
        end
    end
    
    # Print summary of the iterative process
    println("\n=== Iterative Sliding Resolution Summary ===")
    println("Total iterations: $(length(expansion_history))")
    
    for (i, info) in enumerate(expansion_history)
        growth = info.expanded_size / info.original_size
        println("Iteration $i:")
        println("  Size: $(info.original_size) → $(info.expanded_size) ($(round(growth, digits=2))x)")
        println("  Expanded: $(info.columns_expanded) columns, $(info.rows_expanded) rows")
        println("  Maximum difference: $(round(info.max_difference, digits=2))")
    end
    
    return current_pomdp, current_values
end

function expand_grid_with_sliding_resolution(pomdp, center_state, horizon, values, threshold)
    println("\n=== Expanding Grid Using Sliding Resolution ===")
    
    # Step 1: Identify the horizon boundaries
    center_pos = center_state.pos
    x_min = max(1, center_pos[1] - horizon)
    x_max = min(pomdp.map_size[1], center_pos[1] + horizon)
    y_min = max(1, center_pos[2] - horizon)
    y_max = min(pomdp.map_size[2], center_pos[2] + horizon)
    
    println("Horizon boundaries: x=[$(x_min),$(x_max)], y=[$(y_min),$(y_max)]")
    
    # Step 2: Create a direct lookup from state to value
    all_states = ordered_states(pomdp)
    direct_lookup = Dict(s => values[i] for (i, s) in enumerate(all_states))
    
    # Identify all rock combinations within the horizon
    rock_states_seen = Set()
    for s in all_states
        if !isterminal(pomdp, s) && 
           x_min <= s.pos[1] <= x_max && 
           y_min <= s.pos[2] <= y_max
            push!(rock_states_seen, s.rocks)
        end
    end
    rock_combinations = collect(rock_states_seen)
    println("Number of rock combinations in horizon: $(length(rock_combinations))")
    
    # Step 3: Find columns and rows that need expansion
    columns_to_expand = Set{Int}()
    rows_to_expand = Set{Int}()
    max_difference = 0.0
    
    # First pass: collect all columns/rows that need expansion for any rock combination
    for rock_combo in rock_combinations
        # Check columns
        for x in x_min:x_max-1
            # Skip if this column is already marked for expansion
            if x in columns_to_expand
                continue
            end
            
            for y in y_min:y_max
                state1 = RSState(RSPos(x, y), rock_combo)
                state2 = RSState(RSPos(x+1, y), rock_combo)
                
                if haskey(direct_lookup, state1) && haskey(direct_lookup, state2)
                    diff = abs(direct_lookup[state1] - direct_lookup[state2])
                    max_difference = max(max_difference, diff)
                    
                    if diff > threshold
                        push!(columns_to_expand, x)
                        println("Column $x needs expansion: Large difference at position ($x,$y) with rocks $rock_combo (diff=$diff)")
                        break  # Found a difference, no need to check other positions in this column
                    end
                end
            end
        end
        
        # Check rows
        for y in y_min:y_max-1
            # Skip if this row is already marked for expansion
            if y in rows_to_expand
                continue
            end
            
            for x in x_min:x_max
                state1 = RSState(RSPos(x, y), rock_combo)
                state2 = RSState(RSPos(x, y+1), rock_combo)
                
                if haskey(direct_lookup, state1) && haskey(direct_lookup, state2)
                    diff = abs(direct_lookup[state1] - direct_lookup[state2])
                    max_difference = max(max_difference, diff)
                    
                    if diff > threshold
                        push!(rows_to_expand, y)
                        println("Row $y needs expansion: Large difference at position ($x,$y) with rocks $rock_combo (diff=$diff)")
                        break  # Found a difference, no need to check other positions in this row
                    end
                end
            end
        end
    end
    
    println("Columns to expand: $columns_to_expand")
    println("Rows to expand: $rows_to_expand")
    println("Maximum value difference found: $max_difference")
    
    # If no columns or rows need expansion, return the original POMDP
    if isempty(columns_to_expand) && isempty(rows_to_expand)
        return pomdp, values, Dict(), Dict(), Dict(x => x for x in 1:pomdp.map_size[1]), 
               Dict(y => y for y in 1:pomdp.map_size[2]), columns_to_expand, rows_to_expand, max_difference
    end
    
    # Step 4: Create mapping from original coordinates to expanded coordinates
    column_mapping = Dict{Int, Int}()  # original_x => expanded_x
    expanded_x = 1
    for x in 1:pomdp.map_size[1]
        column_mapping[x] = expanded_x
        expanded_x += 1
        if x in columns_to_expand
            expanded_x += 1  # Skip a position for the interpolated column
        end
    end
    
    row_mapping = Dict{Int, Int}()  # original_y => expanded_y
    expanded_y = 1
    for y in 1:pomdp.map_size[2]
        row_mapping[y] = expanded_y
        expanded_y += 1
        if y in rows_to_expand
            expanded_y += 1  # Skip a position for the interpolated row
        end
    end
    
    # New map dimensions
    new_width = pomdp.map_size[1] + length(columns_to_expand)
    new_height = pomdp.map_size[2] + length(rows_to_expand)
    
    println("Original map size: $(pomdp.map_size)")
    println("New map size: ($new_width, $new_height)")
    
    # Step 5: Create reverse mappings to identify which expanded positions correspond to original or interpolated
    reverse_column_map = Dict{Int, Union{Int, Tuple{Int,Int}}}()
    for (orig_x, exp_x) in column_mapping
        reverse_column_map[exp_x] = orig_x
        if orig_x in columns_to_expand
            # This is an interpolated column between orig_x and orig_x+1
            reverse_column_map[exp_x + 1] = (orig_x, orig_x + 1)
        end
    end
    
    reverse_row_map = Dict{Int, Union{Int, Tuple{Int,Int}}}()
    for (orig_y, exp_y) in row_mapping
        reverse_row_map[exp_y] = orig_y
        if orig_y in rows_to_expand
            # This is an interpolated row between orig_y and orig_y+1
            reverse_row_map[exp_y + 1] = (orig_y, orig_y + 1)
        end
    end
    
    # Step 6: Update rock positions
    new_rocks_positions = []
    for (rx, ry) in pomdp.rocks_positions
        # Find the corresponding expanded position
        new_rx = column_mapping[rx]
        new_ry = row_mapping[ry]
        push!(new_rocks_positions, (new_rx, new_ry))
    end
    
    # Step 7: Create the expanded POMDP
    expanded_pomdp = DroneRockSamplePOMDP(
        map_size = (new_width, new_height),
        rocks_positions = new_rocks_positions,
        init_pos = (column_mapping[pomdp.init_pos[1]], row_mapping[pomdp.init_pos[2]]),
        sensor_efficiency = pomdp.sensor_efficiency,
        discount_factor = pomdp.discount_factor,
        good_rock_reward = pomdp.good_rock_reward,
        fly_penalty = pomdp.fly_penalty
    )
    
    # Step 8: Initialize the expanded value function by interpolation
    expanded_states = ordered_states(expanded_pomdp)
    expanded_values = zeros(length(expanded_states))
    
    # Helper function to check if a position is interpolated
    function is_interpolated(ex, ey)
        x_interp = haskey(reverse_column_map, ex) && isa(reverse_column_map[ex], Tuple)
        y_interp = haskey(reverse_row_map, ey) && isa(reverse_row_map[ey], Tuple)
        return x_interp || y_interp
    end
    
    # Helper function to calculate an interpolated value
    function calculate_interpolated_value(ex, ey, rocks)
        if haskey(reverse_column_map, ex) && isa(reverse_column_map[ex], Tuple)
            # This is an interpolated column
            x1, x2 = reverse_column_map[ex]
            
            if haskey(reverse_row_map, ey) && isa(reverse_row_map[ey], Integer)
                # Only x is interpolated
                y = reverse_row_map[ey]
                state1 = RSState(RSPos(x1, y), rocks)
                state2 = RSState(RSPos(x2, y), rocks)
                
                if haskey(direct_lookup, state1) && haskey(direct_lookup, state2)
                    return (direct_lookup[state1] + direct_lookup[state2]) / 2
                end
            else
                # Both x and y are interpolated
                y1, y2 = reverse_row_map[ey]
                corner_states = [
                    RSState(RSPos(x1, y1), rocks),
                    RSState(RSPos(x2, y1), rocks),
                    RSState(RSPos(x1, y2), rocks),
                    RSState(RSPos(x2, y2), rocks)
                ]
                
                valid_values = [direct_lookup[s] for s in corner_states if haskey(direct_lookup, s)]
                if !isempty(valid_values)
                    return mean(valid_values)
                end
            end
        elseif haskey(reverse_row_map, ey) && isa(reverse_row_map[ey], Tuple)
            # Only y is interpolated
            y1, y2 = reverse_row_map[ey]
            x = reverse_column_map[ex]
            
            state1 = RSState(RSPos(x, y1), rocks)
            state2 = RSState(RSPos(x, y2), rocks)
            
            if haskey(direct_lookup, state1) && haskey(direct_lookup, state2)
                return (direct_lookup[state1] + direct_lookup[state2]) / 2
            end
        end
        
        # Default: no valid interpolation possible
        return 0.0
    end
    
    # Fill the expanded values
    interpolated_count = 0
    for (i, s) in enumerate(expanded_states)
        if isterminal(expanded_pomdp, s)
            expanded_values[i] = 0.0
            continue
        end
        
        ex, ey = s.pos
        rocks = s.rocks
        
        if is_interpolated(ex, ey)
            # This is an interpolated state
            expanded_values[i] = calculate_interpolated_value(ex, ey, rocks)
            interpolated_count += 1
        else
            # This is an original state
            if haskey(reverse_column_map, ex) && isa(reverse_column_map[ex], Integer) &&
               haskey(reverse_row_map, ey) && isa(reverse_row_map[ey], Integer)
                # Get the original coordinates
                ox = reverse_column_map[ex]
                oy = reverse_row_map[ey]
                
                # Find the original state
                orig_state = RSState(RSPos(ox, oy), rocks)
                if haskey(direct_lookup, orig_state)
                    expanded_values[i] = direct_lookup[orig_state]
                end
            end
        end
    end
    
    println("Number of interpolated states: $interpolated_count")
    
    # Return the expanded POMDP and related information
    return expanded_pomdp, expanded_values, reverse_column_map, reverse_row_map, column_mapping, 
           row_mapping, columns_to_expand, rows_to_expand, max_difference
end

function solve_expanded_pomdp(expanded_pomdp, expanded_values=nothing)
    # Solve the expanded POMDP using SARSOP
    println("Solving expanded POMDP with SARSOP...")
    
    # Configure the SARSOP solver with the correct parameter names
    solver = SARSOPSolver(
        precision=0.1,        # Target precision
        max_time=30.0,        # Maximum time in seconds (not timeout)
        verbose=true          # Print progress
    )
    
    # Solve the POMDP
    policy = solve(solver, expanded_pomdp)
    
    println("POMDP solved!")
    
    return policy
end

function get_action_for_expanded_state(policy, state)
    # Get the action for a specific state in the expanded POMDP
    action = POMDPs.action(policy, state)
    return action
end

function get_action_for_position_and_rocks(policy, expanded_pomdp, position, rocks)
    # Create a state for the given position and rock states
    state = RSState(RSPos(position...), rocks)
    
    # Create a deterministic belief centered on this state
    belief = SparseCat([state], [1.0])
    
    # Get the action for this belief
    action = POMDPs.action(policy, belief)
    
    return action
end



# Usage example
function test_iterative_refinement()
    # Create a DroneRockSample POMDP instance
    pomdp = DroneRockSamplePOMDP(
        map_size = (10, 10),
        rocks_positions = [(2, 8), (4, 5), (7, 2), (8, 9)],
        sensor_efficiency = 20.0,
        discount_factor = 0.95,
        good_rock_reward = 20.0,
        fly_penalty = -0.2
    )

    # Solve the full POMDP to get initial values
    solver = ValueIterationSolver(max_iterations=100, belres=1e-3)
    mdp = UnderlyingMDP(pomdp)
    policy = solve(solver, mdp)
    values = policy.util
    
    # Set center state and threshold
    center_pos = RSPos(5, 5)
    center_state = RSState(center_pos, SVector{4, Bool}(false, false, false, false))
    threshold = 3.0
    horizon = 3
    
    # Run the iterative sliding resolution process
    final_pomdp, final_values = iterative_sliding_resolution(
        pomdp, center_state, horizon, values, threshold, 3)
    # Solve it with SARSOP
    policy = solve_expanded_pomdp(final_pomdp)

    # Get an action for a specific position and rock states
    current_position = (6, 7)  # Position in the expanded grid
    current_rocks = SVector{4, Bool}(true, false, true, false)  # Rock states
    action = get_action_for_position_and_rocks(policy, final_pomdp, current_position, current_rocks)

    println("\nAction for position $current_position with rocks $current_rocks: $action")

    
    # Print final state space size
    println("\nOriginal POMDP size: $(length(pomdp))")
    println("Final POMDP size: $(length(final_pomdp))")
    println("Size increase: $(round(length(final_pomdp) / length(pomdp), digits=2))x")
    
    return final_pomdp, final_values
end

test_iterative_refinement()
