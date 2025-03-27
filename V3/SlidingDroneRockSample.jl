



module SlidingDroneRockSample

using POMDPs
using POMDPTools
using DiscreteValueIteration
using StaticArrays
using Random
using LinearAlgebra
using Parameters
using Combinatorics

# Import DroneRockSample module and its types
include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: RSPos, RSState, DroneRockSamplePOMDP
import .DroneRockSample: ACTION_SAMPLE, is_fly_action, is_sense_action, N_BASIC_ACTIONS, action_to_direction, action_to_string

export SlidingDroneRockSamplePOMDP, solve_sliding_pomdp, run_sliding_pomdp_simulation

"""
A wrapper around DroneRockSamplePOMDP that implements the Sliding MDP approach.
This allows for dynamic resolution adjustment based on value differences.
"""
mutable struct SlidingDroneRockSamplePOMDP{K}
    # The base POMDP with coarse discretization
    base_pomdp::DroneRockSamplePOMDP{K}
    
    # Current refined POMDP (changes as drone moves)
    current_pomdp::DroneRockSamplePOMDP{K}
    
    # Value function for the current POMDP
    value_function::Vector{Float64}
    
    # Current policy
    policy::ValueIterationPolicy
    
    # Parameters for the sliding approach
    horizon_limit::Float64  # Probability threshold for states in horizon
    splitting_param::Float64  # Value difference threshold for splitting
    
    # Current drone state
    current_state::RSState{K}
    
    # Mapping between original rock positions and refined positions
    rock_position_mapping::Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}
    
    # Refinement level tracker (how many times we've split each cell)
    grid_refinement::Matrix{Int}
end

"""
    create_sliding_drone_rock_sample(map_size, rocks_positions; kwargs...)

Create a SlidingDroneRockSamplePOMDP with the given parameters.
"""
function create_sliding_drone_rock_sample(
    map_size::Tuple{Int,Int}, 
    rocks_positions::Vector{Tuple{Int,Int}};
    init_pos::Tuple{Int,Int}=(1,1),
    sensor_efficiency::Float64=20.0,
    discount_factor::Float64=0.95,
    good_rock_reward::Float64=10.0,
    fly_penalty::Float64=-0.2,
    horizon_limit::Float64=0.7,  # Default probability threshold for horizon
    splitting_param::Float64=2.0  # Default value difference threshold for splitting
)
    # Create the base POMDP with coarse discretization
    base_pomdp = DroneRockSamplePOMDP(
        map_size=map_size,
        rocks_positions=rocks_positions,
        init_pos=RSPos(init_pos...),
        sensor_efficiency=sensor_efficiency,
        discount_factor=discount_factor,
        good_rock_reward=good_rock_reward,
        fly_penalty=fly_penalty
    )
    
    # Initially, the current POMDP is identical to the base POMDP
    current_pomdp = deepcopy(base_pomdp)
    
    # Initialize the rock position mapping 
    # Each original rock maps to itself initially
    rock_position_mapping = Dict{Tuple{Int,Int}, Vector{Tuple{Int,Int}}}()
    for rock_pos in rocks_positions
        rock_position_mapping[rock_pos] = [rock_pos]
    end
    
    # Initialize grid refinement tracker 
    grid_refinement = zeros(Int, map_size)
    
    # Solve the initial POMDP to get value function and policy
    solver = ValueIterationSolver(max_iterations=1000, belres=1e-6)
    policy = solve(solver, UnderlyingMDP(current_pomdp))
    
    # Initialize the current state
    current_state = RSState(RSPos(init_pos...), rand(Bool, length(rocks_positions)) |> SVector{length(rocks_positions),Bool})
    
    # Create and return the SlidingDroneRockSamplePOMDP
    return SlidingDroneRockSamplePOMDP{length(rocks_positions)}(
        base_pomdp,
        current_pomdp,
        policy.util,  # Initial value function
        policy,
        horizon_limit,
        splitting_param,
        current_state,
        rock_position_mapping,
        grid_refinement
    )
end

"""
    determine_horizon(smdp, state)

Determine which states are within the horizon (reachable with probability > threshold).
"""
function determine_horizon(smdp::SlidingDroneRockSamplePOMDP, state::RSState)
    horizon_states = RSState[]
    current_pos = state.pos
    
    # Loop through all states in the current POMDP
    for s in states(smdp.current_pomdp)
        # Skip terminal state
        if isterminal(smdp.current_pomdp, s)
            continue
        end
        
        # Check if this state is reachable with high enough probability
        # For DroneRockSample, states are reachable if they're within flying distance
        if manhattan_distance(current_pos, s.pos) <= 3  # MAX_FLIGHT_DISTANCE
            # States with the same position but different rock states are considered reachable
            if s.pos == current_pos || 
               (manhattan_distance(current_pos, s.pos) <= 3 && 
                any(a -> is_fly_action(a) && 
                    current_pos + action_to_direction(a) == s.pos, 
                    actions(smdp.current_pomdp, state)))
                push!(horizon_states, s)
            end
        end
    end
    
    return horizon_states
end

"""
    manhattan_distance(pos1, pos2)

Calculate Manhattan distance between two positions.
"""
function manhattan_distance(pos1::RSPos, pos2::RSPos)
    return abs(pos1[1] - pos2[1]) + abs(pos1[2] - pos2[2])
end

"""
    find_split_candidates(smdp, horizon_states)

Find pairs of neighboring states with value differences exceeding the splitting threshold.
"""
function find_split_candidates(smdp::SlidingDroneRockSamplePOMDP, horizon_states::Vector{RSState})
    candidates = Tuple{RSState, RSState, Symbol}[]
    
    # Group states by their rock configuration
    rock_groups = Dict()
    for s in horizon_states
        rock_key = s.rocks |> Tuple
        if !haskey(rock_groups, rock_key)
            rock_groups[rock_key] = RSState[]
        end
        push!(rock_groups[rock_key], s)
    end
    
    # Check for value differences among states with the same rock configuration
    for (_, states) in rock_groups
        for i in 1:length(states)
            for j in (i+1):length(states)
                s1, s2 = states[i], states[j]
                
                # Check if states are neighbors (differ by 1 in exactly one coordinate)
                x_diff = abs(s1.pos[1] - s2.pos[1])
                y_diff = abs(s1.pos[2] - s2.pos[2])
                
                if (x_diff == 1 && y_diff == 0) || (x_diff == 0 && y_diff == 1)
                    # Calculate value difference
                    v1 = smdp.value_function[POMDPs.stateindex(smdp.current_pomdp, s1)]
                    v2 = smdp.value_function[POMDPs.stateindex(smdp.current_pomdp, s2)]
                    
                    if abs(v1 - v2) > smdp.splitting_param
                        # Determine split direction
                        direction = if x_diff == 1
                            :x
                        else
                            :y
                        end
                        
                        push!(candidates, (s1, s2, direction))
                    end
                end
            end
        end
    end
    
    return candidates
end

"""
    refine_grid(smdp, split_candidates)

Refine the grid by adding new states between states with high value differences.
"""
function refine_grid(smdp::SlidingDroneRockSamplePOMDP{K}, split_candidates) where K
    if isempty(split_candidates)
        return false  # No refinement needed
    end
    
    # Create a new map with increased size to accommodate new states
    old_map_size = smdp.current_pomdp.map_size
    
    # Count how many new cells we need to add in each dimension
    new_x_cells = 0
    new_y_cells = 0
    
    for (s1, s2, direction) in split_candidates
        if direction == :x
            new_x_cells += 1
        else
            new_y_cells += 1
        end
    end
    
    # Calculate new map size
    new_map_size = (old_map_size[1] + new_x_cells, old_map_size[2] + new_y_cells)
    
    # Create position mapping from old to new grid
    pos_mapping = Dict{RSPos, RSPos}()
    
    # Start with identity mapping
    for x in 1:old_map_size[1]
        for y in 1:old_map_size[2]
            pos_mapping[RSPos(x, y)] = RSPos(x, y)
        end
    end
    
    # Process each split candidate
    x_offset = 0
    y_offset = 0
    
    # First, update position mapping based on splits
    for (s1, s2, direction) in split_candidates
        if direction == :x
            # Split along x-axis
            min_x = min(s1.pos[1], s2.pos[1])
            # Shift all positions after min_x
            for x in old_map_size[1]:-1:(min_x+1)
                for y in 1:old_map_size[2]
                    old_pos = RSPos(x, y)
                    if haskey(pos_mapping, old_pos)
                        pos_mapping[old_pos] = RSPos(x + x_offset + 1, y + y_offset)
                    end
                end
            end
            x_offset += 1
        else
            # Split along y-axis
            min_y = min(s1.pos[2], s2.pos[2])
            # Shift all positions after min_y
            for x in 1:old_map_size[1]
                for y in old_map_size[2]:-1:(min_y+1)
                    old_pos = RSPos(x, y)
                    if haskey(pos_mapping, old_pos)
                        pos_mapping[old_pos] = RSPos(x + x_offset, y + y_offset + 1)
                    end
                end
            end
            y_offset += 1
        end
    end
    
    # Create new rock positions
    new_rocks_positions = RSPos[]
    
    # Map old rock positions to new positions
    for rock_pos in smdp.current_pomdp.rocks_positions
        if haskey(pos_mapping, rock_pos)
            push!(new_rocks_positions, pos_mapping[rock_pos])
        else
            push!(new_rocks_positions, rock_pos)  # Fallback
        end
    end
    
    # Create the refined POMDP
    refined_pomdp = DroneRockSamplePOMDP(
        map_size=new_map_size,
        rocks_positions=new_rocks_positions,
        init_pos=pos_mapping[smdp.current_pomdp.init_pos],
        sensor_efficiency=smdp.current_pomdp.sensor_efficiency,
        discount_factor=smdp.current_pomdp.discount_factor,
        good_rock_reward=smdp.current_pomdp.good_rock_reward,
        fly_penalty=smdp.current_pomdp.fly_penalty
    )
    
    # Map current state to new grid
    new_current_pos = pos_mapping[smdp.current_state.pos]
    new_current_state = RSState(new_current_pos, smdp.current_state.rocks)
    
    # Update the SMDP
    smdp.current_pomdp = refined_pomdp
    smdp.current_state = new_current_state
    
    # Update grid refinement tracker
    new_grid_refinement = zeros(Int, new_map_size)
    for x in 1:old_map_size[1]
        for y in 1:old_map_size[2]
            old_pos = RSPos(x, y)
            if haskey(pos_mapping, old_pos)
                new_pos = pos_mapping[old_pos]
                new_grid_refinement[new_pos[1], new_pos[2]] = smdp.grid_refinement[x, y]
            end
        end
    end
    
    # Now mark the new cells as refined
    for (s1, s2, direction) in split_candidates
        if direction == :x
            x1, x2 = s1.pos[1], s2.pos[1]
            y = s1.pos[2]  # Both have the same y
            min_x, max_x = minmax(x1, x2)
            
            # Find the new position of min_x
            if haskey(pos_mapping, RSPos(min_x, y))
                new_min_x = pos_mapping[RSPos(min_x, y)][1]
                new_y = pos_mapping[RSPos(min_x, y)][2]
                
                # The new cell is right after new_min_x
                new_grid_refinement[new_min_x + 1, new_y] = max(
                    smdp.grid_refinement[min_x, y], 
                    smdp.grid_refinement[max_x, y]
                ) + 1
            end
        else
            x = s1.pos[1]  # Both have the same x
            y1, y2 = s1.pos[2], s2.pos[2]
            min_y, max_y = minmax(y1, y2)
            
            # Find the new position of min_y
            if haskey(pos_mapping, RSPos(x, min_y))
                new_x = pos_mapping[RSPos(x, min_y)][1]
                new_min_y = pos_mapping[RSPos(x, min_y)][2]
                
                # The new cell is right after new_min_y
                new_grid_refinement[new_x, new_min_y + 1] = max(
                    smdp.grid_refinement[x, min_y], 
                    smdp.grid_refinement[x, max_y]
                ) + 1
            end
        end
    end
    
    smdp.grid_refinement = new_grid_refinement
    
    # Solve the new POMDP
    solver = ValueIterationSolver(max_iterations=1000, belres=1e-6)
    smdp.policy = solve(solver, UnderlyingMDP(smdp.current_pomdp))
    smdp.value_function = smdp.policy.util
    
    return true  # Grid was refined
end

"""
    solve_sliding_pomdp(smdp)

Perform one step of the Sliding MDP algorithm.
"""
function solve_sliding_pomdp(smdp::SlidingDroneRockSamplePOMDP)
    # 1. Determine states within horizon
    horizon_states = determine_horizon(smdp, smdp.current_state)
    
    println("Found $(length(horizon_states)) states in horizon")
    
    # 2. Find candidate states for splitting
    split_candidates = find_split_candidates(smdp, horizon_states)
    
    println("Found $(length(split_candidates)) split candidates")
    
    # 3. Refine the grid if needed
    if !isempty(split_candidates)
        refined = refine_grid(smdp, split_candidates)
        if refined
            println("Grid refined. New map size: $(smdp.current_pomdp.map_size)")
        end
    end
    
    # 4. Get the best action for the current state
    current_state_idx = POMDPs.stateindex(smdp.current_pomdp, smdp.current_state)
    best_action = smdp.policy.action_map[POMDPs.action(smdp.policy, smdp.current_state)]
    
    println("Best action: $(action_to_string(smdp.current_pomdp, best_action))")
    
    return best_action
end

"""
    step_sliding_pomdp!(smdp, action)

Execute the given action in the Sliding MDP.
"""
function step_sliding_pomdp!(smdp::SlidingDroneRockSamplePOMDP{K}, action::Int) where K
    # Get the next state distribution
    td = transition(smdp.current_pomdp, smdp.current_state, action)
    
    # Sample a next state
    next_state = rand(td)
    
    # Update the current state
    smdp.current_state = next_state
    
    # Return the new state and reward
    r = reward(smdp.current_pomdp, smdp.current_state, action)
    
    return next_state, r
end

"""
    run_sliding_pomdp_simulation(smdp, max_steps)

Run a simulation using the Sliding MDP approach.
"""
function run_sliding_pomdp_simulation(smdp::SlidingDroneRockSamplePOMDP, max_steps::Int=30)
    total_reward = 0.0
    discount = 1.0
    step_results = []
    
    for step in 1:max_steps
        println("\n--- Step $step ---")
        println("Current state: Position=$(smdp.current_state.pos), Rocks=$(smdp.current_state.rocks)")
        
        # Solve the current POMDP
        action = solve_sliding_pomdp(smdp)
        
        # Take the action
        next_state, r = step_sliding_pomdp!(smdp, action)
        
        # Record results
        push!(step_results, (
            state=smdp.current_state,
            action=action,
            reward=r,
            next_state=next_state,
            map_size=smdp.current_pomdp.map_size,
            grid_refinement=copy(smdp.grid_refinement)
        ))
        
        # Update total reward
        total_reward += discount * r
        discount *= smdp.current_pomdp.discount_factor
        
        # Print information
        println("Action: $(action_to_string(smdp.current_pomdp, action))")
        println("Reward: $r")
        println("New state: Position=$(next_state.pos), Rocks=$(next_state.rocks)")
        
        # Check if we've reached the terminal state
        if isterminal(smdp.current_pomdp, next_state)
            println("Reached terminal state!")
            break
        end
    end
    
    println("\nSimulation complete!")
    println("Total reward: $total_reward")
    
    return step_results, total_reward
end

end  # module