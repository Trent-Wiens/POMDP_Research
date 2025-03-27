module SlidingSARSOP

using POMDPs
using POMDPTools
using NativeSARSOP
using StaticArrays
using LinearAlgebra
using Parameters
using Random
using ParticleFilters  # Add ParticleFilters for belief representations

# Instead of importing the DroneRockSample module, we'll access it through a variable
# that will be set when the module is included

export SlidingSARSOPSolver, SlidingSARSOPPolicy, set_dronerocksample

# Store the DroneRockSample module in a global variable
const DroneRockSampleModule = Ref{Module}()

# Function to set the DroneRockSample module
function set_dronerocksample(drs_module::Module)
    DroneRockSampleModule[] = drs_module
end

"""
    SlidingSARSOPSolver

A solver that applies SARSOP to a localized subset of the state space 
in a DroneRockSamplePOMDP, following the "Sliding MDP" concept.

# Fields
- `horizon_distance::Int`: How many flight actions to consider in the horizon
- `precision::Float64`: Precision parameter for SARSOP
- `timeout::Float64`: Maximum time allowed for SARSOP computation
- `belief_points::Int`: Number of belief points to use in SARSOP
- `include_goal_state::Bool`: Whether to always include the goal (exit) state
"""
@with_kw struct SlidingSARSOPSolver <: Solver
    horizon_distance::Int = 3  # Default horizon distance
    precision::Float64 = 1e-3  # Precision for SARSOP
    timeout::Float64 = 2.0     # Time limit for each SARSOP run (seconds)
    belief_points::Int = 1000  # Number of belief points to sample
    include_goal_state::Bool = true  # Always include goal state in horizon
end

"""
    SlidingSARSOPPolicy

Policy returned by the SlidingSARSOPSolver.

# Fields
- `full_pomdp::DroneRockSamplePOMDP`: The original POMDP
- `solver::SlidingSARSOPSolver`: The solver that created this policy
- `full_belief_updater::Updater`: Belief updater for the full POMDP
"""
# Define a type alias for the POMDP type
# This allows us to use a more generic type in the policy
const DronePOMDPType = POMDP

mutable struct SlidingSARSOPPolicy <: Policy
    full_pomdp::DronePOMDPType
    solver::SlidingSARSOPSolver
    full_belief_updater::Updater
    current_belief::Any  # Current belief state
    current_policy::Any  # Current localized policy
end

# Constructor initializes the belief updater
function SlidingSARSOPPolicy(pomdp::DronePOMDPType, solver::SlidingSARSOPSolver)
    return SlidingSARSOPPolicy(
        pomdp,
        solver,
        DiscreteUpdater(pomdp),  # Use DiscreteUpdater directly
        nothing,  # Current belief (initialized during action)
        nothing   # Current policy (computed on-the-fly)
    )
end

# Implementation of the solve method
function POMDPs.solve(solver::SlidingSARSOPSolver, pomdp::DronePOMDPType)
    return SlidingSARSOPPolicy(pomdp, solver)
end

"""
    create_localized_pomdp(pomdp, center_pos, horizon_distance)

Create a smaller POMDP that only includes states within horizon_distance of center_pos.
"""
function create_localized_pomdp(pomdp::DronePOMDPType, center_pos, 
                                horizon_distance::Int, include_goal::Bool)
    
    # Calculate the bounds of our localized map
    min_x = max(1, center_pos[1] - horizon_distance)
    max_x = min(pomdp.map_size[1], center_pos[1] + horizon_distance)
    min_y = max(1, center_pos[2] - horizon_distance)
    max_y = min(pomdp.map_size[2], center_pos[2] + horizon_distance)
    
    # Only include rocks that are within our horizon
    local_rocks = []
    for (i, rock_pos) in enumerate(pomdp.rocks_positions)
        # Check if rock is within our horizon
        if min_x <= rock_pos[1] <= max_x && min_y <= rock_pos[2] <= max_y
            push!(local_rocks, rock_pos)
        end
    end
    
    # Special case: If no rocks in range, add the closest one
    if isempty(local_rocks) && !isempty(pomdp.rocks_positions)
        distances = [norm(center_pos .- rock_pos) for rock_pos in pomdp.rocks_positions]
        closest_idx = argmin(distances)
        push!(local_rocks, pomdp.rocks_positions[closest_idx])
    end
    
    # # If we're including the goal, make sure our range extends to the map edge
    # if include_goal
    #     max_x = max(max_x, pomdp.map_size[1])
    # end
    
    # Create a localized POMDP with our reduced map
    local_map_size = (max_x - min_x + 1, max_y - min_y + 1)
    
    # Adjust rock positions to the local coordinate system
    local_adjusted_rocks = [(r[1] - min_x + 1, r[2] - min_y + 1) for r in local_rocks]
    
    # Create the localized POMDP
    drs = DroneRockSampleModule[]
    local_pomdp = drs.DroneRockSamplePOMDP(
        map_size = local_map_size,
        rocks_positions = local_adjusted_rocks,
        init_pos = drs.RSPos(center_pos[1] - min_x + 1, center_pos[2] - min_y + 1),
        sensor_efficiency = pomdp.sensor_efficiency,
        bad_rock_penalty = pomdp.bad_rock_penalty,
        good_rock_reward = pomdp.good_rock_reward,
        step_penalty = pomdp.step_penalty,
        sensor_use_penalty = pomdp.sensor_use_penalty,
        fly_penalty = pomdp.fly_penalty,
        exit_reward = pomdp.exit_reward,
        discount_factor = pomdp.discount_factor
    )
    
    return local_pomdp, min_x, min_y
end

"""
    translate_state(local_state, min_x, min_y, K)

Translate a state from the local POMDP coordinate system to the full POMDP.
"""
function translate_state(local_state, min_x::Int, min_y::Int, original_rock_positions, local_rock_positions)
    # Translate the position
    drs = DroneRockSampleModule[]
    global_pos = drs.RSPos(local_state.pos[1] + min_x - 1, local_state.pos[2] + min_y - 1)
    
    # Translate the rock states - need to map from local rocks back to global rocks
    global_rocks = falses(length(original_rock_positions))
    
    for (local_idx, local_pos) in enumerate(local_rock_positions)
        # Find this rock in the original positions
        global_pos_adjusted = (local_pos[1] + min_x - 1, local_pos[2] + min_y - 1)
        for (global_idx, global_pos) in enumerate(original_rock_positions)
            if global_pos == global_pos_adjusted
                # If rock exists in both and is good in local state, mark it good in global
                if local_idx <= length(local_state.rocks) && local_state.rocks[local_idx]
                    global_rocks[global_idx] = true
                end
                break
            end
        end
    end
    
    # Create the translated state
    drs = DroneRockSampleModule[]
    return drs.RSState(global_pos, SVector{length(global_rocks),Bool}(global_rocks))
end

"""
    translate_action(action, local_pomdp, full_pomdp)

Translate an action from the local POMDP to the full POMDP.
"""
function translate_action(action::Int, local_pomdp, full_pomdp, 
                          local_rock_positions, original_rock_positions, min_x::Int, min_y::Int)
    
    # Handle sample action (unchanged)
    if action == 1 # ACTION_SAMPLE
        return action
    end
    
    # Handle fly actions (unchanged)
    drs = DroneRockSampleModule[]
    n_local_basic = drs.N_BASIC_ACTIONS
    n_full_basic = drs.N_BASIC_ACTIONS
    
    if action <= n_local_basic
        return action  # Fly actions are the same in both POMDPs
    end
    
    # Handle sense actions (need to map rock indices)
    local_rock_idx = action - n_local_basic
    
    if local_rock_idx <= length(local_rock_positions)
        local_rock_pos = local_pomdp.rocks_positions[local_rock_idx]
        global_rock_pos = (local_rock_pos[1] + min_x - 1, local_rock_pos[2] + min_y - 1)
        
        # Find matching rock in full POMDP
        for (global_idx, global_pos) in enumerate(original_rock_positions)
            if global_pos == global_rock_pos
                return n_full_basic + global_idx
            end
        end
    end
    
    # Fallback to east-moving action if no translation is found
    return 2  # ACTION_FLY_START (east)
end

"""
    localize_belief(b, local_pomdp, min_x, min_y, original_rock_positions)

Convert a belief from the full POMDP to a belief for the local POMDP.
"""
function localize_belief(b, local_pomdp, min_x::Int, min_y::Int, 
                        original_rock_positions, local_rock_positions)
    
    if b isa AbstractParticleBelief
        local_particles = []
        local_weights = []
        
        # For each particle in the original belief
        for (i, s) in enumerate(particles(b))
            # Only include particles within our local range
            if min_x <= s.pos[1] <= min_x + local_pomdp.map_size[1] - 1 &&
               min_y <= s.pos[2] <= min_y + local_pomdp.map_size[2] - 1
                
                # Translate position to local coordinates
                drs = DroneRockSampleModule[]
                local_pos = drs.RSPos(s.pos[1] - min_x + 1, s.pos[2] - min_y + 1)
                
                # Translate rock states
                local_rocks = falses(length(local_rock_positions))
                
                for (local_idx, local_pos_rocks) in enumerate(local_rock_positions)
                    # Find this rock in the original positions
                    global_pos = (local_pos_rocks[1] + min_x - 1, local_pos_rocks[2] + min_y - 1)
                    for (global_idx, orig_pos) in enumerate(original_rock_positions)
                        if orig_pos == global_pos
                            # If rock exists in both
                            if global_idx <= length(s.rocks)
                                local_rocks[local_idx] = s.rocks[global_idx]
                            end
                            break
                        end
                    end
                end
                
                # Create the local particle
                drs = DroneRockSampleModule[]
                local_state = drs.RSState(local_pos, SVector{length(local_rocks),Bool}(local_rocks))
                push!(local_particles, local_state)
                push!(local_weights, weight(b, i))
            end
        end
        
        # If we filtered out all particles, add one with uniform rock beliefs
        if isempty(local_particles)
            # Use drone's position as default
            drs = DroneRockSampleModule[]
            local_pos = drs.RSPos(local_pomdp.init_pos)
            
            # Create particles with all combinations of rock states
            for rock_states in Iterators.product(fill([false, true], length(local_rock_positions))...)
                drs = DroneRockSampleModule[]
                local_state = drs.RSState(local_pos, SVector{length(local_rock_positions),Bool}(rock_states))
                push!(local_particles, local_state)
                push!(local_weights, 1.0 / 2^length(local_rock_positions))
            end
        end
        
        # Normalize weights
        if !isempty(local_weights)
            local_weights ./= sum(local_weights)
        end
        
        # Create a new particle belief
        return ParticleCollection(local_particles, local_weights)
    else
        # For other belief types, sample from belief and use ParticleCollection
        # (This is a simplification - could be more sophisticated)
        n_samples = 100
        local_particles = []
        
        for _ in 1:n_samples
            s = rand(b)
            
            # Only include particles within our local range
            if min_x <= s.pos[1] <= min_x + local_pomdp.map_size[1] - 1 &&
               min_y <= s.pos[2] <= min_y + local_pomdp.map_size[2] - 1
                
                # Translate position to local coordinates
                drs = DroneRockSampleModule[]
                local_pos = drs.RSPos(s.pos[1] - min_x + 1, s.pos[2] - min_y + 1)
                
                # Translate rock states
                local_rocks = falses(length(local_rock_positions))
                
                for (local_idx, local_pos_rocks) in enumerate(local_rock_positions)
                    # Convert to global coordinates
                    global_pos = (local_pos_rocks[1] + min_x - 1, local_pos_rocks[2] + min_y - 1)
                    for (global_idx, orig_pos) in enumerate(original_rock_positions)
                        if orig_pos == global_pos && global_idx <= length(s.rocks)
                            local_rocks[local_idx] = s.rocks[global_idx]
                            break
                        end
                    end
                end
                
                # Create the local particle
                drs = DroneRockSampleModule[]
                local_state = drs.RSState(local_pos, SVector{length(local_rocks),Bool}(local_rocks))
                push!(local_particles, local_state)
            end
        end
        
        # If we filtered out all particles, add one with uniform rock beliefs
        if isempty(local_particles)
            # Use drone's position as default
            drs = DroneRockSampleModule[]
            local_pos = drs.RSPos(local_pomdp.init_pos)
            
            # Create a particle for each rock state combination
            for rock_states in Iterators.product(fill([false, true], length(local_rock_positions))...)
                drs = DroneRockSampleModule[]
                local_state = drs.RSState(local_pos, SVector{length(local_rock_positions),Bool}(rock_states))
                push!(local_particles, local_state)
            end
        end
        
        return ParticleCollection(local_particles)
    end
end

function get_representative_state(b)
    if b isa AbstractParticleBelief
        # Find highest weight particle
        max_weight = -Inf
        best_idx = 1
        
        for (i, s) in enumerate(particles(b))
            w = weight(b, i)
            if w > max_weight
                max_weight = w
                best_idx = i
            end
        end
        
        return particles(b)[best_idx]
    else
        # For other belief types, use mean belief
        return rand(b)
    end
end

function POMDPs.action(policy::SlidingSARSOPPolicy, b)
    # 1. Find the representative state to center our localized POMDP
    representative = get_representative_state(b)
    center_pos = representative.pos
    
    # 2. Create a localized POMDP around the representative state
    local_pomdp, min_x, min_y = create_localized_pomdp(
        policy.full_pomdp, 
        center_pos, 
        policy.solver.horizon_distance,
        policy.solver.include_goal_state
    )
    
    # Remember the local rock positions for translating back
    local_rock_positions = local_pomdp.rocks_positions
    original_rock_positions = policy.full_pomdp.rocks_positions
    
    # 3. Create a localized belief
    local_belief = localize_belief(
        b, 
        local_pomdp, 
        min_x, 
        min_y, 
        original_rock_positions,
        local_rock_positions
    )
    
    # 4. Solve the localized POMDP
    sarsop = SARSOPSolver(
        precision=policy.solver.precision,
        max_time=policy.solver.timeout,
        verbose=false
    )
    
    local_policy = POMDPs.solve(sarsop, local_pomdp)
    
    # 5. Get action from the local policy for the local belief
    local_action = POMDPs.action(local_policy, local_belief)
    
    # 6. Translate the action back to the original POMDP
    global_action = translate_action(
        local_action, 
        local_pomdp, 
        policy.full_pomdp,
        local_rock_positions,
        original_rock_positions,
        min_x, 
        min_y
    )
    
    # 7. Save the current belief and policy for value computation
    policy.current_belief = b
    policy.current_policy = local_policy
    
    return global_action
end

function POMDPs.value(policy::SlidingSARSOPPolicy, b)
    # If we don't have a current policy, compute one
    if policy.current_policy === nothing || policy.current_belief !== b
        # This will set current_policy as a side effect
        POMDPs.action(policy, b)
    end
    
    # Find the representative state to center our localized POMDP
    representative = get_representative_state(b)
    center_pos = representative.pos
    
    # Create a localized POMDP around the representative state
    local_pomdp, min_x, min_y = create_localized_pomdp(
        policy.full_pomdp, 
        center_pos, 
        policy.solver.horizon_distance,
        policy.solver.include_goal_state
    )
    
    # Remember the local rock positions for translating back
    local_rock_positions = local_pomdp.rocks_positions
    
    # Create a localized belief
    local_belief = localize_belief(
        b, 
        local_pomdp, 
        min_x, 
        min_y, 
        policy.full_pomdp.rocks_positions,
        local_rock_positions
    )
    
    # Get value from the local policy
    return POMDPs.value(policy.current_policy, local_belief)
end

function POMDPs.updater(policy::SlidingSARSOPPolicy)
    return policy.full_belief_updater
end

end # module