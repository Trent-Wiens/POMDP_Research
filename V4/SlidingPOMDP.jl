using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using Cairo
using Parameters

include("DroneRockSample.jl")
# Save a reference to the module
const DRS = DroneRockSample


export SlidingPOMDPSolver, SlidingPOMDPPolicy, set_dronerocksample

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
- `horizon_distance::Int`: Distance from the agent within the horizon
- `precision::Float64`: Precision parameter for SARSOP
- `timeout::Float64`: Maximum time allowed for SARSOP computation
- `belief_points::Int`: Number of belief points to use in SARSOP
- `include_goal_state::Bool`: Whether to always include the goal (exit) state
- 'splitting_parameter::Int': Value difference for splitting the state space
"""
@with_kw struct SlidingSARSOPSolver <: Solver
    horizon_distance::Int = 3  # Default horizon distance
    splitting_parameter::Int = 1 # Default splitting parameter
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

# make sub-pomdp
function make_sub_pomdp(pomdp::DronePOMDPType, state::RSState, horizon_distance::Int)
    # Get the map size
    map_size = pomdp.map_size

    # Get the drone position
    drone_pos = state.pos

    # Get the rock positions
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
    
    # Compute the sub-map size
    sub_map_size = (2*horizon_distance+1, 2*horizon_distance+1)

    # Compute the sub-map bounds
    x_min = max(1, drone_pos.x - horizon_distance)
    x_max = min(map_size[1], drone_pos.x + horizon_distance)
    y_min = max(1, drone_pos.y - horizon_distance)
    y_max = min(map_size[2], drone_pos.y + horizon_distance)

    # Compute the sub-map rocks
    sub_rocks = []
    for rock_pos in local_rocks
        if x_min <= rock_pos.x <= x_max && y_min <= rock_pos.y <= y_max
            push!(sub_rocks, rock_pos)
        end
    end

    # Compute the sub-map initial position
    sub_init_pos = RSPos(drone_pos.x - x_min + 1, drone_pos.y - y_min + 1)

    # Create the sub-map POMDP
    sub_pomdp = DroneRockSampleModule[].
        DroneRockSamplePOMDP(
            map_size = sub_map_size,
            rocks_positions = sub_rocks,
            init_pos = sub_init_pos,
            sensor_efficiency = pomdp.sensor_efficiency,
            discount_factor = pomdp.discount_factor,
            good_rock_reward = pomdp.good_rock_reward,
            fly_penalty = pomdp.fly_penalty
        )

    return sub_pomdp
end

