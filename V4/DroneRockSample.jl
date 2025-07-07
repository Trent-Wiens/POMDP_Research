module DroneRockSample

using LinearAlgebra
using POMDPs
using POMDPTools
using StaticArrays
using Parameters
using Random
using Compose
using Combinatorics
using DiscreteValueIteration
using ParticleFilters

# Import types and functions from RockSample that we can reuse
import RockSample: RSPos, RSState

export
    DroneRockSamplePOMDP,
    RSPos,
    RSState,
    DRSExit,
    DRSExitSolver,
    DRSMDPSolver,
    DRSQMDPSolver,
    ACTION_SAMPLE,
    ACTION_FLY_START,
    N_BASIC_ACTIONS,
    is_fly_action,
    is_sense_action,
    action_to_string,
    POMDPs # Re-export POMDPs to ensure solve is available

"""
    DroneRockSamplePOMDP{K}
A version of RockSamplePOMDP where the agent is a drone that can fly
directly to any location within a distance of 3 cells.

# Fields
- `map_size::Tuple{Int, Int}` size of the map
- `rocks_positions::SVector{K,RSPos}` positions of rocks
- `init_pos::RSPos` initial position of the drone
- `sensor_efficiency::Float64` sensor efficiency parameter
- `bad_rock_penalty::Float64` penalty for sampling a bad rock
- `good_rock_reward::Float64` reward for sampling a good rock
- `step_penalty::Float64` penalty for each step
- `sensor_use_penalty::Float64` penalty for using the sensor
- `exit_reward::Float64` reward for exiting
- `terminal_state::RSState{K}` terminal state
- `indices::Vector{Int}` some indices for state indexing
- `discount_factor::Float64` discount factor
"""
@with_kw struct DroneRockSamplePOMDP{K} <: POMDP{RSState{K}, Int, Int}
    map_size::Tuple{Int, Int} = (5,5)
    max_map_size::Tuple{Int, Int} = (10,10)  #size of the map
    rocks_positions::SVector{K,RSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::RSPos = (1,1)
    sensor_efficiency::Float64 = 20.0
    bad_rock_penalty::Float64 = -5
    good_rock_reward::Float64 = 10.
    step_penalty::Float64 = -0.2
    sensor_use_penalty::Float64 = 0
    wrong_sample::Float64 = -5  # Penalty for useless sampling
    fly_penalty::Float64 = -0.1  # Small penalty for flying (fuel cost)
    exit_reward::Float64 = 10.
    terminal_state::RSState{K} = RSState(RSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    # Some special indices for quickly retrieving the stateindex of any state
    indices::Vector{Int} = cumprod([map_size[1], map_size[2], fill(2, length(rocks_positions))...][1:end-1])
    discount_factor::Float64 = 0.95
end

# Constructor for non-StaticArray rocks_positions
function DroneRockSamplePOMDP(map_size,
                             max_map_size,
                             rocks_positions,
                             args...
                            )
    k = length(rocks_positions)
    return DroneRockSamplePOMDP{k}(map_size,
                                  max_map_size,  # Default to same size for max_map_size
                                  SVector{k,RSPos}(rocks_positions),
                                  args...
                                 )
end

# Generate a random instance with a n×n square map and m rocks
DroneRockSamplePOMDP(map_size::Int, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG) = 
    DroneRockSamplePOMDP((map_size,map_size), rocknum, rng)

# Generate a random instance with a n×m map and l rocks
function DroneRockSamplePOMDP(map_size::Tuple{Int, Int}, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    possible_ps = [(i, j) for i in 1:map_size[1], j in 1:map_size[2]]
    selected = unique(rand(rng, possible_ps, rocknum))
    while length(selected) != rocknum
        push!(selected, rand(rng, possible_ps))
        selected = unique!(selected)
    end
    return DroneRockSamplePOMDP(map_size=map_size, rocks_positions=selected)
end

# Reusing state conversion functions from RockSample
function POMDPs.convert_s(T::Type{<:AbstractArray}, s::RSState, m::DroneRockSamplePOMDP)
    return convert(T, vcat(s.pos, s.rocks))
end

function POMDPs.convert_s(T::Type{RSState}, v::AbstractArray, m::DroneRockSamplePOMDP)
    return RSState(RSPos(v[1], v[2]), SVector{length(v)-2,Bool}(v[i] for i = 3:length(v)))
end

# Constructor with specified rocks positions
DroneRockSamplePOMDP(map_size::Tuple{Int, Int}, rocks_positions::AbstractVector) = 
    DroneRockSamplePOMDP(map_size=map_size,max_map_size=max_map_size, rocks_positions=rocks_positions)

POMDPs.isterminal(pomdp::DroneRockSamplePOMDP, s::RSState) = s.pos == pomdp.terminal_state.pos 
POMDPs.discount(pomdp::DroneRockSamplePOMDP) = pomdp.discount_factor

# Include all the component functions
include("drone-actions.jl")       # Fixed action functions
include("drone-transition.jl")    # Fixed transition function 
include("drone-states.jl")              # States management
include("drone-observations.jl")        # Observations
include("drone-reward.jl")        # Fixed reward function
include("drone-visualization.jl")       # Visualization
include("drone-heuristics.jl")          # Heuristics

end # module