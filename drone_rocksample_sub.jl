module Drone

using LinearAlgebra
using POMDPs
using POMDPTools
using StaticArrays
using Parameters
using Random
using Compose
using Combinatorics
using DiscreteValueIteration
using ParticleFilters           # used in heuristics

export
    DronePOMDP,
    RSPos,
    RSState,
    RSExit,
    RSExitSolver,
    RSMDPSolver,
    RSQMDPSolver

const RSPos = SVector{2, Int}

"""
    RSState{K}
Represents the state in a DronePOMDP problem. 
`K` is an integer representing the number of rocks

# Fields
- `pos::RPos` position of the robot
- `rocks::SVector{K, Bool}` the status of the rocks (false=bad, true=good)
"""
struct RSState{K}
    pos::RSPos 
    rocks::SVector{K, Bool}
end

@with_kw struct DronePOMDP{K} <: POMDP{RSState{K}, Int, Int}
    map_size::Tuple{Int, Int} = (5,5)
    rocks_positions::SVector{K,RSPos} = @SVector([(1,1), (3,3), (4,4)])
    init_pos::RSPos = (1,1)
    sensor_efficiency::Float64 = 20.0
    bad_rock_penalty::Float64 = -10
    good_rock_reward::Float64 = 10.
    step_penalty::Float64 = 0.
    sensor_use_penalty::Float64 = 0.
    exit_reward::Float64 = 10.
    terminal_state::RSState{K} = RSState(RSPos(-1,-1),
                                         SVector{length(rocks_positions),Bool}(falses(length(rocks_positions))))
    # Some special indices for quickly retrieving the stateindex of any state
    indices::Vector{Int} = cumprod([map_size[1], map_size[2], fill(2, length(rocks_positions))...][1:end-1])
    discount_factor::Float64 = 0.95
end

# to handle the case where rocks_positions is not a StaticArray
function DronePOMDP(map_size,
                         rocks_positions,
                         args...
                        )

    k = length(rocks_positions)
    return DronePOMDP{k}(map_size,
                              SVector{k,RSPos}(rocks_positions),
                              args...
                             )
end

# Generate a random instance of Drone(n,m) with a n×n square map and m rocks
DronePOMDP(map_size::Int, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG) = DronePOMDP((map_size,map_size), rocknum, rng)

# Generate a random instance of Drone with a n×m map and l rocks
function DronePOMDP(map_size::Tuple{Int, Int}, rocknum::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    possible_ps = [(i, j) for i in 1:map_size[1], j in 1:map_size[2]]
    selected = unique(rand(rng, possible_ps, rocknum))
    while length(selected) != rocknum
        push!(selected, rand(rng, possible_ps))
        selected = unique!(selected)
    end
    return DronePOMDP(map_size=map_size, rocks_positions=selected)
end

# transform a Drone state to a vector 
function POMDPs.convert_s(T::Type{<:AbstractArray}, s::RSState, m::DronePOMDP)
    return convert(T, vcat(s.pos, s.rocks))
end

# transform a vector to a RSState
function POMDPs.convert_s(T::Type{RSState}, v::AbstractArray, m::DronePOMDP)
    return RSState(RSPos(v[1], v[2]), SVector{length(v)-2,Bool}(v[i] for i = 3:length(v)))
end

"""
    POMDPs.initialstate(pomdp::DronePOMDP{K})

Returns an initial belief distribution over possible states.
"""
function POMDPs.initialstate(pomdp::DronePOMDP{K}) where K
    probs = normalize!(ones(2^K), 1)  # Uniform distribution over rock states
    states = Vector{RSState{K}}(undef, 2^K)

    for (i, rocks) in enumerate(Iterators.product(ntuple(x -> [false, true], K)...))
        states[i] = RSState{K}(pomdp.init_pos, SVector(rocks))
    end

    return SparseCat(states, probs)  # Return a valid belief distribution
end


# To handle the case where the `rocks_positions` is specified
DronePOMDP(map_size::Tuple{Int, Int}, rocks_positions::AbstractVector) = DronePOMDP(map_size=map_size, rocks_positions=rocks_positions)

POMDPs.isterminal(pomdp::DronePOMDP, s::RSState) = s.pos == pomdp.terminal_state.pos 
POMDPs.discount(pomdp::DronePOMDP) = pomdp.discount_factor

include("states.jl") #done
include("actions.jl") #done
include("transition.jl") #done
include("observations.jl") #done
include("rewards.jl") #done 
include("visualization.jl") #in progress
# include("heuristics.jl")

end # module
