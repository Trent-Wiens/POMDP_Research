module DronePOMDP

using LinearAlgebra
using POMDPs
using POMDPTools
using StaticArrays
using Parameters
using Random
using DiscreteValueIteration

export DronePOMDP, DroneState

const DronePos = SVector{3, Float64}  # (x, y, z) position in 3D space

"""
    DroneState
Represents the state of the drone in a 3D space.

# Fields
- `pos::DronePos`: Position of the drone (x, y, z)
"""
struct DroneState
    pos::DronePos
end

@with_kw struct DronePOMDP <: POMDP{DroneState, DronePos, DronePos}
    space_size::Tuple{Int, Int, Int} = (10, 10, 5)  # 3D grid boundaries
    goal::DronePos = DronePos(9.0, 9.0, 3.0)  # Target location
    step_penalty::Float64 = -0.1  # Small penalty for moving
    goal_reward::Float64 = 100.0  # Reward for reaching goal
    discount_factor::Float64 = 0.95  # Future reward discount
end

# Transition function: probability of moving to any other position
function transition(pomdp::DronePOMDP, state::DroneState, action::DronePos)
    positions = [SVector{3, Float64}(x, y, z) for x in 0:pomdp.space_size[1],
                                                    y in 0:pomdp.space_size[2],
                                                    z in 0:pomdp.space_size[3]]
    distances = [norm(pos - action) for pos in positions]
    probabilities = exp.(-distances) / sum(exp.(-distances))  # Higher probability for closer positions
    return positions, probabilities
end

# Observation function: exact position (fully observable)
observation(pomdp::DronePOMDP, state::DroneState, action::DronePos) = state.pos

# Reward function: rewards reaching the goal
function reward(pomdp::DronePOMDP, state::DroneState, action::DronePos)
    if norm(state.pos - pomdp.goal) < 1.0
        return pomdp.goal_reward  # Reward for reaching target
    else
        return pomdp.step_penalty  # Small penalty for each move
    end
end

# Terminal state check
POMDPs.isterminal(pomdp::DronePOMDP, s::DroneState) = norm(s.pos - pomdp.goal) < 1.0
POMDPs.discount(pomdp::DronePOMDP) = pomdp.discount_factor

end  # module
