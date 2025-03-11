const OBSERVATION_NAME = (:good, :bad, :none)

"""
    POMDPs.observations(pomdp::DronePOMDP)

Returns the observation space.
"""
POMDPs.observations(pomdp::DronePOMDP) = 1:3  # `1 = good`, `2 = bad`, `3 = none`

"""
    POMDPs.obsindex(pomdp::DronePOMDP, o::Int)

Return the observation index.
"""
POMDPs.obsindex(pomdp::DronePOMDP, o::Int) = o

"""
    POMDPs.observation(pomdp::DronePOMDP, a::Tuple{Int, Int}, s::RSState)

Returns a probability distribution over possible observations given action `a` and state `s`.
"""
function POMDPs.observation(pomdp::DronePOMDP, a::Tuple{Int, Int}, s::RSState)
    # If the action is movement or sampling, return "none"
    if a == SAMPLE_ACTION || a[1] > 0  # If first element is >0, it's a movement action
        return SparseCat((1,2,3), (0.0, 0.0, 1.0))  # Always "none" when moving or sampling
    end

    # Extract rock index from the sensing action
    rock_ind = a[2]  # Now sensing actions are (0, k) so k is in the second position

    # If the rock index is invalid, return "none"
    if rock_ind < 1 || rock_ind > length(pomdp.rocks_positions)
        return SparseCat((1,2,3), (0.0, 0.0, 1.0))
    end

    # Get the rock's position and compute the distance
    rock_pos = pomdp.rocks_positions[rock_ind]
    dist = norm(rock_pos - s.pos)

    # Compute observation probability based on distance
    efficiency = 0.5 * (1.0 + exp(-dist * log(2) / pomdp.sensor_efficiency))
    rock_state = s.rocks[rock_ind]

    if rock_state  # Rock is good
        return SparseCat((1,2,3), (efficiency, 1.0 - efficiency, 0.0))
    else  # Rock is bad
        return SparseCat((1,2,3), (1.0 - efficiency, efficiency, 0.0))
    end
end