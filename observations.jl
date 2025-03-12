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
    POMDPs.observation(pomdp::DronePOMDP, a::Int, s::RSState)

Returns a probability distribution over possible observations given action `a` and state `s`.
"""
function POMDPs.observation(pomdp::DronePOMDP, a::Int, s::RSState)
    # If the action is movement or sampling, return "none"
    if a == SAMPLE_ACTION || a < SENSING_START_INDEX
        return SparseCat((1,2,3), (0.0, 0.0, 1.0))  # Always "none" when moving or sampling
    end

    # Extract rock index from the sensing action
    rock_ind = a - SENSING_START_INDEX  # Adjusted since sensing starts at 101

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

    # Ensure probabilities sum to 1
    if rock_state  # Rock is good
        probs = [efficiency, 1.0 - efficiency, 0.0]
    else  # Rock is bad
        probs = [1.0 - efficiency, efficiency, 0.0]
    end
    probs ./= sum(probs)  # Normalize to sum to 1

    if a < 10
        println("Action: $a, Rock Index: $rock_ind, Probabilities: $probs, Sum: $(sum(probs))")
    end

    # println("Action: $a, Rock Index: $rock_ind, Probabilities: $probs, Sum: $(sum(probs))")

    return SparseCat((1,2,3), probs)
end