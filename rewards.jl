"""
    POMDPs.reward(pomdp::DronePOMDP, s::RSState, a::Int)

Define rewards for different actions.
"""
function POMDPs.reward(pomdp::DronePOMDP, s::RSState, a::Int)
    r = pomdp.step_penalty  # General step penalty

    # If drone tries to exit (out of bounds), give reward
    if s.pos[1] > pomdp.map_size[1]
        r += pomdp.exit_reward
        return r
    end

    # If sampling, reward or penalize based on rock quality
    if a == SAMPLE_ACTION && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
        if rock_ind !== nothing
            r += s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty
        end
    elseif a >= SENSING_START_INDEX
        # Penalize using the sensor too much (optional)
        r += pomdp.sensor_use_penalty
    else
        # Encourage movement (optional: adjust for optimal behavior)
        r += 1.0  # Small reward for moving
    end
    return r
end