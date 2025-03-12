"""
    POMDPs.reward(pomdp::DronePOMDP, s::RSState, a::Int)

Computes the reward for taking action `a` in state `s` in `DronePOMDP`.
"""
function POMDPs.reward(pomdp::DronePOMDP, s::RSState{K}, a::Int) where K
    r = pomdp.step_penalty  # Base step penalty for any action

    # Convert action index to (x, y)
    action_x, action_y = index_to_action(pomdp, a)

    # Exit reward if flying out of bounds
    if action_x > pomdp.map_size[1] || action_y > pomdp.map_size[2]
        return r + pomdp.exit_reward
    end

    # Sampling reward (if at a rock position)
    if a == SAMPLE_ACTION && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
        if rock_ind !== nothing
            return r + (s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty)
        end
    end

    # Sensor use penalty (if sensing a rock)
    if a >= SENSING_START_INDEX  # Sensing action (101+)
        return r + pomdp.sensor_use_penalty
    end

    return r
end