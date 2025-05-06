function POMDPs.reward(pomdp::DroneRockSamplePOMDP, s::RSState, a::Int)
    r = pomdp.step_penalty
    
    # Calculate next position directly to avoid recursion
    next_pos = if a == ACTION_SAMPLE || a > N_BASIC_ACTIONS
        s.pos  # No movement for sample or sense
    elseif is_fly_action(a)
        direction = action_to_direction(a)
        s.pos + direction
    else
        s.pos  # Fallback
    end
    
    # Check if we're exiting the map
    if next_pos[1] > pomdp.map_size[1]
        r += pomdp.exit_reward
        return r
    end

    # Apply action-specific rewards/penalties
    if a == ACTION_SAMPLE
        if in(s.pos, pomdp.rocks_positions)
            rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
            r += s.rocks[rock_ind] ? pomdp.good_rock_reward : pomdp.bad_rock_penalty
        else
            rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
            r += pomdp.wrong_sample  # Penalize useless sampling
        end
    elseif is_fly_action(a)
        # Flying action - apply small penalty proportional to distance
        direction = action_to_direction(a)
        distance = abs(direction[1]) + abs(direction[2])  # Manhattan distance
        r += pomdp.fly_penalty * distance
    elseif a > N_BASIC_ACTIONS
        # Sensing action
        r += pomdp.sensor_use_penalty
    end
    
    return r
end