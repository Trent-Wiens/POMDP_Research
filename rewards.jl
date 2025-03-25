function POMDPs.reward(pomdp::DronePOMDP, s::RSState, a::Int)
    r = pomdp.step_penalty  # General step penalty

    # Get the next position based on the action
    next_pos = index_to_action(pomdp, a)
    
    # If the next position is beyond the right edge of the map (exit)
    if next_pos[1] > pomdp.map_size[1]
        # Very large penalty for exiting before checking all rocks
        # This approach matches the structure of RockSample but makes exit highly undesirable
        if any(s.rocks)  # If any rocks are still good
            r += -100.0  # Large penalty to avoid exiting with good rocks
        else
            r += pomdp.exit_reward  # Normal exit reward only when all rocks are checked
        end
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
    end
    
    return r
end