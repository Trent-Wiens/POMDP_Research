function POMDPs.transition(pomdp::DronePOMDP{K}, s::RSState{K}, a::Int) where {K}
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end

    # Convert action index to (x, y) position
    target_x, target_y = index_to_action(pomdp, a)

    # If sampling action, update the rock state
    if a == SAMPLE_ACTION
        new_rocks = s.rocks
        for (i, rock_pos) in enumerate(pomdp.rocks_positions)
            if s.pos == rock_pos
                new_rocks = setindex(new_rocks, false, i)  # "Remove" the rock
            end
        end
        return Deterministic(RSState(s.pos, new_rocks))
    end

    # If sensing action, state remains the same
    if a >= SENSING_START_INDEX
        return Deterministic(s)  # Sensing does not change position
    end

    # Otherwise, this is a movement action
    current_x, current_y = s.pos
    distance = norm(RSPos(target_x, target_y) .- RSPos(current_x, current_y), 2)

    # Define the exit area (rightmost column)
    exit_x = pomdp.map_size[1]

    # If the drone tries to reach the exit area, check if all rocks are sampled
    if target_x == exit_x
        # Only allow exit if all rocks have been sampled (are false)
        all_rocks_sampled = !any(s.rocks)
        
        if all_rocks_sampled
            return Deterministic(pomdp.terminal_state)
        else
            # If trying to exit with unsampled rocks, prevent movement
            return Deterministic(s)
        end
    end

    # If within range (≤3 spaces), move with 100% success
    if distance <= 3
        return Deterministic(RSState(RSPos(target_x, target_y), s.rocks))
    end

    # If move is too far, stay in place
    return Deterministic(s)
end