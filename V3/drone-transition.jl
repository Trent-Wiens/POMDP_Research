function POMDPs.transition(pomdp::DroneRockSamplePOMDP{K}, s::RSState{K}, a::Int) where K
    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end
    
    # Calculate next position without recursion
    new_pos = next_drone_position(s, a)
    
    if a == ACTION_SAMPLE && in(s.pos, pomdp.rocks_positions)
        rock_ind = findfirst(isequal(s.pos), pomdp.rocks_positions)
        # set the rock to bad after sampling
        new_rocks = MVector{K, Bool}(undef)
        for r=1:K
            new_rocks[r] = r == rock_ind ? false : s.rocks[r]
        end
        new_rocks = SVector(new_rocks)
    else 
        new_rocks = s.rocks
    end
    
    if new_pos[1] > pomdp.map_size[1]
        # the drone reached the exit area
        new_state = pomdp.terminal_state
    else
        new_pos = RSPos(clamp(new_pos[1], 1, pomdp.map_size[1]), 
                        clamp(new_pos[2], 1, pomdp.map_size[2]))
        new_state = RSState{K}(new_pos, new_rocks)
    end
    
    return Deterministic(new_state)
end

# Renamed to avoid confusion with any existing functions
function next_drone_position(s::RSState, a::Int)
    if a == ACTION_SAMPLE || a > N_BASIC_ACTIONS
        # Drone samples or senses - no movement
        return s.pos
    elseif is_fly_action(a)
        # Drone flies to a new position
        direction = action_to_direction(a)
        return s.pos + direction
    else
        # Should not reach here
        return s.pos
    end
end