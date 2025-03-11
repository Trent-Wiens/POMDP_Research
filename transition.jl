"""
    POMDPs.transition(pomdp::DronePOMDP, s::RSState{K}, a::Tuple{Int,Int}) where {K}

Defines the transition model for the DronePOMDP.
"""
function POMDPs.transition(pomdp::DronePOMDP{K}, s::RSState{K}, a::Tuple{Int, Int}) where {K}

    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end

    # Unpack action
    action_x, action_y = a

    # If sampling (0,0), update the rock states if at a rock position
    if a == (0,0)
        new_rocks = s.rocks
        for (i, rock_pos) in enumerate(pomdp.rocks_positions)
            if s.pos == rock_pos
                new_rocks = setindex(new_rocks, false, i)  # "Remove" the rock
            end
        end
        return Deterministic(RSState(s.pos, new_rocks))
    end

    # If sensing (0,k), state remains the same
    if action_x == 0 && action_y > 0  # Sensing action (0, k)
        return Deterministic(s)  # Sensing does not change the state
    end

    # Otherwise, it’s a movement action to (target_x, target_y)
    target_x, target_y = action_x, action_y
    current_x, current_y = s.pos

    # Compute success probability (farther moves are less likely to succeed)
    distance = norm(RSPos(target_x, target_y) .- RSPos(current_x, current_y), 2)
    success_prob = exp(-0.3 * distance)  # α=0.3 (adjustable parameter)

    # If successful, move to target location
    new_pos = RSPos(target_x, target_y)

    # If failure, move to a nearby random location
    nx, ny = pomdp.map_size
    neighbor_moves = [(dx, dy) for dx in -1:1, dy in -1:1 if (dx, dy) != (0, 0)]
    fallback_positions = [
        (clamp(current_x + dx, 1, nx), clamp(current_y + dy, 1, ny))
        for (dx, dy) in neighbor_moves
    ]
    
    # Create a probability distribution over next states
    next_states = Vector{RSState{K}}()
    probabilities = Vector{Float64}()

    push!(next_states, RSState{K}(new_pos, s.rocks))  # Successful move
    push!(probabilities, success_prob)

    # Failed movement: drone lands in a random adjacent cell
    failure_prob = (1 - success_prob) / length(fallback_positions)
    for fallback in fallback_positions
        push!(next_states, RSState{K}(RSPos(fallback...), s.rocks))
        push!(probabilities, failure_prob)
    end

    return SparseCat(next_states, probabilities)
end