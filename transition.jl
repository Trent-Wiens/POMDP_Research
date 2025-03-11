function POMDPs.transition(pomdp::DronePOMDP, s::RSState{K}, a::Tuple{Int,Int}) where {K}

    if isterminal(pomdp, s)
        return Deterministic(pomdp.terminal_state)
    end

    # Unpack action (target_x, target_y)
    target_x, target_y = a

    # If sensing, the state remains the same
    if a == (SENSE_ACTION, SENSE_ACTION)
        return Deterministic(s)  # Stay in the same state
    end

    # Retrieve current position
    current_x, current_y = s.pos

    # Compute success probability
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