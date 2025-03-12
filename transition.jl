"""
    POMDPs.transition(pomdp::DronePOMDP, s::RSState{K}, a::Int) where {K}

Defines the transition model for the DronePOMDP.
"""
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
        return Deterministic(s)
    end

    # Otherwise, this is a movement action
    current_x, current_y = s.pos
    distance = norm(RSPos(target_x, target_y) .- RSPos(current_x, current_y), 2)

    # If within range, move with 100% success
    if distance <= 3
        return Deterministic(RSState(RSPos(target_x, target_y), s.rocks))
    end

    # If out of range, movement fails - land in a random nearby cell
    nx, ny = pomdp.map_size
    neighbor_moves = [(dx, dy) for dx in -1:1, dy in -1:1 if (dx, dy) != (0, 0)]
    fallback_positions = [
        (clamp(current_x + dx, 1, nx), clamp(current_y + dy, 1, ny))
        for (dx, dy) in neighbor_moves
    ]

    # Create a probability distribution over fallback states
    next_states = Vector{RSState{K}}()
    probabilities = Vector{Float64}()

    for fallback in fallback_positions
        push!(next_states, RSState(RSPos(fallback...), s.rocks))
        push!(probabilities, 1.0 / length(fallback_positions))  # Uniform fallback probability
    end

    return SparseCat(next_states, probabilities)
end