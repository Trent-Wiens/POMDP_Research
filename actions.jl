print("actions\n")

const SAMPLE_ACTION = 1  # Now sampling is indexed as 1
const SENSING_START_INDEX = 101  # Sensing starts at 101+

"""
    action_to_index(pomdp::DronePOMDP, a::Tuple{Int, Int})

Convert an action `(x, y)` into an integer index (starting from 1).
"""
function action_to_index(pomdp::DronePOMDP, a::Tuple{Int, Int})
    nx, ny = pomdp.map_size

    # Sampling action (1 instead of 0)
    if a == (0,0)
        return SAMPLE_ACTION
    end

    # Sensing action (0, k) maps to 101+
    if a[1] == 0 && a[2] > 0
        return SENSING_START_INDEX + a[2]  # Shift down by 1
    end

    # Movement action (x, y) -> Convert to 1-based index
    x, y = a
    return (x - 1) * ny + y + 1  # Shift all indices up by 1
end

"""
    index_to_action(pomdp::DronePOMDP, index::Int)

Convert an integer index back to an `(x, y)` action.
"""
function index_to_action(pomdp::DronePOMDP, index::Int)
    nx, ny = pomdp.map_size

    # Sampling action (1 instead of 0)
    if index == SAMPLE_ACTION
        return (0,0)
    end

    # Sensing action (101+)
    if index >= SENSING_START_INDEX
        return (0, index - SENSING_START_INDEX + 1)  # Shift up by 1
    end

    # Movement action: Convert index back to (x, y)
    index -= 1  # Shift back down
    x = div(index - 1, ny) + 1
    y = mod(index - 1, ny) + 1
    return (x, y)
end

"""
    generate_actions(pomdp::DronePOMDP)

Generate a list of all possible actions using integer indices.
"""
function generate_actions(pomdp::DronePOMDP{K}) where K
    nx, ny = pomdp.map_size
    all_actions = Vector{Int}()

    # Add the sampling action (1)
    push!(all_actions, SAMPLE_ACTION)

    # Add all possible move actions (2 to nx * ny + 1)
    for x in 1:nx, y in 1:ny
        push!(all_actions, action_to_index(pomdp, (x, y)))
    end

    # Add sensing actions (should start at 101 and go up to 101 + K - 1)
    for k in 1:K-1
        push!(all_actions, SENSING_START_INDEX + k)  # Ensure all K sensing actions exist
    end

    # Debugging: Check for missing actions
    expected_n_actions = SENSING_START_INDEX + K - 1  # This should match n_actions(pomdp)
    println("Expected n_actions: $expected_n_actions, Generated Actions: $(length(all_actions))")
    
    return all_actions
end

"""
    POMDPs.actions(pomdp::DronePOMDP)

Return the full action space as integer indices.
"""
POMDPs.actions(pomdp::DronePOMDP) = generate_actions(pomdp)

"""
    POMDPs.actionindex(pomdp::DronePOMDP, a::Int)

Return the integer action index for `a`. 
"""
POMDPs.actionindex(pomdp::DronePOMDP, a::Int) = a  # Already an integer

"""
    POMDPs.action(pomdp::DronePOMDP, a_idx::Int)

Convert an integer action index back to its corresponding `(x,y)` action.
"""
POMDPs.action(pomdp::DronePOMDP, a_idx::Int) = index_to_action(pomdp, a_idx)