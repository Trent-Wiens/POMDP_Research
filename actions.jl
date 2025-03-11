print("actions\n")

const SENSE_ACTION = 0  # Special action for sensing

"""
    generate_actions(pomdp::DronePOMDP)

Generate a list of all possible actions: flying to any position or sensing.
"""
function generate_actions(pomdp::DronePOMDP{K}) where K
    nx, ny = pomdp.map_size
    all_actions = Vector{Tuple{Int,Int}}()

    # Add all possible move actions (flying to any other position)
    for x in 1:nx, y in 1:ny
        push!(all_actions, (x, y))
    end

    return vcat(all_actions, (SENSE_ACTION, SENSE_ACTION))  # Add sensing action
end

"""
    POMDPs.actions(pomdp::DronePOMDP)

Return the full action space, including flying to any location and sensing.
"""
POMDPs.actions(pomdp::DronePOMDP) = generate_actions(pomdp)

"""
    POMDPs.actions(pomdp::DronePOMDP, s::RSState)

Return valid actions from a given state `s`. 
All actions are valid unless the drone is in the terminal state.
"""
function POMDPs.actions(pomdp::DronePOMDP, s::RSState)
    if isterminal(pomdp, s)
        return [(SENSE_ACTION, SENSE_ACTION)]  # Only sensing allowed in the terminal state
    end
    return generate_actions(pomdp)
end

"""
    action_success_probability(pomdp::DronePOMDP, start_pos::RSPos, dest_pos::RSPos)

Calculate the probability of reaching `dest_pos` from `start_pos`. 
Probability decreases with distance.
"""
function action_success_probability(pomdp::DronePOMDP, start_pos::RSPos, dest_pos::RSPos)
    distance = norm(dest_pos .- start_pos, 2)  # Euclidean distance
    alpha = 0.3  # Tuning parameter for probability decay
    return exp(-alpha * distance)  # Higher distance -> lower probability
end



# function POMDPs.actions(pomdp::DronePOMDP{K}) where K
#     println("Testing: DronePOMDP map size is ", pomdp.map_size)
#     return 1:(pomdp.map_size[1] * pomdp.map_size[2])  # Example action space (all grid positions)
# end