# Fixed action structure for the drone version of RockSample
# The drone can fly to any cell within a 3-cell distance 
# in a single action, plus sample and sense actions

const MAX_FLIGHT_DISTANCE = 3

# Define action types
const ACTION_SAMPLE = 1
const ACTION_FLY_START = 2
# Sensing actions will start after the fly actions

# Calculate the number of possible fly actions based on the maximum flight distance
function count_fly_actions(max_distance)
    count = 0
    for dx in -max_distance:max_distance
        for dy in -max_distance:max_distance
            if dx == 0 && dy == 0
                continue  # Skip staying in place
            end
            if abs(dx) + abs(dy) <= max_distance  # Manhattan distance
                count += 1
            end
        end
    end
    return count
end

const N_FLY_ACTIONS = count_fly_actions(MAX_FLIGHT_DISTANCE)
const N_BASIC_ACTIONS = 1 + N_FLY_ACTIONS  # Sample + Fly actions

# Generate all possible flight directions within the max distance
function generate_flight_dirs()
    dirs = RSPos[]
    for dx in -MAX_FLIGHT_DISTANCE:MAX_FLIGHT_DISTANCE
        for dy in -MAX_FLIGHT_DISTANCE:MAX_FLIGHT_DISTANCE
            if dx == 0 && dy == 0
                continue  # Skip staying in place
            end
            if abs(dx) + abs(dy) <= MAX_FLIGHT_DISTANCE  # Manhattan distance
                push!(dirs, RSPos(dx, dy))
            end
        end
    end
    return Tuple(dirs)
end

# Define action directions as a constant - fixed to avoid recursive generation
const ACTION_DIRS = generate_flight_dirs()

# Update the actions function for the drone version
POMDPs.actions(pomdp::DroneRockSamplePOMDP{K}) where K = 1:N_BASIC_ACTIONS+K
POMDPs.actionindex(pomdp::DroneRockSamplePOMDP, a::Int) = a

function POMDPs.actions(pomdp::DroneRockSamplePOMDP{K}, s::RSState{K}) where K
    rock_idx = findfirst(==(s.pos), pomdp.rocks_positions)
    if rock_idx !== nothing
        return 1:N_BASIC_ACTIONS+K  # Allow sampling
    else
        return ACTION_SAMPLE+1:N_BASIC_ACTIONS+K  # Disallow sample
    end
end

# function POMDPs.actions(pomdp::DroneRockSamplePOMDP{K}, s::RSState) where K
#     if in(s.pos, pomdp.rocks_positions) # allow sampling only if on a rock
#         return actions(pomdp)
#     else
#         # sample not available, only flying and sensing
#         return ACTION_SAMPLE+1:N_BASIC_ACTIONS+K
#     end
# end

function POMDPs.actions(pomdp::DroneRockSamplePOMDP, b)
    # All states in a belief should have the same position
    # Get a representative state
    if b isa AbstractParticleBelief
        state = particles(b)[1]  # Use first particle as representative
    else
        state = rand(Random.GLOBAL_RNG, b)
    end
    return actions(pomdp, state)
end

# Function to convert action index to flight direction
function action_to_direction(a::Int)
    if a == ACTION_SAMPLE
        return RSPos(0, 0)  # No movement for sampling
    elseif a >= ACTION_FLY_START && a < ACTION_FLY_START + N_FLY_ACTIONS
        return ACTION_DIRS[a - ACTION_FLY_START + 1]
    else
        return RSPos(0, 0)  # No movement for sensing
    end
end

# Function to check if an action is a flying action
is_fly_action(a::Int) = a >= ACTION_FLY_START && a < ACTION_FLY_START + N_FLY_ACTIONS

# Function to check if an action is a sensing action
is_sense_action(a::Int, K::Int) = a >= N_BASIC_ACTIONS+1 && a <= N_BASIC_ACTIONS+K

# For visualization and debugging
function action_to_string(pomdp::DroneRockSamplePOMDP{K}, a::Int) where K
    if a == ACTION_SAMPLE
        return "Sample"
    elseif is_fly_action(a)
        dir = ACTION_DIRS[a - ACTION_FLY_START + 1]
        return "Fly($(dir[1]),$(dir[2]))"
    elseif is_sense_action(a, K)
        rock_ind = a - N_BASIC_ACTIONS
        return "Sense Rock $rock_ind"
    else
        return "Unknown"
    end
end