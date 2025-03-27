using POMDPs
using POMDPTools
using NativeSARSOP
using Random
using Plots
using Statistics
using StaticArrays

# Include the DroneRockSample module
include("DroneRockSample.jl")
using .DroneRockSample

# Define all constants and helper functions we need locally
# to avoid import issues
const LOCAL_MAX_FLIGHT_DISTANCE = 3
const LOCAL_ACTION_SAMPLE = 1
const LOCAL_ACTION_FLY_START = 2

# Calculate the number of possible fly actions based on the maximum flight distance
function count_local_fly_actions(max_distance)
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

const LOCAL_N_FLY_ACTIONS = count_local_fly_actions(LOCAL_MAX_FLIGHT_DISTANCE)
const LOCAL_N_BASIC_ACTIONS = 1 + LOCAL_N_FLY_ACTIONS  # Sample + Fly actions

# Generate all possible flight directions within the max distance
function generate_local_flight_dirs()
    dirs = []
    for dx in -LOCAL_MAX_FLIGHT_DISTANCE:LOCAL_MAX_FLIGHT_DISTANCE
        for dy in -LOCAL_MAX_FLIGHT_DISTANCE:LOCAL_MAX_FLIGHT_DISTANCE
            if dx == 0 && dy == 0
                continue  # Skip staying in place
            end
            if abs(dx) + abs(dy) <= LOCAL_MAX_FLIGHT_DISTANCE  # Manhattan distance
                push!(dirs, [dx, dy])
            end
        end
    end
    return dirs
end

const LOCAL_ACTION_DIRS = generate_local_flight_dirs()

function local_action_to_direction(a::Int)
    if a == LOCAL_ACTION_SAMPLE
        return [0, 0]  # No movement for sampling
    elseif a >= LOCAL_ACTION_FLY_START && a < LOCAL_ACTION_FLY_START + LOCAL_N_FLY_ACTIONS
        return LOCAL_ACTION_DIRS[a - LOCAL_ACTION_FLY_START + 1]
    else
        return [0, 0]  # No movement for sensing
    end
end

function local_is_fly_action(a::Int)
    return a >= LOCAL_ACTION_FLY_START && a < LOCAL_ACTION_FLY_START + LOCAL_N_FLY_ACTIONS
end

function local_is_sense_action(a::Int, K::Int)
    return a >= LOCAL_N_BASIC_ACTIONS+1 && a <= LOCAL_N_BASIC_ACTIONS+K
end

function analyze_value_function()
    # Create a small DroneRockSample POMDP for analysis
    pomdp = DroneRockSamplePOMDP(
        map_size = (5, 5),
        rocks_positions = [(2, 4), (4, 2)],
        sensor_efficiency = 20.0,
        discount_factor = 0.95,
        good_rock_reward = 20.0,
        fly_penalty = -0.2
    )
    
    println("POMDP created with $(length(pomdp.rocks_positions)) rocks")
    
    # Solve the POMDP using SARSOP
    println("Solving with SARSOP...")
    solver = SARSOPSolver(precision=1e-3, max_time=10.0)
    policy = POMDPs.solve(solver, pomdp)
    
    println("Policy successfully generated!")
    
    # The SARSOP policy contains alpha vectors that represent the value function
    alphas = policy.alphas
    println("Number of alpha vectors: ", length(alphas))
    
    # Extract action mappings
    actions_map = policy.action_map
    println("Action mapping: ", actions_map)
    
    # Debug - look at the possible action range
    println("Action space: ", actions(pomdp))
    println("Local action constants check:")
    println("  LOCAL_ACTION_SAMPLE: ", LOCAL_ACTION_SAMPLE)
    println("  LOCAL_ACTION_FLY_START: ", LOCAL_ACTION_FLY_START)
    println("  LOCAL_N_FLY_ACTIONS: ", LOCAL_N_FLY_ACTIONS)
    println("  LOCAL_N_BASIC_ACTIONS: ", LOCAL_N_BASIC_ACTIONS)
    
    # To understand the value function, we can sample some belief points
    # and compute their values under the policy
    
    # First, let's create a grid of belief points representing different certainties
    # about rock goodness
    
    rock_states = [[false, false], [false, true], [true, false], [true, true]]
    positions = [(x, y) for x in 1:pomdp.map_size[1] for y in 1:pomdp.map_size[2]]
    
    belief_values = []
    positions_list = []
    rock_beliefs = []
    best_actions = []
    
    # Sample a set of deterministic beliefs (perfect knowledge of state)
    for pos in positions
        for rocks in rock_states
            # Create a deterministic belief at this position with these rock states
            state = RSState(RSPos(pos...), SVector{2, Bool}(rocks))
            
            # Create a deterministic belief for this state
            belief = SparseCat([state], [1.0])
            
            # Compute the value of this belief
            value = POMDPs.value(policy, belief)
            
            # Determine the best action for this belief
            action = POMDPs.action(policy, belief)
            action_name = if action == LOCAL_ACTION_SAMPLE
                "Sample"
            elseif local_is_fly_action(action)
                dir = local_action_to_direction(action)
                "Fly($(dir[1]),$(dir[2]))"
            elseif local_is_sense_action(action, length(pomdp.rocks_positions))
                "Sense Rock $(action - LOCAL_N_BASIC_ACTIONS)"
            else
                "Unknown($action)"
            end
            
            push!(belief_values, value)
            push!(positions_list, pos)
            push!(rock_beliefs, rocks)
            push!(best_actions, action_name)
            
            println("Position: $pos, Rocks: $rocks, Value: $value, Best Action: $action_name")
        end
    end
    
    # Now we can make some visualizations of the value function
    
    # 1. Value function heat map for each rock state combination
    for (i, rock_state) in enumerate(rock_states)
        # Filter values for this rock state
        indices = findall(r -> r == rock_state, rock_beliefs)
        pos_subset = positions_list[indices]
        val_subset = belief_values[indices]
        
        # Create a matrix for the heat map
        value_grid = zeros(pomdp.map_size)
        for j in 1:length(pos_subset)
            x, y = pos_subset[j]
            value_grid[x, y] = val_subset[j]
        end
        
        # Plot the heat map
        heatmap(value_grid, 
                title="Value Function - Rocks: $rock_state",
                xlabel="X Position", 
                ylabel="Y Position",
                color=:viridis)
        
        # Mark rock positions
        for (k, rock_pos) in enumerate(pomdp.rocks_positions)
            annotate!(rock_pos[2], rock_pos[1], text("R$k", :white, :center, 8))
        end
        
        # Save the plot
        savefig("value_function_rocks_$(rock_state[1])_$(rock_state[2]).png")
    end
    
    # 2. Action maps for each rock state
    for (i, rock_state) in enumerate(rock_states)
        # Filter actions for this rock state
        indices = findall(r -> r == rock_state, rock_beliefs)
        pos_subset = positions_list[indices]
        act_subset = best_actions[indices]
        
        # Create a visual representation of the policy
        println("\nPolicy for Rock State: $rock_state")
        println("--------------------------")
        
        for y in pomdp.map_size[2]:-1:1
            line = ""
            for x in 1:pomdp.map_size[1]
                idx = findfirst(p -> p == (x, y), pos_subset)
                if idx === nothing
                    line *= " · "
                else
                    action = act_subset[idx]
                    if action == "Sample"
                        line *= " S "
                    elseif startswith(action, "Fly")
                        # Extract direction from Fly(dx,dy)
                        try
                            parts = split(replace(replace(action, "Fly(" => ""), ")" => ""), ",")
                            if length(parts) == 2
                                dx = parse(Int, parts[1])
                                dy = parse(Int, parts[2])
                                if dx > 0 && dy == 0
                                    line *= " → "
                                elseif dx < 0 && dy == 0
                                    line *= " ← "
                                elseif dx == 0 && dy > 0
                                    line *= " ↑ "
                                elseif dx == 0 && dy < 0
                                    line *= " ↓ "
                                elseif dx > 0 && dy > 0
                                    line *= " ↗ "
                                elseif dx > 0 && dy < 0
                                    line *= " ↘ "
                                elseif dx < 0 && dy > 0
                                    line *= " ↖ "
                                elseif dx < 0 && dy < 0
                                    line *= " ↙ "
                                else
                                    line *= " F "
                                end
                            else
                                line *= " F "
                            end
                        catch e
                            line *= " F "
                        end
                    elseif startswith(action, "Sense")
                        try
                            rock_num = parse(Int, split(action, " ")[3])
                            line *= " $(rock_num) "
                        catch e
                            line *= " ? "
                        end
                    else
                        line *= " ? "
                    end
                end
            end
            println(line)
        end
        println("--------------------------")
    end
    
    # 3. Analysis of value distribution
    println("\nValue Distribution Statistics:")
    println("Min value: ", minimum(belief_values))
    println("Max value: ", maximum(belief_values))
    println("Mean value: ", mean(belief_values))
    println("Median value: ", median(belief_values))
    
    # Plot a histogram of values
    histogram(belief_values, 
              bins=20, 
              title="Value Function Distribution",
              xlabel="Value", 
              ylabel="Frequency",
              legend=false)
    savefig("value_distribution.png")
    
    println("\nValue function analysis complete! Check the generated PNG files for visualizations.")
end

# Run the analysis
analyze_value_function()