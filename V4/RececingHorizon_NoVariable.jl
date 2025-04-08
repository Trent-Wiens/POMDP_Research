using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using Cairo
using LinearAlgebra
using Printf
using DiscreteValueIteration
using StaticArrays
using StatsBase: ProbabilityWeights  # For the weighted sampling

# Include the DroneRockSample module
include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: action_to_direction, is_fly_action, ACTION_SAMPLE, N_BASIC_ACTIONS, action_to_string



"""
    RecedingHorizonSolver

A solver that uses a receding horizon approach to solve POMDPs.
It solves the full POMDP at each step but filters actions to stay within horizon.
"""
struct RecedingHorizonSolver <: Solver
    horizon::Int               # Receding horizon distance
    solver::Solver             # Base solver to use for the sub-POMDP
    rng::AbstractRNG           # Random number generator
end

# Create a fixed policy for replay
mutable struct FixedPolicy <: Policy
    steps::Vector{Dict}
    current_step::Int
end

function POMDPs.action(policy::FixedPolicy, b)
    if policy.current_step <= length(policy.steps)
        a = policy.steps[policy.current_step][:a]
        policy.current_step += 1
        return a
    else
        return 0  # Default action (shouldn't be reached)
    end
end

# Default constructor
RecedingHorizonSolver(horizon::Int, solver::Solver) =
    RecedingHorizonSolver(horizon, solver, MersenneTwister(0))

"""
    filter_valid_actions

Filter actions that would move the agent too far from the current position.
"""
function filter_valid_actions(pomdp::DroneRockSamplePOMDP, state, actions_list, horizon)
    valid_actions = Int[]

    for a in actions_list
        # For sample and sense actions, always valid
        if a == ACTION_SAMPLE || a > N_BASIC_ACTIONS
            push!(valid_actions, a)
            continue
        end

        # For fly actions, check if they move too far
        if is_fly_action(a)
            direction = action_to_direction(a)
            distance = abs(direction[1]) + abs(direction[2])  # Manhattan distance

            if distance <= horizon
                push!(valid_actions, a)
            end
        end
    end

    return valid_actions
end

"""
    simulate_with_receding_horizon

Simulate a trajectory using the receding horizon approach.
"""
function simulate_with_receding_horizon(
    solver::RecedingHorizonSolver,
    pomdp::DroneRockSamplePOMDP,
    init_state,
    max_steps::Int=100
)
    current_state = init_state
    trajectory = [(s=current_state, a=0, r=0.0, sp=current_state)]
    total_reward = 0.0
    discount_factor = discount(pomdp)
    discount_cumulative = 1.0

    println("\n=== Starting Receding Horizon Simulation ===")
    println("Initial state: $current_state")

    for step in 1:max_steps
        println("\n--- Step $step ---")
        println("Current state: $current_state")

        if isterminal(pomdp, current_state)
            println("Reached terminal state. Simulation complete.")
            break
        end

        # Get all possible actions for the current state
        all_actions = actions(pomdp, current_state)

        # Filter to only include actions within the horizon
        valid_actions = filter_valid_actions(pomdp, current_state, all_actions, solver.horizon)

        if isempty(valid_actions)
            println("No valid actions within horizon! Using all actions.")
            valid_actions = all_actions
        else
            println("Filtered to $(length(valid_actions)) valid actions within horizon $(solver.horizon).")
        end

        # Solve the full POMDP (shorter time limit)
        adjusted_solver = SARSOPSolver(
            precision=solver.solver.precision,
            max_time=min(solver.solver.max_time, 3.0),
            verbose=solver.solver.verbose
        )
        policy = solve(adjusted_solver, pomdp)

        # Create a deterministic belief centered on the current state
        belief = SparseCat([current_state], [1.0])

        # Get the best action for the current state
        action = POMDPs.action(policy, belief)

        # If the best action is not in valid_actions, choose the best valid action
        if !(action in valid_actions) && !isempty(valid_actions)
            println("Best action $(action_to_string(pomdp, action)) is outside horizon.")

            # Choose a reasonable valid action (can be improved)
            # For simplicity, we'll take the first valid action
            action = valid_actions[1]
            println("Choosing valid action: $(action_to_string(pomdp, action))")
        end

        println("Selected action: $(action_to_string(pomdp, action))")

        # Execute the action
        sp = rand(solver.rng, transition(pomdp, current_state, action))
        r = reward(pomdp, current_state, action, sp)

        println("Next state: $sp")
        println("Reward: $r")

        # Update the total reward with discounting
        total_reward += discount_cumulative * r
        discount_cumulative *= discount_factor

        # Store the step in the trajectory
        push!(trajectory, (s=current_state, a=action, r=r, sp=sp))

        # Update the current state
        current_state = sp
    end

    println("\n=== Simulation Complete ===")
    println("Total reward: $total_reward")
    println("Trajectory length: $(length(trajectory))")

    return (trajectory=trajectory, total_reward=total_reward)
end

"""
    convert_trajectory_to_steps

Convert a trajectory to a series of rendering steps for visualization.
"""
function convert_trajectory_to_steps(trajectory, pomdp)
    steps = []

    for i in 1:length(trajectory)-1
        step = Dict(
            :s => trajectory[i].s,
            :a => trajectory[i].a,
            :r => trajectory[i].r,
            :sp => trajectory[i+1].s
        )
        push!(steps, step)
    end

    return steps
end

"""
    manual_simulate_sarsop

Manually simulate using a SARSOP policy.
"""
function manual_simulate_sarsop(
    pomdp::DroneRockSamplePOMDP,
    policy::AlphaVectorPolicy,
    init_state,
    rng::AbstractRNG,
    max_steps::Int=100
)
    current_state = init_state
    trajectory = [(s=current_state, a=0, r=0.0, sp=current_state)]
    total_reward = 0.0
    discount_factor = discount(pomdp)
    discount_cumulative = 1.0

    println("\n=== Starting Full SARSOP Simulation ===")
    println("Initial state: $current_state")

    for step in 1:max_steps
        println("\n--- Step $step (Full SARSOP) ---")
        println("Current state: $current_state")

        if isterminal(pomdp, current_state)
            println("Reached terminal state. Simulation complete.")
            break
        end

        # Create a deterministic belief centered on the current state
        belief = SparseCat([current_state], [1.0])

        # Get the action from the policy
        action = POMDPs.action(policy, belief)

        println("Selected action: $(action_to_string(pomdp, action))")

        # Execute the action
        sp = rand(rng, transition(pomdp, current_state, action))
        r = reward(pomdp, current_state, action, sp)

        println("Next state: $sp")
        println("Reward: $r")

        # Update the total reward with discounting
        total_reward += discount_cumulative * r
        discount_cumulative *= discount_factor

        # Store step in trajectory
        push!(trajectory, (s=current_state, a=action, r=r, sp=sp))

        # Update the current state
        current_state = sp
    end

    println("\n=== Full SARSOP Simulation Complete ===")
    println("Total reward: $total_reward")

    return (trajectory=trajectory, total_reward=total_reward)
end

"""
    create_comparison_visualization

Create GIFs comparing the full SARSOP solution with the receding horizon approach.
"""
function create_comparison_visualization()
    # Set random seed for reproducibility
    rng = MersenneTwister(42)

    # Create a 7x7 DroneRockSample POMDP with 8 rocks
    rock_positions = [(2,1), (3, 7), (5, 12), (8, 4), (11, 9), (14, 10), (6, 6), (9, 13), (1, 5), (13, 3), (7, 14), (4, 8), (10, 2), (12, 11), (15, 15)]

    pomdp = DroneRockSamplePOMDP(
        map_size=(15, 15),
        rocks_positions=rock_positions,
        init_pos=(1, 1),  # Bottom left corner
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=20.0,
        fly_penalty=-0.2
    )

    println("Created DroneRockSample POMDP with dimensions $(pomdp.map_size) and $(length(pomdp.rocks_positions)) rocks")

    ####################################################################################################
    # 1. Full SARSOP solution

    start_time = time_ns()

    # Create a smaller example first for testing
    # pomdp = DroneRockSamplePOMDP(
    #     map_size=(15, 15),  # Reduced map size for testing
    #     rocks_positions=rock_positions,
    #     sensor_efficiency=20.0,
    #     discount_factor=0.95,
    #     good_rock_reward=20.0,
    #     fly_penalty=-0.2
    # )

    # println("POMDP created successfully")

    # # Print the action space size
    # println("Number of actions: ", length(actions(pomdp)))
    # println("Basic actions: ", N_BASIC_ACTIONS)
    # println("Total rocks: ", length(pomdp.rocks_positions))

    # # Get the list of states
    # states = ordered_states(pomdp)
    # println("Number of states: ", length(states))

    # # Solve the POMDP using SARSOP with shorter time for testing
    # println("Solving with SARSOP...")
    # solver = SARSOPSolver(precision=1e-2, max_time=5.0)  # Reduced precision and time
    # policy = POMDPs.solve(solver, pomdp)  # Explicitly use POMDPs.solve

    # end_time = time_ns()
    # elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
    # println("Elapsed time: $elapsed_time seconds")

    # # Create a GIF of the simulation
    # println("Creating simulation GIF...")
    # sim = GifSimulator(
    #     filename="DroneRockSample.gif",
    #     max_steps=50,  # Reduced steps for testing
    #     rng=MersenneTwister(1),
    #     show_progress=true  # Enable progress display
    # )

    # saved_gif = simulate(sim, pomdp, policy)

    # println("GIF saved to: $(saved_gif.filename)")


    ####################################################################################################

    # 2. Receding Horizon approach
    println("\n=== Running Receding Horizon Solution ===")

    horizon = 3 # horizon size, how many spots to include in the sub POMDP
    pos = [1, 1] # initial position
    rng = MersenneTwister(42) # For reproducibility
    rock_beliefs = Dict() # Track beliefs about each rock's state

    # Initialize rock beliefs - assuming 50/50 chance of good/bad initially
    for (i, rock) in enumerate(rock_positions)
        rock_beliefs[i] = 0.5 # Probability the rock is good
    end

    # For visualization, store the sequence of steps
    all_steps = []

    while true # Continue until we break due to reaching terminal state
        println("\n=== Position: $pos ===")

        # Check if we've reached terminal state
        if pos[1] > pomdp.map_size[1]
            println("Reached exit! Terminal state achieved.")
            break
        end

        # Get receding horizon boundaries
        center_pos = pos
        x_min = max(1, center_pos[1] - horizon)
        x_max = min(pomdp.map_size[1], center_pos[1] + horizon)
        y_min = max(1, center_pos[2] - horizon)
        y_max = min(pomdp.map_size[2], center_pos[2] + horizon)
        x_size = x_max - x_min + 1
        y_size = y_max - y_min + 1

        println("Horizon boundaries: x=[$(x_min),$(x_max)], y=[$(y_min),$(y_max)]")

        # # Get rocks within horizon
        # sub_rocks = []
        # rock_mapping = Dict() # Maps original rock indices to sub-POMDP indices

        # for (i, rock) in enumerate(rock_positions)
        #     if x_min <= rock[1] <= x_max && y_min <= rock[2] <= y_max
        #         # Convert to local coordinates
        #         local_rock = (rock[1] - x_min + 1, rock[2] - y_min + 1)
        #         push!(sub_rocks, local_rock)
        #         rock_mapping[i] = length(sub_rocks) # Original index -> local index
        #         println("Rock at $rock (global) → $local_rock (local) is within horizon")
        #     end
        # end

        # # Convert position to local coordinates
        # local_pos = (pos[1] - x_min + 1, pos[2] - y_min + 1)

        # Get rocks within horizon, but only include uncertain rocks
sub_rocks = []
rock_mapping = Dict() # Maps original rock indices to sub-POMDP indices

# First, add uncertain rocks within the horizon
for (i, rock) in enumerate(rock_positions)
    # Only include rocks we're uncertain about
    # Let's define "uncertain" as not being 95% sure either way
    if (0.05 < rock_beliefs[i] < 0.95) && x_min <= rock[1] <= x_max && y_min <= rock[2] <= y_max
        # Convert to local coordinates
        local_rock = (rock[1] - x_min + 1, rock[2] - y_min + 1)
        push!(sub_rocks, local_rock)
        rock_mapping[i] = length(sub_rocks)
        println("Uncertain rock at $rock (global) → $local_rock (local) is within horizon")
    end
end

# If no uncertain rocks in horizon, find the closest uncertain rock
if isempty(sub_rocks)
    closest_rock_idx = nothing
    closest_distance = Inf
    
    for (i, rock) in enumerate(rock_positions)
        # Only consider uncertain rocks
        if 0.05 < rock_beliefs[i] < 0.95
            # Calculate distance to this rock
            distance = sqrt((rock[1] - pos[1])^2 + (rock[2] - pos[2])^2)
            
            if distance < closest_distance
                closest_distance = distance
                closest_rock_idx = i
            end
        end
    end
    
    # If we found a closest uncertain rock, include it
    if closest_rock_idx !== nothing
        rock = rock_positions[closest_rock_idx]
        # If it's outside our horizon, expand the horizon to include it
        if rock[1] < x_min || rock[1] > x_max || rock[2] < y_min || rock[2] > y_max
            println("No uncertain rocks in horizon, expanding to include closest uncertain rock")
            
            # Recalculate boundaries to include this rock
            x_min = min(x_min, rock[1])
            x_max = max(x_max, rock[1]) 
            y_min = min(y_min, rock[2])
            y_max = max(y_max, rock[2])
            
            # Update dimensions
            x_size = x_max - x_min + 1
            y_size = y_max - y_min + 1
            
            # Convert the rock position to local coordinates
            local_rock = (rock[1] - x_min + 1, rock[2] - y_min + 1)
            push!(sub_rocks, local_rock)
            rock_mapping[closest_rock_idx] = length(sub_rocks)
            println("Added closest uncertain rock at $rock (global) → $local_rock (local)")
        end
    else
        println("No uncertain rocks left! Focusing on exit.")
        # If all rocks are certain, we can focus on reaching the exit
        # No need to add any rocks to the sub-POMDP
    end
end

# Also convert position to local coordinates
local_pos = (pos[1] - x_min + 1, pos[2] - y_min + 1)

        # Create sub POMDP
        sub_pomdp = DroneRockSamplePOMDP(
            map_size=(x_size, y_size),
            rocks_positions=sub_rocks,
            init_pos=local_pos,
            sensor_efficiency=20.0,
            discount_factor=0.95,
            good_rock_reward=20.0,
            fly_penalty=-0.2
        )

        println("Sub-POMDP created with $(length(sub_rocks)) rocks")

        # Create a custom initial belief that incorporates our rock knowledge
        if !isempty(sub_rocks)
            # Create all possible rock states based on our beliefs
            K = length(sub_rocks)
            states = Vector{RSState{K}}()
            probs = Vector{Float64}()

            # For each possible rock configuration
            for rock_config in Iterators.product(fill([false, true], K)...)
                # rock_config is now directly a tuple of booleans, one for each rock
                rocks_state = SVector{K,Bool}(collect(rock_config))

                state = RSState{K}(RSPos(local_pos...), rocks_state)

                # Calculate probability of this configuration
                prob = 1.0
                for (orig_idx, local_idx) in rock_mapping
                    p_good = rock_beliefs[orig_idx]
                    # If rock is good in this config, use p_good, otherwise use 1-p_good
                    prob *= rock_config[local_idx] ? p_good : (1.0 - p_good)
                end

                push!(states, state)
                push!(probs, prob)
            end
            # Normalize probabilities
            probs ./= sum(probs)

            # Create a belief
            belief = SparseCat(states, probs)
        else
            # No rocks in horizon, use default initialization
            belief = POMDPs.initialstate(sub_pomdp)
        end

        # Solve the POMDP
        println("Solving sub-POMDP...")
        solver = SARSOPSolver(precision=1e-2, max_time=5.0)
        policy = POMDPs.solve(solver, sub_pomdp)

        # Get action from policy
        actionNum = action(policy, belief)
        actString = action_to_string(sub_pomdp, actionNum)
        println("Chosen action: $actString")

        # Sample current state and apply action
        current_state = rand(rng, belief)
        next_state_distribution = transition(sub_pomdp, current_state, actionNum)
        next_state = rand(rng, next_state_distribution)

        # Get observation
        obs = rand(rng, POMDPs.observation(sub_pomdp, actionNum, next_state))

        # Update rock beliefs if it was a sensing action
        if actionNum > N_BASIC_ACTIONS
            local_rock_idx = actionNum - N_BASIC_ACTIONS
            # Find which original rock this corresponds to
            for (orig_idx, local_idx) in rock_mapping
                if local_idx == local_rock_idx
                    # Update belief about this rock based on observation
                    if obs == 1 # good rock observation
                        # Update using Bayes rule
                        p_good = rock_beliefs[orig_idx]
                        efficiency = 0.5 * (1.0 + exp(-norm(collect(rock_positions[orig_idx]) - pos) * log(2) / sub_pomdp.sensor_efficiency))

                        # P(good|obs) = P(obs|good)P(good)/P(obs)
                        posterior = (efficiency * p_good) /
                                    (efficiency * p_good + (1.0 - efficiency) * (1.0 - p_good))
                        rock_beliefs[orig_idx] = posterior

                        println("Updated belief for rock $orig_idx: $p_good → $posterior")
                    elseif obs == 2 # bad rock observation
                        p_good = rock_beliefs[orig_idx]
                        efficiency = 0.5 * (1.0 + exp(-norm(collect(rock_positions[orig_idx]) - pos) * log(2) / sub_pomdp.sensor_efficiency))
                        # P(good|obs_bad) = P(obs_bad|good)P(good)/P(obs_bad)
                        posterior = ((1.0 - efficiency) * p_good) /
                                    ((1.0 - efficiency) * p_good + efficiency * (1.0 - p_good))
                        rock_beliefs[orig_idx] = posterior

                        println("Updated belief for rock $orig_idx: $p_good → $posterior")
                    end
                    break
                end
            end
        end

        # Convert next state's position to global coordinates
        new_global_pos = [
            next_state.pos[1] + x_min - 1,
            next_state.pos[2] + y_min - 1
        ]

        # Calculate reward
        r = reward(sub_pomdp, current_state, actionNum, next_state)
        println("Action resulted in reward: $r")

        # Store step for later visualization 
        push!(all_steps, Dict(
            :pos => pos,
            :action => actString,
            :reward => r,
            :rock_beliefs => copy(rock_beliefs)
        ))

        # Update position
        pos = new_global_pos
        println("New position: $pos")

        # Optional: Create visualization for this step
        # This would create a GIF for each sub-POMDP
        # Uncomment if you want a GIF for each step
        # sim = GifSimulator(
        #     filename="step_$(length(all_steps)).gif",
        #     max_steps=50,
        #     rng=MersenneTwister(1),
        #     show_progress=true
        # )
        # simulate(sim, sub_pomdp, policy)

        # For debugging - limit number of steps
        # if length(all_steps) >= 20
            
        #     println("Reached maximum step limit!")
        #     break
        # end
    end

    # Final statistics
    println("\n=== Final Results ===")
    println("Steps taken: $(length(all_steps))")
    println("Final position: $pos")
    println("Total reward: $(sum(step[:reward] for step in all_steps))")

    for (i, step) in enumerate(all_steps)
        println("Step $i: Pos = $(step[:pos]), Action = ", step[:action])
    end

    # Creating a consolidated visualization 
    # (This part depends on how you want to visualize the results)

    # start_time = time_ns()


    # horizon = 3 #horizon size, how many spots to include in the sub POMDP
    # pos = [1, 1] #initial position

    # while pos != [~, -1]

    #     #get receding horizon states

    #     center_pos = pos
    #     x_min = max(1, center_pos[1] - horizon)
    #     x_max = min(pomdp.map_size[1], center_pos[1] + horizon)
    #     y_min = max(1, center_pos[2] - horizon)
    #     y_max = min(pomdp.map_size[2], center_pos[2] + horizon)
    #     x_size = x_max - x_min + 1
    #     y_size = y_max - y_min + 1

    #     println("Horizon boundaries: x=[$(x_min),$(x_max)], y=[$(y_min),$(y_max)]")

    #     # get rocks
    #     sub_rocks = []

    #     for rock in rock_positions
    #         if x_min <= rock[1] <= x_max && y_min <= rock[2] <= y_max
    #             push!(sub_rocks, rock)
    #             println("Rock at $rock is within horizon")
    #         end
    #     end

    #     #create sub POMDP

    #     sub_pomdp = DroneRockSamplePOMDP(
    #         map_size=(x_size, y_size),  # Reduced map size for testing
    #         rocks_positions=sub_rocks,
    #         init_pos=pos,
    #         sensor_efficiency=20.0,
    #         discount_factor=0.95,
    #         good_rock_reward=20.0,
    #         fly_penalty=-0.2
    #     )

    #     println("POMDP created successfully")

    #     # Print the action space size
    #     println("Number of actions: ", length(actions(sub_pomdp)))
    #     println("Basic actions: ", N_BASIC_ACTIONS)
    #     println("Total rocks: ", length(sub_pomdp.rocks_positions))

    #     # Get the list of states
    #     states = ordered_states(sub_pomdp)
    #     println("Number of states: ", length(states))

    #     # Solve the POMDP using SARSOP with shorter time for testing
    #     println("Solving with SARSOP...")
    #     solver = SARSOPSolver(precision=1e-2, max_time=5.0)  # Reduced precision and time
    #     policy = POMDPs.solve(solver, sub_pomdp)  # Explicitly use POMDPs.solve

    #     end_time = time_ns()
    #     elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
    #     println("Elapsed time: $elapsed_time seconds")

    #     belief = POMDPs.initialstate(sub_pomdp)
    #     actionNum = action(policy, belief)
    #     actString = action_to_string(sub_pomdp, actionNum)
    #     println(actString)

    #     println("actionNum = $(actionNum)")

    #     # Debug the structure of the belief object
    #     println("Belief type: ", typeof(belief))
    #     println("Available fields: ", fieldnames(typeof(belief)))
    #     for fn in fieldnames(typeof(belief))
    #         println("  $fn: ", getfield(belief, fn))
    #     end

    #     # Sample a specific state from your belief
    #     # current_state = rand(rng, belief.vals, ProbabilityWeights(belief.probs))
    #     current_state = rand(rng, belief)

    #     println("current_state = $(current_state)")


    #     # Get the next state distribution
    #     next_state_distribution = transition(sub_pomdp, current_state, actionNum)

    #     println("current_state = $(current_state)")

    #     # Sample a next state
    #     next_state = rand(rng, next_state_distribution)

    #     println("next_state = $(next_state)")

    #     pos = next_state.pos

    #     println(pos)



    #     # Get an observation
    #     observation = rand(rng, POMDPs.observation(sub_pomdp, actionNum, next_state))
    #     # Create a belief updater
    #     updater = POMDPTools.DiscreteUpdater(sub_pomdp)

    #     # Update the belief
    #     new_belief = update(updater, belief, actionNum, observation)

    #     println("new_belief = $(new_belief)")



    #     # Create a GIF of the simulation
    #     println("Creating simulation GIF...")
    #     sim = GifSimulator(
    #         filename="sub_pomdp.gif",
    #         max_steps=50,  # Reduced steps for testing
    #         rng=MersenneTwister(1),
    #         show_progress=true  # Enable progress display
    #     )

    #     saved_gif = simulate(sim, sub_pomdp, policy)

    #     println(saved_gif)

    #     println("GIF saved to: $(saved_gif.filename)")

    #     exit()

    # end


    # rh_solver = RecedingHorizonSolver(
    #     horizon,
    #     SARSOPSolver(precision=0.1, max_time=3.0, verbose=true),
    #     MersenneTwister(42)
    # )

    # Simulate with the receding horizon approach
    # rh_sim = simulate_with_receding_horizon(rh_solver, pomdp, init_state)

    # Convert trajectories to steps for rendering
    # full_steps = convert_trajectory_to_steps(full_sim.trajectory, pomdp)
    # rh_steps = convert_trajectory_to_steps(rh_sim.trajectory, pomdp)

    # Create GIFs for visualization
    # println("\n=== Creating GIFs ===")

    # # Full SARSOP GIF
    # full_sim_gif = GifSimulator(
    #     filename="FullSARSOP.gif",
    #     max_steps=length(full_steps),
    #     rng=MersenneTwister(1)
    # )

    # # Create and simulate with fixed policies
    # full_policy_replay = FixedPolicy(full_steps, 1)
    # rh_policy_replay = FixedPolicy(rh_steps, 1)

    # # Create GIFs
    # # simulate(full_sim_gif, pomdp, full_policy_replay, updater(full_policy), initialstate(pomdp))

    # # Receding Horizon GIF
    # rh_sim_gif = GifSimulator(
    #     filename="RecedingHorizon.gif",
    #     max_steps=length(rh_steps),
    #     rng=MersenneTwister(1)
    # )

    # simulate(rh_sim_gif, pomdp, rh_policy_replay, updater(full_policy), initialstate(pomdp))

    # println("GIFs created:")
    # println("  Full SARSOP: FullSARSOP.gif")
    # println("  Receding Horizon: RecedingHorizon.gif")

    # Return results
    return (
        pomdp=pomdp,
        # full_trajectory=full_sim.trajectory,
        # rh_trajectory=rh_sim.trajectory,
        # full_reward=full_sim.total_reward,
        # rh_reward=rh_sim.total_reward
    )
end

# Run the visualization
results = create_comparison_visualization()