using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using Cairo
using LinearAlgebra
using StaticArrays

include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: action_to_direction, is_fly_action, ACTION_SAMPLE, N_BASIC_ACTIONS, action_to_string

"""
    Receding horizon solver
"""


struct RecedingHorizonSolver <: Solver
    horizon::Int        # Size of horizon
    solver::Solver      # Sovled used in sub-POMDP
end

function create_comparison_visualization()
    # Set random seed for reproducibility
    rng = MersenneTwister(42)

    # Create a 7x7 DroneRockSample POMDP with 8 rocks
    rock_positions = [(2, 1), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6,12), (12, 14)]

    # (5, 12), (14, 10), (11, 9), (9, 13), (13, 3), (7, 14), (12, 11), (15, 15)

    pomdp = DroneRockSamplePOMDP(
        map_size=(15, 15),
        max_map_size=(15, 15),
        rocks_positions=rock_positions,
        init_pos=(1, 1),  # Bottom left corner
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=20.0,
        fly_penalty=-0.2
    )

    println("Created DroneRockSample POMDP with dimensions $(pomdp.map_size) and $(length(pomdp.rocks_positions)) rocks")

    # 1. Solve using SARSOP regularly

    #skip for now

    #2. Solve using receding horizon

    println("Solving with Receding Horizon...")

    horizon = 3  # Set horizon size
    pos = [1, 1]
    rock_beliefs = Dict()

    for rock in pomdp.rocks_positions
        rock_beliefs[rock] = 0.5
    end

    steps = []

    count = 0
    term = false

    while count < 100 && term == false

        if pos[1] > pomdp.map_size[1]
            term = true
            break
        end

        # set horizon boundaries

        center_pos = pos
        xmin = max(1, center_pos[1] - horizon)
        ymin = max(1, center_pos[2] - horizon)
        xmax = min(pomdp.map_size[1], center_pos[1] + horizon)
        ymax = min(pomdp.map_size[2], center_pos[2] + horizon)
        xsize = xmax - xmin + 1
        y_size = ymax - ymin + 1

        println("Horizon boundaries: xmin = $(xmin), ymin = $(ymin), xmax = $(xmax), ymax = $(ymax)")

        sub_rocks = []
        rock_mapping = Dict()

        #add uncertain rocks to sub-POMDP
        for rock in pomdp.rocks_positions
            if rock[1] >= xmin && rock[1] <= xmax && rock[2] >= ymin && rock[2] <= ymax && 0.1 < rock_beliefs[rock] < 0.99
                push!(sub_rocks, rock)
                rock_mapping[rock] = length(sub_rocks)  # Map original rock to its index in sub-POMDP
            end
        end

        #get next closest rock

        if length(sub_rocks) == 0
            println("No rocks in sub-POMDP, adding next closest rock.")
            closestRock = nothing
            dist = Inf

            for rock in pomdp.rocks_positions

                # println("Checking rock: $rock with belief $(rock_beliefs[rock])")

                if 0.1 < rock_beliefs[rock] < 0.99
                    thisDist = norm([rock[1] - pos[1], rock[2] - pos[2]])

                    if thisDist < dist
                        dist = thisDist
                        closestRock = rock
                    end
                end


            end

            #if no rocks are uncertain, just add no rocks, otherwise add closest rock

            if closestRock !== nothing
                push!(sub_rocks, closestRock)
                rock_mapping[closestRock] = length(sub_rocks)  # Map original rock to its index in sub-POMDP                
            end

        end

        # Create sub-POMDP with current horizon and uncertain rocks

        sub_pomdp = DroneRockSamplePOMDP(
            map_size=(xsize, y_size),
            max_map_size=pomdp.map_size,  #keep track of original map size
            rocks_positions=sub_rocks,
            init_pos=(pos[1] - xmin + 1, pos[2] - ymin + 1),  # Adjust initial position for sub-POMDP
            sensor_efficiency=pomdp.sensor_efficiency,
            discount_factor=pomdp.discount_factor,
            good_rock_reward=pomdp.good_rock_reward,
            fly_penalty=pomdp.fly_penalty
        )

        # Update rock beliefs in sub-POMDP

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

                state = RSState{K}(RSPos(sub_pomdp.init_pos...), rocks_state)

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

        println("Created sub-POMDP with dimensions $(sub_pomdp.map_size) and $(length(sub_pomdp.rocks_positions)) rocks")

        # Solve sub-POMDP using SARSOP

        solver = SARSOPSolver(precision=1e-2, max_time=5.0)
        sub_policy = solve(solver, sub_pomdp)

        println("Sub-POMDP solved.")

        #get the next action based on current belief

        chosenAct = action(sub_policy, belief)
        actionString = action_to_string(sub_pomdp, chosenAct)

        println("chosen action: $(chosenAct) ($actionString) ")

        current_state = rand(rng, belief)
        next_state_distribution = transition(sub_pomdp, current_state, chosenAct)
        next_state = rand(rng, next_state_distribution)

        # Get observation
        obs = rand(rng, POMDPs.observation(sub_pomdp, chosenAct, next_state))

        # Update rock beliefs if it was a sensing action
        if chosenAct > N_BASIC_ACTIONS
            local_rock_idx = chosenAct - N_BASIC_ACTIONS
            # Find which original rock this corresponds to
            for (orig_idx, local_idx) in rock_mapping
                local_rock_pos = sub_pomdp.rocks_positions[local_rock_idx]
                rock_num = rock_mapping[local_rock_pos]
                if local_idx == local_rock_idx
                    # Update belief about this rock based on observation
                    if obs == 1 # good rock observation
                        # Update using Bayes rule
                        p_good = rock_beliefs[orig_idx]
                        efficiency = 0.5 * (1.0 + exp(-norm(collect(rock_positions[rock_num]) - pos) * log(2) / sub_pomdp.sensor_efficiency))

                        # P(good|obs) = P(obs|good)P(good)/P(obs)
                        posterior = (efficiency * p_good) /
                                    (efficiency * p_good + (1.0 - efficiency) * (1.0 - p_good))
                        rock_beliefs[orig_idx] = posterior

                        println("Updated belief for rock $orig_idx: $p_good → $posterior")
                    elseif obs == 2 # bad rock observation
                        p_good = rock_beliefs[orig_idx]
                        efficiency = 0.5 * (1.0 + exp(-norm(collect(rock_positions[rock_num]) - pos) * log(2) / sub_pomdp.sensor_efficiency))
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

        if chosenAct == ACTION_SAMPLE #sample rock and remove it from the set of rock_positions
            rock_idx = findfirst(isequal(next_state.pos), pomdp.rocks_positions)
            # println(rock_idx)
            # println(typeof(pos))
            # println(pos[1])
            # println(pos[2])
            # println(rock_positions[rock_idx][1])
            # println(rock_positions[rock_idx][2])
            # println(typeof(rock_positions[rock_idx]))
            if rock_idx !== nothing && pos[1] == rock_positions[rock_idx][1] && pos[2] == rock_positions[rock_idx][2]
                deleteat!(rock_positions, rock_idx) # Remove the sampled rock from the global rock positions
                println("Sampled rock at position $(next_state.pos), removing from beliefs.")
            else
                println("No rock at sampled position $(next_state.pos), no update to beliefs.")
            end
            
        end


        # # Update rock beliefs if it was a sensing action
        # if chosenAct > N_BASIC_ACTIONS
        #     local_rock_idx = chosenAct - N_BASIC_ACTIONS
        #     # println("Local rock index: $local_rock_idx")
        #     # Find which original rock this corresponds to
        #     # for (orig_idx, local_idx) in rock_mapping
        #     # println("pos = $(pos)")
        #     local_rock_pos = sub_pomdp.rocks_positions[local_rock_idx]
        #     # println("Local rock position: $local_rock_pos")
        #     rock_num = rock_mapping[local_rock_pos]
        #     # println("Rock number in original POMDP: $rock_num")
        #         # if pos == local_rock_pos
        #             # Update belief about this rock based on observation
        #             if obs == 1 # good rock observation
        #                 # Update using Bayes rule
        #                 p_good = rock_beliefs[local_rock_pos]
        #                 efficiency = 0.5 * (1.0 + exp(-1 * norm(collect(rock_positions[rock_num]) .- pos) * log(2) / sub_pomdp.sensor_efficiency))

        #                 # P(good|obs) = P(obs|good)P(good)/P(obs)
        #                 posterior = (efficiency * p_good) /
        #                             (efficiency * p_good + (1.0 - efficiency) * (1.0 - p_good))
        #                 rock_beliefs[local_rock_pos] = posterior

        #                 println("Updated belief for rock $local_rock_pos: $p_good → $posterior")
        #             elseif obs == 2 # bad rock observation
        #                 p_good = rock_beliefs[local_rock_pos]
        #                 # rock_pos = collect(rock_positions[local_rock_pos])
        #                 # println("Rock position: $local_rock_pos")
        #                 # print(rock_pos)
        #                 # print(pos)
        #                 efficiency = 0.5 * (1.0 + exp(-1 * norm(collect(rock_positions[rock_num]) .- pos) * log(2) / sub_pomdp.sensor_efficiency))
        #                 # P(good|obs_bad) = P(obs_bad|good)P(good)/P(obs_bad)
        #                 posterior = ((1.0 - efficiency) * p_good) /
        #                             ((1.0 - efficiency) * p_good + efficiency * (1.0 - p_good))
        #                 rock_beliefs[local_rock_pos] = posterior

        #                 println("Updated belief for rock $local_rock_pos: $p_good → $posterior")
        #             end
        #             # break
        #         # else
        #             # println("No rock at local index $local_rock_idx")
        #         # end
        #     # end
        # end

        println("Next state sub-pomdp pos: $(next_state.pos)")

        # Convert next state's position to global coordinates
        new_global_pos = [
            next_state.pos[1] + xmin - 1,
            next_state.pos[2] + ymin - 1
        ]

        #clamp the position to the map size
        global_clamp = (new_global_pos[1], min(new_global_pos[2], pomdp.map_size[2]))
        global_clamp = (max(new_global_pos[1], 1), max(new_global_pos[2], 1))

        new_global_pos = [
            global_clamp[1],
            global_clamp[2]
        ]

        println("New global position: $new_global_pos")


        # Calculate reward
        r = reward(sub_pomdp, current_state, chosenAct, next_state)
        println("Action resulted in reward: $r")

        # # Store step for later visualization 
        # push!(all_steps, Dict(
        #     :pos => pos,
        #     :action => actString,
        #     :reward => r,
        #     :rock_beliefs => copy(rock_beliefs)
        # ))

        # Update position
        pos = new_global_pos

        println("=========================================================")




    end

end


result = create_comparison_visualization()





