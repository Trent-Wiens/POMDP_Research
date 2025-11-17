using POMDPs
using POMDPTools
using POMDPTools.Policies: PlaybackPolicy, RandomPolicy
using POMDPGifs
using POMDPModels
using NativeSARSOP
using Random
using RockSample
using Cairo
# using DiscreteValueIteration
# using Plots
# using LinearAlgebra
# using Statistics

pomdp = RockSamplePOMDP(map_size = (15, 15),
	rocks_positions = [(6, 14), (12, 12), (2, 10), (4,3), (5,10), (8,9)],
	# rocks_positions = [(6,6)], 
	init_pos = (1,1),
	sensor_efficiency = 20.0,
	discount_factor = 0.95,
	good_rock_reward = 20.0,
	bad_rock_penalty = -5.0
)

actionListNums = [3
3
3
9
2
2
9
3
3
6
2
2
2
3
3
11
2
10
2
2
2
5
5
5
1
8
3
2
2
2
2
1
3
3
3
3
3
3
7
7
3
3
3
3]


    is_zero_based = false

    # --- map indices -> concrete RockSample actions ---
    A = collect(actions(pomdp))  # action catalog in THIS POMDP instance (ordering can vary!)
    idxs = is_zero_based ? (actionListNums .+ 1) : copy(actionListNums)

    for (k,i) in enumerate(idxs)
        if !(1 <= i <= length(A))
            error("Index $(actionListNums[k]) (mapped to $i) is out of bounds for this POMDP's action set of length $(length(A)).")
        end
    end
    
    seq_actions = A[idxs]             # the concrete actions to play
    backup = RandomPolicy(pomdp)
    policy = PlaybackPolicy(seq_actions, backup)
    
    # --- render GIF for exactly the length of your sequence ---
    sim = GifSimulator(
        filename="ActionSequence.gif",
        max_steps=length(seq_actions),
        rng=MersenneTwister(1),
        show_progress=true
    )
    gif = simulate(sim, pomdp, policy, DiscreteUpdater(pomdp))
    println("GIF saved to: $(gif.filename)")


