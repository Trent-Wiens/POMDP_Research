"""
@article{egorov2017pomdps,
  author  = {Maxim Egorov and Zachary N. Sunberg and Edward Balaban and Tim A. Wheeler and Jayesh K. Gupta and Mykel J. Kochenderfer},
  title   = {{POMDP}s.jl: A Framework for Sequential Decision Making under Uncertainty},
  journal = {Journal of Machine Learning Research},
  year    = {2017},
  volume  = {18},
  number  = {26},
  pages   = {1-5},
  url     = {http://jmlr.org/papers/v18/16-300.html}
}
"""

using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using RockSample
using Cairo
using DiscreteValueIteration

function make_sub_POMDP(pos, map_size, rock_pos, rock_probs)

    horizon = 3

    maxx = pos[1] + horizon
    minx = pos[1] - horizon
    maxy = pos[2] + horizon
    miny = pos[2] - horizon

    # display(rock_pos)

    # Ensure the sub-POMDP's map size is within the bounds of the main POMDP

    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

    sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

    sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)

    display(sub_rocks)
    # display(sub_map_size)
    # exit()

    sub_pomdp = RockSamplePOMDP(map_size = sub_map_size,
                                rocks_positions = sub_rocks, 
                                init_pos = pos)

    # display(sub_pomdp)

    numRock = length(sub_pomdp.rocks_positions)

    rockings = zeros(numRock)

    for i = 1:numRock #get the probabilities for just the rocks in the subpomdp
        posit = sub_pomdp.rocks_positions[i] 
        idx = findfirst(==(posit), rock_probs.vals)       
        rockings[i] = isnothing(idx) ? 0.0 : rock_probs.probs[idx]
    end

    notRockings = ones(numRock) .- rockings #get the not rock probabilities

    states = ordered_states(sub_pomdp) #get all the states in the subpomdp
    indc = findall(s -> s.pos == pos, states) #get the indices of the states that share the same position as the init pos
    init_states = states[indc]
    init_probs = zeros(length(init_states)) #initialise the probabilities for the initial states

    scaling = 1 / (numRock * 2^(numRock - 1)) #scaling factor to ensure the probabilities sum to 1
    j = 1;

    for s in init_states

        mask = s.rocks

        init_probs[j] = sum(mask[i] == 0 ? notRockings[i] : rockings[i] for i in 1:numRock) * scaling

        j = j + 1
    end

    init_state = SparseCat(init_states, init_probs)

    # print(init_state)

    return sub_pomdp, init_state, rock_probs

end

function get_next_init_state(policy, pomdp, rock_probs)

    # create an updater for the pollicy
    up = updater(policy)
    # get the initial belief state
    b0 = initialize_belief(up, initialstate(pomdp)) # initialize belief state
    # display(b0)

    # init_state = initialstate(sub_pomdp)
    # next_action = action(policy, init_state) 

    state = nothing
    action = nothing
    obs = nothing
    rew = nothing

    # simulate the first step after the inital state
    for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps=1)
        println("in state $s")
        println("took action $a")
        println("received observation $o and reward $r")
        println("=====================================================")

        state = s
        action = a
        obs = o
        rew = r
    end


    # get the next state after the iniital state
    trans = transition(pomdp, state, action)

    display(trans)

    #initialise the belief after the first action has been taken
    b1 = update(up, b0, action, obs)

    # trim the belief state to only states that are in the same position as the next belief
    S = eltype(b1.state_list)
    next_states = S[]                  
    next_probs  = Float64[]           

    # Collect states/probs that share the current position
    for (s, p) in zip(b1.state_list, b1.b)
        if s.pos == trans.val.pos
            push!(next_states, s)
            push!(next_probs,  p)
        end
    end

    # display(next_states)
    # display(next_probs) 
    
    thisSum = 0
    
    for r = 1:length(pomdp.rocks_positions)
        
        i = 1
        thisSum = 0
        for s in next_states
    
            if s.rocks[r] == true
                thisSum += next_probs[i]
                # println("true")
                # println(next_probs[i])
                # println(thisSum)
                # println("-----")
            else
                # println("false") 
                # println(thisSum)
                # println("-----")
            end
    
            i += 1
        end
    
        # println("total $r: $thisSum")
    
        # probs[r] = thisSum
    
        thisRock = pomdp.rocks_positions[r]
    
        ind = findfirst(==(thisRock), rock_probs.vals)
    
        # print(thisRock)
        # println(ind)
    
        rock_probs.probs[ind] = thisSum    
    
        # print(probs)
    
    end
    next_init_state = SparseCat(next_states, next_probs)
    # display(next_init_state)

    #return the belief state
    return next_init_state.vals[1].pos
    # return SparseCat(next_states, next_probs), rock_probs

end

rng = MersenneTwister(1)

start_time = time_ns() #start time

# initial large POMDP of the whole space
pomdp = RockSamplePOMDP(map_size = (15,15),
                        rocks_positions=[(2, 1), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6,12), (12, 14)],
                        sensor_efficiency=20.0,
                        discount_factor=0.95,
                        good_rock_reward = 20.0
                        )

# states = ordered_states(pomdp)
# display(states)

#initialize rock_probs for belief state
rock_probs = SparseCat(pomdp.rocks_positions, [0.5 for _ in 1:length(pomdp.rocks_positions)])

# # make sub pomdp based on starting position
# sub_pomdp, init_state = make_sub_POMDP([3,4], pomdp.map_size, pomdp.rocks_positions, rock_probs)
# POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = init_state


# # subpomdp_sates = ordered_states(sub_pomdp)
# # display(sub_pomdp)

# # solve sub POMDP
# solver = SARSOPSolver(precision=1e-3; max_time=10.0, verbose=false) #use SARSOP solver
# policy = solve(solver, sub_pomdp) # get policy using SARSOP solver

# #get the next initial belief state
# next_init_state, rock_probs = get_next_init_state(policy, sub_pomdp, rock_probs)

# display(rock_probs)

# # set the initialstate to the next_init_state
# POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = next_init_state

# pos = next_init_state.vals[1].pos # get the position of the next initial state

for i in 1:20

    global pomdp
    global rock_probs
    global next_init_state
    global pos
    if i == 1
        pos = [1,1]
    else
        # pos = next_init_state.vals[1].pos
    end

    println("Iteration: $i")
    println("Current Position: $pos")

    sub_pomdp, init_state, rock_probs= make_sub_POMDP(pos, pomdp.map_size, pomdp.rocks_positions, rock_probs)

    POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = init_state

    solver = SARSOPSolver(precision=1e-3; max_time=10.0, verbose=false) #use SARSOP solver
    policy = solve(solver, sub_pomdp) # get policy using SARSOP solver

    #get the next initial belief state
    pos = get_next_init_state(policy, sub_pomdp, rock_probs)

    display(rock_probs)
    # POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = next_init_state

    # pos = next_init_state.vals[1].pos # get the position of the next initial state


    println("============================================")

end

# to next pomdp

# for i in 1:20

#     println("Iteration: $i")

#     println("Current Position: $pos")
    
#     # make sub pomdp
#     sub_pomdp = make_sub_POMDP(pos, pomdp.map_size, pomdp.rocks_positions)

#     # subpomdp_sates = ordered_states(sub_pomdp)
#     # display(sub_pomdp)

#     # make next_init_state

    
    
#     # solve sub POMDP
#     # solver = SARSOPSolver(precision=1e-3; max_time=10.0) #use SARSOP solver
#     policy = solve(solver, sub_pomdp) # get policy using SARSOP solver
    
#     #get the next initial belief state
#     next_init_state = get_next_init_state(policy, sub_pomdp)

#     pos = next_init_state.vals[1].pos # get the position of the next initial state
    
#     # set the initialstate to the next_init_state
#     POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = next_init_state


# end

# take the action

# If O(sub pomdp) < O(pomdp) and O(receding horizon process) < O(pomdp)
# O(sub pomdp) + O(receding horizon process) < O(pomdp) (?)


# display(next_action)
# display(init_state)

# take action

# repeat

# end_time = time_ns()
# elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
# println("Elapsed time: $elapsed_time seconds")

# sim = GifSimulator(; filename="RockSample.gif", max_steps=1, rng=MersenneTwister(1), show_progress=true)
# saved_gif = simulate(sim, sub_pomdp, policy)

# println("gif saved to: $(saved_gif.filename)")




