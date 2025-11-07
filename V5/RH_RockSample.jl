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
using POMDPModels
using NativeSARSOP
using Random
using RockSample
using Cairo
using DiscreteValueIteration
using Plots
using LinearAlgebra
using Statistics
# using QMDP
# using DataFrames
# using CSV

function global2local(pos, sub_map)

	local_x = pos[1] - sub_map[1] + 1
	local_y = pos[2] - sub_map[2] + 1

	return [local_x, local_y]

end

function local2global(pos, sub_map)

	global_x = pos[1] + sub_map[1] - 1
	global_y = pos[2] + sub_map[2] - 1

	return [global_x, global_y]

end

function actionNum2word(action)

	if action == 1
		return "Sample"
	elseif action == 2
		return "North"
	elseif action == 3
		return "East"
	elseif action == 4
		return "South"
	elseif action == 5
		return "West"
	elseif action >= 6
		return "Sense Rock $(action - 5)"
	else
		return "Invalid Action"
	end

end

function obsNum2word(obs)

	if obs == 1
		return "Good"
	elseif obs == 2
		return "Bad"
	elseif obs == 3
		return "None"
	else
		return "Invalid Observation"
	end

end

function add_nearest_rock(sub_rocks, rock_pos)

	# println("new rock being added")

	not_in_sub = [x for x in rock_pos if !(Tuple(x) in sub_rocks)]
	minDist = 10000000
	chosen_rock = nothing
	for rock in not_in_sub
		thisDist = abs(pos[1] - rock[1]) + abs(pos[2] - rock[2])
		if thisDist < minDist
			minDist = thisDist
			chosen_rock = rock
			break
		end
	end

	push!(sub_rocks, Tuple(chosen_rock))

	return sub_rocks


end

function make_sub_POMDP(pos, map_size, rock_pos, rock_probs, pomdp)
    horizon = 3

    maxx = min(pos[1] + horizon, map_size[1])
    minx = max(pos[1] - horizon, 1)
    maxy = min(pos[2] + horizon, map_size[2])
    miny = max(pos[2] - horizon, 1)

    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

	# rocks_reloaded = SVector(SVector(1, 2), SVector(3, 7), SVector(6, 6), SVector(1, 5))


	# rock_dists = [abs(pos[1] - rock[1]) + abs(pos[2] - rock[2]) for rock in rock_pos]


    # Start with rocks already inside the horizon (GLOBAL coords)
    sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

	# sub_rock_indices = [findfirst(x -> Tuple(x) == rock, rock_pos) for rock in sub_rocks]

	println("subrocks: ", sub_rocks)




    # sub_rocks = map(x -> Tuple(x), sub_rocks) |> collect  # Vector{Tuple{Int,Int}}

    # # Build a helper to read the current global posterior for any rock coordinate
    global_vals_as_tuples = Tuple.(rock_probs.vals)
    get_prob = r -> begin
        gi = findfirst(==(r), global_vals_as_tuples)
        gi === nothing ? 0.0 : rock_probs.probs[gi]
    end

    # # Current posteriors for the in-horizon rocks (if empty, treat as zeros)
    # sub_probs = isempty(sub_rocks) ? Float64[] : [get_prob(r) for r in sub_rocks]

    # # Identify rocks OUTSIDE horizon and their probs
    # out_of_horizon = [(x, y) for (x, y) in rock_pos if !((sub_map[1] ≤ x ≤ sub_map[3]) && (sub_map[2] ≤ y ≤ sub_map[4]))]
    # out_of_horizon = map(x -> Tuple(x), out_of_horizon) |> collect
    # out_probs = Dict(r => get_prob(r) for r in out_of_horizon)

    # # === Receding-horizon augmentation rule ===
    # # If EVERY rock in the current horizon is ≤ low_thresh, we try to pull in
    # # up to `max_extra` rocks from outside with prob ≥ high_thresh,
    # # prioritizing nearest (Manhattan) to the agent.
    # need_boost = !isempty(sub_probs) && all(p -> p ≤ low_thresh, sub_probs)

    # if need_boost && !isempty(out_of_horizon)
    #     # Candidates: outside rocks with prob >= high_thresh and not already included
    #     candidates = [r for r in out_of_horizon if out_probs[r] ≥ high_thresh && !(r in sub_rocks)]

    #     if !isempty(candidates)
    #         # Sort by Manhattan distance to current agent pos; tie-break by higher prob
    #         sort!(candidates,
    #               by = r -> (abs(pos[1]-r[1]) + abs(pos[2]-r[2]), -out_probs[r]))
    #         # Add up to max_extra
    #         for r in Iterators.take(candidates, max_extra)
    #             push!(sub_rocks, r)
    #             @info "Added high-prob rock outside horizon" rock=r prob=out_probs[r]
    #         end
    #     else
    #         @info "No outside rocks meet high_thresh=$(high_thresh) to add."
    #     end
    # end

    # if isempty(sub_rocks)
    #     # pick nearest nonzero-prob outside rock if any
    #     nz = [(r, out_probs[r]) for r in out_of_horizon if out_probs[r] > 0.0]
    #     if !isempty(nz)
    #         dists = [abs(pos[1]-r[1]) + abs(pos[2]-r[2]) for (r, _) in nz]
    #         idx = argmin(dists)
    #         push!(sub_rocks, nz[idx][1])
    #         @info "Fallback: added nearest nonzero-prob outside rock" rock=nz[idx][1] prob=nz[idx][2]
    #     else
    #         @warn "No rocks available to include in sub-POMDP."
    #     end
    # end

	# make sure there are rocks in the sub pomdp
	# if isempty(sub_rocks)
	# 	println("no rocks in sub")
	# 	sub_rocks = add_nearest_rock(sub_rocks, rock_pos)
	# end

	#make sure each rock has a big enough prob
	rock_thresh = 0.25

	numrocks = length(rock_pos)

	rockpos = rock_probs.vals
	rockprob = rock_probs.probs

	highrocks = Tuple{Int64, Int64}[]
	lowrocks = Tuple{Int64, Int64}[]

	for i in 1:length(rockprob)

		if rockprob[i] > rock_thresh
			push!(highrocks, Tuple(rockpos[i]))

		else
			push!(lowrocks, Tuple(rockpos[i]))

		end

	end

	println(highrocks)
	println(lowrocks)

	# println(typeof(highrocks))

	# valuesiosca = intersect(highrocks, sub_rocks)

	# println(typeof(sub_rocks), typeof(highrocks))

	# # print(typeof(sub_rocks))

	# println("rocks not in highrocks: ", valuesiosca)


	if isempty(intersect(highrocks, sub_rocks))

		println("rock not in high rocks")

		# sub_rocks does not contain any highrocks (above the threshold)

		sub_rocks = add_nearest_rock(sub_rocks, highrocks)

	end



    # Sub-map size
    sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)

    # Convert GLOBAL sub_rocks to LOCAL coordinates for the sub-POMDP
    local_sub_rocks = Tuple.(global2local.(sub_rocks, Ref(sub_map)))

	# print(local_sub_rocks)

    # Local init pos
    locpos = global2local(pos, sub_map)

	# print(sub_rocks)

    # Build the sub-POMDP
    sub_pomdp = RockSamplePOMDP(
        map_size = sub_map_size,
        rocks_positions = collect(local_sub_rocks),   # Vector{Tuple{Int,Int}}
        init_pos = locpos,
        sensor_efficiency = pomdp.sensor_efficiency,
        discount_factor = pomdp.discount_factor,
        good_rock_reward = pomdp.good_rock_reward,
        bad_rock_penalty = pomdp.bad_rock_penalty,
        step_penalty = pomdp.step_penalty,
        exit_reward = pomdp.exit_reward,
        sensor_use_penalty = pomdp.sensor_use_penalty
    )

	println("sub_pomdp rocks: ", sub_pomdp.rocks_positions)

    # # Map each LOCAL rock index in the sub-POMDP back to its GLOBAL index (for posterior updates later)
    numRock = length(sub_pomdp.rocks_positions)
    global_idx_map = Vector{Union{Int, Nothing}}(undef, numRock)
    for i in 1:numRock
        g = sub_rocks[i]  # global tuple aligned with local_sub_rocks[i]
        global_idx_map[i] = findfirst(==(g), global_vals_as_tuples)
    end

    # Build initial belief over the sub-POMDP states using global posteriors
    rockings = zeros(numRock)
    for i in 1:numRock
        gi = global_idx_map[i]
        rockings[i] = (gi === nothing) ? 0.0 : rock_probs.probs[gi]
    end
    notRockings = ones(numRock) .- rockings

    states = ordered_states(sub_pomdp)
    indc = findall(s -> s.pos == locpos, states)
    init_states = states[indc]
    init_probs = zeros(length(init_states))

    j = 1
    for s in init_states
        mask = s.rocks
        init_probs[j] = prod(mask[i] == 0 ? notRockings[i] : rockings[i] for i in 1:numRock)
        j += 1
    end

    total_prob = sum(init_probs)
    if total_prob <= 0
        init_probs .= 1.0 / max(length(init_probs), 1)
    else
        init_probs ./= total_prob
    end

    init_state = SparseCat(init_states, init_probs)
    return sub_pomdp, init_state, rock_probs, sub_map, global_idx_map
end

# function make_sub_POMDP(pos, map_size, rock_pos, rock_probs, pomdp)

# 	horizon = 10

# 	maxx = min(pos[1] + horizon, map_size[1])
# 	minx = max(pos[1] - horizon, 1)
# 	maxy = min(pos[2] + horizon, map_size[2])
# 	miny = max(pos[2] - horizon, 1)

# 	# display("minx = $minx, maxx = $maxx, miny = $miny, maxy = $maxy")

# 	# display(rock_pos)

# 	# Ensure the sub-POMDP's map size is within the bounds of the main POMDP

# 	sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

# 	# display("Sub-map boundaries: $sub_map")
# 	# display("current position: $pos")
# 	# pos = global2local(pos, sub_map)
# 	# display("Local position in sub-map: $pos")


# 	sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

# 	# sub_rocks = rock_pos

# 	# println(typeof(sub_rocks))

# 	# Convert to Vector{Tuple{Int64, Int64}}
# 	# sub_rocks = map(x -> Tuple(x), sub_rocks)

# 	# If you explicitly want a Vector type
# 	# converted = collect(map(Tuple, sub_rocks))

# 	# display(sub_rocks)
# 	# display(typeof(sub_rocks))

# 	indices = findall(x -> Tuple(x) in sub_rocks, rock_probs.vals)
# 	# display(indices)

# 	probs = rock_probs.probs[indices]


# 	if sum(probs) == 0
# 		sub_rocks_notinHoriz = [(x, y) for (x, y) in rock_pos if !((sub_map[1] ≤ x ≤ sub_map[3]) && (sub_map[2] ≤ y ≤ sub_map[4]))]

# 		if isempty(sub_rocks_notinHoriz)
# 			println("!!No out-of-horizon rocks to add!!")
# 		else
# 			global_vals_as_tuples = Tuple.(rock_probs.vals)
# 			get_prob = r -> begin
# 				gi = findfirst(==(r), global_vals_as_tuples)
# 				gi === nothing ? 0.0 : rock_probs.probs[gi]
# 			end

# 			eps = 0.0
# 			candidates_pos = [(r, get_prob(r)) for r in sub_rocks_notinHoriz if get_prob(r) > eps]

# 			nearest_by = (rs) -> begin
# 				ds = [abs(pos[1] - r[1]) + abs(pos[2] - r[2]) for (r, _) in rs]
# 				argmin(ds)
# 			end

# 			if !isempty(candidates_pos)
# 				idx = nearest_by(candidates_pos)
# 				nearest_rock = candidates_pos[idx][1]
# 			else
# 				all_probs = [get_prob(r) for r in sub_rocks_notinHoriz]
# 				if isempty(all_probs)
# 					# should not happen thanks to outer guard, but safe fallback
# 					nearest_rock = sub_rocks[1]
# 				else
# 					maxp = maximum(all_probs)
# 					best = [(r, p) for (r, p) in zip(sub_rocks_notinHoriz, all_probs) if p == maxp]
# 					idx = nearest_by(best)
# 					nearest_rock = best[idx][1]
# 				end
# 			end

# 			push!(sub_rocks, nearest_rock)
# 			println("!!New Rock Added!! $(nearest_rock)")
# 		end
# 	end

# 	sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)

# 	# println("Sub-map size: $sub_map_size")

# 	println("Rocks: $sub_rocks")
# 	# display(sub_rocks)
# 	# display(sub_map_size)

# 	local_sub_rocks = []

# 	for rock in sub_rocks
# 		new_rock = global2local(rock, sub_map)
# 		push!(local_sub_rocks, Tuple(new_rock))
# 	end

# 	# display("Local rock positions in sub-map: $local_sub_rocks")

# 	locpos = global2local(pos, sub_map)

# 	# println("localpos: $locpos")

# 	# println("pos: $pos")
# 	# println("local pos: $locpos")
# 	# display(sub_map)

# 	sub_pomdp = RockSamplePOMDP(map_size = sub_map_size,
# 		rocks_positions = local_sub_rocks,
# 		init_pos = locpos,
# 		sensor_efficiency = pomdp.sensor_efficiency,
# 		discount_factor = pomdp.discount_factor,
# 		good_rock_reward = pomdp.good_rock_reward,
# 		bad_rock_penalty = pomdp.bad_rock_penalty,
# 		step_penalty = pomdp.step_penalty,
# 		exit_reward = pomdp.exit_reward, # always allow exit
# 		# exit_reward = is_global_right_edge ? pomdp.exit_reward : 0.0,
# 		sensor_use_penalty = pomdp.sensor_use_penalty) # if available

# 	# display(ordered_states(sub_pomdp))

# 	# display("rocks in subpomdp: $(sub_pomdp.rocks_positions)")

# 	numRock = length(sub_pomdp.rocks_positions)
# 	# Build a mapping from each LOCAL sub-POMDP rock to its GLOBAL index in rock_probs
# 	global_vals_as_tuples = Tuple.(rock_probs.vals)
# 	global_idx_map = Vector{Union{Int, Nothing}}(undef, numRock)
# 	for i in 1:numRock
# 		# sub_pomdp.rocks_positions are LOCAL tuples; sub_rocks[i] are GLOBAL tuples aligned
# 		g = sub_rocks[i]
# 		global_idx_map[i] = findfirst(==(g), global_vals_as_tuples)
# 	end

# 	# Read priors for local rocks using the global indices
# 	rockings = zeros(numRock)
# 	for i in 1:numRock
# 		gi = global_idx_map[i]
# 		rockings[i] = (gi === nothing) ? 0.0 : rock_probs.probs[gi]
# 	end

# 	notRockings = ones(numRock) .- rockings
# 	# println("Rock probabilities in subpomdp: $rockings")
# 	# println("Not rock probabilities in subpomdp: $notRockings")

# 	states = ordered_states(sub_pomdp) #get all the states in the subpomdp
# 	# display(states)
# 	indc = findall(s -> s.pos == locpos, states) #get the indices of the states that share the same position as the init pos
# 	init_states = states[indc]
# 	# display(init_states)
# 	init_probs = zeros(length(init_states)) #initialise the probabilities for the initial states

# 	j = 1;

# 	for s in init_states

# 		mask = s.rocks

# 		init_probs[j] = prod(mask[i] == 0 ? notRockings[i] : rockings[i] for i in 1:numRock) #* scaling
# 		# println("State: $(s), Probability: $(init_probs[j])")

# 		j = j + 1
# 	end

# 	total_prob = sum(init_probs)
# 	if total_prob <= 0
# 		# fallback: uniform over init states
# 		init_probs .= 1.0 / length(init_probs)
# 	else
# 		init_probs ./= total_prob
# 	end



# 	init_state = SparseCat(init_states, init_probs)
# 	return sub_pomdp, init_state, rock_probs, sub_map, global_idx_map

# end

function get_next_init_state(policy, thisPomdp, rock_probs, sub_map, actionList, init_state, global_idx_map)

	# create an updater for the pollicy
	initpos = thisPomdp.init_pos
	# println("Initial Position in Sub-POMDP: $initpos")
	up = updater(policy)
	# get the initial belief state
	# b0 = initialize_belief(up, initialstate(thisPomdp)) # initialize belief state
	b0 = init_state
	# show(stdout, "text/plain", b0)

	# init_state = initialstate(thisPomdp)
	# println("Initial state: $init_state")

	# init_action = POMDPs.action(policy, b0)

	# println("Initial action: $init_action")


	state = nothing
	action = nothing
	obs = nothing
	rew = nothing

	# simulate the first step after the inital state
	for (s, a, o, r) in stepthrough(thisPomdp, policy, "s,a,o,r", max_steps = 1)
		println("in state $s")
		println("took action $(actionNum2word(a))")
		println("received observation $(obsNum2word(o)) and reward $r")
		println("----------------------------------------------------")

		state = s
		action = a
		obs = o
		rew = r
	end


	# display("State after action: $state")
	# display("Action taken: $action")

	# display("type of state: $(typeof(state))")

	global glob = state

	# all_true_state = state
	# all_true_state.rocks .= true

	# display("All true state: $all_true_state")



	# # get the next state after the iniital state

	# display("+++++++++++++++++++++++++++++++")
	# display(thisPomdp.map_size)
	# display(pos)
	# display(state)
	# display(action)
	# display("+++++++++++++++++++++++++++++++")

	trans = transition(thisPomdp, state, action)

	actionWord = actionNum2word(action)

	push!(actionList, actionWord)

	if POMDPs.isterminal(thisPomdp, trans.val)

		return "TERMINAL"

	end

	# locpos = trans.val.pos

	# display("Transition result: $trans")

	# println("local trans pos: $locpos")

	#initialise the belief after the first action has been taken
	b1 = update(up, b0, action, obs)

	val = value(policy, b0)
	val2 = value(policy, b1)


	# print("---------------+-----------------------+---------------------\n")

	# b_west = update(up, b0, 5, 3) #west, none
	# b_east = update(up, b0, 3, 3) #east, none
	# b_north = update(up, b0, 2, 3) #north, none
	# b_south = update(up, b0, 4, 3) #south, none
	# b_sample = update(up, b0, 1, 3) #sample, none

	# v_west = value(policy, b_west)
	# v_east = value(policy, b_east)
	# v_north = value(policy, b_north)
	# v_south = value(policy, b_south)
	# v_sample = value(policy, b_sample)
	# println("Value of west belief: $v_west")
	# println("Value of east belief: $v_east")
	# println("Value of north belief: $v_north")
	# println("Value of south belief: $v_south")
	# println("Value of sample belief: $v_sample")

	# #sense rock beliefs
	# for i = 1:length(thisPomdp.rocks_positions)
	# 	if rock_probs.probs[i] == 0
	# 		continue
	# 	end
	# 	b_sense_good = update(up, b0, 5 + i, 1) #sense rock i, good
	# 	b_sense_bad = update(up, b0, 5 + i, 2) #sense rock i, bad
	# 	v_sense_good = value(policy, b_sense_good)
	# 	v_sense_bad = value(policy, b_sense_bad)
	# 	println("Value of sense rock $i good belief: $v_sense_good")
	# 	println("Value of sense rock $i bad belief: $v_sense_bad")
	# end

	# print("---------------+-----------------------+---------------------\n")



	# println("Value of initial belief: $val")
	# println("Value of next belief: $val2")

	# trim the belief state to only states that are in the same position as the next belief
	S           = eltype(b1.state_list)
	next_states = S[]
	next_probs  = Float64[]

	# display(b1)
	# display(trans.val.pos)


	# Collect states/probs that share the current position
	for (s, p) in zip(b1.state_list, b1.b)
		if s.pos == trans.val.pos
			push!(next_states, s)
			push!(next_probs, p)
		end
	end

	# display(next_states)
	# display(next_probs)

	thisSum = 0

	# for r ∈ 1:length(thisPomdp.rocks_positions)

	# 	# display("Evaluating rock $r at position $(thisPomdp.rocks_positions[r])")

	# 	i = 1
	# 	thisSum = 0
	# 	for s in next_states

	# 		# display("Considering state $s with probability $(next_probs[i])")

	# 		if s.rocks[r] == true
	# 			thisSum += next_probs[i]
	# 			# println("true")
	# 			# println(next_probs[i])
	# 			# println(thisSum)
	# 			# println("-----")
	# 		else
	# 			# println("false") 
	# 			# println(thisSum)
	# 			# println("-----")
	# 		end

	# 		i += 1
	# 	end

	# 	# println("total $r: $thisSum")

	# 	# probs[r] = thisSum

	# 	thisRock = thisPomdp.rocks_positions[r]

	# 	ind = findfirst(==(thisRock), rock_probs.vals)

	# 	# print(thisRock)
	# 	# println(ind)

	# 	rock_probs.probs[ind] = thisSum

	# 	# print(probs)

	# end

	for r in 1:length(thisPomdp.rocks_positions)  # r is LOCAL rock index
		thisSum = 0.0
		for (i, s) in enumerate(next_states)
			if s.rocks[r] == true
				thisSum += next_probs[i]
			end
		end
		gi = global_idx_map[r]  # global index in rock_probs, or nothing
		if gi === nothing
			@warn "Rock mapping failed; local rock not in global list (check tuple vs SVector types)" local_r=thisPomdp.rocks_positions[r] global_vals=rock_probs.vals
		else
			rock_probs.probs[gi] = thisSum
		end
	end
	next_init_state = SparseCat(next_states, next_probs)
	# display(next_init_state)

	position = next_init_state.vals[1].pos

	# println("Local Position: $position ")

	globpos = local2global(position, sub_map)
	# println("Global Position: $globpos")

	#return the global position
	return globpos

end

# rng = MersenneTwister(1)

# start_time = time_ns() #start time

#loop through avery initial position

# Master tables to collect all runs
# Master long-format tables
# actions_df   = DataFrame(RunLabel = String[], RunIdx = Int[], Step = Int[], Action = String[])
# positions_df = DataFrame(RunLabel = String[], RunIdx = Int[], Step = Int[], Position = String[])

# for j ∈ 1:49

# x = (j - 1) % 7 + 1   # column
# y = div(j - 1, 7) + 1 # row
# pos = [x, y]

# initial_pos = copy(pos)
# run_label = "$(initial_pos[1]),$(initial_pos[2])"

# println("Starting new initial position: $pos")

actionListList = []
# println("\n\n\n\n\n\n\n\n\n")


# initial large POMDP of the whole space
# pomdp = RockSamplePOMDP(map_size = (15, 15),
# 	rocks_positions = [(1,2), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6, 12), (12, 14)],
# 	sensor_efficiency = 20.0,
# 	discount_factor = 0.95,
# 	good_rock_reward = 20.0,
# )

pomdp = RockSamplePOMDP(map_size = (7, 7),
	rocks_positions = [(3, 7), (6, 6), (1, 5), (4,3)],
	# rocks_positions = [(6,6)], 
	init_pos = (1,1),
	sensor_efficiency = 20.0,
	discount_factor = 0.95,
	good_rock_reward = 20.0,
	bad_rock_penalty = -5.0
)
#initialize rock_probs for belief state
rock_probs = SparseCat(pomdp.rocks_positions, [0.5 for _ in 1:length(pomdp.rocks_positions)])

actionList = []
posList = []

# new_pomdp = RockSamplePOMDP(map_size = (7, 7),
# 	rocks_positions = [(3, 7), (4,3)],
# 	init_pos = (1,1),
# 	sensor_efficiency = 20.0,
# 	discount_factor = 0.95,
# 	good_rock_reward = 20.0,
# 	bad_rock_penalty = -5.0
# )



# # # solve full POMDP and create GIF
# solver = SARSOPSolver(precision = 1e-3; max_time = 10.0, verbose = false) #use SARSOP solver
# policy = solve(solver, new_pomdp) # get policy using SARSOP solver

# println("Creating simulation GIF...")
# sim = GifSimulator(
# 	filename="DroneRockSample.gif",
# 	max_steps=100,  # Reduced steps for testing
# 	rng=MersenneTwister(1),
# 	show_progress=true  # Enable progress display
# )
# saved_gif = simulate(sim, new_pomdp, policy)
# println("GIF saved to: $(saved_gif.filename)")

# exit()

# # you already have: policy = solve(solver, pomdp)
# as = policy.alphas
# K = length(as)
# N = length(as[1])
# @assert all(length(α)==N for α in as)

# w, h = pomdp.map_size
# positions = [(x, y) for x in 1:w for y in 1:h]



# # Build a lookup: (x, y, is_good)::(Int,Int,Bool) -> state index
# states_list = collect(states(pomdp))
# index_by = Dict{Tuple{Int, Int, Bool}, Int}()
# for s in states_list
# 	x = s.pos[1];
# 	y = s.pos[2]
# 	good = s.rocks[1] == true
# 	idx = stateindex(pomdp, s)
# 	index_by[(x, y, good)] = idx
# end

# # collect (pos => (ig, ib)) only for positions that have both good and bad states
# valid = Tuple{Tuple{Int, Int}, Tuple{Int, Int}}[]
# for (x, y) in positions
# 	ig = get(index_by, (x, y, true), nothing)
# 	ib = get(index_by, (x, y, false), nothing)
# 	if ig !== nothing && ib !== nothing
# 		push!(valid, ((x, y), (ig, ib)))
# 	end
# end

# if isempty(valid)
#     @warn "No valid (pos,good/bad) states found to plot. \
# This can happen if stateindex throws for all positions (e.g., custom enumerator, \
# terminal-only policy, or mismatched RSState). Double-check RSState and stateindex."
# else
#     ts = range(0.0, 1.0, length = 200)  # p = Pr(rock is good)

#     # --- Flip the grid visually: sort by y descending, then x ascending ---
#     # (1,1) ends up bottom-left; (w,h) top-right.
#     sort!(valid, by = v -> (h - v[1][2], v[1][1]))

#     max_plots = length(valid)
#     rows = max(1, ceil(Int, sqrt(max_plots)))
#     cols = max(1, ceil(Int, max_plots / rows))

#     # We'll build the main collage with *no* legends/titles.
#     mainplt = plot(layout = (rows, cols), size = (1800, 1200), legend = false)

#     # Prepare action mapping for labels and consistent colors
#     has_actions = hasproperty(policy, :action_map)
#     acts = has_actions ? policy.action_map : fill(0, length(as))  # 0 -> unknown action if missing

#     # Map action -> label and color (stable across subplots)
#     using StatsBase
#     uniq_actions = unique(acts)
#     pal = palette(:auto)
#     color_for = Dict{Int,Any}()
#     for (i, a) in enumerate(uniq_actions)
#         color_for[a] = pal[mod1(i, length(pal))]
#     end
#     action_label(a::Int) = a == 0 ? "α (no map)" : actionNum2word(a)

#     # Draw all subplots
#     for (idx, (pos, (ig, ib))) in enumerate(valid)
#         # each α line along p, and max envelope
#         y_lines = [[α[ig]*p + α[ib]*(1 - p) for p in ts] for α in as]
#         y_env   = [maximum((α[ig]*p + α[ib]*(1 - p) for α in as)) for p in ts]

#         # First alpha (no label, thin, translucent)
#         plot!(mainplt, ts, y_lines[1], lw = 2, label = false,
#               xlabel = "", ylabel = "", subplot = idx)

#         # Rest of alphas with action-based labels/colors
#         for k in 2:length(as)
#             a = acts[k]
#             plot!(mainplt, ts, y_lines[k], lw = 2, label = false,
#                   color = get(color_for, a, :black), subplot = idx)
#         end

#         # Upper envelope (quiet gray), no legend
#         plot!(mainplt, ts, y_env, alpha = 0.25, lw = 10, label = false,
#               subplot = idx, color = :yellow)
#     end

#     # --- Build a separate legend panel (single global legend) ---
#     # We draw one tiny plot that only exists to show one line per action with labels.
#     legplt = plot(size = (400, 1200), legend = :topleft,
#                   framestyle = :none, xticks = false, yticks = false)
#     for a in uniq_actions
#         # Dummy data to create a legend entry in the desired color
#         plot!(legplt, ts, fill(0.0, length(ts)),
#               lw = 2, label = action_label(a), color = color_for[a])
#     end
# 	plot!(legplt, ts, fill(0.0, length(ts)),
#       lw = 10, alpha = 0.18, color = :yellow, label = "Upper Envelope")
#     # Also add an entry for the envelope if you want it in the legend:
#     # plot!(legplt, ts, fill(0.0, length(ts)), lw = 3, color = :gray, label = "upper envelope")

#     # Compose main plot + legend panel
#     # plt = plot(mainplt, legplt, layout = grid(1, 2, widths = [0.80, 0.20]))

	


# display(plt)
# end

# savefig(plt, "alpha_vectors.png")

# # After you finish creating `mainplt` and `legplt` (before combining them into plt)
# outdir = "alpha_vectors"
# # Create one legend block with all action labels + envelope
# legplt = plot(size = (400, 400), legend = :bottom,
#               framestyle = :none, xticks = false, yticks = false)
# for a in uniq_actions
#     plot!(legplt, ts, fill(0.0, length(ts)),
#           lw = 2, label = action_label(a), color = color_for[a])
# end
# plot!(legplt, ts, fill(0.0, length(ts)),
#       lw = 10, alpha = 0.18, color = :yellow, label = "upper envelope")

# # save each subplot individually
# for (idx, (pos, (ig, ib))) in enumerate(valid)
#     # extract the subplot as a standalone plot
#     subplt = plot(size=(800,600), legend=:bottomright)
    
#     # envelope first (highlighter)
#     y_env = [maximum((α[ig]*p + α[ib]*(1 - p) for α in as)) for p in ts]
#     plot!(subplt, ts, y_env, lw=10, alpha=0.18, color=:yellow, label="upper envelope")

#     # plot all α lines, color-coded by action
#     for (k, α) in enumerate(as)
#         a = acts[k]
#         plot!(subplt, ts, [α[ig]*p + α[ib]*(1 - p) for p in ts],
#               lw=2, color=get(color_for, a, :black), label=action_label(a))
#     end

#     # add titles and axis labels
#     plot!(subplt, xlabel="p(rock good)", ylabel="Value",
#           title="Position $(pos)", legendfontsize=8)

#     # save file with coordinate in name
#     filename = joinpath(outdir, "alphas_pos_$(pos[1])_$(pos[2]).png")
#     savefig(subplt, filename)
#     println("Saved $(filename)")

# end



# exit()

for i in 1:50

	global pos
	if i == 1
		pos = pomdp.init_pos
	end

	display(rock_probs)


	println("Iteration: $i")
	println("Current Position: $pos")

	sub_pomdp, init_state, rock_probs, sub_map, global_idx_map = make_sub_POMDP(pos, pomdp.map_size, pomdp.rocks_positions, rock_probs, pomdp)

	display(init_state)

	# redefine initial state from the original function from RockSample
	POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = init_state

	solver = SARSOPSolver(precision = 1e-3; max_time = 10.0, verbose = false) #use SARSOP solver
	# solver = QMDPSolver(max_iterations=20, belres=1e-3, verbose=false) #use QMDP solver
	policy = solve(solver, sub_pomdp) # get policy using SARSOP solver

	#get the next initial belief state
	pos = get_next_init_state(policy, sub_pomdp, rock_probs, sub_map, actionList, init_state, global_idx_map)
	push!(posList, pos)

	# if pos == "TERMINAL" || i == 20
	# 	println("Reached terminal state. Exiting loop.")
	# 	println("Actions taken: ")
	# 	count = 1;
	# 	for a in actionList
	# 		println("	", a, " -> ", posList[count])
	# 		count += 1
	# 	end
	# 	break
	# end

	if pos == "TERMINAL" || i == 50
		println("Reached terminal state. Exiting loop.")
		println("Actions taken: ")

		# how many steps have positions recorded
		nsteps = min(length(actionList), length(posList))

		for step_idx in 1:nsteps
			a = actionList[step_idx]
			p = posList[step_idx]  # expected like [x, y]
			println("    ", a, " -> ", p)

			# # Append to actions (long)
			# push!(actions_df, (run_label, j, step_idx, a))

			# # Append to positions (long) as "[x,y]" string
			# if p isa Vector{Int} && length(p) == 2
			# 	push!(positions_df, (run_label, j, step_idx, "[$(p[1]),$(p[2])]"))
			# end
		end

		break
	end



	println("============================================")

end

# push!(actionListList, actionList)


# end

# Actions: rows = Step, columns = RunLabel, values = Action
# wide_actions = unstack(actions_df, :Step, :RunLabel, :Action)

# # Positions: rows = Step, columns = RunLabel, values = "[x,y]" string
# wide_positions = unstack(positions_df, :Step, :RunLabel, :Position)

# CSV.write("wide_actions_1.csv", wide_actions)
# CSV.write("wide_positions_1.csv", wide_positions)


