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
using QMDP

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

function make_sub_POMDP(pos, map_size, rock_pos, rock_probs, pomdp)

	horizon = 3

	maxx = min(pos[1] + horizon, map_size[1])
	minx = max(pos[1] - horizon, 1)
	maxy = min(pos[2] + horizon, map_size[2])
	miny = max(pos[2] - horizon, 1)

	# display("minx = $minx, maxx = $maxx, miny = $miny, maxy = $maxy")

	# display(rock_pos)

	# Ensure the sub-POMDP's map size is within the bounds of the main POMDP

	sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

	# display("Sub-map boundaries: $sub_map")
	# display("current position: $pos")
	# pos = global2local(pos, sub_map)
	# display("Local position in sub-map: $pos")


	sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

	# display(sub_rocks)
	# display(typeof(sub_rocks))

	indices = findall(x -> Tuple(x) in sub_rocks, rock_probs.vals)
	# display(indices)

	probs = rock_probs.probs[indices]
	# display(probs)
	if sum(probs) == 0
		# display(probs)

		sub_rocks_notinHoriz = [(x, y) for (x, y) in rock_pos if !((sub_map[1] ≤ x ≤ sub_map[3]) && (sub_map[2] ≤ y ≤ sub_map[4]))]

		# display(sub_rocks_notinHoriz)

		#find the nearest rock outside the horizon and add it to sub_rocks

		dists = [abs(pos[1] - r[1]) + abs(pos[2] - r[2]) for r in sub_rocks_notinHoriz if !((sub_map[1] ≤ r[1] ≤ sub_map[3]) && (sub_map[2] ≤ r[2] ≤ sub_map[4]))]

		nearest_rock = sub_rocks_notinHoriz[argmin(dists)]
		# display(nearest_rock)
		# display(typeof(nearest_rock))

		# nearest_rock = Tuple(nearest_rock)

		push!(sub_rocks, nearest_rock)


	end

	sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)

	is_global_right_edge = (sub_map[3] == map_size[1])


	println("Rocks: $sub_rocks")
	# display(sub_rocks)
	# display(sub_map_size)

	locpos = global2local(pos, sub_map)

	# println("localpos: $locpos")

	# println("pos: $pos")
	# println("local pos: $locpos")
	# display(sub_map)

	sub_pomdp = RockSamplePOMDP(map_size = sub_map_size,
		rocks_positions = sub_rocks,
		init_pos = locpos,
		sensor_efficiency = pomdp.sensor_efficiency,
		discount_factor = pomdp.discount_factor,
		good_rock_reward = pomdp.good_rock_reward,
		bad_rock_penalty = pomdp.bad_rock_penalty,
		exit_reward = pomdp.exit_reward,
		# exit_reward = is_global_right_edge ? pomdp.exit_reward : 0.0,
		sensor_use_penalty = pomdp.sensor_use_penalty) # if available

	# display(sub_pomdp)

	numRock = length(sub_pomdp.rocks_positions)
	# display(numRock)

	rockings = zeros(numRock)

	for i ∈ 1:numRock #get the probabilities for just the rocks in the subpomdp
		posit = sub_pomdp.rocks_positions[i]
		idx = findfirst(==(posit), rock_probs.vals)
		rockings[i] = isnothing(idx) ? 0.0 : rock_probs.probs[idx]
	end

	notRockings = ones(numRock) .- rockings #get the not rock probabilities
	# println("Rock probabilities in subpomdp: $rockings")
	# println("Not rock probabilities in subpomdp: $notRockings")

	states = ordered_states(sub_pomdp) #get all the states in the subpomdp
	# display(states)
	indc = findall(s -> s.pos == locpos, states) #get the indices of the states that share the same position as the init pos
	init_states = states[indc]
	# display(init_states)
	init_probs = zeros(length(init_states)) #initialise the probabilities for the initial states

	j = 1;

	for s in init_states

		mask = s.rocks

		init_probs[j] = prod(mask[i] == 0 ? notRockings[i] : rockings[i] for i in 1:numRock) #* scaling
		# println("State: $(s), Probability: $(init_probs[j])")

		j = j + 1
	end

	total_prob = sum(init_probs)
	if total_prob <= 0
	    # fallback: uniform over init states
	    init_probs .= 1.0 / length(init_probs)
	else
	    init_probs ./= total_prob
	end

	init_state = SparseCat(init_states, init_probs)

	# print(init_state)

	return sub_pomdp, init_state, rock_probs, sub_map

end

function get_next_init_state(policy, thisPomdp, rock_probs, sub_map, actionList, init_state)

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
	# next_action = POMDPs.action(policy, init_state)
	# println("next action: $next_action")


	state = nothing
	action = nothing
	obs = nothing
	rew = nothing

	# simulate the first step after the inital state
	for (s, a, o, r) in stepthrough(thisPomdp, policy, "s,a,o,r", max_steps = 1)
		# println("in state $s")
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

	println("Value of initial belief: $val")
	println("Value of next belief: $val2")

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

	for r ∈ 1:length(thisPomdp.rocks_positions)

		# display("Evaluating rock $r at position $(thisPomdp.rocks_positions[r])")

		i = 1
		thisSum = 0
		for s in next_states

			# display("Considering state $s with probability $(next_probs[i])")

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

		thisRock = thisPomdp.rocks_positions[r]

		ind = findfirst(==(thisRock), rock_probs.vals)

		# print(thisRock)
		# println(ind)

		rock_probs.probs[ind] = thisSum

		# print(probs)

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

rng = MersenneTwister(1)

start_time = time_ns() #start time

# initial large POMDP of the whole space
# pomdp = RockSamplePOMDP(map_size = (15, 15),
# 	rocks_positions = [(1,2), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6, 12), (12, 14)],
# 	sensor_efficiency = 20.0,
# 	discount_factor = 0.95,
# 	good_rock_reward = 20.0,
# )

pomdp = RockSamplePOMDP(map_size = (7, 7),
	rocks_positions = [(1,2), (3, 7), (6, 6), (1, 5)],
	sensor_efficiency = 20.0,
	discount_factor = 0.95,
	good_rock_reward = 20.0,
	bad_rock_penalty = -5.0,
)
#initialize rock_probs for belief state
rock_probs = SparseCat(pomdp.rocks_positions, [0.5 for _ in 1:length(pomdp.rocks_positions)])

actionList = []
posList = []

# # solve full POMDP and create GIF
# solver = SARSOPSolver(precision=1e-3; max_time=10.0, verbose=false) #use SARSOP solver
# solver = QMDPSolver(max_iterations=20, belres=1e-3, verbose=false) #use QMDP solver
# policy = solve(solver, pomdp) # get policy using SARSOP solver

# println("Creating simulation GIF...")
# sim = GifSimulator(
# 	filename="DroneRockSample.gif",
# 	max_steps=100,  # Reduced steps for testing
# 	rng=MersenneTwister(1),
# 	show_progress=true  # Enable progress display
# )
# saved_gif = simulate(sim, pomdp, policy)
# println("GIF saved to: $(saved_gif.filename)")

for i in 1:20

	global pos
	if i == 1
		pos = [1,5]
	end

	display(rock_probs)


	println("Iteration: $i")
	println("Current Position: $pos")

	sub_pomdp, init_state, rock_probs, sub_map = make_sub_POMDP(pos, pomdp.map_size, pomdp.rocks_positions, rock_probs, pomdp)

	# display(init_state)

	# redefine initial state from the original function from RockSample
	POMDPs.initialstate(p::RockSamplePOMDP{K}) where K = init_state

	solver = SARSOPSolver(precision = 1e-3; max_time = 10.0, verbose = false) #use SARSOP solver
	# solver = QMDPSolver(max_iterations=20, belres=1e-3, verbose=false) #use QMDP solver
	policy = solve(solver, sub_pomdp) # get policy using SARSOP solver

	#get the next initial belief state
	pos = get_next_init_state(policy, sub_pomdp, rock_probs, sub_map, actionList, init_state)
	push!(posList, pos)

	if pos == "TERMINAL" || i == 20
		println("Reached terminal state. Exiting loop.")
		println("Actions taken: ")
		count = 1;
		for a in actionList
			println("	",	a, " -> ", posList[count])
			count += 1
		end
		break
	end

	println("============================================")

end





