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

function make_sub_POMDP(pos, map_size, rock_pos)

    horizon = 3

    maxx = pos[1] + horizon
    minx = pos[1] - horizon
    maxy = pos[2] + horizon
    miny = pos[2] - horizon

    # Ensure the sub-POMDP's map size is within the bounds of the main POMDP

    sub_map = (max(minx, 1), max(miny, 1), min(maxx, map_size[1]), min(maxy, map_size[2]))

    sub_rocks = [(x, y) for (x, y) in rock_pos if sub_map[1] ≤ x ≤ sub_map[3] && sub_map[2] ≤ y ≤ sub_map[4]]

    sub_map_size = (sub_map[3] - sub_map[1] + 1, sub_map[4] - sub_map[2] + 1)

    # display(sub_rocks)
    # display(sub_map_size)
    # exit()

    sub_pomdp = RockSamplePOMDP(map_size = sub_map_size,
                            rocks_positions = sub_rocks, 
                            init_pos = pos)

    return sub_pomdp

end

rng = MersenneTwister(1)

start_time = time_ns()

pomdp = RockSamplePOMDP(map_size = (15,15),
                        rocks_positions=[(2, 1), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6,12), (12, 14)],
                        sensor_efficiency=20.0,
                        discount_factor=0.95,
                        good_rock_reward = 20.0)

states = ordered_states(pomdp)
# display(states)

# make sub pomdp

sub_pomdp = make_sub_POMDP([3,4], pomdp.map_size, pomdp.rocks_positions)

subpomdp_sates = ordered_states(sub_pomdp)

# display(sub_pomdp)

# solve sub POMDP

solver = SARSOPSolver(precision=1e-3; max_time=10.0) #use SARSOP solver
policy = solve(solver, sub_pomdp) # get policy using SARSOP solver

# init_state = initialstate(sub_pomdp)
# next_action = action(policy, init_state) 

state = nothing
action = nothing
obs = nothing
rew = nothing

# simulate the first step after the inital state

for (s, a, o, r) in stepthrough(sub_pomdp, policy, "s,a,o,r", max_steps=1)
    println("in state $s")
    println("took action $a")
    println("received observation $o and reward $r")

    state = s
    action = a
    obs = o
    rew = r
end

# get the next state after the iniital state
trans = transition(sub_pomdp, state, action)

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
