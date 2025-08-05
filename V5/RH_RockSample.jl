using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using RockSample
using Cairo

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

start_time = time_ns()

pomdp = RockSamplePOMDP(map_size = (15,15),
                        rocks_positions=[(2, 1), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6,12), (12, 14)],
                        sensor_efficiency=20.0,
                        discount_factor=0.95,
                        good_rock_reward = 20.0)

states = ordered_states(pomdp)
# display(states)

# make sub pomdp

sub_pomdp = make_sub_POMDP([1,1], pomdp.map_size, pomdp.rocks_positions)

# display(sub_pomdp)

# solve sub POMDP

solver = SARSOPSolver(precision=1e-3; max_time=10.0)
policy = solve(solver, sub_pomdp)

init_state = initialstate(sub_pomdp)

next_action = action(policy, init_state) 

# take action

# repeat

# end_time = time_ns()
# elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
# println("Elapsed time: $elapsed_time seconds")

# sim = GifSimulator(; filename="RockSample.gif", max_steps=30, rng=MersenneTwister(1), show_progress=false)
# saved_gif = simulate(sim, sub_pomdp, policy)

# println("gif saved to: $(saved_gif.filename)")