using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using RockSample
using Cairo

start_time = time_ns()

pomdp = RockSamplePOMDP(map_size = (100,100),
                        rocks_positions=[(20,100),(40,80),(70,20)],
                        sensor_efficiency=20.0,
                        discount_factor=0.95,
                        good_rock_reward = 20.0)

states = ordered_states(pomdp)
# display(states)

solver = SARSOPSolver(precision=1e-3; max_time=10.0)
policy = solve(solver, pomdp)

end_time = time_ns()
elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
println("Elapsed time: $elapsed_time seconds")

sim = GifSimulator(; filename="RockSample.gif", max_steps=30, rng=MersenneTwister(1), show_progress=false)
saved_gif = simulate(sim, pomdp, policy)

println("gif saved to: $(saved_gif.filename)")