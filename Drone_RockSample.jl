include("drone_rocksample_sub.jl")  # Load the DronePOMDP module
using .Drone  # Reference the local module

using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using Cairo
using StaticArrays


# Start timing
start_time = time_ns()

# Create the DronePOMDP instance
pomdp = DronePOMDP(map_size = (10,10),
                    rocks_positions=[(2,10), (4,8), (7,2)],
                    sensor_efficiency=20.0,
                    discount_factor=0.95,
                    good_rock_reward = 20.0)

# # Solve the POMDP using SARSOPSolver
# solver = SARSOPSolver(precision=1e-3, max_time=10.0)
# policy = solve(solver, pomdp)

solver = SARSOPSolver(precision=1e-3, max_time=10.0)
policy = solve(solver, pomdp)

# Use initial belief
belief = POMDPs.initialstate(pomdp)

a_idx = action(policy, belief)  # Now passing a belief instead of a state
# a = index_to_action(pomdp, a_idx)  # Convert back to (x,y)
println("Action chosen by policy: ", a_idx)

# # End timing
end_time = time_ns()
elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
println("Elapsed time: $elapsed_time seconds")

# Simulate and save a GIF
sim = GifSimulator(; filename="DronePOMDP.gif", max_steps=30, rng=MersenneTwister(1), show_progress=false)
saved_gif = simulate(sim, pomdp, policy)

# # Confirm GIF was saved
# println("GIF saved to: $(saved_gif.filename)")