using POMDPs
using POMDPTools
using POMDPGifs
using NativeSARSOP
using Random
using Cairo

# Include the DroneRockSample module with fixed files
include("DroneRockSample.jl")
using .DroneRockSample

start_time = time_ns()

# Create a smaller example first for testing
pomdp = DroneRockSamplePOMDP(
    map_size = (10, 10),  # Reduced map size for testing
    rocks_positions = [(2, 8), (4, 5), (7, 2), (8,9)],
    sensor_efficiency = 20.0,
    discount_factor = 0.95,
    good_rock_reward = 20.0,
    fly_penalty = -0.2
)

println("POMDP created successfully")

# Print the action space size
println("Number of actions: ", length(actions(pomdp)))
println("Basic actions: ", N_BASIC_ACTIONS)
println("Total rocks: ", length(pomdp.rocks_positions))

# Get the list of states
states = ordered_states(pomdp)
println("Number of states: ", length(states))

# Solve the POMDP using SARSOP with shorter time for testing
println("Solving with SARSOP...")
solver = SARSOPSolver(precision=1e-2, max_time=5.0)  # Reduced precision and time
policy = POMDPs.solve(solver, pomdp)  # Explicitly use POMDPs.solve

end_time = time_ns()
elapsed_time = (end_time - start_time) / 1e9  # Convert from nanoseconds to seconds
println("Elapsed time: $elapsed_time seconds")

# Create a GIF of the simulation
println("Creating simulation GIF...")
sim = GifSimulator(
    filename = "DroneRockSample.gif", 
    max_steps = 15,  # Reduced steps for testing
    rng = MersenneTwister(1), 
    show_progress = true  # Enable progress display
)

saved_gif = simulate(sim, pomdp, policy)

println("GIF saved to: $(saved_gif.filename)")