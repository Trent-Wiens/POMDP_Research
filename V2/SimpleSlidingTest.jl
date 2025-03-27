using POMDPs
using POMDPTools
using NativeSARSOP
using Random
using ParticleFilters  # Added for belief representation

# Include the DroneRockSample module first
include("DroneRockSample.jl")
# Save a reference to the module
const DRS = DroneRockSample

# Include our SlidingSARSOP implementation
include("SlidingSARSOP.jl")
# Set the DroneRockSample module in SlidingSARSOP
SlidingSARSOP.set_dronerocksample(DRS)

println("Creating a simple DroneRockSample POMDP...")
pomdp = DRS.DroneRockSamplePOMDP(
    map_size = (10, 10),
    rocks_positions = [(2,8), (4, 5), (7,2), (8,9)],
    sensor_efficiency = 20.0,
    good_rock_reward = 20.0
)

println("Creating standard SARSOP solver...")
standard_solver = SARSOPSolver(precision=1e-2, max_time=5.0)
println("Solving with standard SARSOP...")
standard_policy = POMDPs.solve(standard_solver, pomdp)

println("Creating SlidingSARSOP solver...")
sliding_solver = SlidingSARSOP.SlidingSARSOPSolver(
    horizon_distance = 4,
    precision = 1e-2,
    timeout = 1.0
)
println("Solving with SlidingSARSOP...")
sliding_policy = POMDPs.solve(sliding_solver, pomdp)

# Run a quick simulation to test
println("\nRunning a simulation with each solver...")
rng = MersenneTwister(123)

# Test with standard policy
standard_hr = HistoryRecorder(max_steps=10, rng=copy(rng))
standard_history = simulate(standard_hr, pomdp, standard_policy)
standard_reward = discounted_reward(standard_history)
standard_steps = length(standard_history)

# Test with sliding policy
sliding_hr = HistoryRecorder(max_steps=10, rng=copy(rng))
sliding_history = simulate(sliding_hr, pomdp, sliding_policy)
sliding_reward = discounted_reward(sliding_history)
sliding_steps = length(sliding_history)

println("Results:")
println("Standard SARSOP: Reward = $standard_reward, Steps = $standard_steps")
println("Sliding SARSOP: Reward = $sliding_reward, Steps = $sliding_steps")

println("\nSimple test complete!")