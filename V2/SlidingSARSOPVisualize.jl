using POMDPs
using POMDPTools
using NativeSARSOP
using POMDPGifs
using Random
using ParticleFilters
using Cairo

# Include the DroneRockSample module first
include("DroneRockSample.jl")
# Save a reference to the module
const DRS = DroneRockSample

# Include our SlidingSARSOP implementation
include("SlidingSARSOP.jl")
# Set the DroneRockSample module in SlidingSARSOP
SlidingSARSOP.set_dronerocksample(DRS)

println("Creating DroneRockSample POMDP for visualization...")
pomdp = DRS.DroneRockSamplePOMDP(
    map_size = (7, 7),
    rocks_positions = [(2, 4), (4, 2), (5, 5), (6, 3)],
    sensor_efficiency = 20.0,
    discount_factor = 0.95,
    good_rock_reward = 20.0,
    fly_penalty = -0.2
)

println("Solving with standard SARSOP...")
standard_solver = SARSOPSolver(precision=1e-2, max_time=10.0)
standard_policy = POMDPs.solve(standard_solver, pomdp)

println("Solving with SlidingSARSOP...")
sliding_solver = SlidingSARSOP.SlidingSARSOPSolver(
    horizon_distance = 4,
    precision = 1e-2,
    timeout = 2.0
)
sliding_policy = POMDPs.solve(sliding_solver, pomdp)

# Create GIFs of both policies
println("Creating GIFs to visualize the policies...")

# Set a fixed seed for reproducibility
seed = 42
max_steps = 20

# Standard SARSOP simulation
standard_sim = GifSimulator(
    filename = "StandardSARSOP.gif",
    max_steps = max_steps,
    rng = MersenneTwister(seed),
    show_progress = true
)
standard_gif = simulate(standard_sim, pomdp, standard_policy)
println("Standard SARSOP GIF saved to: $(standard_gif.filename)")

# Sliding SARSOP simulation
sliding_sim = GifSimulator(
    filename = "SlidingSARSOP.gif",
    max_steps = max_steps,
    rng = MersenneTwister(seed),  # Same seed for fair comparison
    show_progress = true
)
sliding_gif = simulate(sliding_sim, pomdp, sliding_policy)
println("Sliding SARSOP GIF saved to: $(sliding_gif.filename)")

# Show summary of trajectories
println("\nCreating GIFs with belief visualization...")

# Standard SARSOP with belief
standard_belief_sim = GifSimulator(
    filename = "StandardSARSOP_with_belief.gif",
    max_steps = max_steps,
    rng = MersenneTwister(seed),
    show_progress = true,
    render_kwargs = (viz_belief=true,)
)
standard_belief_gif = simulate(standard_belief_sim, pomdp, standard_policy)
println("Standard SARSOP with belief GIF saved to: $(standard_belief_gif.filename)")

# Sliding SARSOP with belief
sliding_belief_sim = GifSimulator(
    filename = "SlidingSARSOP_with_belief.gif",
    max_steps = max_steps,
    rng = MersenneTwister(seed),
    show_progress = true,
    render_kwargs = (viz_belief=true,)
)
sliding_belief_gif = simulate(sliding_belief_sim, pomdp, sliding_policy)
println("Sliding SARSOP with belief GIF saved to: $(sliding_belief_gif.filename)")

println("\nVisualization complete! Check the generated GIF files to see the policies in action.")