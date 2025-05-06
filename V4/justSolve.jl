using POMDPs
using POMDPTools
using NativeSARSOP
using POMDPGifs
using Random
using Cairo

# Include the DroneRockSample module
include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: action_to_string

function run_full_sarsop()
    rng = MersenneTwister(42)

    # Define the POMDP
    rock_positions = [(2,1), (3, 7),  (8, 4),  (6, 6),  (1, 5),   (4, 8), (10, 2)]
    pomdp = DroneRockSamplePOMDP(
        map_size=(10, 10),
        rocks_positions=rock_positions,
        init_pos=(1, 1),
        sensor_efficiency=20.0,
        discount_factor=0.95,
        good_rock_reward=20.0,
        fly_penalty=-0.2
    )

    println("Created DroneRockSample POMDP with dimensions $(pomdp.map_size) and $(length(pomdp.rocks_positions)) rocks")

    # Solve using SARSOP
    println("Solving with SARSOP...")
    solver = SARSOPSolver(precision=1e-2, max_time=5.0)
    policy = solve(solver, pomdp)

    # Simulate the solution
    println("Creating simulation GIF...")
    sim = GifSimulator(
        filename="FullSARSOP.gif",
        max_steps=50,
        rng=MersenneTwister(1),
        show_progress=true
    )
    simulate(sim, pomdp, policy)
    println("GIF saved to: $(sim.filename)")
end

run_full_sarsop()
