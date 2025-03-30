using POMDPs
using POMDPTools
using StaticArrays

# Include the DroneRockSample module
include("DroneRockSample.jl")
using .DroneRockSample

# Include your SlidingPOMDP implementation
include("SlidingPOMDP.jl")
import .SlidingPOMDP: make_sub_pomdp, set_dronerocksample

# Set the DroneRockSample module reference
set_dronerocksample(DroneRockSample)

# Create a DroneRockSample POMDP instance
pomdp = DroneRockSamplePOMDP(
    map_size = (10, 10),
    rocks_positions = [(2, 8), (4, 5), (7, 2), (8, 9)],
    sensor_efficiency = 20.0,
    discount_factor = 0.95,
    good_rock_reward = 20.0,
    fly_penalty = -0.2
)

# Print original POMDP info
println("Original POMDP:")
println("Map size: $(pomdp.map_size)")
println("Rocks: $(pomdp.rocks_positions)")
println("State space size: $(length(pomdp))")

# Create a state for testing
test_state = RSState(RSPos(5, 5), SVector{4, Bool}(false, false, false, false))
println("\nTest state position: $(test_state.pos)")

# Test with different horizon distances
# for horizon in [1, 3]
#     println("\n--- Testing with horizon = $horizon ---")
horizon = 3
    
    # Call your make_sub_pomdp function
    sub_pomdp = SlidingPOMDP.make_sub_pomdp(pomdp, test_state, horizon)
    
    # Print sub-POMDP info
    println("Sub-POMDP:")
    println("Map size: $(sub_pomdp.map_size)")
    println("Rocks: $(sub_pomdp.rocks_positions)")
    println("State space size: $(length(sub_pomdp))")
    
    # Calculate the size reduction
    size_ratio = length(sub_pomdp) / length(pomdp)
    println("Size reduction: $(round(100 * (1 - size_ratio), digits=2))%")
    
    # Verify the expected size
    expected_width = min(2*horizon + 1, 10)
    expected_height = min(2*horizon + 1, 10)
    println("Expected map size: ($expected_width, $expected_height)")
    if sub_pomdp.map_size == (expected_width, expected_height)
        println("✓ Map size matches expected dimensions")
    else
        println("✗ Map size doesn't match expected dimensions")
    end
# end