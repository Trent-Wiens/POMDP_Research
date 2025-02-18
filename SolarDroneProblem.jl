using POMDPs, QuickPOMDPs, Plots, POMDPTools

using LocalApproximationValueIteration, LocalFunctionApproximation, GridInterpolations, NearestNeighbors, Distances, Rectangle

using Distributions

# Function to compute the probability of moving to a new position
function move_probability(start, target, battery)
    x1, y1 = (start - 1) % 4 + 1, div(start - 1, 4) + 1
    x2, y2 = (target - 1) % 4 + 1, div(target - 1, 4) + 1
    # println(x1,y1,x2,y2)
    distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)  # Euclidean distance
    # println(distance)
    if distance == 0
        return 0.0  # No self-loop probability
    end
    return (1 / distance) * battery # Scale by battery level
end

# dist = move_probability(1,8, 1)

# print(dist)

# exit()

# Define MDP
mdp = QuickMDP(
    function gen(s, a, rng)
        pos, b, d = s  # Extract position, battery, and time
        dt = 1
        
            # Compute movement probabilities: 1/distance scaling
            move_probs = Dict()
            total_prob = 0.0

            for target_pos in 1:16
                distance = abs(pos - target_pos)  # Manhattan distance in 1D grid
                if distance > 0
                    p = 1.0 / distance  # Inverse distance weighting
                    move_probs[target_pos] = p
                    total_prob += p
                end
            end

            # Normalize probabilities
            for k in keys(move_probs)
                move_probs[k] /= total_prob
            end

            # Convert dictionary to probability distribution
            positions = collect(keys(move_probs))
            probs = Float64.(collect(values(move_probs)))

            # Sample new position based on computed probabilities
            new_pos = rand(rng, Categorical(probs))

        function charge(t)
            return max(0.05 * sin(π * t / 24), 0) # Battery charge function
        end
        
        if a == 17  # Charging action
            b_new = min(1.0, b + Float64(charge(d) * dt))
        else
            b_new = max(0.0, Float64(b) - 0.02 * sqrt((pos - new_pos)^2)) # Battery usage scales with move distance
        end
        
        d_new = (d + dt) % 24

        # Reward function
        if b_new == 0.0
            r = -100  # Large penalty for running out of battery
        elseif new_pos == 16  # Assume position 16 is the goal
            r = 100  # Large reward for reaching the goal
        else
            r = -1  # Small penalty for efficiency
        end
        
        return (sp = [Float64(new_pos), Float64(b_new), Float64(d_new)], r=r)
    end,
    
    actions = vcat(1:16, [17]),  # Moving to any position 1-16 or charging (17)
    
    initialstate = [[1.0, 1.0, 8.0]],  # Start at position 1, full battery, time = 8
    
    discount = .95,
    
    isterminal = s -> s[2] <= 0 || s[1] == 16,  # Terminal if battery is empty or goal is reached
    
    render = function render(step)
        pos, b, d = step.s
        grid_x = (pos - 1) % 4 + 1
        grid_y = div(pos - 1, 4) + 1
        scatter([grid_x], [grid_y], color=:blue, legend=false, label="Drone")
        title!("Time: $(round(d, digits=1))h | Battery: $(round(b, digits=2))")
    end
)

begin
	min_position = 1
	max_position = 16

	min_batt = 0
	max_batt = 1

    min_time = 0
    max_time = 24
end;

grid = RectangleGrid(range(min_position, stop=max_position, length=16), range(min_batt, stop=max_batt, length=21), range(min_time, stop=max_time, length=25));

interpolation = LocalGIFunctionApproximator(grid)

# Run Local Approximation Value Iteration (LAVI)
solver = LocalApproximationValueIterationSolver(interpolation, max_iterations = 1000, is_mdp_generative = true, n_generative_samples = 1)

policy = solve(solver, mdp)

println(typeof(policy))

for i = 1:25
    println("Time: $i")
    for j = 1:21
        this_state = [1,j,i]
        best_action = action(policy, this_state)
        println("   Batt: $j | Action: $best_action")
    end
   
end

# for pos in 1:16
#     for battery in range(0.0, 1.0, length=5)  # Reduce to 5 levels for readability
#         for time in range(0, 24, length=5)   # Reduce to 5 time points
#             state = [pos, battery, time]
#             best_action = action(policy, state)
#             println("State: $state -> Best Action: $best_action")
#         end
#     end
# end