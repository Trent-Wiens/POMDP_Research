using POMDPs, Distributions, PrettyTables, DataFrames

function dist(pos, target)
    
    x1, y1 = (pos - 1) % 4 + 1, div(pos - 1, 4) + 1
    x2, y2 = (target - 1) % 4 + 1, div(target - 1, 4) + 1
    distance = sqrt((x1 - x2)^2 + (y1 - y2)^2)  # Euclidean distance
    if distance == 0
        return 0.0  # No self-loop probability
    end
    return distance


end

states = [(pos, battery, time) for pos in 1:16 for battery in [33,66,100] for time in [0,1]]

actions = vcat(1:16, [17])

function transitionMat()

    P = Dict()

    for (pos, battery, time) in states
        for a in actions
            next_states = Dict()

            if a == 17 #charge
                next_bat = 100
                next_time = time + 1
                next_states[(pos, next_bat, next_time)] = 1

            else #moving
                norm_dist = Normal(0,3)

                tot_prob = 0
                for target_pos in 1:16
                    distance = dist(pos, target_pos)
                    if distance > 0
                        p = cdf(norm_dist, distance) - cdf(norm_dist, distance - 1)
                        next_states[(target_pos, max(0,battery - distance), time + 1)] = p
                        tot_prob += p
                    end
                end

                for k in keys(next_states)
                    next_states[k] /= tot_prob
                end
            end

            P[(pos, battery, time), a] = next_states


        end
    end

    return P

end

P = transitionMat()
using PrettyTables, DataFrames

# Define fixed conditions
fixed_battery = 66
fixed_time = 1
action = 1
positions = 1:16  # 16 positions in the grid

# Initialize transition matrix (16x16 filled with zeros)
TPM = fill(0.0, 16, 16)

# Populate TPM
for pos in positions
    state = (pos, fixed_battery, fixed_time)
    if haskey(P, (state, action))  # Ensure the (state, action) pair exists
        transitions = P[(state, action)]
        for (next_state, prob) in transitions
            next_pos = next_state[1]  # Extract only the position
            TPM[pos, next_pos] = prob  # Fill matrix
        end
    end
end

# Convert to DataFrame for readability
df_TPM = DataFrame(TPM, :auto)
rename!(df_TPM, string.(1:16))  # Column names as positions
insertcols!(df_TPM, 1, :From => 1:16)  # Add row labels

# Print nicely
println("Transition Probability Matrix for Action = 1, Battery = 66, Time = 1:")
pretty_table(df_TPM, header=["From ↓ \\ To →", string.(1:16)...])

# # Get all state-action pairs where action is 1
# filtered_keys = filter(k -> k[2] == 1, keys(P))  # (state, action) where action = 1

# # Extract transition probabilities for each state-action pair
# for key in filtered_keys
#     println("TPM for State: ", key[1], " | Action: 1")
#     pretty_table(hcat(collect(keys(P[key])), collect(values(P[key]))), 
#                  header=["Next State", "Probability"])
#     println()
# end
