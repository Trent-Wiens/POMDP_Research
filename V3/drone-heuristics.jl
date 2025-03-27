# Heuristics for DroneRockSample

# A fixed action policy which always takes the fastest route to exit
struct DRSExitSolver <: Solver end
struct DRSExit <: Policy
    exit_return::Vector{Float64}
end

POMDPs.solve(::DRSExitSolver, m::DroneRockSamplePOMDP) = DRSExit([discount(m)^(ceil((m.map_size[1]-x)/MAX_FLIGHT_DISTANCE)) * m.exit_reward for x in 1:m.map_size[1]])
POMDPs.solve(solver::DRSExitSolver, m::UnderlyingMDP{P}) where P <: DroneRockSamplePOMDP = solve(solver, m.pomdp)

POMDPs.value(p::DRSExit, s::RSState) = s.pos[1] == -1 ? 0.0 : p.exit_return[s.pos[1]]

function POMDPs.value(p::DRSExit, b::AbstractParticleBelief)
    utility = 0.0
    for (i, s) in enumerate(particles(b))
        if s.pos[1] != -1 # if s is not terminal
            utility += weight(b, i) * p.exit_return[s.pos[1]]
        end
    end
    return utility / weight_sum(b)
end

# Find the best fly action to move east (toward exit)
function find_east_action()
    best_action = ACTION_FLY_START
    best_eastward = 0
    
    for i in 1:N_FLY_ACTIONS
        dir = ACTION_DIRS[i]
        if dir[1] > best_eastward
            best_eastward = dir[1]
            best_action = ACTION_FLY_START + i - 1
        end
    end
    
    return best_action
end

# Always choose the action that moves furthest east (toward exit)
POMDPs.action(p::DRSExit, b) = find_east_action()

# Dedicated MDP solver for DroneRockSample
struct DRSMDPSolver <: Solver
    include_Q::Bool
end

DRSMDPSolver(;include_Q=false) = DRSMDPSolver(include_Q)

POMDPs.solve(solver::DRSMDPSolver, m::DroneRockSamplePOMDP) = solve(solver, UnderlyingMDP(m))

function POMDPs.solve(solver::DRSMDPSolver, m::UnderlyingMDP{P}) where P <: DroneRockSamplePOMDP
    util = drs_mdp_utility(m.pomdp)
    if solver.include_Q
        return solve(ValueIterationSolver(init_util=util, include_Q=true), m)
    else
        return ValueIterationPolicy(m, utility=util, include_Q=false)
    end
end

# Dedicated QMDP solver for DroneRockSample
struct DRSQMDPSolver <: Solver end

function POMDPs.solve(::DRSQMDPSolver, m::DroneRockSamplePOMDP)
    vi_policy = solve(DRSMDPSolver(include_Q=true), m)
    return AlphaVectorPolicy(m, vi_policy.qmat, vi_policy.action_map)
end

# Calculate optimal utility for drone version - we need to adapt this for drone movement
function drs_mdp_utility(m::DroneRockSamplePOMDP{K}) where K
    util = zeros(length(states(m)))
    
    # For the drone, we need to recalculate discounts based on flight capabilities
    # Now we can move up to MAX_FLIGHT_DISTANCE cells per action
    steps_to_exit = [ceil((m.map_size[1] - x) / MAX_FLIGHT_DISTANCE) for x in 1:m.map_size[1]]
    discounts = discount(m) .^ steps_to_exit
    
    # Rewards for exiting
    exit_returns = [discounts[x] * m.exit_reward for x in 1:m.map_size[1]]

    # Calculate the optimal utility for states having no good rocks
    rocks = falses(K)
    for x in 1:m.map_size[1]
        for y in 1:m.map_size[2]
            util[stateindex(m, RSState(RSPos(x,y), SVector{K,Bool}(rocks)))] = exit_returns[x]
        end
    end

    # The optimal utility with k good rocks is derived from states with k-1 good rocks
    # For the drone, we need to account for direct flight to any rock within MAX_FLIGHT_DISTANCE
    for good_rock_num in 1:K
        for good_rocks in combinations(1:K, good_rock_num)
            rocks = falses(K)
            for good_rock in good_rocks
                rocks[good_rock] = true
            end
            
            for x in 1:m.map_size[1]
                for y in 1:m.map_size[2]
                    best_return = exit_returns[x]
                    
                    for good_rock in good_rocks
                        rock_pos = m.rocks_positions[good_rock]
                        dist_to_good_rock = max(
                            ceil((abs(x - rock_pos[1]) + abs(y - rock_pos[2])) / MAX_FLIGHT_DISTANCE),
                            1
                        )
                        
                        rocks[good_rock] = false
                        sample_return = discount(m)^dist_to_good_rock * (
                            m.good_rock_reward + 
                            discount(m) * util[stateindex(m, RSState(rock_pos, SVector{K,Bool}(rocks)))]
                        )
                        rocks[good_rock] = true
                        
                        if sample_return > best_return
                            best_return = sample_return
                        end
                    end
                    
                    util[stateindex(m, RSState(RSPos(x,y), SVector{K,Bool}(rocks)))] = best_return
                end
            end
        end
    end

    return util
end