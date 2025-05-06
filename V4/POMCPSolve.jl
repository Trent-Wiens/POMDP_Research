using POMDPs
using POMCPOW
using POMDPModels
using POMDPTools


include("DroneRockSample.jl")
using .DroneRockSample
import .DroneRockSample: action_to_string

function run_pomcp_simulation()

    solver = POMCPOWSolver(criterion=MaxUCB(20.0))
    pomdp = TigerPOMDP() # from POMDPModels
    planner = solve(solver, pomdp)
    
    hr = HistoryRecorder(max_steps=100)
    hist = simulate(hr, pomdp, planner)
    for (s, b, a, r, sp, o) in hist
        @show s, a, r, sp
    end
    
    rhist = simulate(hr, pomdp, RandomPolicy(pomdp))
    println("""
        Cumulative Discounted Reward (for 1 simulation)
            Random: $(discounted_reward(rhist))
            POMCPOW: $(discounted_reward(hist))
        """)

end

run_pomcp_simulation()