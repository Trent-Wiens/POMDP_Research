# Receding-Horizon Solver for RockSample (POMDPs.jl)
# -------------------------------------------------
# This script performs model-predictive (receding-horizon) planning on the
# RockSample POMDP using an online planner (POMCPOW) with a fixed lookahead
# depth H. At each real step, we:
#   1) Re-plan from the current belief with depth H and a compute budget.
#   2) Execute only the first action.
#   3) Update the belief with the observation and repeat.
#
# Requirements (Julia ≥ 1.9):
# ] add POMDPs POMDPModels POMDPPolicies POMDPTools POMCPOW Random
# (Optionally) ] add SARSOP  # if you want to compare offline policies
#
# To run:
#   julia --project=. rocksample_receding_horizon.jl

import Pkg
Pkg.activate(@__DIR__)     # use the project in the script's folder
Pkg.instantiate()          # install the exact deps recorded in Project/Manifest

using Random
using POMDPs
using POMDPModels
using POMDPTools
using POMDPPolicies
using POMCPOW

# -------------------------
# Problem Setup
# -------------------------
# Common RockSample instances: RockSample(7, 8, rocks_vec)
# where rocks_vec is a list of rock (x,y) coordinates (1-based, (1,1) at SW).
# Here we use the classic RS(7,8) with 4 rocks.

function make_problem(; seed=7)
    rng = MersenneTwister(seed)
    width, height = 7, 8
    rocks = [(1,2), (3,7), (6,6), (1,5)]
    # Sensor parameters are from POMDPModels defaults (good enough for demo)
    rs = RockSample(width, height, rocks)
    γ = discount(rs)  # already set by model (default 0.95)
    return rs, rng
end

# -------------------------
# Receding-Horizon Agent
# -------------------------
# We’ll use POMCPOW as the online planner. The two key RH knobs are:
#   - H: planning horizon (max search depth)
#   - N: number of tree queries per real step (compute budget)
# You can also tune exploration constants and rollout depth.

Base.@kwdef mutable struct RHConfig
    H::Int = 6           # lookahead depth (planning horizon)
    N::Int = 10_000      # tree queries per step (compute budget)
    k_obs::Float64 = 4.0 # POMCPOW observation widening (default reasonable)
    k_act::Float64 = 4.0 # POMCPOW action widening
    c::Float64 = 10.0    # UCB exploration constant
    rollout_h::Int = 6   # rollout depth for default policy
end

# A simple heuristic rollout policy: use a random policy with a slight bias
# to moving east (toward the exit) if available.
struct EastBiasedRandomPolicy{P} <: Policy
    base::P
end

function POMDPs.action(p::EastBiasedRandomPolicy, b)
    A = actions(p.base.pomdp)
    # Bias: if :east (or East()) is present, choose it more often.
    # RockSample uses Symbol actions like :north, :south, :east, :west,
    # and :sample, :check_i for sensing rock i.
    if :east in A
        # 50% east, 50% uniform over others
        if rand() < 0.5
            return :east
        end
        others = collect(a for a in A if a != :east)
        return rand(others)
    else
        return rand(A)
    end
end

# Build a POMCPOW policy for the current belief state with our RHConfig.
function build_planner(pomdp::POMDP, cfg::RHConfig, rng)
    # Default policy used in rollouts (heuristic policy defined above)
    rp = RandomPolicy(pomdp; rng=rng)
    dp = EastBiasedRandomPolicy(rp)

    return POMCPOWPlanner(
        pomdp,
        tree_queries=cfg.N,
        max_depth=cfg.H,
        k_obs=cfg.k_obs,
        k_action=cfg.k_act,
        c=cfg.c,
        default_action=first(actions(pomdp)), # fallback
        estimate_value=RolloutEstimator(dp; rollout_depth=cfg.rollout_h, rng=rng),
        rng=rng,
    )
end

# -------------------------
# Belief Utilities
# -------------------------
# RockSample in POMDPModels has discrete latent state. We’ll use a discrete
# belief updater. Initial belief comes from the model.

function initial_belief(pomdp::POMDP)
    d = initialstate(pomdp)  # distribution (works on current POMDPs.jl)
    return initialize_belief(DiscreteUpdater(pomdp), d)
end
# -------------------------
# Simulation Loop (MPC/RH)
# -------------------------

function run_episode(; cfg=RHConfig(), seed=7, max_steps=200, verbose=true)
    pomdp, rng = make_problem(seed=seed)

    up = DiscreteUpdater(pomdp)
    b = initial_belief(pomdp)
    s = rand(rng, particles(b))  # sample a state for the simulator

    total_reward = 0.0
    γ = discount(pomdp)

    for t in 1:max_steps
        planner = build_planner(pomdp, cfg, rng)
        a = action(planner, b)  # plan from current belief (re-solve each step)

        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, a, rng)
        total_reward += r

        # Belief update
        b = update(up, b, a, o)

        if verbose
            println("Step $(t): action=$(a), reward=$(r), obs=$(o)")
        end

        # Check termination: RockSample terminal on exiting the east edge
        if isterminal(pomdp, sp)
            s = sp
            break
        end

        s = sp
    end

    return total_reward
end

# -------------------------
# Demo / main
# -------------------------
if abspath(PROGRAM_FILE) == @__FILE__
    cfg = RHConfig(H=6, N=12_000, rollout_h=8)
    println("Running a single receding-horizon episode…")
    R = run_episode(cfg=cfg, seed=42, max_steps=100, verbose=true)
    println("Total undiscounted reward: ", R)

    # You can also do a quick sweep over budgets/horizons
    println("\nBudget/Depth sweep (3 seeds)…")
    for H in (4, 6, 8)
        for N in (5_000, 10_000, 20_000)
            cfg = RHConfig(H=H, N=N, rollout_h=H)
            rewards = [run_episode(cfg=cfg, seed=10+i, max_steps=120, verbose=false) for i in 0:2]
            println("H=$(H), N=$(N) ⇒ mean R=$(round(mean(rewards); digits=2)) ± $(round(std(rewards); digits=2))")
        end
    end
end
