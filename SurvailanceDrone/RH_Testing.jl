using POMDPs
using POMDPTools
using POMDPModels
using DroneSurveillance
using NativeSARSOP
using Random
using LinearAlgebra
using StaticArrays
using Statistics

# --- 1. SETUP THE PROBLEM ---

# We use a small map so the Global Solver can actually finish.
# RHC is designed for maps where Global fails (e.g., 20x20), 
# but we need a small map to have a "Ground Truth" to compare against.
MAP_SIZE = (10, 10) 
FOV = (3, 3) # Camera size
HORIZON = 1  # 1 radius = 3x3 window (matches FOV)

# Define the Master POMDP
global_pomdp = DroneSurveillancePOMDP(
    size = MAP_SIZE,
    region_A = [1, 1],
    region_B = [MAP_SIZE[1], MAP_SIZE[2]],
    fov = FOV,
    agent_policy = :restricted,
    discount_factor = 0.95
)

# --- 2. DEFINE THE TWO METHODS ---

# Method A: Standard Global Solve (The "Gold Standard")
println("Pre-solving Global Policy (Baseline)...")
# Note: precision is high here to get the "best possible" value
global_solver = SARSOPSolver(precision=1e-3, max_time=10.0, verbose=false)
global_policy = solve(global_solver, global_pomdp)

function run_global_sim(pomdp, policy, rng)
    # Standard POMDP simulation
    sim = RolloutSimulator(max_steps=50, rng=rng)
    return simulate(sim, pomdp, policy)
end

# Method B: Your Receding Horizon Control (RHC)
# (Includes the Helper functions from previous turns)
function make_sub_POMDP(quad_pos, agent_pos, global_pomdp)
    horizon = HORIZON
    g_size = global_pomdp.size
    
    minx = max(quad_pos[1] - horizon, 1)
    maxx = min(quad_pos[1] + horizon, g_size[1])
    miny = max(quad_pos[2] - horizon, 1)
    maxy = min(quad_pos[2] + horizon, g_size[2])

    sub_bounds = (minx, miny, maxx, maxy)
    sub_size = (maxx - minx + 1, maxy - miny + 1)

    function get_clamped_local(global_p, bounds, local_size)
        lx = global_p[1] - bounds[1] + 1
        ly = global_p[2] - bounds[2] + 1
        cx = clamp(lx, 1, local_size[1])
        cy = clamp(ly, 1, local_size[2])
        return [cx, cy]
    end

    local_region_A = get_clamped_local(global_pomdp.region_A, sub_bounds, sub_size)
    local_region_B = get_clamped_local(global_pomdp.region_B, sub_bounds, sub_size)

    sub_pomdp = DroneSurveillancePOMDP(
        size = sub_size,
        region_A = local_region_A,
        region_B = local_region_B,
        fov = global_pomdp.fov,
        agent_policy = global_pomdp.agent_policy,
        camera = global_pomdp.camera,
        discount_factor = global_pomdp.discount_factor
    )

    local_quad = global2local(quad_pos, sub_bounds)
    local_agent = get_clamped_local(agent_pos, sub_bounds, sub_size)
    
    # WRAP IN DETERMINISTIC BELIEF
    init_belief = Deterministic(DSState(local_quad, local_agent))

    return sub_pomdp, init_belief, sub_bounds
end

function global2local(pos, sub_map_bounds)
    return [pos[1] - sub_map_bounds[1] + 1, pos[2] - sub_map_bounds[2] + 1]
end

function run_rhc_sim(pomdp, rng)
    # Start States
    # We must randomize start state same way Global does to be fair, 
    # but for simplicity here we pick fixed start or sample from initial dist.
    init_dist = initialstate(pomdp)
    start_state = rand(rng, init_dist)
    
    quad_pos = copy(start_state.quad)
    agent_pos = copy(start_state.agent)
    
    accumulated_reward = 0.0
    discount = 1.0
    gamma = pomdp.discount_factor
    
    for t in 1:50
        # 1. Plan
        sub_pomdp, sub_b0, bounds = make_sub_POMDP(quad_pos, agent_pos, pomdp)
        
        # Fast, low-precision solve for online use
        local_solver = SARSOPSolver(precision=0.1, max_time=0.5, verbose=false) 
        local_policy = solve(local_solver, sub_pomdp)
        
        # 2. Action
        action_idx = action(local_policy, sub_b0)
        
        # 3. Transition (Execute on Global)
        # We need the reward from the global environment
        # POMDPs.gen returns (obs, reward, next_state) - we just need r and sp
        s = DSState(quad_pos, agent_pos)
        sp, o, r = @gen(:sp, :o, :r)(pomdp, s, action_idx, rng)
        
        accumulated_reward += r * discount
        discount *= gamma
        
        # Update positions
        quad_pos = sp.quad
        agent_pos = sp.agent
        
        if isterminal(pomdp, sp)
            break
        end
    end
    
    return accumulated_reward
end

# --- 3. RUN THE EXPERIMENT ---

N_SIMS = 20
println("\nRunning Comparison over $N_SIMS simulations...")

global_scores = Float64[]
rhc_scores = Float64[]

for i in 1:N_SIMS
    # Important: Use same seed for paired T-test logic if possible, 
    # but here just ensuring independent randomness is fine.
    seed = i + 1000 * Random.rand(1:1000)
    
    # Run Global
    r_global = run_global_sim(global_pomdp, global_policy, MersenneTwister(seed))
    push!(global_scores, r_global)
    
    # Run RHC
    r_rhc = run_rhc_sim(global_pomdp, MersenneTwister(seed))
    push!(rhc_scores, r_rhc)
    
    print(".") # Progress bar
end

# --- 4. RESULTS ---

println("\n\n=== RESULTS (Map: $MAP_SIZE) ===")
println("Global Baseline Mean Reward: $(mean(global_scores)) ± $(std(global_scores)/sqrt(N_SIMS))")
println("RHC (Online) Mean Reward:    $(mean(rhc_scores)) ± $(std(rhc_scores)/sqrt(N_SIMS))")

ratio = mean(rhc_scores) / mean(global_scores) * 100
println("RHC Performance Ratio: $(round(ratio, digits=1))% of optimal")

# Simple ASCII Bar Chart
function draw_bar(val, max_val)
    len = Int(round((val/max_val) * 20))
    return "[" * "#"^len * " "^(20-len) * "]"
end

max_r = max(mean(global_scores), mean(rhc_scores))
println("\nVisual Comparison:")
println("Global: ", draw_bar(mean(global_scores), max_r))
println("RHC:    ", draw_bar(mean(rhc_scores), max_r))
