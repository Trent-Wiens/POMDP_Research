"""
Clean rewrite of the receding‑horizon RockSample experiment with:
- Correct local↔global mapping
- No belief truncation between windows (carry full posterior)
- Distance‑aware exit‑reward shaping (suppresses local exit if sampling now is better)
- Local→global rock index mapping everywhere
- Minimal prints that explain decisions

This reproduces the original behavior (solve sub‑POMDPs around the current
position and advance one step at a time) but fixes the beeline‑to‑exit issue.
"""

using POMDPs
using POMDPTools
using NativeSARSOP
using RockSample
using Random
using DataFrames

# ---- Initialstate hook for sub-POMDP belief (defined at top level) ----
if !isdefined(Main, :RS_INIT_BELIEF)
    const RS_INIT_BELIEF = Ref{Any}(nothing)
    POMDPs.initialstate(::RockSamplePOMDP) = RS_INIT_BELIEF[]
end

# ------------------------ Utilities ------------------------ #

# Normalize a 2-vector (Tuple, Vector, SVector) to a plain (Int,Int) Tuple
as_xy_tuple(x) = (Int(x[1]), Int(x[2]))
# Get GLOBAL rock positions as (Int,Int) tuples for reliable indexing/matching
global_pos_tuples(rock_probs::SparseCat) = map(as_xy_tuple, collect(rock_probs.vals))

"""Global → local grid coordinates for a sub-window `sub_map=(xmin,ymin,xmax,ymax)`."""
function global2local(pos::AbstractVector{<:Integer}, sub_map::NTuple{4,Int})
    return [pos[1] - sub_map[1] + 1, pos[2] - sub_map[2] + 1]
end

"""Local → global grid coordinates for a sub-window `sub_map=(xmin,ymin,xmax,ymax)`."""
function local2global(pos::AbstractVector{<:Integer}, sub_map::NTuple{4,Int})
    return [pos[1] + sub_map[1] - 1, pos[2] + sub_map[2] - 1]
end

actionNum2word(a) = a == 1 ? "Sample" : a == 2 ? "North" : a == 3 ? "East" : a == 4 ? "South" : a >= 6 ? "Sense Rock $(a-5)" : a == 5 ? "West" : "?"
obsNum2word(o)    = o == 1 ? "Good"   : o == 2 ? "Bad"    : o == 3 ? "None"       : "?"

# -------------------- Sub-POMDP construction -------------------- #

"""
Construct a sub‑POMDP centered around global `pos` with Chebyshev radius `horizon`.
Also returns an initial **belief** over sub‑states derived from the global rock
probabilities (assumed independent per rock), and the `(xmin,ymin,xmax,ymax)`
sub‑map tuple for later mapping.

`rock_probs` is a SparseCat over GLOBAL rock coordinates: vals = [(x,y), ...],
probs = [p_good, ...].
"""
function make_sub_pomdp(
    pos,
    map_size,
    rock_pos,
    rock_probs::SparseCat,
    pomdp;
    horizon::Int = 3,
)
    # Window bounds in GLOBAL coordinates
    xmin = max(pos[1] - horizon, 1)
    xmax = min(pos[1] + horizon, map_size[1])
    ymin = max(pos[2] - horizon, 1)
    ymax = min(pos[2] + horizon, map_size[2])
    sub_map = (xmin, ymin, xmax, ymax)

    # Rocks that fall inside this window (GLOBAL coords)
    sub_rocks = [(x,y) for (x,y) in rock_pos if xmin ≤ x ≤ xmax && ymin ≤ y ≤ ymax]
    gvals_tuple = global_pos_tuples(rock_probs)

    # If none have nonzero prior in window, pull the nearest outside rock into the model
    idx_in = findall(x -> x in sub_rocks, rock_probs.vals)
    if !isempty(idx_in)
        # ok
    elseif !isempty(rock_pos)
        # nearest (Manhattan) rock outside the window
        outside = [(x,y) for (x,y) in rock_pos if !(xmin ≤ x ≤ xmax && ymin ≤ y ≤ ymax)]
        if !isempty(outside)
            dists = [abs(pos[1]-r[1]) + abs(pos[2]-r[2]) for r in outside]
            push!(sub_rocks, outside[argmin(dists)])
        end
    end

    # Sub‑map size and local init position
    sub_size = (xmax - xmin + 1, ymax - ymin + 1)
    locpos   = global2local(pos, sub_map)

    # ---- Distance‑aware exit shaping ---- #
    is_global_right_edge = xmax == map_size[1]
    exit_reward_local = is_global_right_edge ? pomdp.exit_reward : 0.0

    # If standing ON a rock (compare in GLOBAL coords) compute EV(sample)
    rock_here = findfirst(==(Tuple(pos)), sub_rocks)
    if rock_here !== nothing
        gi = findfirst(==(sub_rocks[rock_here]), gvals_tuple) # global rock index
        if gi !== nothing
            p_good = rock_probs.probs[gi]
            ev_sample = p_good * pomdp.good_rock_reward + (1-p_good) * pomdp.bad_rock_penalty
            # steps to move off the right edge of THIS window
            dist_to_exit = (xmax - pos[1]) + 1
            discounted_exit = is_global_right_edge ? (pomdp.discount_factor^dist_to_exit) * pomdp.exit_reward : 0.0
            if ev_sample >= discounted_exit
                exit_reward_local = 0.0
                @info "Shaping: suppress exit" p_good ev_sample dist_to_exit discounted_exit
            else
                @info "Shaping: keep exit" p_good ev_sample dist_to_exit discounted_exit
            end
        end
    end

    # Also suppress exit if ANY nearby rock in window has EV(sample) exceeding discounted exit after walking to it
    begin
        best_ev_margin = -Inf
        for (idx, rxy) in enumerate(sub_rocks)
            gi = findfirst(==(rxy), gvals_tuple)
            gi === nothing && continue
            p_good = rock_probs.probs[gi]
            ev_sample = p_good * pomdp.good_rock_reward + (1 - p_good) * pomdp.bad_rock_penalty
            # distance in steps (Manhattan) from current pos to that rock, plus one step to sample
            dist = abs(pos[1]-rxy[1]) + abs(pos[2]-rxy[2]) + 1
            # discounted value of exiting from current position (two-stage baseline)
            discounted_exit_from_here = is_global_right_edge ? (pomdp.discount_factor^((xmax - pos[1]) + 1)) * pomdp.exit_reward : 0.0
            # bias toward sampling sooner: more steps reduce advantage
            margin = ev_sample - discounted_exit_from_here * (pomdp.discount_factor^(max(dist-1,0)))
            if margin > best_ev_margin
                best_ev_margin = margin
            end
        end
        if best_ev_margin > 0
            exit_reward_local = 0.0
            @info "Shaping: suppress exit due to better rock in window" best_ev_margin
        end
    end

    sub_pomdp = RockSamplePOMDP(
        map_size = sub_size,
        rocks_positions = sub_rocks,
        init_pos = locpos,
        sensor_efficiency = pomdp.sensor_efficiency,
        discount_factor = pomdp.discount_factor,
        good_rock_reward = pomdp.good_rock_reward,
        bad_rock_penalty = pomdp.bad_rock_penalty,
        exit_reward = exit_reward_local,
        sensor_use_penalty = pomdp.sensor_use_penalty,
    )

    # ---- Build initial belief over SUB states from global rock priors ---- #
    states = ordered_states(sub_pomdp)
    # Keep only states that match the local starting position
    init_states = [s for s in states if s.pos == locpos]
    numR = length(sub_pomdp.rocks_positions)

    # Map local rock order → global order so we can read p_good
    local2global = [findfirst(==(sub_pomdp.rocks_positions[i]), gvals_tuple) for i in 1:numR]
    p_good_local = [gi === nothing ? 0.0 : rock_probs.probs[gi] for gi in local2global]

    init_probs = zeros(Float64, length(init_states))
    for (j,s) in enumerate(init_states)
        # s.rocks is a BitVector/Array{Bool}: true means rock is GOOD in that state
        # P(s) = ∏_r p if s.rocks[r] else (1-p)
        prodp = 1.0
        @inbounds for r in 1:numR
            p = p_good_local[r]
            prodp *= s.rocks[r] ? p : (1 - p)
        end
        init_probs[j] = prodp
    end
    s = sum(init_probs)
    if s <= 0
        init_probs .= 1/length(init_probs)
    else
        init_probs ./= s
    end
    init_belief = SparseCat(init_states, init_probs)

    return sub_pomdp, init_belief, sub_map
end

# -------------------- One-step advance + belief carry -------------------- #

"""
Advance one step using `policy` on `thisPomdp` from `init_belief`.
Update global `rock_probs` by marginalizing over the posterior, choose the
next GLOBAL position via MAP over the full posterior, and return the next pos
or "TERMINAL" if the transition ended the episode.
"""
function step_and_carry!(
    policy,
    thisPomdp,
    rock_probs::SparseCat,
    sub_map,
    init_belief::SparseCat,
    actionList::Vector{String},

)
    up = updater(policy)
    b0 = init_belief

    state = nothing; action = nothing; obs = nothing; rew = nothing
    for (s,a,o,r) in stepthrough(thisPomdp, policy, "s,a,o,r", max_steps=1)
        println("took action ", actionNum2word(a))
        println("received observation ", obsNum2word(o), " and reward ", r)
        println("----------------------------------------------------")
        state, action, obs, rew = s,a,o,r
    end

    trans = transition(thisPomdp, state, action)
    push!(actionList, actionNum2word(action))
    if POMDPs.isterminal(thisPomdp, trans.val)
        return "TERMINAL"
    end

    b1 = update(up, b0, action, obs)
    println("Value of initial belief: ", value(policy, b0))
    println("Value of next belief: ", value(policy, b1))

    # ---- Update GLOBAL rock probabilities using action-aware logic ---- #
    numR = length(thisPomdp.rocks_positions)
    gvals_tuple = global_pos_tuples(rock_probs)
    
    # Helper: map a local rock index -> global index
    local_to_global = Dict{Int,Int}()
    for r in 1:numR
        local_xy = thisPomdp.rocks_positions[r]
        local_xy_T = (local_xy[1], local_xy[2])
        gi = findfirst(==(local_xy_T), gvals_tuple)
        if gi !== nothing
            local_to_global[r] = gi
        else
            @warn "Rock mapping failed; local rock not found in global list" local_xy_T global_vals=gvals_tuple
        end
    end
    
    # Detect which rock (if any) is referenced by the action
    # action ids: 1=Sample, 2=N, 3=E, 4=S, 5=W, 6+=Sense Rock (a-5)
    affected_global_index = nothing
    if action == 1
        # Sample: if we're standing on a rock in LOCAL coords, set that rock's prob to 0 (removed)
        here_local = state.pos
        r_on_here = findfirst(r -> (thisPomdp.rocks_positions[r][1] == here_local[1] &&
                                    thisPomdp.rocks_positions[r][2] == here_local[2]), 1:numR)
        if r_on_here !== nothing && haskey(local_to_global, r_on_here)
            affected_global_index = local_to_global[r_on_here]
            rock_probs.probs[affected_global_index] = 0.0
            @info "Belief update (sampled): set rock prob to 0" local_xy_T=thisPomdp.rocks_positions[r_on_here] gi=affected_global_index
        end
    elseif action >= 6
        # Sense Rock k: only update that one using the posterior b1
        k_local = action - 5
        if 1 <= k_local <= numR && haskey(local_to_global, k_local)
            affected_global_index = local_to_global[k_local]
            mass = 0.0
            for (s,p) in zip(b1.state_list, b1.b)
                if s.rocks[k_local] == true
                    mass += p
                end
            end
            rock_probs.probs[affected_global_index] = mass
            @info "Belief update (sensed): posterior mass of GOOD" local_xy_T=thisPomdp.rocks_positions[k_local] gi=affected_global_index new_mass=mass
        end
    else
        # Movement only: do not try to re-estimate rock goodness from b1; carry beliefs forward unchanged.
        @info "Belief update: movement step only; leaving rock priors unchanged"
    end

    # ---- Choose next GLOBAL position via MAP over positions from b1 ---- #
    pos_mass = Dict{Tuple{Int,Int},Float64}()
    for (s,p) in zip(b1.state_list, b1.b)
        k = (s.pos[1], s.pos[2])
        pos_mass[k] = get(pos_mass, k, 0.0) + p
    end
    best_key = nothing
    best_p = -Inf
    for (k,p) in pos_mass
        if p > best_p
            best_p = p
            best_key = k
        end
    end
    local_next = [best_key[1], best_key[2]]
    global_next = local2global(local_next, sub_map)
    @info "Next position (MAP)" local_next global_next
    return global_next
end

# ----------------------------- Main ----------------------------- #

function main()
    rng = MersenneTwister(1)

    # Full POMDP (7×7 example from your log)
    pomdp = RockSamplePOMDP(
        map_size = (7,7),
        rocks_positions = [(1,2), (3,7), (6,6), (1,5)],
        init_pos = [6,6],
        sensor_efficiency = 20.0,
        discount_factor = 0.95,
        good_rock_reward = 20.0,
        bad_rock_penalty = -5.0,
        exit_reward = 10.0,
    )

    # Global rock priors (p_good)
    rock_probs = SparseCat(pomdp.rocks_positions, fill(0.5, length(pomdp.rocks_positions)))

    pos = copy(pomdp.init_pos)
    actionList = String[]
    posList = Any[]

    for i in 1:20
        println("============================================")
        display(rock_probs)
        println("Iteration: $i\nCurrent Position: $pos")

        sub_pomdp, init_belief, sub_map = make_sub_pomdp(pos, pomdp.map_size, pomdp.rocks_positions, rock_probs, pomdp)

        # Set the initial belief for this sub-POMDP run (consumed by NativeSARSOP)
        RS_INIT_BELIEF[] = init_belief

        solver = SARSOPSolver(precision=1e-3; max_time=10.0, verbose=false)
        policy = solve(solver, sub_pomdp)

        nextpos = step_and_carry!(policy, sub_pomdp, rock_probs, sub_map, init_belief, actionList)
        push!(posList, nextpos)
        if nextpos == "TERMINAL"
            println("Reached terminal state. Exiting loop.")
            println("Actions taken:")
            for (k,a) in enumerate(actionList)
                p = k <= length(posList) ? posList[k] : "?"
                println("    ", a, " -> ", p)
            end
            break
        else
            pos = nextpos
        end
    end
end

main()
