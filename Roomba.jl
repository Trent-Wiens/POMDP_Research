using POMDPs
using POMDPTools
using POMDPGifs
using BasicPOMCP
using Random
using ParticleFilters
using Cairo
using LinearAlgebra


# If you don't have RoombaPOMDPs installed, uncomment the following two lines
# using Pkg
# Pkg.add(url="https://github.com/sisl/RoombaPOMDPs.git")
using RoombaPOMDPs

# Let's only consider discrete actions
roomba_actions = [RoombaAct(2.0, 0.0), RoombaAct(2.0, 0.7), RoombaAct(2.0, -0.7)]

pomdp = RoombaPOMDP(;
    sensor=Bumper(),
    mdp=RoombaMDP(;
        config=2,
        discount=0.99,
        contact_pen=-0.1,
        aspace=roomba_actions
    )
)

# Define the belief updater
num_particles = 20000
v_noise_coefficient = 0.0
om_noise_coefficient = 0.4
resampler=LowVarianceResampler(num_particles)
rng = MersenneTwister(1)
belief_updater = RoombaParticleFilter(
    pomdp, num_particles, v_noise_coefficient,
    om_noise_coefficient,resampler, rng
)

# Custom update function for the particle filter
function POMDPs.update(up::RoombaParticleFilter, b::ParticleCollection, a, o)
    pm = up._particle_memory
    wm = up._weight_memory
    ps = []
    empty!(pm)
    empty!(wm)
    all_terminal = true
    for s in particles(b)
        if !isterminal(up.model, s)
            all_terminal = false
            a_pert = RoombaAct(a.v + (up.v_noise_coeff * (rand(up.rng) - 0.5)), a.omega + (up.om_noise_coeff * (rand(up.rng) - 0.5)))
            sp = @gen(:sp)(up.model, s, a_pert, up.rng)
            weight_sp = pdf(observation(up.model, sp), o)
            if weight_sp > 0.0
                push!(ps, s)
                push!(pm, sp)
                push!(wm, weight_sp)
            end
        end
    end

    while length(pm) < up.n_init
        a_pert = RoombaAct(a.v + (up.v_noise_coeff * (rand(up.rng) - 0.5)), a.omega + (up.om_noise_coeff * (rand(up.rng) - 0.5)))
        s = isempty(ps) ? rand(up.rng, b) : rand(up.rng, ps)
        sp = @gen(:sp)(up.model, s, a_pert, up.rng)
        weight_sp = obs_weight(up.model, s, a_pert, sp, o)
        if weight_sp > 0.0
            push!(pm, sp)
            push!(wm, weight_sp)
        end
    end

    # if all particles are terminal, issue an error
    if all_terminal
        error("Particle filter update error: all states in the particle collection were terminal.")
    end

    # return ParticleFilters.ParticleCollection(deepcopy(pm))
    return ParticleFilters.resample(up.resampler,
                    WeightedParticleBelief(pm, wm, sum(wm), nothing),
                    up.rng)
end

solver = POMCPSolver(;
    tree_queries=20000,
    max_depth=150,
    c = 10.0,
    rng=MersenneTwister(1)
)

planner = solve(solver, pomdp)

sim = GifSimulator(;
    filename="examples/EscapeRoomba.gif",
    max_steps=100,
    rng=MersenneTwister(3),
    show_progress=false,
    fps=5)
saved_gif = simulate(sim, pomdp, planner, belief_updater)

println("gif saved to: $(saved_gif.filename)")