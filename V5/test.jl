using Random
using POMDPs
using POMDPTools
using RockSample
using NativeSARSOP

rng   = MersenneTwister(1)
pomdp = RockSamplePOMDP(map_size = (15,15),
                        rocks_positions=[(2, 1), (3, 7), (8, 4), (6, 6), (1, 5), (4, 8), (10, 2), (6,12), (12, 14)],
                        sensor_efficiency=20.0,
                        discount_factor=0.95,
                        good_rock_reward = 20.0
                        )
                        
policy = solve(SARSOPSolver(precision=1e-3; max_time=10.0), pomdp)          # or your solver

b = initialstate(pomdp)                     # belief for action selection
s = rand(rng, initialstate(pomdp))           # a concrete underlying state
a = action(policy, b)                        # choose action from BELIEF

nt = POMDPs.gen(pomdp, s, a, rng)            # <- use the function, not @gen
sp, o, r = nt.sp, nt.o, nt.r
up = updater(policy)                         # or DiscreteUpdater(pomdp)
b_next = update(up, b, a, o)