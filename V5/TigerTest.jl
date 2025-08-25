import POMDPs
using POMDPs: POMDP
using POMDPTools: Deterministic, Uniform, SparseCat

struct TigerPOMDP <: POMDP{String, String, String}
    p_correct::Float64
    indices::Dict{String, Int}

    TigerPOMDP(p_correct=0.85) = new(p_correct, Dict("left"=>1, "right"=>2, "listen"=>3))
end

POMDPs.states(m::TigerPOMDP) = ["left", "right"]
POMDPs.actions(m::TigerPOMDP) = ["left", "right", "listen"]
POMDPs.observations(m::TigerPOMDP) = ["left", "right"]
POMDPs.discount(m::TigerPOMDP) = 0.95
POMDPs.stateindex(m::TigerPOMDP, s) = m.indices[s]
POMDPs.actionindex(m::TigerPOMDP, a) = m.indices[a]
POMDPs.obsindex(m::TigerPOMDP, o) = m.indices[o]

function POMDPs.transition(m::TigerPOMDP, s, a)
    if a == "listen"
        return Deterministic(s) # tiger stays behind the same door
    else # a door is opened
        return Uniform(["left", "right"]) # reset
    end
end

function POMDPs.observation(m::TigerPOMDP, a, sp)
    if a == "listen"
        if sp == "left"
            return SparseCat(["left", "right"], [m.p_correct, 1.0-m.p_correct])
        else
            return SparseCat(["right", "left"], [m.p_correct, 1.0-m.p_correct])
        end
    else
        return Uniform(["left", "right"])
    end
end

function POMDPs.reward(m::TigerPOMDP, s, a)
    if a == "listen"
        return -1.0
    elseif s == a # the tiger was found
        return -100.0
    else # the tiger was escaped
        return 10.0
    end
end

POMDPs.initialstate(m::TigerPOMDP) = Uniform(["left", "right"])
# output