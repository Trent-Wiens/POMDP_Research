# Observations for DroneRockSample - similar to the original RockSample

const OBSERVATION_NAME = (:good, :bad, :none)

POMDPs.observations(pomdp::DroneRockSamplePOMDP) = 1:3
POMDPs.obsindex(pomdp::DroneRockSamplePOMDP, o::Int) = o

function POMDPs.observation(pomdp::DroneRockSamplePOMDP{K}, a::Int, s::RSState) where K
    if a <= N_BASIC_ACTIONS
        # No observation for sample or fly actions
        return SparseCat((1,2,3), (0.0,0.0,1.0)) # for type stability
    else
        # Sensing action
        rock_ind = a - N_BASIC_ACTIONS 
        rock_pos = pomdp.rocks_positions[rock_ind]
        dist = norm(rock_pos - s.pos)
        efficiency = 0.5*(1.0 + exp(-dist*log(2)/pomdp.sensor_efficiency))
        rock_state = s.rocks[rock_ind]
        if rock_state
            return SparseCat((1,2,3), (efficiency, 1.0 - efficiency, 0.0))
        else
            return SparseCat((1,2,3), (1.0 - efficiency, efficiency, 0.0))
        end
    end
end