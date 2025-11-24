using DroneSurveillance
using POMDPs
using POMDPTools

# import a solver from POMDPs.jl e.g. SARSOP
using NativeSARSOP
using Random

# for visualization
using POMDPGifs
import Cairo

function actionNum2word(a::Int64)
    if a == 1
        return :north
    elseif a == 2
        return :east
    elseif a == 3
        return :south
    elseif a == 4
        return :west
    elseif a == 5
        return :hover
    else
        error("invalid action number")
    end
end

function observationNum2word(o::Int64, CamType)
    if CamType == QuadCam()
        if o == 1
            return :SW
        elseif o == 2
            return :NW
        elseif o == 3
            return :NE
        elseif o == 4
            return :SE
        elseif o == 5
            return :DET
        elseif o == 6
            return :OUT
        else
            error("invalid observation number for QuadCam")
        end

    elseif CamType == PerfectCam()
        if o == 1
            return :N
        elseif o == 2
            return :E
        elseif o == 3
            return :S
        elseif o == 4
            return :W
        elseif o == 5
            return :DET
        elseif o == 6
            return :NE
        elseif o == 7
            return :SE
        elseif o == 8
            return :SW
        elseif o == 9
            return :NW
        elseif o == 10
            return :OUT
        else
            error("invalid observation number for PerfectCam")
        end
    else
        error("invalid camera type")
    end
end

pomdp = DroneSurveillancePOMDP(
    camera = PerfectCam(),
    region_A = [2,2]
) # initialize the problem 

solver = SARSOPSolver(precision=1e-3; verbose = true) # configure the solver
policy = solve(solver, pomdp) # solve the problem


actionList = []



println("Creating simulation GIF...")
sim = GifSimulator(
	filename="DroneSurveillance.gif",
	max_steps=100,  # Reduced steps for testing
	rng=MersenneTwister(1),
	show_progress=true  # Enable progress display
)
saved_gif = simulate(sim, pomdp, policy)
println("GIF saved to: $(saved_gif.filename)")

# for (s, a, o, r) in stepthrough(pomdp, policy, "s,a,o,r", max_steps = 20)
#     println("in state $s")
#     println("took action $(actionNum2word(a))")
#     println("received observation $(observationNum2word(o,pomdp.camera)) and reward $r")
#     println("----------------------------------------------------")
# end



regionApos = [1,1]

# makegif(pomdp, policy, filename="out.gif")



