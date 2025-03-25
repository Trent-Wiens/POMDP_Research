# Visualization for DroneRockSample

function POMDPTools.render(pomdp::DroneRockSamplePOMDP, step;
    viz_rock_state=true,
    viz_belief=true,
    pre_act_text=""
)
    nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
    cells = []
    for x in 1:nx-1, y in 1:ny-1
        ctx = cell_ctx((x, y), (nx, ny))
        cell = compose(ctx, rectangle(), fill("white"))
        push!(cells, cell)
    end
    grid = compose(context(), linewidth(0.5mm), stroke("gray"), cells...)
    outline = compose(context(), linewidth(1mm), rectangle())

    # Draw rocks
    rocks = []
    for (i, (rx, ry)) in enumerate(pomdp.rocks_positions)
        ctx = cell_ctx((rx, ry), (nx, ny))
        clr = "black"
        if viz_rock_state && get(step, :s, nothing) !== nothing
            clr = step[:s].rocks[i] ? "green" : "red"
        end
        rock = compose(ctx, ngon(0.5, 0.5, 0.3, 6), stroke(clr), fill("gray"))
        push!(rocks, rock)
    end
    rocks = compose(context(), rocks...)
    exit_area = render_exit((nx, ny))

    agent = nothing
    action = nothing
    if get(step, :s, nothing) !== nothing
        agent_ctx = cell_ctx(step[:s].pos, (nx, ny))
        agent = render_drone(agent_ctx)  # Changed to render a drone instead of a rover
        if get(step, :a, nothing) !== nothing
            action = render_action(pomdp, step)
        end
    end
    action_text = render_action_text(pomdp, step, pre_act_text)

    belief = nothing
    if viz_belief && (get(step, :b, nothing) !== nothing)
        belief = render_belief(pomdp, step)
    end
    
    sz = min(w, h)
    return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), action, agent, belief,
        exit_area, rocks, action_text, grid, outline)
end

function cell_ctx(xy, size)
    nx, ny = size
    x, y = xy
    return context((x - 1) / nx, (ny - y - 1) / ny, 1 / nx, 1 / ny)
end

function render_belief(pomdp::DroneRockSamplePOMDP, step)
    rock_beliefs = get_rock_beliefs(pomdp, get(step, :b, nothing))
    nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
    belief_outlines = []
    belief_fills = []
    for (i, (rx, ry)) in enumerate(pomdp.rocks_positions)
        ctx = cell_ctx((rx, ry), (nx, ny))
        clr = "black"
        belief_outline = compose(ctx, rectangle(0.1, 0.87, 0.8, 0.07), stroke("gray31"), fill("gray31"))
        belief_fill = compose(ctx, rectangle(0.1, 0.87, rock_beliefs[i] * 0.8, 0.07), stroke("lawngreen"), fill("lawngreen"))
        push!(belief_outlines, belief_outline)
        push!(belief_fills, belief_fill)
    end
    return compose(context(), belief_fills..., belief_outlines...)
end

function get_rock_beliefs(pomdp::DroneRockSamplePOMDP{K}, b) where K
    rock_beliefs = zeros(Float64, K)
    for (sᵢ, bᵢ) in weighted_iterator(b)
        rock_beliefs[sᵢ.rocks.==1] .+= bᵢ
    end
    return rock_beliefs
end

function render_exit(size)
    nx, ny = size
    x = nx
    y = ny
    ctx = context((x - 1) / nx, (ny - y) / ny, 1 / nx, 1)
    rot = Rotation(pi / 2, 0.5, 0.5)
    txt = compose(ctx, text(0.5, 0.5, "EXIT AREA", hcenter, vtop, rot),
        stroke("black"),
        fill("black"),
        fontsize(20pt))
    return compose(ctx, txt, rectangle(), fill("red"))
end

# Render a drone instead of a rover
function render_drone(ctx)
    # Draw a simple drone with propellers
    body = compose(context(), circle(0.5, 0.5, 0.25), fill("skyblue"), stroke("black"))
    
    # Four propellers around the drone body
    prop1 = compose(context(), ellipse(0.25, 0.25, 0.1, 0.05), fill("gray"), stroke("black"))
    prop2 = compose(context(), ellipse(0.75, 0.25, 0.1, 0.05), fill("gray"), stroke("black"))
    prop3 = compose(context(), ellipse(0.25, 0.75, 0.1, 0.05), fill("gray"), stroke("black"))
    prop4 = compose(context(), ellipse(0.75, 0.75, 0.1, 0.05), fill("gray"), stroke("black"))
    
    return compose(ctx, body, prop1, prop2, prop3, prop4)
end

function render_action_text(pomdp::DroneRockSamplePOMDP, step, pre_act_text)
    action_text = "Terminal"
    
    if get(step, :a, nothing) !== nothing
        if step.a == ACTION_SAMPLE
            action_text = "Sample"
        elseif is_fly_action(step.a)
            dir = action_to_direction(step.a)
            action_text = "Fly($(dir[1]),$(dir[2]))"
        elseif is_sense_action(step.a, length(pomdp.rocks_positions))
            action_text = "Sensing Rock $(step.a - N_BASIC_ACTIONS)"
        end
    end
    
    action_text = pre_act_text * action_text

    _, ny = pomdp.map_size
    ny += 1
    ctx = context(0, (ny - 1) / ny, 1, 1 / ny)
    txt = compose(ctx, text(0.5, 0.5, action_text, hcenter),
        stroke("black"),
        fill("black"),
        fontsize(20pt))
    return compose(ctx, txt, rectangle(), fill("white"))
end

function render_action(pomdp::DroneRockSamplePOMDP, step)
    if step.a == ACTION_SAMPLE
        ctx = cell_ctx(step.s.pos, pomdp.map_size .+ (1, 1))
        if in(step.s.pos, pomdp.rocks_positions)
            rock_ind = findfirst(isequal(step.s.pos), pomdp.rocks_positions)
            clr = step.s.rocks[rock_ind] ? "green" : "red"
        else
            clr = "black"
        end
        return compose(ctx, ngon(0.5, 0.5, 0.1, 6), stroke("gray"), fill(clr))
    elseif is_fly_action(step.a)
        # Draw flight path
        dir = action_to_direction(step.a)
        dest_pos = step.s.pos + dir
        
        # Convert positions to coordinates
        nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
        start_pos = ((step.s.pos[1] - 0.5) / nx, (ny - step.s.pos[2] - 0.5) / ny)
        end_pos = ((dest_pos[1] - 0.5) / nx, (ny - dest_pos[2] - 0.5) / ny)
        
        sz = min(w, h)
        return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), 
                       line([start_pos, end_pos]), 
                       stroke("skyblue"), 
                       linewidth(0.02w))
    elseif step.a > N_BASIC_ACTIONS
        # Draw sensing beam
        rock_ind = step.a - N_BASIC_ACTIONS
        rock_pos = pomdp.rocks_positions[rock_ind]
        nx, ny = pomdp.map_size[1] + 1, pomdp.map_size[2] + 1
        rock_pos = ((rock_pos[1] - 0.5) / nx, (ny - rock_pos[2] - 0.5) / ny)
        rob_pos = ((step.s.pos[1] - 0.5) / nx, (ny - step.s.pos[2] - 0.5) / ny)
        sz = min(w, h)
        return compose(context((w - sz) / 2, (h - sz) / 2, sz, sz), 
                       line([rob_pos, rock_pos]), 
                       stroke("orange"), 
                       linewidth(0.01w))
    end
    return nothing
end