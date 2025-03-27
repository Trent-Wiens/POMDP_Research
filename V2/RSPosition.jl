# RSPosition.jl
module RSPositionModule

export RSPosition

# Define RSPosition structure to support variable resolution
struct RSPosition
    x::Float64  # Actual continuous position
    y::Float64
    grid_x::Int  # Discretized grid position
    grid_y::Int
    resolution::Float64  # Current resolution level
end

# Constructor from discrete coordinates
function RSPosition(x::Int, y::Int, resolution::Float64=1.0)
    return RSPosition(Float64(x), Float64(y), x, y, resolution)
end

# Constructor from tuple
function RSPosition(pos::Tuple{Int,Int}, resolution::Float64=1.0)
    return RSPosition(Float64(pos[1]), Float64(pos[2]), pos[1], pos[2], resolution)
end

# Operators for compatibility with original RSPos
import Base: +, -, ==

function +(a::RSPosition, b::RSPosition)
    # Add continuous positions
    new_x = a.x + b.x
    new_y = a.y + b.y
    # Calculate new grid positions
    res = min(a.resolution, b.resolution)
    grid_x = Int(floor(new_x / res) + 1)
    grid_y = Int(floor(new_y / res) + 1)
    return RSPosition(new_x, new_y, grid_x, grid_y, res)
end

function +(a::RSPosition, b::Tuple{Int,Int})
    # Add tuple to position
    new_x = a.x + b[1]
    new_y = a.y + b[2]
    # Calculate new grid positions
    grid_x = Int(floor(new_x / a.resolution) + 1)
    grid_y = Int(floor(new_y / a.resolution) + 1)
    return RSPosition(new_x, new_y, grid_x, grid_y, a.resolution)
end

function -(a::RSPosition, b::RSPosition)
    # Subtract continuous positions
    new_x = a.x - b.x
    new_y = a.y - b.y
    # Calculate new grid positions
    res = min(a.resolution, b.resolution)
    grid_x = Int(floor(new_x / res) + 1)
    grid_y = Int(floor(new_y / res) + 1)
    return RSPosition(new_x, new_y, grid_x, grid_y, res)
end

# Equality operator
function ==(a::RSPosition, b::RSPosition)
    # Check if grid positions match
    return a.grid_x == b.grid_x && a.grid_y == b.grid_y
end

function ==(a::RSPosition, b::Tuple{Int,Int})
    # Check if grid position matches tuple
    return a.grid_x == b[1] && a.grid_y == b[2]
end

end # module