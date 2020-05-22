using Metrics
using Test
using Random

Random.seed!(0)

@testset "Metrics.jl" begin
    include("regression.jl")
    include("classification.jl")
    include("rank.jl")
    include("nlp.jl")
    include("cv.jl")
end
