using Metrics
using Test

@testset "Metrics.jl" begin
    include("regression.jl")
    include("classification.jl")
    include("rank.jl")
    include("nlp.jl")
    include("cv.jl")
    include("utils.jl")
end
