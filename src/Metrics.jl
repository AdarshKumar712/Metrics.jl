module Metrics
# Module for DL Metrics
using StatsBase, DataFrames
using Flux: Crossentropy

include("./CV_Metrics/CVMetrics.jl")
include("./NLP_Metrics/NLPMetrics.jl")
inlcude("Classification.jl")
include("Regression.jl")
include("Ranking_n_Statistical.jl")
include("utils.jl")

end
