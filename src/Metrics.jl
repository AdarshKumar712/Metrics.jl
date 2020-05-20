module Metrics
# Module for DL Metrics

export mae, mse, msle, male, r2_score, adjusted_r2_score

using StatsBase, DataFrames
using Flux: crossentropy
using DataStructures: OrderedDict

include("./CV_Metrics/CVMetrics.jl")
include("./NLP_Metrics/NLPMetrics.jl")
include("Classification.jl")
include("Regression.jl")
include("Ranking_n_Statistical.jl")
include("utils.jl")

end
