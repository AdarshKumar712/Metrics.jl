module Metrics
# Module for DL Metrics

export mae, mse, msle, male, r2_score, adjusted_r2_score

export binary_accuracy, confusion_matrix, categorical_accuracy, sparse_categorical, top_k_categorical, top_k_sparse_categorical, precision, recall, sensitivity, detection_rate, f_beta_score, specificity, false_alarm_rate, cohen_kappa, statsfromTFPN, classwise_stats, global_stats   

export IoU, PSNR

export bleu_score, rouge, rouge_l_summary_level 

export ranking_stats_k, avg_precision

export report_stats

using StatsBase, DataFrames
using DataStructures: OrderedDict

include("./CV_Metrics/CVMetrics.jl")
include("./NLP_Metrics/NLPMetrics.jl")
include("Classification.jl")
include("Regression.jl")
include("Ranking_n_Statistical.jl")
include("utils.jl")

end
