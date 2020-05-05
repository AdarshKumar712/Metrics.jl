# Util functions
using DataFrames
include("./Classification.jl")

"""
    report_stats(y_pred, y_true; classwise_stats=true, avg_type="macro", sample_weights=nothing)

A utility function that prints the statistics summary of the model based on provided `y_pred` and `y_true`. 

# Arguments:
 - `y_pred`: predicted values  
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed.
 - `classwise_stats=true`: if set `true`, prints classwise stats along with global stats.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.

"""
function report_stats(y_pred, y_true; classwise_stats=true, avg_type="macro", sample_weights=nothing)
    g_stats = global_stats(y_pred, y_true, avg_type = avg_type)
    print("++++++++++++++++++++ Confusion Matrix +++++++++++++++++++\n")
    cm = g_stats["Confusion_Matrix"]
    for i in 1:size(cm,1)
        print(" ")
        for j in 1:size(cm,2)
            print(cm[i,j]," ")
        end
        print("\n")
    end
    print("\n")
    print("++++++++++++++++++++ Global Statistics ++++++++++++++++++++\n")
    print("  Accuracy: ", g_stats["Accuracy"],"\n")
    print("  Precision: ", g_stats["Precision"],"\n")
    print("  Recall: ", g_stats["Recall"],"\n")
    print("  F1_score: ", g_stats["F1_score"],"\n")
    print("  Specificity: ", g_stats["Specificity"],"\n")
    print("  False alarm rate: ", g_stats["False_alarm_rate"],"\n")
    print("\n")
    if (classwise_stats == true)
        c_stats = classwise_stats(y_pred, y_true)
        stats = DataFrame(Class=[], Confusion_Matrix=[], Accuracy=[], Precision=[], Recall=[], F1_score=[], Specificity=[])
        for i in 1: length(c_stats)
            push!(stats, Dict(:Class=>i,c_stats[i]...))
        end
        print(stats)
    end
end

# TODO report_stats_neat() - a function to print the Statistics summary more neatly

# TODO stats_to_csv() - function to save stats to a csv file.
