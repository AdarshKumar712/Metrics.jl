# Util functions
using DataFrames
include("./Classification.jl")

function report_stats(y_pred, y_true; classwise_stats=true, avg_type="macro", sample_weights=nothing)
    g_stats = Global_Stats(y_pred, y_true, avg_type = avg_type)
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
    print("++++++++++++++++++++ Global Stats ++++++++++++++++++++\n")
    print("  Accuracy: ", g_stats["Accuracy"],"\n")
    print("  Precision: ", g_stats["Precision"],"\n")
    print("  Recall: ", g_stats["Recall"],"\n")
    print("  F1_score: ", g_stats["F1_score"],"\n")
    print("  Specificity: ", g_stats["Specificity"],"\n")
    print("  False alarm rate: ", g_stats["False_alarm_rate"],"\n")
    print("\n")
    if (classwise_stats == true)
        c_stats = Classwise_Stats(y_pred, y_true)
        stats = DataFrame(Class=[], Confusion_Matrix=[], Accuracy=[], Precision=[], Recall=[], F1_score=[], Specificity=[])
        for i in 1: length(c_stats)
            push!(stats, Dict(:Class=>i,c_stats[i]...))
        end
        print(stats)
    end
end
