using Plots
using StatsBase: mean

# Classification Metrics and Plots
# This files contains various metrics used in Classifiaction problems

# Onehot_encode
function onehot_encode(y, labels)
   onehot_arr = zeros(length(labels), length(y))
   for i in 1:length(y)
       onehot_arr[Int(y[i]), i] = 1
   end
   onehot_arr
end

# Onecold
function onecold(y)
    onecold_arr = zeros(size(y,2))
    argmax_vec = argmax(y, dims = 1)
    for i in 1:size(y,2)
        onecold_arr[i] = argmax_vec[i].I[1]
    end
    return onecold_arr
end

# Confusion Matrix
function confusion_matrix(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    label_count = size(y_true, 1)
    ŷ = onehot_encode(onecold(y_pred), 1:label_count)
    return ŷ * transpose(y_true) 
end
  
# TFPN  
function TFPN(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    label_count = size(y_true, 1)
    TP = zeros(label_count) ; TN = zeros(label_count)
    FP = zeros(label_count) ; FN = zeros(label_count)
    ConfusionMatrix = confusion_matrix(y_pred, y_true)
    for c in 1 : label_count
        TP[c] = ConfusionMatrix[c,c]
        FP[c] = sum(ConfusionMatrix[:,c]) - TP[c]
        FN[c] = sum(ConfusionMatrix[c,:]) - TP[c]
        TN[c] = sum(ConfusionMatrix) - TP[c] - FP[c] - FN[c]
    end
    return ConfusionMatrix,TP, TN, FP, FN
end

# Averaged Binary Accuracy 
function binary_accuracy(y_pred, y_true; threshold = 0.5)
    @assert size(y_pred) == size(y_true)
    return sum((y_pred .>= threshold) .== y_true) / size(y_true, 1)
end

# Average Categorical Accuracy
function categorical_accuracy(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    return sum(onecold(y_pred) .== onecold(y_true)) / size(y_true, 2)
end

# Average Sparse Categorical Accuracy
function sparse_categorical(y_pred, y_true)
    @assert size(y_pred, 2) == length(y_true)
    return sum(onecold(y_pred) .== y_true) / size(y_true, 1)
end

# Top-k Categorical
function top_k_categorical(y_pred, y_true, k=3)
    @assert size(y_pred) == size(y_true)
    count = 0
    sparse_y = onecold(y_true)
    for i in 1:size(y_true, 2)
        top_k = partialsortperm(y_pred[:,i], 1:k, lt= >)
        for j in 1:k
            if top_k[j] == sparse_y[i]
                count+=1
                break
            end
        end
    end
    return count / length(sparse_y) 
end

# Top-k-sparse_categorical
function top_k_sparse_categorical(y_pred, y_true; k=3)
    count = 0
    for i in 1:length(y_true)
        top_k = partialsortperm(y_pred[:,i], 1:k, lt= >)
        for j in 1:k
            if top_k[j] == y_true[i]
                count+=1
                break
            end
        end
    end
    return count / length(y_true)
end

# Calculate Instances for each class in y
function calc_instances(y)
   instances = zeros(size(y, 1))
   sparse_y = onecold(y)
   for i in 1:length(instances)
       instances[i] = sum(sparse_y .== i)
   end
   return instances
end


# Precision
function Precision(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    _,TP, TN, FP, FN = TFPN(y_pred, y_true)
    # Macro-averaged Precision
    if avg_type == "macro"
        return mean(TP ./ (TP .+ FP .+ eps(eltype(TP))))
    # Micro-averaged Precision
    elseif avg_type == "micro"   
        return mean(TP) / (mean(TP) + mean(FP))
    # Weighted-Averaged Precision
    else
        weights = []
        if sample_weights != nothing
            weights = sample_weights
        else
            weights = calc_instances(y_true) / size(y_true, 2)
        end
        return mean((TP ./ (TP .+ FP .+ eps(eltype(TP)))) .* weights)
    end
end

# Recall
function Recall(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    _,TP, TN, FP, FN = TFPN(y_pred, y_true)
    # Macro-averaged Precision
    if avg_type == "macro"
        return mean(TP ./ (TP .+ FN .+ eps(eltype(TP))))    
    # Micro-averaged Precision
    elseif avg_type == "micro"   
        return mean(TP) / (mean(TP) + mean(FN))
    # Weighted-Averaged Precision
    else
        weights = []
        if sample_weights != nothing
            weights = sample_weights
        else
            weights = calc_instances(y_true) / size(y_true, 2)
        end
        return mean((TP ./ (TP .+ FN .+ eps(eltype(TP)))) .* weights)
    end
end

# F_beta score
function F_beta_score(y_pred, y_true; β=1, avg_type="macro", sample_weights=nothing)
    recall_ = Recall(y_pred, y_true, avg_type=avg_type, sample_weights=sample_weights)
    precision_ = Precision(y_pred, y_true, avg_type=avg_type, sample_weights=sample_weights)
    return (1 + β^2) * precision_ * recall_ / (precision_ + (β^2) * recall_ + eps(eltype(recall)))
end

# Sensitivity
const Sensitivity = Recall
# Detection Rate
const Detection_rate = Recall

# Specificity
function Specificity(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    _, TP, TN, FP, FN = TFPN(y_pred, y_true)
    # Macro-averaged Precision
    if avg_type == "macro"
        return mean(TN ./ (TN .+ FP .+ eps(eltype(TP))))    
    # Micro-averaged Precision
    elseif avg_type == "micro"   
        return mean(TN) / (mean(TN) + mean(FP))
    # Weighted-Averaged Precision
    else
        weights = []
        if sample_weights != nothing
            weights = sample_weights
        else
            weights = calc_instances(y_true) / size(y_true, 2)
        end
        return mean((TN ./ (TN .+ FP .+ eps(eltype(TP)))) .* weights)
    end
end

# False Alarm Rate
function False_alarm_rate(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    return 1 - Specificity(y_pred, y_true, avg_type, sample_weights)
end

# Cohen's Kappa
function Cohen_Kappa(y_pred, y_true)
    _, tp, tn, fp, fn = TFPN(y_pred, y_true)
    mrg_a = ((tp .+ fn) .* (tp .+ fp)) ./ (tp .+ fn .+ fp .+ tn)
    mrg_b = ((fp .+ tn) .* (fn .+ tn)) ./ (tp .+ fn .+ fp .+ tn)
    expec_agree = (mrg_a .+ mrg_b) ./ (tp .+ fn .+ fp .+ tn)
    obs_agree = (tp .+ tn) ./ (tp .+ fn .+ fp .+ tn)
    cohens_kappa = mean((obs_agree .- expec_agree) ./ (1 .- expec_agree))
    return cohens_kappa
end

# StatsfromTFPN
function StatsfromTFPN(TP, TN, FP, FN)
    Confusion_Matrix = reshape([TP, FP, FN, TN], 2, 2)
    Precision = TP / (TP + FP + eps(eltype(TP)))
    Recall = TP / (TP + FN + eps(eltype(TP)))
    Specificity = TN / (TN + FP + eps(eltype(TP)))
    F1_score = 2 * (Precision * Recall) / (Precision + Recall + eps(eltype(TP)))
    Accuracy = (TP + TN) / (TP + TN + FP + FN)
    False_alarm_rate = 1 - Specificity
    return Dict(:Confusion_Matrix => Confusion_Matrix,
                :Precision => Precision, :Recall => Recall,
                :Specificity => Specificity, :F1_score => F1_score,
                :Accuracy => Accuracy)
end

# Classwise Stats
function Classwise_Stats(y_pred, y_true)
    _, TP, TN, FP, FN = TFPN(y_pred, y_true)
    ClasswiseStats = Dict() 
    for i in 1:size(y_true, 1)
        ClasswiseStats[i] = StatsfromTFPN(TP[i], TN[i], FP[i], FN[i])
    end
    return ClasswiseStats
end

# Global Stats
# TODO: add weighted stats option as above functions
function Global_Stats(y_pred, y_true; avg_type="macro")
    confusion_matrix_, TP, TN, FP, FN = TFPN(y_pred, y_true) 
    if avg_type == "macro"
        precision = mean(TP ./ (TP .+ FP .+ eps(eltype(TP))))
        recall = mean(TP ./ (TP .+ FN .+ eps(eltype(TP))))
        f1_score = 2 * precision * recall / (precision + recall + eps(eltype(TP)))
        specificity = mean(TN ./ (TN .+ FP .+ eps(eltype(TP))))
        accuracy = categorical_accuracy(y_pred, y_true)
        false_alarm_rate = 1 - specificity
        return Dict("Confusion_Matrix" => confusion_matrix_,
                "Precision" => precision, "Recall" => recall,
                "Specificity" => specificity, "F1_score" => f1_score,
                "Accuracy" => accuracy, "False_alarm_rate" => false_alarm_rate)
    elseif avg_type == "micro"
        precision = mean(TP) / (mean(TP) + mean(FP))
        recall = mean(TP) / (mean(TP) + mean(FN))
        f1_score = 2 * precision * recall / (precision + recall + eps(eltype(TP)))
        specificity = mean(TN)/ (mean(TN) + mean(FP))
        accuracy = categorical_accuracy(y_pred, y_true)
        false_alarm_rate = 1 - specificity
        return Dict("Confusion_Matrix" => confusion_matrix_,
                "Precision" => precision, "Recall" => recall,
                "Specificity" => specificity, "F1_score" => f1_score,
                "Accuracy" => accuracy, "False_alarm_rate" => false_alarm_rate)
    end
end

# TODO
# Concordance and Discordance
    
# Receiver Operating Characterstic (ROC) Curve

# AUC






