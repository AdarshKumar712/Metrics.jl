# Classification Metrics and Plots
# This files contains various metrics used in Classifiaction problems

# One Hot encode
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

"""
    confusion_matrix(y_pred, y_true)

Function to create a confusion_matrix for classification problems based on provided `y_pred` and `y_true`. Expects `y_true`, to be onehot_enocded already.
"""
function confusion_matrix(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    label_count = size(y_true, 1)
    ŷ = onehot_encode(onecold(y_pred), 1:label_count)
    return ŷ * transpose(y_true) 
end
  
"""
    TFPN(y_pred, y_true)

Returns `Confusion Matrix` and `True Positive`, `True Negative`, `False Positive` and `False Negative` for each class based on `y_pred` and `y_true`. Expects `y_true`, to be onehot_enocded already.  
""" 
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

"""
    binary_accuracy(y_pred, y_true; threshold=0.5)

Calculates Averaged Binary Accuracy based on `y_pred` and `y_true`. Argument `threshold` is used to specify the minimum predicted probability `y_pred` required to be labelled as `1`. Default value set as `0.5`.
""" 
function binary_accuracy(y_pred, y_true; threshold=0.5)
    @assert size(y_pred) == size(y_true)
    return sum((y_pred .>= threshold) .== y_true) / size(y_true, 1)
end

"""
    categorical_accuracy(y_pred, y_true)

Calculates Averaged Categorical Accuracy based on `y_pred` and `y_true`.
"""
function categorical_accuracy(y_pred, y_true)
    @assert size(y_pred) == size(y_true)
    return sum(onecold(y_pred) .== onecold(y_true)) / size(y_true, 2)
end

"""
    sparse_categorical(y_pred, y_true)

Calculated Sparse Categorical Accuracy based on `y_pred` and `y_true`. It evaluates the maximal true value is equal to the index of the maximal predicted value. Here, `y_true` is expected to provide only an integer as label for each data element (ie. not one hot encoded). 
"""
function sparse_categorical(y_pred, y_true)
    @assert size(y_pred, 2) == length(y_true)
    return sum(onecold(y_pred) .== y_true) / size(y_true, 1)
end

"""
    top_k_categorical(y_pred, y_true; k=3)

Evaluates if the index of true value is equal to any of the indices of top k predicted values. Default value of `k` set to `3`. 
"""
function top_k_categorical(y_pred, y_true; k=3)
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


"""
    top_k_sparse_categorical(y_pred, y_true; k=3)

Evaluates if the true value is equal to any of the indices of top k predicted values. Default value of `k` set to `3`. Similar to `sparse_categorical`, expects the `y_true` to provide only an integer as label for each data element (ie. not one hot encoded).
"""
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

"""
    precision(y_pred, y_true; avg_type="macro", sample_weights=nothing)

Computes the precision of the predictions with respect to the labels. 

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.
 
"""
function precision(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    _,TP, TN, FP, FN = TFPN(y_pred, y_true)
    # Macro-averaged Precision
    if avg_type == "macro"
        return mean(TP ./ (TP .+ FP .+ eps(eltype(TP))))
    # Micro-averaged Precision
    elseif avg_type == "micro"   
        return mean(TP) / (mean(TP) + mean(FP))
    # Weighted-Averaged Precision
    elseif avg_type == "weighted"
        weights = []
        if sample_weights != nothing
            weights = sample_weights
        else
            weights = calc_instances(y_true) / size(y_true, 2)
        end
        return mean((TP ./ (TP .+ FP .+ eps(eltype(TP)))) .* weights)
    end
end

"""
    recall(y_pred, y_true; avg_type="macro", sample_weights=nothing)

Computes the recall of the predictions with respect to the labels.

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.

Aliases: `sensitivity` and `detection_rate`
"""
function recall(y_pred, y_true; avg_type="macro", sample_weights=nothing)
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
const Sensitivity = recall
const Detection_rate = recall

"""
    f_beta_score(y_pred, y_true; β=1, avg_type="macro", sample_weights=nothing)

Compute fbeta score. The F_beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed.
 - `β=1`: the weight of precision in the combined score. If `β<1`, more weight given to `precision`, while `β>1` favors recall.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.

"""
function f_beta_score(y_pred, y_true; β=1, avg_type="macro", sample_weights=nothing)
    recall_ = recall(y_pred, y_true, avg_type=avg_type, sample_weights=sample_weights)
    precision_ = precision(y_pred, y_true, avg_type=avg_type, sample_weights=sample_weights)
    return (1 + β^2) * precision_ * recall_ / (precision_ + (β^2) * recall_ + eps(eltype(recall)))
end


"""
    specificity(y_pred, y_true; avg_type="macro", sample_weights=nothing)

Computes the specificity of the predictions with respect to the labels.

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.
"""
function specificity(y_pred, y_true; avg_type="macro", sample_weights=nothing)
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

"""
    false_alarm_rate(y_pred, y_true; avg_type="macro", sample_weights=nothing)

Computes the false_alarm_raye of the predictions with respect to the labels as `1 - specificity(y_pred, y_true, avg_type, sample_weights)`

# Arguments
 - `y_pred`: predicted values.
 - `y_true`: ground truth values on the basis of which predicted values are to be assessed.
 - `avg_type="macro"`: Type of average to be used while calculating precision of multiclass models. Can take values as `macro`, `micro` and `weighted`. Default set to `macro`.
 - `sample_weights`: Class weights to be provided when `avg_type` is set to `weighted`. Useful in case of imbalanced classes.

See also: [`specificity`](@ref)
"""
function false_alarm_rate(y_pred, y_true; avg_type="macro", sample_weights=nothing)
    return 1 - specificity(y_pred, y_true, avg_type, sample_weights)
end

"""
    cohen_kappa(y_pred, y_true)

Measures the agreement between two raters (predicted and ground truth, here) who each classify N items into C mutually exclusive categories, using the observed data to calculate the probabilities of each observer randomly seeing each category.
If the raters are in complete agreement then κ = 1. If there is no agreement among the raters other than what would be expected by chance,
κ = 0.

Ref: [Cohen's Kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa)
"""
function cohen_kappa(y_pred, y_true)
    _, tp, tn, fp, fn = TFPN(y_pred, y_true)
    mrg_a = ((tp .+ fn) .* (tp .+ fp)) ./ (tp .+ fn .+ fp .+ tn)
    mrg_b = ((fp .+ tn) .* (fn .+ tn)) ./ (tp .+ fn .+ fp .+ tn)
    expec_agree = (mrg_a .+ mrg_b) ./ (tp .+ fn .+ fp .+ tn)
    obs_agree = (tp .+ tn) ./ (tp .+ fn .+ fp .+ tn)
    cohens_kappa = mean((obs_agree .- expec_agree) ./ (1 .- expec_agree))
    return cohens_kappa
end

"""
    statsfromTFPN(TP, TN, FP, FN)

Computes statistics in case of binary classification or one-vs-all statsitics in case of multiclass classification.

# Arguments:
 - `TP`: true positive values
 - `TN`: true negative values
 - `FP`: false positive values
 - `FN`: false negative values

Return the result stats as a dictionary.
"""
function statsfromTFPN(TP, TN, FP, FN)
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

"""
    classwise_stats(y_pred, y_true)

Computes statistics for each of the class for multiclass classification based on provided `y_pred` and `y_true`.

Return the result stats as a dictionary.
"""
function Classwise_Stats(y_pred, y_true)
    _, TP, TN, FP, FN = TFPN(y_pred, y_true)
    ClasswiseStats = Dict() 
    for i in 1:size(y_true, 1)
        ClasswiseStats[i] = StatsfromTFPN(TP[i], TN[i], FP[i], FN[i])
    end
    return ClasswiseStats
end

"""
    global_stats(y_pred, y_true; avg_type="macro")

Computes the overall statistics based on provided `y_pred` and `y_true`. `avg_type` allows to specify the type of average to be used while evaluating the stats. Currently, it can take values as "macro" or "micro".

Return the result stats as a dictionary.
"""
function global_stats(y_pred, y_true; avg_type="macro")
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
    # TODO: add weighted stats option as above functions
end

# TODO
# Concordance and Discordance
    
# Receiver Operating Characterstic (ROC) Curve

# AUC






