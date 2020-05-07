# Ranking based Metrics for Recommender Systems

"""
    ranking_stats_k(y_rec, y_rel, k = 10)

Evaluates the relevancy of top k recommendations using `precison@k`, `recall@k` and `f1_score@k`. Returns result as a `Dict`.

Here, `y_rec` are predicted probabilities for recommendation and `y_rel` defines as `1` if particular result is relevant, else `0`.
The shape of `y_rec` is expected to be (1, N_elements).<br>

`precison_k` is evaluated as `Recommended_items_that_are_relevant / Total_Recommended_items`
`recall_l` is evaluated as `Recommended_items_that_are_relevant / Total_Relevant_items`
`f1_k` is evaluated as `2 * Recommended_items_that_are_relevant / (Total_Recommended_items + Total_Relevant_items)`
"""
function ranking_stats_k(y_rec, y_rel, k = 10)
    @assert size(y_rec) == size(y_rel)
    if size(y_rec,1) !=1
       error("Input should be of the shape (1, N), N as Number of elements")
    end
    top_k = partialsortperm(y_rec[1,:], 1:k, lt= >)
    y_rec_k, y_rel_k = y_rec[:, top_k], y_rel[:, top_k]
    tp = sum(y_rel_k .== 1)
    total_positive = sum(y_rel.==1)
    precision_k = k!=0 ? tp / k : 1
    recall_k = total_positive!=0 ? tp / total_positive : 1
    f1_k = (total_positive + k!=0) ? (2 * tp / (total_positive + k)) : 1 
    return Dict("precision_k" => precision_k,
                "recall_k" => recall_k,
                "f1_k" => f1_k)
end

"""
    avg_precision(y_rec, y_rel, k = 10)

Evaluates how much of the relevant documents are concentrated in the highest ranked predictions. 

Calculated as
    ∑(Recall@i - Recall@i-1)* Precision@i for i = (1, 2, 3....k)

Here, `y_rec` are predicted probabilities for recommendation and `y_rel` defines as `1` if particular result is relevant, else `0`.
The shape of `y_rec` is expected to be (1, N_elements)
"""
function avg_precision(y_rec, y_rel, k = 10)
    @assert size(y_rec) == size(y_rel)
    if size(y_rec,1) !=1
       error("Input should be of the shape (1, N), N as Number of elements")
    end
    top_k = partialsortperm(y_rec[1,:], 1:k, lt= >)
    y_rec_k, y_rel_k = y_rec[:, top_k], y_rel[:, top_k]
    correct_prediction = 0
    running_sum = 0
    for i in 1:k
        if y_rel_k[i] == 1
            correct_prediction +=1
            running_sum += correct_prediction / i
        end
    end
    return running_sum / sum(y_rel .== 1)
end

# TODO

# DCG
# NDCG


