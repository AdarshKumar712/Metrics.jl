var documenterSearchIndex = {"docs":
[{"location":"#Metrics.jl-1","page":"Home","title":"Metrics.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"","category":"page"},{"location":"#","page":"Home","title":"Home","text":"Modules = [Metrics]","category":"page"},{"location":"#Metrics.Classwise_Stats-Tuple{Any,Any}","page":"Home","title":"Metrics.Classwise_Stats","text":"classwise_stats(y_pred, y_true)\n\nComputes statistics for each of the class for multiclass classification based on provided y_pred and y_true.\n\nReturn the result stats as a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.Perplexity-Tuple{Any,Any}","page":"Home","title":"Metrics.Perplexity","text":"perplexity(y_pred, y_true)\n\nReturns the exponentiation of crossentropy based on y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.TFPN-Tuple{Any,Any}","page":"Home","title":"Metrics.TFPN","text":"TFPN(y_pred, y_true)\n\nReturns Confusion Matrix and True Positive, True Negative, False Positive and False Negative for each class based on y_pred and y_true. Expects y_true, to be onehot_enocded already.  \n\n\n\n\n\n","category":"method"},{"location":"#Metrics._f_p_r_lcs-Tuple{Any,Any,Any}","page":"Home","title":"Metrics._f_p_r_lcs","text":"_f_p_r_lcs(llcs, m, n)\n\nComputes the LCS-based F-measure score\n\nArguments:\n\nllcs: Length of LCS\nm: number of words in reference summary\nn: number of words in candidate summary\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._get_ngrams-Tuple{Any,Any}","page":"Home","title":"Metrics._get_ngrams","text":"_get_ngrams(n, text)\n\nCalcualtes n-grams. Returns a set of n-grams.\n\nArguments:\n\nn: provide which n-grams to calculate\ntext: An array of tokens\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._get_word_ngrams-Tuple{Any,Any}","page":"Home","title":"Metrics._get_word_ngrams","text":"_get_word_ngrams(n, sentences)\n\nCalculates word n-grams for multiple sentences.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._lcs-Tuple{Any,Any}","page":"Home","title":"Metrics._lcs","text":"_lcs(x, y)\n\nUtility function to compute the length of the longest common subsequence (lcs) between two strings. The implementation below uses a DP programming algorithm and runs in O(nm) time where n = len(x) and m = len(y).\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._len_lcs-Tuple{Any,Any}","page":"Home","title":"Metrics._len_lcs","text":"_len_lcs(x, y)\n\nReturns the length of the Longest Common Subsequence between sequences x and y.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._recon_lcs-Tuple{Any,Any}","page":"Home","title":"Metrics._recon_lcs","text":"_recons_lcs(x, y)\n\nReturns the Longest Subsequence between x and y.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._split_into_words-Tuple{Any}","page":"Home","title":"Metrics._split_into_words","text":"_split_into_words(sentences)\n\nSplits multiple sentences into words and flattens the result\n\n\n\n\n\n","category":"method"},{"location":"#Metrics._union_lcs-Tuple{Any,Any}","page":"Home","title":"Metrics._union_lcs","text":"_union_lcs(evaluated_sentences, reference_sentence)\n\nReturns LCSu(ri, C) which is the LCS score of the union longest common subsequence between reference sentence ri and candidate summary C.\n\nArguments:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentence: one of the sentences in the reference summaries\n\nFor example, if ri= w1 w2 w3 w4 w5, and C contains two sentences: c1 = w1 w2 w6 w7 w8 and c2 = w1 w3 w8 w9 w5, then the longest common subsequence of ri and c1 is “w1 w2” and the longest common subsequence of ri and c2 is “w1 w3 w5”. The union longest common subsequence of ri, c1, and c2 is “w1 w2 w3 w5” and LCSu(ri, C) = 4/5.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.adjusted_r2_score-Tuple{Any,Any,Any}","page":"Home","title":"Metrics.adjusted_r2_score","text":"adjusted_r2_score(y_pred, y_true, n)\n\nModified version of r2_score that has been adjusted for the number of predictors in the model. Here the argument n is for the number of predictors(or independent variables in X). \n\nSee also: r2_score\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.avg_precision","page":"Home","title":"Metrics.avg_precision","text":"avg_precision(y_rec, y_rel, k = 10)\n\nEvaluates how much of the relevant documents are concentrated in the highest ranked predictions. \n\nCalculated as     ∑(Recall@i - Recall@i-1)* Precision@i for i = (1, 2, 3....k)\n\nHere, y_rec are predicted probabilities for recommendation and y_rel defines as 1 if particular result is relevant, else 0. The shape of y_rec is expected to be (1, N_elements)\n\n\n\n\n\n","category":"function"},{"location":"#Metrics.binary_accuracy-Tuple{Any,Any}","page":"Home","title":"Metrics.binary_accuracy","text":"binary_accuracy(y_pred, y_true; threshold=0.5)\n\nCalculates Averaged Binary Accuracy based on y_pred and y_true. Argument threshold is used to specify the minimum predicted probability y_pred required to be labelled as 1. Default value set as 0.5.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.bleu_score-Tuple{Any,Any}","page":"Home","title":"Metrics.bleu_score","text":"bleu_score(reference_corpus, translation_corpus; max_order=4, smooth=false)\n\nComputes BLEU score of translated segments against one or more references. Returns the BLEU score, n-gram precisions, brevity penalty,  geometric mean of n-gram precisions, translationlength and  referencelength\n\nArguments\n\nreference_corpus: list of lists of references for each translation. Each reference should be tokenized into a list of tokens.\ntranslation_corpus: list of translations to score. Each translation should be tokenized into a list of tokens.\nmax_order: maximum n-gram order to use when computing BLEU score. \nsmooth=false: whether or not to apply. Lin et al. 2004 smoothing.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.categorical_accuracy-Tuple{Any,Any}","page":"Home","title":"Metrics.categorical_accuracy","text":"categorical_accuracy(y_pred, y_true)\n\nCalculates Averaged Categorical Accuracy based on y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.cohen_kappa-Tuple{Any,Any}","page":"Home","title":"Metrics.cohen_kappa","text":"cohen_kappa(y_pred, y_true)\n\nMeasures the agreement between two raters (predicted and ground truth, here) who each classify N items into C mutually exclusive categories, using the observed data to calculate the probabilities of each observer randomly seeing each category. If the raters are in complete agreement then κ = 1. If there is no agreement among the raters other than what would be expected by chance, κ = 0.\n\nRef: Cohen's Kappa\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.confusion_matrix-Tuple{Any,Any}","page":"Home","title":"Metrics.confusion_matrix","text":"confusion_matrix(y_pred, y_true)\n\nFunction to create a confusionmatrix for classification problems based on provided `ypredandytrue. Expectsytrue`, to be onehot_enocded already.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.f_beta_score-Tuple{Any,Any}","page":"Home","title":"Metrics.f_beta_score","text":"f_beta_score(y_pred, y_true; β=1, avg_type=\"macro\", sample_weights=nothing)\n\nCompute fbeta score. The F_beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\nβ=1: the weight of precision in the combined score. If β<1, more weight given to precision, while β>1 favors recall.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.false_alarm_rate-Tuple{Any,Any}","page":"Home","title":"Metrics.false_alarm_rate","text":"false_alarm_rate(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the falsealarmraye of the predictions with respect to the labels as 1 - specificity(y_pred, y_true, avg_type, sample_weights)\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\nSee also: specificity\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.get_ngrams-Tuple{Any,Any}","page":"Home","title":"Metrics.get_ngrams","text":"get_ngrams(segment, max_order)\n\nExtracts all n-grams upto a given maximum order from an input segment. Returns the counter containing all n-grams upto max_order in segment with a count of how many times each n-gram occurred.\n\nArguments\n\nsegment: text segment from which n-grams will be extracted.\nmax_order: maximum length in tokens of the n-grams returned by this methods.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.global_stats-Tuple{Any,Any}","page":"Home","title":"Metrics.global_stats","text":"global_stats(y_pred, y_true; avg_type=\"macro\")\n\nComputes the overall statistics based on provided y_pred and y_true. avg_type allows to specify the type of average to be used while evaluating the stats. Currently, it can take values as \"macro\" or \"micro\".\n\nReturn the result stats as a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.mae-Tuple{Any,Any}","page":"Home","title":"Metrics.mae","text":"mae(y_pred, y_true)\n\nMean Absolute Error. Calculated as sum(|y_true .- y_pred|) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.male-Tuple{Any,Any}","page":"Home","title":"Metrics.male","text":"male(y_pred, y_true)\n\nMean Absolute Logarithmic Error. Calculated as sum(|log.(y_true) .- log.(y_pred)|) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.mse-Tuple{Any,Any}","page":"Home","title":"Metrics.mse","text":"mse(y_pred, y_true)\n\nMean Squared Error. Calculated as sum((y_true .- y_pred).^2) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.msle-Tuple{Any,Any}","page":"Home","title":"Metrics.msle","text":"msle(y_pred, y_true)\n\nMean Absolute Logarithmic Error. Calculated as sum((log.(y_true) .- log.(y_pred)).^2) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.precision-Tuple{Any,Any}","page":"Home","title":"Metrics.precision","text":"precision(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the precision of the predictions with respect to the labels. \n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.r2_score-Tuple{Any,Any}","page":"Home","title":"Metrics.r2_score","text":"r2_score(y_pred, y_true)\n\nCalculates the r2 (Coefficient of Determination) score for the provided y_pred and y_true. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a r2_score of 0.0.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.ranking_stats_k","page":"Home","title":"Metrics.ranking_stats_k","text":"ranking_stats_k(y_rec, y_rel, k = 10)\n\nEvaluates the relevancy of top k recommendations using precison@k, recall@k and f1_score@k. Returns result as a Dict.\n\nHere, y_rec are predicted probabilities for recommendation and y_rel defines as 1 if particular result is relevant, else 0. The shape of y_rec is expected to be (1, N_elements).<br>\n\nprecison_k is evaluated as Recommended_items_that_are_relevant / Total_Recommended_items recall_l is evaluated as Recommended_items_that_are_relevant / Total_Relevant_items f1_k is evaluated as 2 * Recommended_items_that_are_relevant / (Total_Recommended_items + Total_Relevant_items)\n\n\n\n\n\n","category":"function"},{"location":"#Metrics.recall-Tuple{Any,Any}","page":"Home","title":"Metrics.recall","text":"recall(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the recall of the predictions with respect to the labels.\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\nAliases: sensitivity and detection_rate\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.report_stats-Tuple{Any,Any}","page":"Home","title":"Metrics.report_stats","text":"report_stats(y_pred, y_true; classwise_stats=true, avg_type=\"macro\", sample_weights=nothing)\n\nA utility function that prints the statistics summary of the model based on provided y_pred and y_true. \n\nArguments:\n\ny_pred: predicted values  \ny_true: ground truth values on the basis of which predicted values are to be assessed.\nclasswise_stats=true: if set true, prints classwise stats along with global stats.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.rouge-Tuple{Any,Any}","page":"Home","title":"Metrics.rouge","text":"rouge(hypotheses, references)\n\nCalculates average rouge scores for a list of hypotheses and references.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.rouge_l_sentence_level-Tuple{Any,Any}","page":"Home","title":"Metrics.rouge_l_sentence_level","text":"rouge_l_sentence_level(evaluated_sentences, reference_sentences)\n\nComputes ROUGE-L (sentence level) of two text collections of sentences.\n\nCalculated according to:   Rlcs = LCS(X,Y)/m,   Plcs = LCS(X,Y)/n,   Flcs = ((1 + beta^2)*Rlcs*Plcs) / (Rlcs + (beta^2) * P_lcs)\n\nwhere:   X = reference summary   Y = Candidate summary   m = length of reference summary   n = length of candidate summary\n\nArgumnets:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentences: the sentences from the referene set\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.rouge_l_summary_level-Tuple{Any,Any}","page":"Home","title":"Metrics.rouge_l_summary_level","text":"rouge_l_summary_level(evaluated_sentences, reference_sentences)\n\nComputes ROUGE-L (summary level) of two text collections of sentences.\n\nCalculated according to:   Rlcs = SUM(1, u)[LCS<union>(ri,C)]/m   Plcs = SUM(1, u)[LCS<union>(ri,C)]/n   Flcs = ((1 + beta^2)*Rlcs*Plcs) / (Rlcs + (beta^2) * P_lcs)\n\nwhere:   SUM(i,u) = SUM from i through u   u = number of sentences in reference summary   C = Candidate summary made up of v sentences   m = number of words in reference summary   n = number of words in candidate summary\n\nArguments:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentence: the sentences in the reference summaries\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.rouge_n-Tuple{Any,Any}","page":"Home","title":"Metrics.rouge_n","text":"rouge_n(evaluated_sentences, reference_sentences; n=2)\n\nComputes ROUGE-N of two text collections of sentences. Returns f1, precision, recall for ROUGE-N.\n\nArguments:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentences: the sentences from the referene set\nn: size of ngram.  Defaults to 2.\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/   papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.sparse_categorical-Tuple{Any,Any}","page":"Home","title":"Metrics.sparse_categorical","text":"sparse_categorical(y_pred, y_true)\n\nCalculated Sparse Categorical Accuracy based on y_pred and y_true. It evaluates the maximal true value is equal to the index of the maximal predicted value. Here, y_true is expected to provide only an integer as label for each data element (ie. not one hot encoded). \n\n\n\n\n\n","category":"method"},{"location":"#Metrics.specificity-Tuple{Any,Any}","page":"Home","title":"Metrics.specificity","text":"specificity(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the specificity of the predictions with respect to the labels.\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.statsfromTFPN-NTuple{4,Any}","page":"Home","title":"Metrics.statsfromTFPN","text":"statsfromTFPN(TP, TN, FP, FN)\n\nComputes statistics in case of binary classification or one-vs-all statsitics in case of multiclass classification.\n\nArguments:\n\nTP: true positive values\nTN: true negative values\nFP: false positive values\nFN: false negative values\n\nReturn the result stats as a dictionary.\n\n\n\n\n\n","category":"method"},{"location":"#Metrics.top_k_categorical-Tuple{Any,Any}","page":"Home","title":"Metrics.top_k_categorical","text":"top_k_categorical(y_pred, y_true; k=3)\n\nEvaluates if the index of true value is equal to any of the indices of top k predicted values. Default value of k set to 3. \n\n\n\n\n\n","category":"method"},{"location":"#Metrics.top_k_sparse_categorical-Tuple{Any,Any}","page":"Home","title":"Metrics.top_k_sparse_categorical","text":"top_k_sparse_categorical(y_pred, y_true; k=3)\n\nEvaluates if the true value is equal to any of the indices of top k predicted values. Default value of k set to 3. Similar to sparse_categorical, expects the y_true to provide only an integer as label for each data element (ie. not one hot encoded).\n\n\n\n\n\n","category":"method"}]
}
