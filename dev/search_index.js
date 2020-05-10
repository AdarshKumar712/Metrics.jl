var documenterSearchIndex = {"docs":
[{"location":"rank/#Ranking-Metrics-1","page":"Ranking Metrics","title":"Ranking Metrics","text":"","category":"section"},{"location":"rank/#","page":"Ranking Metrics","title":"Ranking Metrics","text":"These functions are helpful in evaluation of ranking problem like as in case of Recommendation Systems, where you need to rank your recommendations based on the predicted values.","category":"page"},{"location":"rank/#","page":"Ranking Metrics","title":"Ranking Metrics","text":"Metrics.avg_precision\nMetrics.ranking_stats_k","category":"page"},{"location":"rank/#Metrics.avg_precision","page":"Ranking Metrics","title":"Metrics.avg_precision","text":"avg_precision(y_rec, y_rel, k = 10)\n\nEvaluates how much of the relevant documents are concentrated in the highest ranked predictions. \n\nCalculated as     ∑(Recall@i - Recall@i-1)* Precision@i for i = (1, 2, 3....k)\n\nHere, y_rec are predicted probabilities for recommendation and y_rel defines as 1 if particular result is relevant, else 0. The shape of y_rec is expected to be (1, N_elements)\n\n\n\n\n\n","category":"function"},{"location":"rank/#Metrics.ranking_stats_k","page":"Ranking Metrics","title":"Metrics.ranking_stats_k","text":"ranking_stats_k(y_rec, y_rel, k = 10)\n\nEvaluates the relevancy of top k recommendations using precison@k, recall@k and f1_score@k. Returns result as a Dict.\n\nHere, y_rec are predicted probabilities for recommendation and y_rel defines as 1 if particular result is relevant, else 0. The shape of y_rec is expected to be (1, N_elements).<br>\n\nprecison_k is evaluated as Recommended_items_that_are_relevant / Total_Recommended_items.\nrecall_l is evaluated as Recommended_items_that_are_relevant / Total_Relevant_items.\nf1_k is evaluated as 2 * Recommended_items_that_are_relevant / (Total_Recommended_items + Total_Relevant_items).\n\n\n\n\n\n","category":"function"},{"location":"nlp/#Natural-Language-Procession-Metrics-1","page":"NLP Metrics","title":"Natural Language Procession Metrics","text":"","category":"section"},{"location":"nlp/#","page":"NLP Metrics","title":"NLP Metrics","text":"Once you have trained your NLP model, you need to evaluate the performance of the model. This package provide various metrics functions with which we can evaluate and assess the accuracy of the NLP model. However, their usefulness depends on the type of NLP problem you are working on.","category":"page"},{"location":"nlp/#BLEU-Score-1","page":"NLP Metrics","title":"BLEU Score","text":"","category":"section"},{"location":"nlp/#","page":"NLP Metrics","title":"NLP Metrics","text":"Metrics.bleu_score","category":"page"},{"location":"nlp/#Metrics.bleu_score","page":"NLP Metrics","title":"Metrics.bleu_score","text":"bleu_score(reference_corpus, translation_corpus; max_order=4, smooth=false)\n\nComputes BLEU score of translated segments against one or more references. Returns the BLEU score, n-gram precisions, brevity penalty,  geometric mean of n-gram precisions, translationlength and  referencelength\n\nArguments\n\nreference_corpus: list of lists of references for each translation. Each reference should be tokenized into a list of tokens.\ntranslation_corpus: list of translations to score. Each translation should be tokenized into a list of tokens.\nmax_order: maximum n-gram order to use when computing BLEU score. \nsmooth=false: whether or not to apply. Lin et al. 2004 smoothing.\n\n\n\n\n\n","category":"function"},{"location":"nlp/#Rouge-Score-1","page":"NLP Metrics","title":"Rouge Score","text":"","category":"section"},{"location":"nlp/#","page":"NLP Metrics","title":"NLP Metrics","text":"Metrics.rouge_n\nMetrics.rouge_l_sentence_level\nMetrics.rouge_l_summary_level\nMetrics.rouge","category":"page"},{"location":"nlp/#Metrics.rouge_n","page":"NLP Metrics","title":"Metrics.rouge_n","text":"rouge_n(evaluated_sentences, reference_sentences; n=2)\n\nComputes ROUGE-N of two text collections of sentences. Returns f1, precision, recall for ROUGE-N.\n\nArguments:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentences: the sentences from the referene set\nn: size of ngram.  Defaults to 2.\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/   papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"function"},{"location":"nlp/#Metrics.rouge_l_sentence_level","page":"NLP Metrics","title":"Metrics.rouge_l_sentence_level","text":"rouge_l_sentence_level(evaluated_sentences, reference_sentences)\n\nComputes ROUGE-L (sentence level) of two text collections of sentences.\n\nCalculated according to:   Rlcs = LCS(X,Y)/m,   Plcs = LCS(X,Y)/n,   Flcs = ((1 + beta^2)*Rlcs*Plcs) / (Rlcs + (beta^2) * P_lcs)\n\nwhere:   X = reference summary   Y = Candidate summary   m = length of reference summary   n = length of candidate summary\n\nArgumnets:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentences: the sentences from the referene set\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"function"},{"location":"nlp/#Metrics.rouge_l_summary_level","page":"NLP Metrics","title":"Metrics.rouge_l_summary_level","text":"rouge_l_summary_level(evaluated_sentences, reference_sentences)\n\nComputes ROUGE-L (summary level) of two text collections of sentences.\n\nCalculated according to:   Rlcs = SUM(1, u)[LCS<union>(ri,C)]/m   Plcs = SUM(1, u)[LCS<union>(ri,C)]/n   Flcs = ((1 + beta^2)*Rlcs*Plcs) / (Rlcs + (beta^2) * P_lcs)\n\nwhere:   SUM(i,u) = SUM from i through u   u = number of sentences in reference summary   C = Candidate summary made up of v sentences   m = number of words in reference summary   n = number of words in candidate summary\n\nArguments:\n\nevaluated_sentences: the sentences that have been picked by the summarizer\nreference_sentence: the sentences in the reference summaries\n\nSource: (http://research.microsoft.com/en-us/um/people/cyl/download/papers/rouge-working-note-v1.3.1.pdf)\n\n\n\n\n\n","category":"function"},{"location":"nlp/#Metrics.rouge","page":"NLP Metrics","title":"Metrics.rouge","text":"rouge(hypotheses, references)\n\nCalculates average rouge scores for a list of hypotheses and references.\n\n\n\n\n\n","category":"function"},{"location":"nlp/#Other-Metrics-1","page":"NLP Metrics","title":"Other Metrics","text":"","category":"section"},{"location":"nlp/#","page":"NLP Metrics","title":"NLP Metrics","text":"Metrics.perplexity","category":"page"},{"location":"nlp/#Metrics.perplexity","page":"NLP Metrics","title":"Metrics.perplexity","text":"perplexity(y_pred, y_true)\n\nReturns the exponentiation of crossentropy based on y_pred and y_true.\n\n\n\n\n\n","category":"function"},{"location":"cv/#Computer-Vision-Metrics-1","page":"CV Metrics","title":"Computer Vision Metrics","text":"","category":"section"},{"location":"cv/#","page":"CV Metrics","title":"CV Metrics","text":"These Metrics are meant to evaluate the Computer Vision Algorithms / Models. Their main purpose include analysing the quality of generated images, extent of accuracy in case of Object Detection and so on.","category":"page"},{"location":"cv/#Functions-1","page":"CV Metrics","title":"Functions","text":"","category":"section"},{"location":"cv/#","page":"CV Metrics","title":"CV Metrics","text":"Metrics.PSNR\nMetrics.IoU","category":"page"},{"location":"cv/#Metrics.PSNR","page":"CV Metrics","title":"Metrics.PSNR","text":"PSNR(img1, img2)\n\nComputes peak-signal-to-noise ratio, in decibels, between two images img1 and img2. The higher the PSNR, the better the quality of the compressed, or reconstructed image.\n\n\n\n\n\n","category":"function"},{"location":"cv/#Metrics.IoU","page":"CV Metrics","title":"Metrics.IoU","text":"IoU(bb1, bb2)\n\nCalculate the Intersection over Union (IoU) of two axis-aligned bounding boxes bb1 and bb2.\n\nHere, bb1 and bb2 are provided as Dict with keys = {\"x1\", \"x2\", \"y1\", \"y2\"}, where x1, y1 are coordinates of top-left corner, and x2 and y2 are coordinates of bottom-right corner.  \n\n\n\n\n\n","category":"function"},{"location":"classification/#Classification-Metrics-1","page":"Classification Metrics","title":"Classification Metrics","text":"","category":"section"},{"location":"classification/#","page":"Classification Metrics","title":"Classification Metrics","text":"This package allows you to use a variety of Classification metrics for the performance analysis of Classification models based on the provided y_true and y_pred. The metrics that you choose to evaluate your machine learning model is very important. Choice of metrics influences how the performance of machine learning algorithms is measured and compared. For most of these function, it is expected that the provided ","category":"page"},{"location":"classification/#Functions-1","page":"Classification Metrics","title":"Functions","text":"","category":"section"},{"location":"classification/#","page":"Classification Metrics","title":"Classification Metrics","text":"Metrics.binary_accuracy\nMetrics.categorical_accuracy\nMetrics.cohen_kappa\nMetrics.confusion_matrix\nMetrics.f_beta_score\nMetrics.false_alarm_rate\nMetrics.precision\nMetrics.recall\nMetrics.sparse_categorical\nMetrics.specificity\nMetrics.top_k_categorical\nMetrics.top_k_sparse_categorical","category":"page"},{"location":"classification/#Metrics.binary_accuracy","page":"Classification Metrics","title":"Metrics.binary_accuracy","text":"binary_accuracy(y_pred, y_true; threshold=0.5)\n\nCalculates Averaged Binary Accuracy based on y_pred and y_true. Argument threshold is used to specify the minimum predicted probability y_pred required to be labelled as 1. Default value set as 0.5.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.categorical_accuracy","page":"Classification Metrics","title":"Metrics.categorical_accuracy","text":"categorical_accuracy(y_pred, y_true)\n\nCalculates Averaged Categorical Accuracy based on y_pred and y_true.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.cohen_kappa","page":"Classification Metrics","title":"Metrics.cohen_kappa","text":"cohen_kappa(y_pred, y_true)\n\nMeasures the agreement between two raters (predicted and ground truth, here) who each classify N items into C mutually exclusive categories, using the observed data to calculate the probabilities of each observer randomly seeing each category. If the raters are in complete agreement then κ = 1. If there is no agreement among the raters other than what would be expected by chance, κ = 0.\n\nRef: Cohen's Kappa\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.confusion_matrix","page":"Classification Metrics","title":"Metrics.confusion_matrix","text":"confusion_matrix(y_pred, y_true)\n\nFunction to create a confusionmatrix for classification problems based on provided `ypredandytrue. Expectsytrue`, to be onehot_enocded already.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.f_beta_score","page":"Classification Metrics","title":"Metrics.f_beta_score","text":"f_beta_score(y_pred, y_true; β=1, avg_type=\"macro\", sample_weights=nothing)\n\nCompute fbeta score. The F_beta score is the weighted harmonic mean of precision and recall, reaching its optimal value at 1 and its worst value at 0.\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\nβ=1: the weight of precision in the combined score. If β<1, more weight given to precision, while β>1 favors recall.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.false_alarm_rate","page":"Classification Metrics","title":"Metrics.false_alarm_rate","text":"false_alarm_rate(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the falsealarmraye of the predictions with respect to the labels as 1 - specificity(y_pred, y_true, avg_type, sample_weights)\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\nSee also: specificity\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.precision","page":"Classification Metrics","title":"Metrics.precision","text":"precision(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the precision of the predictions with respect to the labels. \n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.recall","page":"Classification Metrics","title":"Metrics.recall","text":"recall(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the recall of the predictions with respect to the labels.\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\nAliases: sensitivity and detection_rate\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.sparse_categorical","page":"Classification Metrics","title":"Metrics.sparse_categorical","text":"sparse_categorical(y_pred, y_true)\n\nCalculated Sparse Categorical Accuracy based on y_pred and y_true. It evaluates the maximal true value is equal to the index of the maximal predicted value. Here, y_true is expected to provide only an integer as label for each data element (ie. not one hot encoded). \n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.specificity","page":"Classification Metrics","title":"Metrics.specificity","text":"specificity(y_pred, y_true; avg_type=\"macro\", sample_weights=nothing)\n\nComputes the specificity of the predictions with respect to the labels.\n\nArguments\n\ny_pred: predicted values.\ny_true: ground truth values on the basis of which predicted values are to be assessed.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.top_k_categorical","page":"Classification Metrics","title":"Metrics.top_k_categorical","text":"top_k_categorical(y_pred, y_true; k=3)\n\nEvaluates if the index of true value is equal to any of the indices of top k predicted values. Default value of k set to 3. \n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.top_k_sparse_categorical","page":"Classification Metrics","title":"Metrics.top_k_sparse_categorical","text":"top_k_sparse_categorical(y_pred, y_true; k=3)\n\nEvaluates if the true value is equal to any of the indices of top k predicted values. Default value of k set to 3. Similar to sparse_categorical, expects the y_true to provide only an integer as label for each data element (ie. not one hot encoded).\n\n\n\n\n\n","category":"function"},{"location":"classification/#Combined-Stats-1","page":"Classification Metrics","title":"Combined Stats","text":"","category":"section"},{"location":"classification/#","page":"Classification Metrics","title":"Classification Metrics","text":"There are some functions that return you the overall analysis of the model performance within a single function. They are:","category":"page"},{"location":"classification/#","page":"Classification Metrics","title":"Classification Metrics","text":"Metrics.statsfromTFPN\nMetrics.classwise_stats\nMetrics.global_stats","category":"page"},{"location":"classification/#Metrics.statsfromTFPN","page":"Classification Metrics","title":"Metrics.statsfromTFPN","text":"statsfromTFPN(TP, TN, FP, FN)\n\nComputes statistics in case of binary classification or one-vs-all statsitics in case of multiclass classification.\n\nArguments:\n\nTP: true positive values\nTN: true negative values\nFP: false positive values\nFN: false negative values\n\nReturn the result stats as a dictionary.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.classwise_stats","page":"Classification Metrics","title":"Metrics.classwise_stats","text":"classwise_stats(y_pred, y_true)\n\nComputes statistics for each of the class for multiclass classification based on provided y_pred and y_true.\n\nReturn the result stats as a dictionary.\n\n\n\n\n\n","category":"function"},{"location":"classification/#Metrics.global_stats","page":"Classification Metrics","title":"Metrics.global_stats","text":"global_stats(y_pred, y_true; avg_type=\"macro\")\n\nComputes the overall statistics based on provided y_pred and y_true. avg_type allows to specify the type of average to be used while evaluating the stats. Currently, it can take values as \"macro\" or \"micro\".\n\nReturn the result stats as a dictionary.\n\n\n\n\n\n","category":"function"},{"location":"regression/#Regression-Metrics-1","page":"Regression Metrics","title":"Regression Metrics","text":"","category":"section"},{"location":"regression/#","page":"Regression Metrics","title":"Regression Metrics","text":"These functions are useful for the analysis of regression type of models, that deal with the continuous data. Metrics package provides the following metrics for the evaluation of regression models based on provided y_true and y_pred:","category":"page"},{"location":"regression/#","page":"Regression Metrics","title":"Regression Metrics","text":"Metrics.mae\nMetrics.mse\nMetrics.male\nMetrics.msle\nMetrics.r2_score\nMetrics.adjusted_r2_score","category":"page"},{"location":"regression/#Metrics.mae","page":"Regression Metrics","title":"Metrics.mae","text":"mae(y_pred, y_true)\n\nMean Absolute Error. Calculated as sum(|y_true .- y_pred|) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"function"},{"location":"regression/#Metrics.mse","page":"Regression Metrics","title":"Metrics.mse","text":"mse(y_pred, y_true)\n\nMean Squared Error. Calculated as sum((y_true .- y_pred).^2) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"function"},{"location":"regression/#Metrics.male","page":"Regression Metrics","title":"Metrics.male","text":"male(y_pred, y_true)\n\nMean Absolute Logarithmic Error. Calculated as sum(|log.(y_true) .- log.(y_pred)|) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"function"},{"location":"regression/#Metrics.msle","page":"Regression Metrics","title":"Metrics.msle","text":"msle(y_pred, y_true)\n\nMean Absolute Logarithmic Error. Calculated as sum((log.(y_true) .- log.(y_pred)).^2) / length(y_true) based on provided y_pred and y_true.\n\n\n\n\n\n","category":"function"},{"location":"regression/#Metrics.r2_score","page":"Regression Metrics","title":"Metrics.r2_score","text":"r2_score(y_pred, y_true)\n\nCalculates the r2 (Coefficient of Determination) score for the provided y_pred and y_true. Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). A constant model that always predicts the expected value of y, disregarding the input features, would get a r2_score of 0.0.\n\n\n\n\n\n","category":"function"},{"location":"regression/#Metrics.adjusted_r2_score","page":"Regression Metrics","title":"Metrics.adjusted_r2_score","text":"adjusted_r2_score(y_pred, y_true, n)\n\nModified version of r2_score that has been adjusted for the number of predictors in the model. Here the argument n is for the number of predictors(or independent variables in X). \n\nSee also: r2_score\n\n\n\n\n\n","category":"function"},{"location":"#Metrics.jl-1","page":"Home","title":"Metrics.jl","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"This package is a collection of diverse Metrics to analyse the performance of variety of Machine Learning and Deep Learning Models. These include Regression Models, NLP Models, CV Models, Recommendation Models and further utilities to support better user interface.","category":"page"},{"location":"#Installation-1","page":"Home","title":"Installation","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"To install Metrics.jl, you need to fill in the following code into the Julia Prompt:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"] add Metrics","category":"page"},{"location":"#Basic-Usage-1","page":"Home","title":"Basic Usage","text":"","category":"section"},{"location":"#","page":"Home","title":"Home","text":"Suppose you are working on a Classification Problem. Once you have your model ready, and you would like to evaluate your model's performance.  One of the most obvious way would be to use the magnitude of loss function as evaluation metrics. However, a better way to accomplish this could be to use Metrics.jl, using the following commands:","category":"page"},{"location":"#","page":"Home","title":"Home","text":"using Metrics\n\nMetrics.report_stats(y_pred, y_true)  # where y_pred are the predicted values and y_true are onehot_encoded ground truth values.\n","category":"page"},{"location":"#","page":"Home","title":"Home","text":"This will print the performance statistics of the model, based on the provided y_pred and y_true values. This statistics include the Confusion Matrix, Accuracy, Precision, Recall, F1 Score and much more. Where Metrics.jl provide you option to get complete evaluation of the model using multiple statistics functions within a single function, it also provide you option to use these statistics functions individually as per your choice. Their usage, you can find further in this documentation.","category":"page"},{"location":"utils/#Utilities-1","page":"Utils","title":"Utilities","text":"","category":"section"},{"location":"utils/#","page":"Utils","title":"Utils","text":"These utility functions are meant to provide easy comprehended overall evaluation of Models and Algorithms. For now, the support is limited to Classification Models, but extended further in future.","category":"page"},{"location":"utils/#","page":"Utils","title":"Utils","text":"Metrics.report_stats","category":"page"},{"location":"utils/#Metrics.report_stats","page":"Utils","title":"Metrics.report_stats","text":"report_stats(y_pred, y_true; classwise_stats=true, avg_type=\"macro\", sample_weights=nothing)\n\nA utility function that prints the statistics summary of the model based on provided y_pred and y_true. \n\nArguments:\n\ny_pred: predicted values  \ny_true: ground truth values on the basis of which predicted values are to be assessed.\nclasswise_stats=true: if set true, prints classwise stats along with global stats.\navg_type=\"macro\": Type of average to be used while calculating precision of multiclass models. Can take values as macro, micro and weighted. Default set to macro.\nsample_weights: Class weights to be provided when avg_type is set to weighted. Useful in case of imbalanced classes.\n\n\n\n\n\n","category":"function"}]
}