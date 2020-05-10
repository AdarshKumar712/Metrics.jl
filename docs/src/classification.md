# Classification Metrics

This package allows you to use a variety of Classification metrics for the performance analysis of Classification models based on the provided `y_true` and `y_pred`. The metrics that you choose to evaluate your machine learning model is very important. Choice of metrics influences how the performance of machine learning algorithms is measured and compared. For most of these function, it is expected that the provided 

## Functions 
 
```@docs
Metrics.binary_accuracy
Metrics.categorical_accuracy
Metrics.cohen_kappa
Metrics.confusion_matrix
Metrics.f_beta_score
Metrics.false_alarm_rate
Metrics.precision
Metrics.recall
Metrics.sparse_categorical
Metrics.specificity
Metrics.top_k_categorical
Metrics.top_k_sparse_categorical
```

## Combined Stats

There are some functions that return you the overall analysis of the model performance within a single function. They are:

```@docs
Metrics.statsfromTFPN
Metrics.classwise_stats
Metrics.global_stats
```
