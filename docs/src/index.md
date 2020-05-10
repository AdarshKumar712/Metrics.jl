# Metrics.jl

This package is a collection of diverse Metrics to analyse the performance of variety of Machine Learning and Deep Learning Models. These include `Regression Models`, `NLP Models`, `CV Models`, `Recommendation Models` and further utilities to support better user interface.

## Installation

To install `Metrics.jl`, you need to fill in the following code into the Julia Prompt:
```
] add Metrics
```

## Basic Usage

Suppose you are working on a Classification Problem. Once you have your model ready, and you would like to evaluate your model's performance. 
One of the most obvious way would be to use the `magnitude` of loss function as evaluation metrics. However, a better way to accomplish this could be to use `Metrics.jl`, using the following commands:

```
using Metrics

Metrics.report_stats(y\_pred, y_true)  # where y\_pred are the predicted values and y\_true are onehot_encoded ground truth values.

```

This will print the performance statistics of the model, based on the provided y\_pred and y\_true values. This statistics include the `Confusion Matrix`, `Accuracy`, `Precision`, `Recall`, `F1 Score` and much more. Where `Metrics.jl` provide you option to get complete evaluation of the model using multiple statistics functions within a single function, it also provide you option to use these statistics functions individually as per your choice. Their usage, you can find further in this documentation.
