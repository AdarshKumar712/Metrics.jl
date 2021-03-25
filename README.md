# Metrics

<!-- [![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://AdarshKumar712.github.io/Metrics.jl/stable) -->
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://AdarshKumar712.github.io/Metrics.jl/dev)
[![Build Status](https://travis-ci.com/AdarshKumar712/Metrics.jl.svg?branch=master)](https://travis-ci.com/AdarshKumar712/Metrics.jl)

A collection of diverse metrics to analyse performance of Machine Leanring and Deep Learning Models. This includes a variety of functions for `Classification`, `Regression`, `Natural Language Processing`, `Computer Vision` and `Ranking` Models and also utilities for better user support.

# Introduction

## Installation
To install Metrics.jl, you need to fill in the following code into the Julia Prompt
``` julia
] add Metrics
```
or 
``` julia
using Pkg
Pkg.add("Metrics")
```

## Examples

``` julia
using Metrics

# get accuracy with default threshold = 0.5
acc = Metrics.binary_accuracy(y_pred, y_true)

# get complete stats including Confusion Matrix, Accuracy, Precision, Recall, F1 Score, etc. 
Metrics.report_stats(y_pred, y_true)  # where y_pred are the predicted values and y_true are onehot_encoded ground truth values
```

## More information
For more details about the package and the functions, check out the [documentation](https://adarshkumar712.github.io/Metrics.jl/stable/).
In case you have any questions, you can tag me (@Adarsh Kumar) in Julia's slack, or you can just create an issue on Github.  

## References
1. https://github.com/caseykneale/ChemometricsTools.jl
2. https://github.com/tensorflow/nmt/blob/master/nmt/scripts/bleu.py#L56
3. https://github.com/JuliaML/MLMetrics.jl
