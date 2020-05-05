using Flux: crossentropy
# NLP Metrics
# NLP_Metrics to support advanced analysis of NLP Models 

"""
    perplexity(y_pred, y_true)

Returns the exponentiation of `crossentropy` based on `y_pred` and `y_true`.
"""
function Perplexity(y_pred, y_true)
    return exp(crossentropy(y_pred, y_true))
end


