using Flux: crossentropy
# NLP Metrics
# NLP_Metrics to support advanced analysis of NLP Models 

# Perplexity
function Perplexity(y_pred, y_true)
    return exp(crossentropy(y_pred, y_true))
end




