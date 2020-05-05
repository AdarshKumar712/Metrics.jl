# Regression Metrics

"""
    mae(y_pred, y_true)

Mean Absolute Error. Calculated as `sum(|y_true .- y_pred|) / length(y_true)` based on provided `y_pred` and `y_true`.
"""
function mae(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum(abs.(y_true .- y_pred)) / length(y_true)
end

"""
    mse(y_pred, y_true)

Mean Squared Error. Calculated as `sum((y_true .- y_pred).^2) / length(y_true)` based on provided `y_pred` and `y_true`.
"""
function mse(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum((y_true .- y_pred).^2) / length(y_true)
end

"""
    male(y_pred, y_true)

Mean Absolute Logarithmic Error. Calculated as `sum(|log.(y_true) .- log.(y_pred)|) / length(y_true)` based on provided `y_pred` and `y_true`.
"""
function male(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum(abs.(log.(y_true) .- log.(y_pred)))  / length(y_true)
end

"""
    msle(y_pred, y_true)

Mean Absolute Logarithmic Error.
Calculated as `sum((log.(y_true) .- log.(y_pred)).^2) / length(y_true)` based on provided `y_pred` and `y_true`.
"""
function msle(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum((log.(y_true) .- log.(y_pred)).^2)  / length(y_true)
end

"""
    r2_score(y_pred, y_true)

Calculates the r2 (Coefficient of Determination) score for the provided `y_pred` and `y_true`.
Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse).
A constant model that always predicts the expected value of y, disregarding the input features, would get a r2_score of `0.0`.
"""
function r2_score(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    mean = sum(y_true) / length(y_true)
    ss_total = sum((y_true .- mean).^2)
    return 1 - ss_res/(ss_total + eps(eltype(y_pred)))
end

"""
    adjusted_r2_score(y_pred, y_true, n)

Modified version of `r2_score` that has been adjusted for the number of predictors in the model. Here the argument `n` is for the number of predictors(or independent variables in X). 

See also: [`r2_score`](@ref)
"""
function adjusted_r2_score(y_pred, y_true, n)		# n -> number of predictors(independent variables in X)
    @assert length(y_true) == length(y_pred)
    score = r2_score(y_pred, y_true)
    return 1 - ((1 - score) * (length(y_true) -1)) / ( length(y_true) - n -1)
end

