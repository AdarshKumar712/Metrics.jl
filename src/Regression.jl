# Regression Metrics

# MAE
function mae(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum(abs.(y_true .- y_pred)) / length(y_true)
end

# MSE_
function mse(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum((y_true .- y_pred).^2) / length(y_true)
end

# Mean Absolute Logarithmc Error
function male(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum(abs.(log.(y_true) .- log.(y_pred)))  / length(y_true)
end

# Mean Squared Logarithmic Error
function msle(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    return sum((log.(y_true) .- log.(y_pred)).^2)  / length(y_true)
end

# R_squared
function r2_score(y_pred, y_true)
    @assert length(y_true) == length(y_pred)
    ss_res = sum((y_true .- y_pred).^2)
    mean = sum(y_true) / length(y_true)
    ss_total = sum((y_true .- mean).^2)
    return 1 - ss_res/(ss_total + eps(eltype(y_pred)))
end

# Adjusted R_squared
function adjusted_r2_score(y_pred, y_true, n)		# n -> number of predictors(independent variables in X)
    @assert length(y_true) == length(y_pred)
    score = r2_score(y_pred, y_true)
    return 1 - ((1 - score) * (length(y_true) -1)) / ( length(y_true) - n -1)
end


