using Metrics
using Metrics: precision

@testset "Classification" begin
    
    yb_true = [1, 1, 0, 0]
    yb_pred = [0.9, 0.1, 0.1, 0.1]
    y_true = Metrics.onehot_encode([1, 2, 1, 3, 0], 0:3)
    y_sparse = [1, 2, 1, 3, 0]
    y_pred = [.1 .8 .05 .05; .05 .5 .35 .1; .1 .6 .2 .1; .0 .1 .6 .3; .85 .1 .0 .05]'
    weights = [0.3, 0.1, 0.4, 0.2]

    @test binary_accuracy(yb_pred, yb_true) == 0.75
    @test categorical_accuracy(y_pred, y_true) == 0.6
    @test sparse_categorical(y_pred, y_sparse) == 0.6
    
    @test top_k_categorical(y_pred, y_true; k=2) == 1.0
    @test top_k_sparse_categorical(y_pred, y_sparse; k=2) == 1.0

    @test precision(y_pred, y_true) == 0.4999999999999999
    @test precision(y_pred, y_true; avg_type="micro") == 0.4
    @test precision(y_pred, y_true; avg_type="weighted") == 0.09999999999999998
    @test precision(y_pred, y_true; avg_type="weighted", sample_weights=weights) == 0.14999999999999997


    @test recall(y_pred, y_true) == 0.33333333333333326
    @test recall(y_pred, y_true; avg_type="micro") == 0.4
    @test recall(y_pred, y_true; avg_type="weighted") == 0.06666666666666665
    @test recall(y_pred, y_true; avg_type="weighted", sample_weights=weights) == 0.08333333333333331

    @test f_beta_score(y_pred, y_true) == 0.3999999999999998
    @test f_beta_score(y_pred, y_true; β=2) == 0.4545454545454544

    @test specificity(y_pred, y_true) == 0.825
    @test specificity(y_pred, y_true; avg_type="micro") == 0.8
    @test specificity(y_pred, y_true; avg_type="weighted") == 0.19
    @test specificity(y_pred, y_true; avg_type="weighted", sample_weights=weights) == 0.22249999999999998
    
    @test cohen_kappa(y_pred, y_true) == 0.23051948051948054
end
