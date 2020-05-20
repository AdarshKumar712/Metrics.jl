using Metrics

@testset "Regression" begin
    y_true = [1, 1, 0, 0, 1]
    y_pred = [0.9, 0.9, 0.1, 0.2, 0.88]

    @test mae(y_pred, y_true) == 0.12399999999999997
    @test mse(y_pred, y_true) == 0.01688
    @test male(y_pred, y_true) == 13.70276763512634
    @test msle(y_pred, y_true) == 464.8426856611348
    
    @test r2_score(y_pred, y_true) == 0.9296666666666666
    @test adjusted_r2_score(y_pred, y_true, 3) == 0.7186666666666666
end
