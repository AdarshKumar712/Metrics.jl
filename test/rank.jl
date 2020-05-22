using Metrics

@testset "Ranking and Statistical" begin

    y_rec = [0.85, 0.44, 0.71, 0.23, 0.90, 0.63, 0.16, 0.27]
    y_rel = [1, 0, 0, 1, 1, 1, 0, 0]
    output_rank_stats = Dict("precision_k" => 0.5, "recall_k" => 0.75, "f1_k" => 0.6)

    @testset "rank_stats_k" begin
        rank_out = ranking_stats_k(y_rec, y_rel; k=6)
        for i in rank_out.keys
            @test rank_out[i] == output_rank_stats[i]
        end
    end

    @test avg_precision(y_rec, y_rel; k=6) == 0.6875
end
