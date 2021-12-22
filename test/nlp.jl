using Metrics
using DataStructures: OrderedDict

@testset "NLP" begin
    
    ref_corpus = [["Example of bleu score"], ["This is an apple"]]
    translated_corpus = ["Example to bleu score", "This no a apple"]
    
    res = bleu_score(ref_corpus, translated_corpus) .≈ (0.7253666236200925, [0.9444444444444444, 0.7941176470588235, 0.6875, 0.6], 0.9726044771163485, 0.7457981540149954, 36, 37)
    @test all(res)
    
    hypothesis = ["Example for bleu score", "This cz an apple"]
    ref_corpus = ["Example of bleu score", "This is an apple"]
    output = OrderedDict(
      "rouge_1 / f_score"=> 0.75,
      "rouge_1 / r_score"=> 0.75,
      "rouge_1 / p_score"=> 0.75,
      "rouge_2 / f_score"=> 0.3333333333328333,
      "rouge_2 / r_score"=> 0.3333333333333333,
      "rouge_2 / p_score"=> 0.3333333333333333,
      "rouge_l / f_score"=> 0.75,
      "rouge_l / r_score"=> 0.75,
      "rouge_l / p_score"=> 0.75)
    
    @testset "rouge" begin
        rouge_out = rouge(hypothesis, ref_corpus)
        for key in keys(output)
              @test rouge_out[key] ≈ output[key]
        end

        @test rouge_l_summary_level(hypothesis, ref_corpus) == (0.2499999999995, 0.25, 0.25)
    end
end
