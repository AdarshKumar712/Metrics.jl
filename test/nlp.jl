using Metrics
using DataStructures: OrderedDict

@testset "NLP/BLEU" begin
    max_order = 4
    # test token-based ngrams
    ngrams = Metrics.get_ngrams(split("it is a dog "), max_order)
    actual_orders = Set(length.(keys(ngrams)))
    
    @test length(intersect(actual_orders, 1:max_order)) == max_order
    @test length(setdiff(actual_orders, 1:max_order)) == 0

    # test character-based ngrams
    ngrams = Metrics.get_ngrams("it is a dog ", max_order)
    actual_orders = Set(length.(keys(ngrams)))

    @test length(intersect(actual_orders, 1:max_order)) == max_order
    @test length(setdiff(actual_orders, 1:max_order)) == 0

    # NLTK sample https://www.nltk.org/api/nltk.translate.bleu_score.html
    reference1 = [
        "It", "is", "a", "guide", "to", "action", "that",
        "ensures", "that", "the", "military", "will", "forever",
        "heed", "Party", "commands"
    ]
    reference2 = [
        "It", "is", "the", "guiding", "principle", "which",
        "guarantees", "the", "military", "forces", "always",
        "being", "under", "the", "command", "of", "the",
        "Party"
    ]
    reference3 = [
        "It", "is", "the", "practical", "guide", "for", "the",
        "army", "always", "to", "heed", "the", "directions",
        "of", "the", "party"
    ]

    hypothesis1 = [
        "It", "is", "a", "guide", "to", "action", "which",
        "ensures", "that", "the", "military", "always",
        "obeys", "the", "commands", "of", "the", "party"
    ]

    score = bleu_score([[reference1, reference2, reference3]], [hypothesis1])
    @test isapprox(score.bleu, 0.5045, atol=1e-4) #(NLTK)

    ref_corpus = [["Example of bleu score"], ["This is an apple"]]
    translated_corpus = ["Example to bleu score", "This no a apple"]

    res = bleu_score(ref_corpus, translated_corpus)
    @test collect(res) ≈ 
      [0.7253666236200925, [0.9444444444444444, 0.7941176470588235, 0.6875, 0.6], 0.9726044771163485, 0.7457981540149954, 36, 37]
end

@testset "NLP/ROUGE" begin
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
