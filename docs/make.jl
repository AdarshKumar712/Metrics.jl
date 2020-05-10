using Documenter, Metrics

makedocs(;
    modules=[Metrics],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
        "Regression Metrics" => "regression.md",
        "Classification Metrics" => "classification.md",
        "NLP Metrics" => "nlp.md",
        "CV Metrics" => "cv.md",
        "Ranking Metrics" => "rank.md",
        "Utils" => "utils.md"
    ],
    repo="https://github.com/AdarshKumar712/Metrics.jl/blob/{commit}{path}#L{line}",
    sitename="Metrics.jl",
    authors="Adarsh Kumar <Adarshkumar712.ak@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/AdarshKumar712/Metrics.jl",
    target = "build",
    push_preview = true
)
