using Documenter, Metrics

makedocs(;
    modules=[Metrics],
    format=Documenter.HTML(),
    pages=[
        "Home" => "index.md",
    ],
    repo="https://github.com/AdarshKumar712/Metrics.jl/blob/{commit}{path}#L{line}",
    sitename="Metrics.jl",
    authors="Adarshkumar712 <Adarshkumar712.ak@gmail.com>",
    assets=String[],
)

deploydocs(;
    repo="github.com/AdarshKumar712/Metrics.jl",
)
