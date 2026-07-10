using Documenter, SteadyStateDiffEq

DocMeta.setdocmeta!(
    SteadyStateDiffEq, :DocTestSetup, :(using SteadyStateDiffEq); recursive = true
)

makedocs(
    sitename = "SteadyStateDiffEq.jl",
    authors = "Chris Rackauckas",
    clean = true,
    doctest = false,
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/SteadyStateDiffEq/stable/"
    ),
    pages = [
        "Home" => "index.md",
    ]
)

deploydocs(
    repo = "github.com/SciML/SteadyStateDiffEq.jl";
    push_preview = true
)
