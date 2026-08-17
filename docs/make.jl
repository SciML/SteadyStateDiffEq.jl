using Documenter, SteadyStateDiffEq

DocMeta.setdocmeta!(
    SteadyStateDiffEq, :DocTestSetup,
    :(using SciMLBase: SteadyStateProblem, solve; using SteadyStateDiffEq);
    recursive = true
)

makedocs(
    sitename = "SteadyStateDiffEq.jl",
    authors = "Chris Rackauckas",
    clean = true,
    format = Documenter.HTML(
        canonical = "https://docs.sciml.ai/SteadyStateDiffEq/stable/"
    ),
    pages = [
        "Home" => "index.md",
        "Developer API" => "developer_api.md",
    ]
)

deploydocs(
    repo = "github.com/SciML/SteadyStateDiffEq.jl";
    push_preview = true
)
