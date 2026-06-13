using SafeTestsets, Test
using SciMLTesting

run_tests(;
    core = () -> begin
        @time begin
            @time @safetestset "Core Tests" begin
                include("core.jl")
            end
            @time @safetestset "Autodiff Tests" begin
                include("autodiff.jl")
            end
        end
    end,
    groups = Dict(
        # Declared `env` => activates test/qa and runs only for GROUP="QA", never
        # under "All" (matching the original `if GROUP == "QA"` branch).
        "QA" => (;
            env = joinpath(@__DIR__, "qa"),
            body = () -> begin
                @time @safetestset "Quality Assurance" begin
                    include("qa/qa.jl")
                end
            end,
        ),
    ),
)
