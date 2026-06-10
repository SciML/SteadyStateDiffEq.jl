using Pkg
using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

function activate_qa_env()
    Pkg.activate("qa")
    Pkg.develop(PackageSpec(path = dirname(@__DIR__)))
    return Pkg.instantiate()
end

@time begin
    if GROUP == "All" || GROUP == "Core"
        @time @safetestset "Core Tests" begin
            include("core.jl")
        end
        @time @safetestset "Autodiff Tests" begin
            include("autodiff.jl")
        end
    end

    if GROUP == "QA"
        activate_qa_env()
        @time @safetestset "Quality Assurance" begin
            include("qa/qa.jl")
        end
    end
end
