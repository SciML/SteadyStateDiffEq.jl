using Pkg
using SafeTestsets, Test

const GROUP = get(ENV, "GROUP", "All")

function activate_nopre_env()
    Pkg.activate("nopre")
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

    if GROUP == "nopre" && isempty(VERSION.prerelease)
        activate_nopre_env()
        @time @safetestset "JET Static Analysis" begin
            include("nopre/jet_tests.jl")
        end
    end
end
