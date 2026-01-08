using SafeTestsets, Test

@testset "SteadyStateDiffEq.jl" begin
    @safetestset "Core Tests" begin
        include("core.jl")
    end
    @safetestset "Autodiff Tests" begin
        include("autodiff.jl")
    end
    @safetestset "JET Static Analysis" begin
        include("jet.jl")
    end
end
