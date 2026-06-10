using Aqua, JET, SteadyStateDiffEq, Test

@testset "Aqua" begin
    Aqua.test_all(SteadyStateDiffEq; deps_compat = false)
    @test_broken false  # Aqua deps_compat: missing [compat] for stdlib LinearAlgebra — see https://github.com/SciML/SteadyStateDiffEq.jl/issues/134
end

@testset "JET" begin
    JET.test_package(SteadyStateDiffEq; target_defined_modules = true)
end
