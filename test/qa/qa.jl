using Aqua, JET, SteadyStateDiffEq, Test

@testset "Aqua" begin
    Aqua.test_all(SteadyStateDiffEq)
end

@testset "JET" begin
    JET.test_package(SteadyStateDiffEq; target_defined_modules = true)
end
