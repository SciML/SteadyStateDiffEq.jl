import SciMLBase
using SciMLBase: AbstractSteadyStateProblem, ReturnCode, SteadyStateProblem, build_solution,
    solve
using SteadyStateDiffEq: SteadyStateDiffEqAlgorithm
using Test

struct GenericSteadyStateAlgorithm <: SteadyStateDiffEqAlgorithm end

function SciMLBase.__solve(
        prob::AbstractSteadyStateProblem,
        ::GenericSteadyStateAlgorithm,
        args...;
        kwargs...
    )
    u = copy(prob.u0)
    return build_solution(
        prob, GenericSteadyStateAlgorithm(), u, zero.(u);
        retcode = ReturnCode.Success
    )
end

@testset "SteadyStateDiffEqAlgorithm generic extension contract" begin
    prob = SteadyStateProblem((u, p, t) -> u, [1.0, -1.0])
    sol = solve(prob, GenericSteadyStateAlgorithm())

    @test sol.prob === prob
    @test sol.alg isa GenericSteadyStateAlgorithm
    @test sol.u == prob.u0
    @test sol.resid == zero.(prob.u0)
    @test sol.retcode == ReturnCode.Success
end
