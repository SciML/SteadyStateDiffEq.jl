using OrdinaryDiffEq
using NonlinearSolve
using SciMLBase
using SteadyStateDiffEq
using Test

function precompile_workload_rhs!(du, u, p, t)
    du[1] = 2.0 - 2.0 * u[1]
    du[2] = u[1] - 4.0 * u[2]
    return nothing
end

@testset "Precompile workload steady-state path" begin
    prob = SteadyStateProblem(precompile_workload_rhs!, zeros(2))

    @test SSRootfind() isa SSRootfind
    @test DynamicSS() isa DynamicSS
    @test SICNM() isa SICNM
    @test NonlinearProblem(prob) isa NonlinearProblem

    rootfind_sol = solve(prob, SSRootfind(NewtonRaphson()))
    @test successful_retcode(rootfind_sol.retcode)
    @test rootfind_sol.u ≈ [1.0, 0.25]

    dynamic_sol = solve(
        prob,
        DynamicSS(Tsit5());
        abstol = 1.0e-8,
        reltol = 1.0e-8
    )
    @test successful_retcode(dynamic_sol.retcode)
    @test dynamic_sol.u ≈ [1.0, 0.25] atol = 1.0e-6
end
