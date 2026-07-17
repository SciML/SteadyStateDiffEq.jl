using SteadyStateDiffEq, OrdinaryDiffEq, NonlinearSolve, Test
using LinearAlgebra: norm
using NonlinearSolve.NonlinearSolveBase: NormTerminationMode, AbsNormTerminationMode,
    RelNormSafeTerminationMode, AbsNormSafeBestTerminationMode

# Note: Rodas3d (OrdinaryDiffEqRosenbrock) is the ODE solver SICNM was constructed
# around; any stiffly accurate mass-matrix-capable Rosenbrock method works, and these
# tests use Rodas5P since it is re-exported by the main OrdinaryDiffEq package.

@testset "SICNM SteadyStateProblem iip/oop" begin
    f_iip = (du, u, p, t) -> begin
        du[1] = 2 - 2u[1]
        du[2] = u[1] - 4u[2]
        nothing
    end
    f_oop = (u, p, t) -> [2 - 2u[1], u[1] - 4u[2]]

    for (f, u0) in ((f_iip, zeros(2)), (f_oop, zeros(2)))
        prob = SteadyStateProblem(f, u0)
        sol = solve(prob, SICNM(Rodas5P()))
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.u ≈ [1.0, 0.25] atol = 1.0e-6
        @test norm(sol.resid, Inf) < 1.0e-8
    end
end

@testset "SICNM NonlinearProblem" begin
    g_oop = (u, p) -> [u[1]^2 + p[1] * u[2] - 2, u[2]^3 - u[1] + 1]
    prob = NonlinearProblem(g_oop, [1.0, 1.0], [1.0])
    sol = solve(prob, SICNM(Rodas5P()))
    @test SciMLBase.successful_retcode(sol.retcode)
    @test norm(sol.resid, Inf) < 1.0e-8

    g_iip = (res, u, p) -> begin
        res[1] = u[1]^2 + p[1] * u[2] - 2
        res[2] = u[2]^3 - u[1] + 1
        nothing
    end
    prob = NonlinearProblem(g_iip, [1.0, 1.0], [1.0])
    sol = solve(prob, SICNM(Rodas5P()))
    @test SciMLBase.successful_retcode(sol.retcode)
    @test norm(sol.resid, Inf) < 1.0e-8
end

@testset "SICNM robustness where Newton diverges" begin
    # Newton's method diverges for atan from |u0| ≳ 1.4 (the iterates blow up
    # until the Jacobian underflows to singular); the continuous Newton flow
    # converges from any starting point
    g = (u, p) -> atan.(u)
    prob = NonlinearProblem(g, [3.0])
    sol = solve(prob, SICNM(Rodas5P()))
    @test SciMLBase.successful_retcode(sol.retcode)
    @test abs(sol.u[1]) < 1.0e-8

    # Powell's badly scaled problem
    g2 = (u, p) -> [1.0e4 * u[1] * u[2] - 1, exp(-u[1]) + exp(-u[2]) - 1.0001]
    prob2 = NonlinearProblem(g2, [0.0, 1.0])
    sol2 = solve(prob2, SICNM(Rodas5P()))
    @test SciMLBase.successful_retcode(sol2.retcode)
    @test norm(g2(sol2.u, nothing), Inf) < 1.0e-8
end

@testset "SICNM termination modes and tspan failure" begin
    f = (u, p, t) -> [2 - 2u[1], u[1] - 4u[2]]
    prob = SteadyStateProblem(f, zeros(2))

    for termination_condition in (
            NormTerminationMode(SteadyStateDiffEq.infnorm),
            AbsNormTerminationMode(SteadyStateDiffEq.infnorm),
            RelNormSafeTerminationMode(SteadyStateDiffEq.infnorm),
            AbsNormSafeBestTerminationMode(SteadyStateDiffEq.infnorm),
        )
        sol = solve(prob, SICNM(Rodas5P()); termination_condition)
        @test SciMLBase.successful_retcode(sol.retcode)
        @test sol.u ≈ [1.0, 0.25] atol = 1.0e-6
    end

    # too-short tspan must report failure
    sol = solve(prob, SICNM(Rodas5P(); tspan = 1.0e-4))
    @test !SciMLBase.successful_retcode(sol.retcode)
end

@testset "SICNM Float32 and save_idxs" begin
    f = (u, p, t) -> [2 - 2u[1], u[1] - 4u[2]]
    prob = SteadyStateProblem(f, zeros(Float32, 2))
    sol = solve(prob, SICNM(Rodas5P()); abstol = 1.0f-5)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test eltype(sol.u) == Float32

    prob64 = SteadyStateProblem(f, zeros(2))
    sol_scalar = solve(prob64, SICNM(Rodas5P()); save_idxs = 1)
    @test sol_scalar[1] ≈ 1.0 atol = 1.0e-6
end
