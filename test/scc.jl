using SteadyStateDiffEq, NonlinearSolve, Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SCCNonlinearSolve: SCCAlg
using SciMLBase: LinearProblem, NonlinearProblem, SCCNonlinearProblem

function coupled_scc_problem(iip, use_vector)
    f = if iip
        (res, u, p) -> (res .= [u[1]^2 + u[2] - 3, u[1] + u[2]^2 - 5])
    else
        (u, p) -> [u[1]^2 + u[2] - 3, u[1] + u[2]^2 - 5]
    end
    firstprob = NonlinearProblem(f, [0.8, 1.8])
    secondprob = NonlinearProblem((u, p) -> [u[1]^2 - sum(p)], [1.5], zeros(2))
    probs = (firstprob, secondprob)
    updates = (Returns(nothing), (p, sols) -> (p .= sols[1].u))
    return SCCNonlinearProblem(
        use_vector ? collect(probs) : probs,
        use_vector ? collect(updates) : updates
    )
end

@testset "SSRootfind forwards SCCNonlinearProblem" begin
    @testset "iip=$iip vector=$use_vector alg=$alg" for iip in (false, true),
            use_vector in (false, true),
            alg in (
                SSRootfind(), SSRootfind(NewtonRaphson()),
                SSRootfind(SCCAlg()), SSRootfind(SCCAlg(; nlalg = NewtonRaphson())),
            )

        prob = coupled_scc_problem(iip, use_vector)
        sol = solve(prob, alg; abstol = 1.0e-12, reltol = 1.0e-12)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2, sqrt(3)] atol = 1.0e-9
        @test maximum(abs, sol.resid) < 1.0e-9
        @test sol.prob === prob
        @test sol.original.prob isa SCCNonlinearProblem
        @test sol.alg.alg === sol.original.alg
        @test sol.u === sol.original.u
        @test sol.resid === sol.original.resid
        @test sol.retcode === sol.original.retcode
        @test sol.stats === sol.original.stats
        @test sol.left === sol.original.left
        @test sol.right === sol.original.right
    end

    @testset "Failure propagation" begin
        prob = SCCNonlinearProblem(
            (NonlinearProblem((u, p) -> u .^ 2 .- 2, [10.0]),),
            (Returns(nothing),)
        )
        sol = solve(prob, SSRootfind(NewtonRaphson()); maxiters = 1, abstol = 1.0e-12)
        @test !successful_retcode(sol)
        @test sol.retcode === sol.original.retcode
    end

    @testset "Ordinary steady-state wrapping" for alg in (
            SSRootfind(), SSRootfind(NewtonRaphson()),
        )
        prob = SteadyStateProblem((u, p, t) -> 1 .- u, [0.0])
        sol = solve(prob, alg)
        @test successful_retcode(sol)
        @test sol.u ≈ [1.0]
        @test sol.prob === prob
        @test sol.alg.alg === sol.original.alg
        @test sol.stats === sol.original.stats
    end
end

@testset "SSRootfind on a ModelingToolkit SCC decomposition" begin
    @variables a(t) b(t) x(t) y(t) c(t) d(t) [irreducible = true]
    @named model = System(
        [
            D(a) ~ 5 - 3a - b,
            D(b) ~ 5 - a - 2b,
            D(x) ~ a + b - x^2 - y,
            D(y) ~ 3a + b - x - y^2,
            D(c) ~ x + y + 7 - 2c - d,
            D(d) ~ 2x + y + 11 - c - 3d,
        ], t
    )

    # Keep the linear blocks so both SCC block kinds are exercised.
    reassemble_alg = ModelingToolkit.StructuralTransformations.DefaultReassembleAlgorithm(
        inline_linear_sccs = false
    )
    sssys = mtkcompile(NonlinearSystem(model); reassemble_alg)
    guesses = [a => 0.8, b => 1.8, x => 0.8, y => 1.8, c => 2.8, d => 3.8]
    sccprob = SCCNonlinearProblem(sssys, guesses; combine_sccs = false)

    @test length(sccprob.probs) == 3
    @test sccprob.probs[1] isa LinearProblem
    @test sccprob.probs[2] isa NonlinearProblem
    @test sccprob.probs[3] isa LinearProblem

    states = [a, b, x, y, c, d]
    expected = [1, 2, 1, 2, 3, 4]

    direct = solve(sccprob, NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
    @test successful_retcode(direct)
    @test direct[states] ≈ expected atol = 1.0e-9

    @testset "alg=$alg" for alg in (
            SSRootfind(), SSRootfind(NewtonRaphson()),
            SSRootfind(SCCAlg()), SSRootfind(SCCAlg(; nlalg = NewtonRaphson())),
        )
        wrapped = solve(sccprob, alg; abstol = 1.0e-12, reltol = 1.0e-12)
        @test successful_retcode(wrapped)
        @test wrapped[states] ≈ expected atol = 1.0e-9
        @test maximum(abs, wrapped.resid) < 1.0e-9
        @test wrapped.prob === sccprob
        @test wrapped.original.prob isa SCCNonlinearProblem
        @test wrapped.u ≈ direct.u atol = 1.0e-9
    end
end
