using SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq, Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SCCNonlinearSolve: SCCAlg
using SciMLBase: LinearProblem, NonlinearProblem, SCCNonlinearProblem, SteadyStateSolution

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

# `DynamicSS` integrates `u' = g(u)` where `g` is the concatenated block
# residual. `b - A*u` is the residual of a linear block, so `A u = b` with a
# positive-definite `A` gives attracting dynamics; the nonlinear block solves
# `x^2 + y = p[1]`, `x + y^2 = p[2]` with `p` fed by the first block's trial
# state, giving the root `(1, 2, 1, 2)` for `p = (3, 5)`.
function dynamicss_scc_problem(iip, use_vector)
    linprob = LinearProblem([3.0 1.0; 1.0 2.0], [5.0, 5.0]; u0 = [0.8, 1.8])
    f = if iip
        (res, u, p) -> (res .= [p[1] - u[1]^2 - u[2], p[2] - u[1] - u[2]^2])
    else
        (u, p) -> [p[1] - u[1]^2 - u[2], p[2] - u[1] - u[2]^2]
    end
    nlprob = NonlinearProblem(f, [0.8, 1.8], zeros(2))
    update = (p, sols) -> (
        p[1] = sols[1].u[1] + sols[1].u[2];
        p[2] = 3sols[1].u[1] + sols[1].u[2];
        nothing
    )
    probs = (linprob, nlprob)
    updates = (Returns(nothing), update)
    return SCCNonlinearProblem(
        use_vector ? collect(probs) : probs,
        use_vector ? collect(updates) : updates
    )
end

@testset "DynamicSS integrates an SCCNonlinearProblem" begin
    @testset "iip=$iip vector=$use_vector" for iip in (false, true),
            use_vector in (false, true)

        prob = dynamicss_scc_problem(iip, use_vector)
        sol = solve(prob, DynamicSS(Tsit5()); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2, 1, 2] atol = 1.0e-8
        @test sol.prob === prob
        @test sol.original isa SteadyStateSolution
        @test sol.original.original isa SciMLBase.AbstractODESolution
    end

    @testset "Non-converging residual" begin
        # `u' = 1 + u` diverges, so steady-state termination never fires.
        prob = SCCNonlinearProblem(
            (LinearProblem([-1.0;;], [1.0]; u0 = [0.0]),), (Returns(nothing),)
        )
        sol = solve(prob, DynamicSS(Tsit5(), tspan = 10.0))
        @test !successful_retcode(sol)
        @test sol.prob === prob
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

@testset "DynamicSS on a ModelingToolkit SCC decomposition" begin
    @variables a(t) b(t) x(t) [irreducible = true]
    # `DynamicSS` integrates `u' = resid(u)` with each state paired positionally
    # to a residual equation, so the steady state has to be attracting under
    # that field. Scalar blocks keep MTK's variable-to-equation pairing
    # unambiguous; a coupled block can pair crosswise and repel.
    @named model = System(
        [
            D(a) ~ 5 - 3a - b,
            D(b) ~ 5 - a - 2b,
            D(x) ~ a + b - x^3,
        ], t
    )

    reassemble_alg = ModelingToolkit.StructuralTransformations.DefaultReassembleAlgorithm(
        inline_linear_sccs = false
    )
    sssys = mtkcompile(NonlinearSystem(model); reassemble_alg)
    guesses = [a => 0.8, b => 1.8, x => 0.8]
    sccprob = SCCNonlinearProblem(sssys, guesses; combine_sccs = false)

    @test length(sccprob.probs) == 2
    @test sccprob.probs[1] isa LinearProblem
    @test sccprob.probs[2] isa NonlinearProblem

    direct = solve(sccprob, NewtonRaphson(); abstol = 1.0e-12, reltol = 1.0e-12)
    @test successful_retcode(direct)

    sol = solve(sccprob, DynamicSS(Tsit5()); abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol.u ≈ direct.u atol = 1.0e-8
    @test sol[[a, b, x]] ≈ [1, 2, cbrt(3)] atol = 1.0e-8
    @test sol.prob === sccprob
    @test sol.original isa SteadyStateSolution
    @test sol.original.original isa SciMLBase.AbstractODESolution
end

@testset "DynamicSS on a SteadyStateProblem with SCC initialization" begin
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
    sys = mtkcompile(model)
    op = [a => 0.8, b => 1.8, x => 0.8, y => 1.8, c => 2.8, d => 3.8]
    prob = SteadyStateProblem(sys, op)

    @test prob.f.initialization_data.initializeprob isa SCCNonlinearProblem

    sol = solve(prob, DynamicSS(Tsit5()); abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol[[a, b, x, y, c, d]] ≈ [1, 2, 1, 2, 3, 4] atol = 1.0e-8
end
