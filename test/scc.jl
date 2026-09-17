using SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq, Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SCCNonlinearSolve: SCCAlg
using SciMLBase: HomotopyProblem, LinearProblem, NonlinearProblem, SCCNonlinearProblem,
    SteadyStateSolution

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

# `DynamicSS` solves the blocks sequentially: the `LinearProblem` block is
# solved directly (`A u = b` gives `a = 1, b = 2`), then its solution updates the
# nonlinear block's parameters through `explicitfuns!` to `p = (3, 5)`, and the
# block residual `x' = 3 - x^2 - y`, `y' = 5 - x - y^2` is integrated to the
# root `(1, 2)`.
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

@testset "DynamicSS sequentially solves an SCCNonlinearProblem" begin
    @testset "iip=$iip vector=$use_vector" for iip in (false, true),
            use_vector in (false, true)

        prob = dynamicss_scc_problem(iip, use_vector)
        sol = solve(prob, DynamicSS(Tsit5()); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2, 1, 2] atol = 1.0e-8
        @test sol.prob === prob
        # `original` is the per-block solutions: a direct linear solve and a
        # `DynamicSS` steady-state solve.
        @test sol.original isa Tuple{SciMLBase.LinearSolution, NonlinearSolution}
        @test sol.original[2].original isa SteadyStateSolution
        @test sol.original[2].original.original isa SciMLBase.AbstractODESolution
    end

    @testset "HomotopyProblem block" for iip in (false, true)
        # The target system is `u' = 2 - u^2` at `λ = λspan[2]` (`1 - u^2` at
        # `λspan[1]`), so integrating the wrong endpoint would give 1 instead
        # of √2.
        f = if iip
            (du, u, p, λ) -> (du .= (λ * 2 + (1 - λ)) .- u .^ 2)
        else
            (u, p, λ) -> (λ * 2 + (1 - λ)) .- u .^ 2
        end
        prob = SCCNonlinearProblem(
            (HomotopyProblem(f, [0.5]),), (Returns(nothing),)
        )
        sol = solve(prob, DynamicSS(Tsit5()); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [sqrt(2)] atol = 1.0e-8
    end

    @testset "LinearProblem blocks are solved directly" begin
        # `A = -1` makes `u' = b - A*u = 1 + u` repelling, but the block is
        # solved directly rather than integrated, so it still converges.
        prob = SCCNonlinearProblem(
            (LinearProblem([-1.0;;], [1.0]; u0 = [0.0]),), (Returns(nothing),)
        )
        sol = solve(prob, DynamicSS(Tsit5(), tspan = 10.0))
        @test successful_retcode(sol)
        @test sol.u ≈ [-1.0]
        @test sol.original isa Tuple{SciMLBase.LinearSolution}
    end

    @testset "Non-converging residual" begin
        # `u' = 1 + u` diverges, so steady-state termination never fires.
        prob = SCCNonlinearProblem(
            (NonlinearProblem((u, p) -> 1.0 .+ u, [0.0]),), (Returns(nothing),)
        )
        sol = solve(prob, DynamicSS(Tsit5(), tspan = 10.0))
        @test !successful_retcode(sol)
        @test sol.prob === prob
    end
end

# `SICNM` takes the same sequential route: the `LinearProblem` block is solved
# directly and the nonlinear block runs its own continuous-Newton flow on the
# block residual `g(x) = (3 - x^2 - y, 5 - x - y^2)`, converging to `(1, 2)`.
@testset "SICNM sequentially solves an SCCNonlinearProblem" begin
    @testset "iip=$iip vector=$use_vector" for iip in (false, true),
            use_vector in (false, true)

        prob = dynamicss_scc_problem(iip, use_vector)
        sol = solve(prob, SICNM(Rodas5P()); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2, 1, 2] atol = 1.0e-8
        @test sol.prob === prob
        @test sol.original isa Tuple{SciMLBase.LinearSolution, NonlinearSolution}
        @test sol.original[2].original isa SteadyStateSolution
        @test sol.original[2].original.original isa SciMLBase.AbstractODESolution
    end

    @testset "HomotopyProblem block" for iip in (false, true)
        # The target system is `u^2 = 2` at `λ = λspan[2]` (`u^2 = 1` at
        # `λspan[1]`), so evaluating the wrong endpoint would give 1 instead
        # of √2.
        f = if iip
            (du, u, p, λ) -> (du .= (λ * 2 + (1 - λ)) .- u .^ 2)
        else
            (u, p, λ) -> (λ * 2 + (1 - λ)) .- u .^ 2
        end
        prob = SCCNonlinearProblem(
            (HomotopyProblem(f, [0.5]),), (Returns(nothing),)
        )
        sol = solve(prob, SICNM(Rodas5P()); abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [sqrt(2)] atol = 1.0e-8
    end

    @testset "LinearProblem blocks are solved directly" begin
        prob = SCCNonlinearProblem(
            (LinearProblem([-1.0;;], [1.0]; u0 = [0.0]),), (Returns(nothing),)
        )
        sol = solve(prob, SICNM(Rodas5P()))
        @test successful_retcode(sol)
        @test sol.u ≈ [-1.0]
        @test sol.original isa Tuple{SciMLBase.LinearSolution}
    end

    @testset "Unsolvable residual" begin
        # `u^2 + 1 = 0` has no real root: the Newton flow is driven into the
        # singularity at `u = 0`, so the DAE solve cannot reach steady state.
        prob = SCCNonlinearProblem(
            (NonlinearProblem((u, p) -> u .^ 2 .+ 1, [1.0]),), (Returns(nothing),)
        )
        sol = solve(prob, SICNM(Rodas5P()))
        @test !successful_retcode(sol)
        @test sol.prob === prob
    end
end

# A `SteadyStateProblem` recording an `SCCNonlinearProblem` lowering is solved
# through it: `DynamicSS` runs the sequential block solve and `SSRootfind`
# forwards the lowering to the nonlinear solver. The solution is expressed on
# the lowering (whose ordering need not match `prob.u0`), so `sol.prob` is the
# `SCCNonlinearProblem`.
@testset "SteadyStateProblem with an SCC lowering" begin
    f_iip(du, u, p, t) = (du .= 1 .- u)
    f_oop(u, p, t) = 1 .- u
    @testset "iip=$iip builder=$builder alg=$alg" for iip in (false, true),
            builder in (false, true),
            alg in (
                DynamicSS(Tsit5()),
                SICNM(Rodas5P()),
                SSRootfind(NewtonRaphson()),
                SSRootfind(SCCAlg(; nlalg = NewtonRaphson())),
            )

        sccprob = dynamicss_scc_problem(iip, false)
        lowered = builder ? (prob -> sccprob) : sccprob
        prob = SteadyStateProblem(
            iip ? f_iip : f_oop, [0.0, 0.0]; lowered_problem = lowered
        )
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2, 1, 2] atol = 1.0e-8
        @test sol.prob === sccprob
        @test sol.original !== nothing
    end

    # `remake`d values reach a callable lowering through the materialized
    # problem, so the block solve sees the new operating point.
    sccprob = dynamicss_scc_problem(false, false)
    prob = SteadyStateProblem(
        f_oop, [0.0, 0.0]; lowered_problem = prob -> sccprob
    )
    @test NonlinearProblem(prob) === sccprob
    prob2 = remake(prob; u0 = [5.0, 6.0])
    @test NonlinearProblem(prob2) === sccprob
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
    # Sequential solve: the linear block is a direct solve and the nonlinear
    # block is integrated to steady state by `DynamicSS`.
    @test sol.original[1] isa SciMLBase.LinearSolution
    @test sol.original[2] isa NonlinearSolution
    @test sol.original[2].original isa SteadyStateSolution
    @test sol.original[2].original.original isa SciMLBase.AbstractODESolution
end

@testset "SteadyStateProblem solves through an SCC lowering" begin
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

    states = [a, b, x, y, c, d]
    expected = [1, 2, 1, 2, 3, 4]

    @test prob.lowered_problem !== nothing

    # The `SSRootfind` solve happens on the `SCCNonlinearProblem` lowering, and
    # `sol` is expressed on it.
    sol = solve(
        prob, SSRootfind(NewtonRaphson()); abstol = 1.0e-12, reltol = 1.0e-12
    )
    @test successful_retcode(sol)
    @test sol[states] ≈ expected atol = 1.0e-9
    @test sol.prob isa SCCNonlinearProblem
end

@testset "DynamicSS on a SteadyStateProblem with an SCC lowering" begin
    # Scalar nonlinear block `x' = 3 - x^3` has a unique real root `∛3`, so the
    # per-block pseudo-transient solve and the monolithic integration converge
    # to the same steady state.
    @variables a(t) b(t) x(t) [irreducible = true]
    @named model = System(
        [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(x) ~ a + b - x^3], t
    )
    sys = mtkcompile(model)
    prob = SteadyStateProblem(sys, [a => 0.8, b => 1.8, x => 0.8])

    sol = solve(prob, DynamicSS(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol[[a, b, x]] ≈ [1, 2, cbrt(3)] atol = 1.0e-8
    @test sol.prob isa SCCNonlinearProblem
    @test sol.original isa Tuple{SciMLBase.LinearSolution, NonlinearSolution}
end

@testset "SICNM on a SteadyStateProblem with an SCC lowering" begin
    # Scalar nonlinear block `3 - x^3` has a unique real root `∛3` that is
    # attracting under SICNM's continuous-Newton flow.
    @variables a(t) b(t) x(t) [irreducible = true]
    @named model = System(
        [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(x) ~ a + b - x^3], t
    )
    sys = mtkcompile(model)
    prob = SteadyStateProblem(sys, [a => 0.8, b => 1.8, x => 0.8])

    sol = solve(prob, SICNM(Rodas5P()); abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol[[a, b, x]] ≈ [1, 2, cbrt(3)] atol = 1.0e-8
    @test sol.prob isa SCCNonlinearProblem
    @test sol.original isa Tuple{SciMLBase.LinearSolution, NonlinearSolution}
end
