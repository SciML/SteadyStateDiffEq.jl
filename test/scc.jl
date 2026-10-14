using SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq, Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SCCNonlinearSolve: SCCAlg
using SymbolicIndexingInterface: state_values, variable_index, variable_symbols
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


# An `SCCSteadyStateProblem` built from the same blocks solves through the same
# sequential/block or SCC-solver paths, with the solution expressed on the SCC
# problem itself.
@testset "SCCSteadyStateProblem from blocks" begin
    @testset "iip=$iip vector=$use_vector alg=$alg" for iip in (false, true),
            use_vector in (false, true),
            alg in (
                DynamicSS(Tsit5()),
                SICNM(Rodas5P()),
                SSRootfind(SCCAlg()),
                SSRootfind(SCCAlg(; nlalg = NewtonRaphson())),
                NewtonRaphson(),
                nothing,
            )

        prob = SciMLBase.SCCSteadyStateProblem(dynamicss_scc_problem(iip, use_vector))
        sol = alg === nothing ? solve(prob) :
            solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.prob === prob
        @test sol.u ≈ [1, 2, 1, 2] atol = 1.0e-8
        @test sol.u isa Vector{Float64}
        @test sol.resid isa Vector{Float64}
        @test length(sol.resid) == length(sol.u)
        @test sol.resid ≈ zero(sol.resid) atol = 1.0e-7
        if alg isa Union{DynamicSS, SICNM}
            @test sol.original isa Tuple
            @test sol.original[1] isa SciMLBase.LinearSolution
        else
            @test sol.original isa NonlinearSolution
            @test sol.original.prob isa SCCNonlinearProblem
        end
    end

    # `save_idxs` selects entries of the compressed SCC state/resid layout and
    # keeps the full solution in `original`.
    @testset "save_idxs" for alg in (DynamicSS(Tsit5()), NewtonRaphson())
        prob = SciMLBase.SCCSteadyStateProblem(dynamicss_scc_problem(false, false))
        full = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        saved = solve(
            prob, alg; save_idxs = [2, 4], abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test saved.u ≈ full.u[[2, 4]]
        @test saved.resid ≈ full.resid[[2, 4]]
        @test saved.prob === prob
        @test saved.original.u ≈ full.u
    end

    # A `tspan` too short to settle gives a failure retcode, like DynamicSS on
    # a `SteadyStateProblem`.
    @testset "short tspan" for alg in (DynamicSS(Tsit5(); tspan = 1.0e-3), SICNM(Rodas5P(); tspan = 1.0e-3))
        prob = SciMLBase.SCCSteadyStateProblem(dynamicss_scc_problem(false, false))
        sol = solve(prob, alg)
        @test !successful_retcode(sol)
        @test sol.prob === prob
    end
end

@testset "SSRootfind on a ModelingToolkit SCCSteadyStateProblem" begin
    @variables a(t) b(t) x(t) [irreducible = true]
    @named model = System(
        [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(x) ~ a + b - x^3], t
    )
    sys = mtkcompile(model)
    prob = SciMLBase.SCCSteadyStateProblem(sys, [a => 0.8, b => 1.8, x => 0.8])

    sol = solve(prob, SSRootfind(NewtonRaphson()); abstol = 1.0e-12, reltol = 1.0e-12)
    @test successful_retcode(sol)
    @test sol[[a, b, x]] ≈ [1, 2, cbrt(3)] atol = 1.0e-8
    @test sol.prob === prob
end

@testset "DynamicSS on a ModelingToolkit SCCSteadyStateProblem" begin
    # Scalar nonlinear block `x' = 3 - x^3` has a unique real root `∛3`, so the
    # per-block pseudo-transient solve and the monolithic integration converge
    # to the same steady state.
    @variables a(t) b(t) x(t) [irreducible = true]
    @named model = System(
        [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(x) ~ a + b - x^3], t
    )
    sys = mtkcompile(model)
    prob = SciMLBase.SCCSteadyStateProblem(sys, [a => 0.8, b => 1.8, x => 0.8])

    sol = solve(prob, DynamicSS(); abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol[[a, b, x]] ≈ [1, 2, cbrt(3)] atol = 1.0e-8
    @test sol.prob === prob
    @test sol.original isa Tuple{SciMLBase.LinearSolution, NonlinearSolution}
end

@testset "SICNM on a ModelingToolkit SCCSteadyStateProblem" begin
    # Scalar nonlinear block `3 - x^3` has a unique real root `∛3` that is
    # attracting under SICNM's continuous-Newton flow.
    @variables a(t) b(t) x(t) [irreducible = true]
    @named model = System(
        [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(x) ~ a + b - x^3], t
    )
    sys = mtkcompile(model)
    prob = SciMLBase.SCCSteadyStateProblem(sys, [a => 0.8, b => 1.8, x => 0.8])

    sol = solve(prob, SICNM(Rodas5P()); abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol[[a, b, x]] ≈ [1, 2, cbrt(3)] atol = 1.0e-8
    @test sol.prob === prob
    @test sol.original isa Tuple{SciMLBase.LinearSolution, NonlinearSolution}
end

# Systems shared by the two solution-layout testsets below: `op` is the
# operating point, `states`/`expected` the steady-state values, `observed` an
# `sym => value` pair of the `osys` system's observed `XY`.
function layout_cases()
    @parameters ca = 1.0 cb = 2.0
    @variables x(t) y(t) z(t)
    chain = mtkcompile(
        System([D(x) ~ ca - x, D(y) ~ x - cb * y, D(z) ~ y - z], t; name = :chain)
    )

    @parameters α = 1.5 β = 1.0 γ = 3.0 δ = 1.0
    @variables u(t) v(t)
    lv = mtkcompile(
        System([D(u) ~ α * u - β * u * v, D(v) ~ -γ * v + δ * u * v], t; name = :lv)
    )

    @variables a(t) b(t) mx(t) [irreducible = true]
    mix = mtkcompile(
        System(
            [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(mx) ~ a + b - mx^3], t;
            name = :mix
        )
    )

    @parameters kp kd k1 k2
    @variables X(t) Y(t) XY(t)
    osys = complete(
        System(
            [D(X) ~ kp - kd * X - k1 * X + k2 * Y, D(Y) ~ 1 + k1 * X - k2 * Y - Y], t;
            observed = [XY ~ X + Y], name = :osys
        )
    )
    Xss, Yss = [0.35 -0.5; -0.25 1.5] \ [1.0, 1.0]

    # `DynamicSS(Tsit5())` over `tspan = Inf` does not reach its terminator on
    # this system — the integrator exhausts maxiters while the residual is still
    # well above the tolerance — so the pseudo-transient runs use a finite span.
    allalgs = (
        NewtonRaphson(), SSRootfind(NewtonRaphson()), DynamicSS(Tsit5(); tspan = 100),
        SICNM(Rodas5P()),
    )
    nlalgs = (NewtonRaphson(), SSRootfind(NewtonRaphson()))
    return (
        (;
            sys = chain, op = [x => 0.1, y => 0.2, z => 0.3], states = [x, y, z],
            expected = [1.0, 0.5, 0.5], algs = allalgs, observed = nothing,
        ),
        # Lotka–Volterra's nonzero steady state is a center, so the
        # pseudo-transient methods do not apply.
        (;
            sys = lv, op = [u => 2.9, v => 1.6], states = [u, v],
            expected = [3.0, 1.5], algs = nlalgs, observed = nothing,
        ),
        (;
            sys = mix, op = [a => 0.8, b => 1.8, mx => 0.8], states = [a, b, mx],
            expected = [1.0, 2.0, cbrt(3)], algs = allalgs, observed = nothing,
        ),
        (;
            sys = osys,
            op = [X => 4.0, Y => 5.0, kp => 1.0, kd => 0.1, k1 => 0.25, k2 => 0.5],
            states = [X, Y], expected = [Xss, Yss], algs = allalgs,
            observed = XY => Xss + Yss,
        ),
    )
end

# https://github.com/SciML/SteadyStateDiffEq.jl/issues/171
# A `SteadyStateProblem` solves in its own layout: `sol.u`/`sol.resid` have the
# length and ordering of `prob.u0`. Nonlinear algorithms return their solution
# on the converted `NonlinearProblem`; the steady-state algorithms on `prob`.
@testset "SteadyStateProblem solutions are in the problem's layout" begin
    @testset "case=$(nameof(case.sys)) alg=$(nameof(typeof(alg)))" for
        case in layout_cases(), alg in case.algs

        (; sys, op, states, expected, observed) = case
        prob = SteadyStateProblem(sys, op)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.prob isa (
            alg isa SciMLBase.AbstractNonlinearAlgorithm ?
                NonlinearProblem : SteadyStateProblem
        )
        @test length(sol.u) == length(prob.u0)
        @test sol.u[variable_index.((sys,), states)] ≈ expected atol = 1.0e-7
        @test sol[states] ≈ expected atol = 1.0e-7
        observed === nothing || @test sol[first(observed)] ≈ last(observed) atol = 1.0e-7
        @test length(sol.resid) == length(prob.u0)
        @test sol.resid ≈ zero(sol.resid) atol = 1.0e-7

        if alg isa Union{DynamicSS, SICNM}
            saved = solve(
                prob, alg; save_idxs = [2], abstol = 1.0e-10, reltol = 1.0e-10
            )
            @test saved.u ≈ sol.u[2:2] atol = 1.0e-7
            @test length(saved.resid) == 1
        end
    end

    # The residual solves `x` to an integer literal; the solution must keep
    # `prob.u0`'s type.
    @variables ex(t)
    csys = complete(System([D(ex) ~ 1 - ex], t; name = :csys))
    @testset "eltype, alg=$(nameof(typeof(alg)))" for alg in (
            SSRootfind(Broyden()), DynamicSS(Tsit5()), SICNM(Rodas5P()),
        )
        prob = SteadyStateProblem(csys, [ex => 0.2])
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.u isa typeof(prob.u0)
        @test sol.u ≈ [1.0]
    end
end

# The same systems via `SCCSteadyStateProblem`: the solution is expressed on the
# compressed block layout, symbolic indexing resolves eliminated variables as
# observed, and `save_idxs` indexes that layout.
@testset "SCCSteadyStateProblem from ModelingToolkit" begin
    @testset "case=$(nameof(case.sys)) alg=$(nameof(typeof(alg)))" for
        case in layout_cases(), alg in case.algs

        (; sys, op, states, expected, observed) = case
        prob = SciMLBase.SCCSteadyStateProblem(sys, op)
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        @test successful_retcode(sol)
        @test sol.prob === prob
        @test length(sol.u) == length(state_values(prob))
        @test sol[states] ≈ expected atol = 1.0e-7
        observed === nothing || @test sol[first(observed)] ≈ last(observed) atol = 1.0e-7
        for s in variable_symbols(prob)
            @test sol.u[variable_index(prob, s)] == sol[s]
        end
        @test length(sol.resid) == length(sol.u)
        @test sol.resid ≈ zero(sol.resid) atol = 1.0e-7

        isempty(state_values(prob)) && continue
        saved = solve(
            prob, alg; save_idxs = [1], abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test saved.u ≈ sol.u[1:1] atol = 1.0e-7
        s1 = first(variable_symbols(prob))
        saved = solve(
            prob, alg; save_idxs = [s1], abstol = 1.0e-10, reltol = 1.0e-10
        )
        @test saved.u ≈ [sol[s1]] atol = 1.0e-7
        @test saved.original.u ≈ sol.u atol = 1.0e-7
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
    # Sequential solve: the linear block is a direct solve and the nonlinear
    # block is integrated to steady state by `DynamicSS`.
    @test sol.original[1] isa SciMLBase.LinearSolution
    @test sol.original[2] isa NonlinearSolution
    @test sol.original[2].original isa SteadyStateSolution
    @test sol.original[2].original.original isa SciMLBase.AbstractODESolution
end
