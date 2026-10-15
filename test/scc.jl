using SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq, LinearAlgebra, Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SCCNonlinearSolve: SCCAlg
import SymbolicIndexingInterface
using SymbolicIndexingInterface: parameter_symbols, state_values, variable_index, variable_symbols
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
        @test state_values(saved.prob) == state_values(prob)[[2, 4]]
        @test solve(
            saved.prob, alg; abstol = 1.0e-10, reltol = 1.0e-10
        ).u ≈ saved.u
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

    # `DynamicSS` integrates each block's residual to steady state. With the
    # explicit `Tsit5` the step size is pinned at the stability limit near the
    # fixed point, so the residual plateaus at a tolerance-sized noise floor
    # right at the termination threshold and flips between `Success` and
    # `Failure` under small perturbations. `Rodas5P` is not step-size limited at
    # the fixed point, so the residual keeps decaying until the terminator
    # fires (here at t ≈ 14).
    allalgs = (
        NewtonRaphson(), SSRootfind(NewtonRaphson()), DynamicSS(Rodas5P()),
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
        # States not in `save_idxs` and observed expressions are unresolvable on
        # the saved-state layout; parameters still index through it.
        for s in Iterators.drop(variable_symbols(prob), 1)
            @test_throws ErrorException saved[s]
        end
        observed === nothing || @test_throws ErrorException saved[first(observed)]
        if !isempty(parameter_symbols(prob))
            p1 = first(parameter_symbols(prob))
            @test saved.ps[p1] == sol.ps[p1]
        end

        # A symbolic `save_idxs` selection is applied in selection order.
        rev = reverse(variable_symbols(prob))
        saved = solve(
            prob, alg; save_idxs = rev, abstol = 1.0e-10, reltol = 1.0e-10
        )
        for s in rev
            @test saved[s] ≈ sol[s] atol = 1.0e-7
        end
        re = solve(saved.prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        for s in rev
            @test re[s] ≈ sol[s] atol = 1.0e-7
        end
        @test re.u ≈ saved.u
    end
end

@testset "SCCSteadyStateProblem save_idxs symbolic indexing" begin
    # A single 2-state linear block with solution x = 1, y = 2; `sys` provides
    # symbolic names so `saved[s]` resolves against the saved-state layout.
    sysc = SymbolicIndexingInterface.SymbolCache([:x, :y], [], nothing)
    linblock() = LinearProblem(Diagonal(ones(2)), [1.0, 2.0]; u0 = [10.0, 20.0])

    @testset "alg=$(nameof(typeof(alg)))" for alg in (
            NewtonRaphson(), SSRootfind(NewtonRaphson()),
            DynamicSS(Rodas5P()), SICNM(Rodas5P()),
        )
        prob = SCCSteadyStateProblem(
            (linblock(),), (Returns(nothing),), nothing, true; sys = sysc
        )
        saved = solve(prob, alg; save_idxs = [2, 1])
        @test saved.u == [2, 1]
        @test saved[:x] == 1
        @test saved[:y] == 2
        @test saved[[:x, :y]] == [1, 2]
        @test saved.original[:x] == 1
        @test saved.original[:y] == 2

        # Re-solving the saved problem works in the saved layout: the same
        # selection is applied again, and `u0`/`save_idxs` are interpreted in
        # saved positions.
        re2 = solve(saved.prob, alg)
        @test re2.u == [2, 1]
        @test re2[:x] == 1
        @test re2[:y] == 2
        @test solve(saved.prob, alg; save_idxs = [:y]).u == [2]
        @test state_values(saved.prob) == state_values(prob)[[2, 1]]
        @test SymbolicIndexingInterface.getu(saved.prob, :y)(saved.prob) ==
            SymbolicIndexingInterface.getu(prob, :y)(prob)
        @test SymbolicIndexingInterface.getu(saved.prob, :x)(saved.prob) ==
            SymbolicIndexingInterface.getu(prob, :x)(prob)

        pre_remake = copy(state_values(prob))
        r = remake(saved.prob; u0 = [30.0, 40.0])
        @test state_values(r) == [30.0, 40.0]
        @test state_values(prob) == pre_remake
        @test solve(r, alg).u == [2, 1]

        saved = solve(prob, alg; save_idxs = [:y])
        @test saved.u == [2]
        @test saved[:y] == 2
        @test_throws ErrorException saved[:x]
        @test saved.original[:x] == 1

        re = solve(saved.prob, alg)
        @test re.u == [2]
        @test re[:y] == 2
        @test_throws ErrorException re[:x]
        @test SymbolicIndexingInterface.getu(saved.prob, :y)(saved.prob) ==
            SymbolicIndexingInterface.getu(prob, :y)(prob)
    end
end

@testset "stateless and empty SCC blocks" begin
    @testset "alg=$(nameof(typeof(alg)))" for alg in (
            DynamicSS(Rodas5P()), SICNM(Rodas5P()),
        )
        # A stateless block with a nonempty residual is rejected — the residual
        # is still evaluated once.
        for iip in (false, true)
            counter = Ref(0)
            blk = if iip
                NonlinearProblem(
                    NonlinearFunction(
                        (res, u, p) -> (counter[] += 1; res .= [1.0]);
                        resid_prototype = [0.0]
                    ), Float64[]
                )
            else
                NonlinearProblem(
                    NonlinearFunction(
                        (u, p) -> (counter[] += 1; [1.0]); resid_prototype = [0.0]
                    ), Float64[]
                )
            end
            for T in (SCCSteadyStateProblem, SCCNonlinearProblem)
                prob = T((blk,), (Returns(nothing),), nothing, true)
                counter[] = 0
                @test_throws DimensionMismatch solve(prob, alg)
                @test counter[] > 0
            end
        end

        # A stateless block whose residual is empty returns Success; the
        # residual is evaluated rather than assumed zero.
        counter = Ref(0)
        blk = NonlinearProblem(
            NonlinearFunction(
                (u, p) -> (counter[] += 1; Float64[]); resid_prototype = Float64[]
            ), Float64[]
        )
        for T in (SCCSteadyStateProblem, SCCNonlinearProblem)
            prob = T((blk,), (Returns(nothing),), nothing, true)
            counter[] = 0
            sol = solve(prob, alg)
            @test successful_retcode(sol)
            @test counter[] > 0
        end

        # An empty linear block is a no-op, alone or before another block.
        lblk() = LinearProblem(zeros(0, 0), Float64[]; u0 = Float64[])
        for T in (SCCSteadyStateProblem, SCCNonlinearProblem)
            sol = solve(T((lblk(),), (Returns(nothing),), nothing, true), alg)
            @test successful_retcode(sol)
            @test sol.u == Float64[]
            prob = T(
                (lblk(), LinearProblem([2.0;;], [4.0]; u0 = [0.0])),
                (Returns(nothing), Returns(nothing)), nothing, true
            )
            sol = solve(prob, alg)
            @test successful_retcode(sol)
            @test sol.u ≈ [2.0]
        end
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

    # Bare and `AbstractNonlinearAlgorithm` solves have no `__solve` dispatch of
    # their own here; they land on the generic nonlinear-solve path, which
    # converts back through `prob.u0` — absent on `SCCNonlinearProblem`. `solve`
    # forwards the lowering to its own `solve` dispatch (the SCC solver)
    # instead.
    @testset "bare and nonlinear-alg solve" for iip in (false, true),
            builder in (false, true),
            alg in (nothing, NewtonRaphson())

        sccprob = dynamicss_scc_problem(iip, false)
        lowered = builder ? (prob -> sccprob) : sccprob
        prob = SteadyStateProblem(
            iip ? f_iip : f_oop, [0.0, 0.0]; lowered_problem = lowered
        )
        sol = alg === nothing ? solve(prob) : solve(prob, alg)
        @test successful_retcode(sol)
        @test sol.u ≈ [1, 2, 1, 2] atol = 1.0e-9
    end

    # Non-SCC lowerings and problems without a lowering keep the upstream
    # nonlinear-solve path.
    @testset "non-SCC lowering" begin
        nlprob = NonlinearProblem((u, p) -> [u[1]^2 - 4], [1.0])
        prob = SteadyStateProblem(f_oop, [0.0]; lowered_problem = nlprob)
        sol = solve(prob)
        @test successful_retcode(sol)
        @test sol.u ≈ [2.0] atol = 1.0e-9

        plain = SteadyStateProblem(f_oop, [0.0])
        psol = solve(plain)
        @test successful_retcode(psol)
        @test psol.u ≈ [1.0] atol = 1.0e-9
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

# A `SteadyStateProblem` may record a non-SCC `lowered_problem`; it solves
# through the lowering and the solution is expressed on the lowered problem's
# own coordinates.
@testset "SteadyStateProblem with a generic lowering" begin
    ssys = SymbolicIndexingInterface.SymbolCache([:x, :y, :z], [], nothing)
    lowsys = SymbolicIndexingInterface.SymbolCache([:y], [], nothing)
    low = NonlinearProblem(
        SciMLBase.NonlinearFunction((u, p) -> [2 - u[1]]; sys = lowsys), [0.0]
    )
    for lp in (low, prob -> low), alg in (SSRootfind(NewtonRaphson()), SSRootfind())
        prob = SteadyStateProblem(
            SciMLBase.ODEFunction((u, p, t) -> [1 - u[1], 2 - u[2], 3 - u[3]]; sys = ssys),
            zeros(3); lowered_problem = lp
        )
        sol = solve(prob, alg)
        @test sol.u ≈ [2]
        @test variable_symbols(sol) == [:y]
        @test sol[:y] ≈ 2
        @test sol.prob === low
    end
end
