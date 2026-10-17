function __build_ssrootfind_solution(prob, nlsol)
    return SciMLBase.build_solution(
        prob, SSRootfind(nlsol.alg), nlsol.u, nlsol.resid;
        nlsol.retcode, nlsol.stats, nlsol.left, nlsol.right, original = nlsol
    )
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::SSRootfind,
        args...; kwargs...
    )
    nlprob = NonlinearProblem(prob)
    fwd = kwargs
    if nlprob isa SciMLBase.SCCNonlinearProblem
        # The nonlinear solve pipeline injects `alias`/`verbose` as nonlinear
        # specifier types, which `LinearProblem` blocks in the SCC solve do not
        # understand.
        fwd = (; (n => v for (n, v) in pairs(kwargs) if n !== :alias && n !== :verbose)...)
    end
    nlsol = solve(nlprob, alg.alg, args...; fwd...)
    if prob isa SteadyStateProblem && prob.lowered_problem !== nothing
        sol = __scc_solution_on_problem(prob, SSRootfind(nlsol.alg), nlsol)
        sol !== nothing && return sol
        # Without symbolic information the lowering's state cannot be mapped
        # back, so the solution is expressed on the lowering itself.
        return __build_ssrootfind_solution(nlprob, nlsol)
    end
    return __build_ssrootfind_solution(prob, nlsol)
end

# An SCCNonlinearProblem has no top-level `u0`/`kwargs` fields, so it cannot go
# through the generic AbstractNonlinearProblem solve preprocessing. Forward it
# directly to the wrapped algorithm instead, which dispatches to the SCC solver
# loaded downstream (SCCNonlinearSolve.jl).
function SciMLBase.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::SSRootfind,
        args...; kwargs...
    )
    nlsol = solve(prob, alg.alg, args...; kwargs...)
    return __build_ssrootfind_solution(prob, nlsol)
end

# `DynamicSS`/`SICNM` on an `SCCNonlinearProblem` solve the blocks sequentially
# in SCC order: each block's parameter cache is updated from the upstream
# solutions through `explicitfuns!`, `LinearProblem` blocks are solved directly,
# and the remaining blocks are integrated to steady state on the block
# residual. `explicitfuns!` mutate the block caches, matching the
# `SCCNonlinearSolve` protocol; each upstream solution is stripped to a plain
# solution so generated explicit functions can index `sols[j][k]`/`sols[j].u`.
function SciMLBase.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::Union{DynamicSS, SICNM},
        args...; kwargs...
    )
    sols = ()
    for i in eachindex(prob.probs)
        block = prob.probs[i]
        SciMLBase.invoke_with_despecialized_parameters(
            prob.explicitfuns![i], (parameter_values(block), sols)
        )
        _sol = if block isa LinearProblem
            A, b = block.A, block.b
            # `remake` recomputes `A` and `b` through a `SymbolicLinearInterface`
            # from the cache that `explicitfun` just updated; for a plain
            # `LinearProblem` it is an identity copy.
            linsol = SciMLBase.solve(remake(block; A, b), nothing)
            resid = linsol.resid === nothing ? A * linsol.u - b : linsol.resid
            SciMLBase.build_linear_solution(
                nothing, linsol.u, resid, nothing; retcode = linsol.retcode
            )
        else
            # `HomotopyProblem` has no `solve` preprocessing support
            # (`get_concrete_problem` is only defined for `NonlinearProblem`
            # and `NonlinearLeastSquaresProblem`), so it goes through `__solve`
            # directly.
            sssol = if block isa SciMLBase.HomotopyProblem
                SciMLBase.__solve(block, alg, args...; kwargs...)
            else
                SciMLBase.solve(block, alg, args...; kwargs...)
            end
            SciMLBase.build_solution(
                block, nothing, sssol.u, sssol.resid;
                retcode = sssol.retcode, original = sssol
            )
        end
        sols = (sols..., _sol)
    end

    u = vcat(map(sol -> sol.u, sols)...)
    resid = vcat(map(sol -> sol.resid, sols)...)
    retcode = SciMLBase.ReturnCode.Success
    for s in sols
        if !SciMLBase.successful_retcode(s)
            retcode = s.retcode
            break
        end
    end
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = sols)
end

# The `SCCNonlinearProblem` lowering recorded on a `SteadyStateProblem`, or
# `nothing` when there is none.
function __scc_lowering(prob)
    lp = prob isa SteadyStateProblem ? prob.lowered_problem : nothing
    lp === nothing && return nothing
    lp isa SciMLBase.AbstractSciMLProblem || (lp = lp(prob))
    return lp isa SciMLBase.SCCNonlinearProblem ? lp : nothing
end

# A `SteadyStateProblem` that records an `SCCNonlinearProblem` lowering `lp` is
# solved block-sequentially in the lowering's ordering instead of one
# monolithic integration.
function __solve_scc_lowering(
        prob, lp, alg, args...; save_idxs = nothing, kwargs...
    )
    sccsol = solve(lp, alg, args...; kwargs...)
    sol = __scc_solution_on_problem(prob, sccsol.alg, sccsol)
    sol === nothing && (sol = sccsol)
    save_idxs === nothing && return sol
    return SciMLBase.build_solution(
        sol.prob, sol.alg, sol.u[save_idxs], sol.resid[save_idxs];
        retcode = sol.retcode, sol.stats, original = sol
    )
end

# The SCC lowering of a symbolic `SteadyStateProblem` is built on a separately
# compiled residual system, so its state generally differs from `prob`'s in
# length and order. When `prob` is symbolic and the lowering's index provider
# resolves every unknown of `prob`, the solution is mapped back onto `prob` so
# that `sol.u` lines up with `prob.u0` and `sol` indexes like `prob`. Returns
# `nothing` when no such mapping exists.
function __scc_solution_on_problem(prob::SteadyStateProblem, alg, sccsol)
    SciMLBase.has_sys(prob.f) && prob.f.sys !== nothing || return nothing
    syms = variable_symbols(prob)
    length(syms) == length(prob.u0) || return nothing
    all(s -> is_variable(sccsol, s) || is_observed(sccsol, s), syms) || return nothing
    u = getsym(sccsol, syms)(sccsol)
    u = convert(typeof(prob.u0), collect(u))
    resid = if isinplace(prob)
        du = similar(u)
        prob.f(du, u, prob.p, Inf)
        du
    else
        prob.f(u, prob.p, Inf)
    end
    return SciMLBase.build_solution(
        prob, alg, u, resid; sccsol.retcode, sccsol.stats, sccsol.left,
        sccsol.right, original = sccsol
    )
end

# A `SteadyStateProblem`'s stored SCC lowering materializes through
# `SciMLBase.NonlinearProblem(prob)` during `solve`. When it is an
# `SCCNonlinearProblem`, the generic nonlinear-solve path re-dispatches
# `__solve` on it and the fallbacks try to convert it back via `prob.u0` — a
# field it does not have — so the lowering goes through its own `solve`
# dispatch instead, which routes bare and `AbstractNonlinearAlgorithm` solves
# to the SCC solver (`SCCNonlinearSolve.jl`, loaded downstream). Algorithms
# that do not lower to a nonlinear solve (`DynamicSS`, ODE algorithms) keep
# their own `__solve` dispatch; non-SCC lowerings keep the upstream path
# verbatim via `invoke`.
function SciMLBase.solve(prob::SteadyStateProblem, args...; kwargs...)
    alg = isempty(args) ? nothing : first(args)
    alg === nothing && (alg = get(kwargs, :alg, nothing))
    alg === nothing && prob.kwargs !== nothing &&
        (alg = get(prob.kwargs, :alg, nothing))
    if alg !== nothing && !(alg isa SciMLBase.AbstractNonlinearAlgorithm)
        return invoke(
            solve, Tuple{SciMLBase.AbstractNonlinearProblem, Vararg{Any}},
            prob, args...; kwargs...
        )
    end
    # Concretize like `NonlinearSolveBase.solve_up` (evaluates `u0(p, Inf)`-style
    # `u0`s, promotes integer `u0`s, applies `u0`/`p` solve kwargs) so a
    # callable `lowered_problem` sees the updated operating point.
    _prob = NonlinearSolveBase.get_concrete_problem(prob; kwargs...)
    nlprob = NonlinearProblem(_prob)
    nlprob isa SciMLBase.SCCNonlinearProblem && return solve(nlprob, args...; kwargs...)
    return invoke(
        solve, Tuple{SciMLBase.AbstractNonlinearProblem, Vararg{Any}},
        _prob, args...; kwargs...
    )
end

# `init(prob, ::DynamicSS/SICNM)` returns one of these caches; `solve!` finishes
# it with the same steady-state finalization as `solve`.
@concrete struct SteadyStateSCCCache
    prob
    lowered_problem
    alg
    args
    kwargs
end

function SciMLBase.solve!(cache::SteadyStateSCCCache)
    return __solve_scc_lowering(
        cache.prob, cache.lowered_problem, cache.alg, cache.args...; cache.kwargs...
    )
end

@concrete struct SteadyStateODECache
    prob
    alg
    integrator
    setup
    save_idxs
end

function SciMLBase.solve!(cache::SteadyStateODECache)
    odesol = solve!(cache.integrator)
    return __steady_state_solution(
        cache.prob, cache.alg, cache.setup, odesol, cache.save_idxs
    )
end

SciMLBase.step!(cache::SteadyStateODECache, args...) = step!(cache.integrator, args...)

__get_tspan(u0, alg::Union{DynamicSS, SICNM}) = __get_tspan(u0, alg.tspan)
__get_tspan(u0, tspan::Tuple) = tspan
function __get_tspan(u0, tspan::Number)
    return convert.(
        SciMLBase.value(real(eltype(u0))), (SciMLBase.value(zero(tspan)), tspan)
    )
end

function __without_verbose(kwargs)
    return (; (name => value for (name, value) in pairs(kwargs) if name !== :verbose)...)
end

# Shared `DynamicSS` setup for `__solve` and `__init`: the ODE problem integrating
# `prob`'s residual to steady state and its termination callback. SCC lowerings are
# handled by `__solve_scc_lowering` before this is reached.
function __dynamicss_ode_setup(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::DynamicSS;
        abstol, reltol, odesolve_kwargs, termination_condition, alias, kwargs...
    )
    tspan = __get_tspan(prob.u0, alg)

    f = if prob isa SteadyStateProblem
        prob.f
    elseif prob isa SciMLBase.HomotopyProblem
        # The `λ = λspan[2]` end of the homotopy is the actual system.
        λ1 = last(prob.λspan)
        if isinplace(prob)
            (du, u, p, t) -> prob.f(du, u, p, λ1)
        else
            (u, p, t) -> prob.f(u, p, λ1)
        end
    elseif prob isa NonlinearProblem
        if isinplace(prob)
            (du, u, p, t) -> prob.f(du, u, p)
        else
            (u, p, t) -> prob.f(u, p)
        end
    end

    if isinplace(prob)
        du = similar(prob.u0)
        f(du, prob.u0, prob.p, first(tspan))
    else
        du = f(prob.u0, prob.p, first(tspan))
    end

    tc_cache = init(prob, termination_condition, du, prob.u0; abstol, reltol)
    abstol = NonlinearSolveBase.get_abstol(tc_cache)
    reltol = NonlinearSolveBase.get_reltol(tc_cache)

    function terminate_function(u, t, integrator)
        return tc_cache(get_du(integrator), integrator.u, integrator.uprev, t)
    end

    callback = TerminateSteadyState(
        abstol, reltol, terminate_function;
        wrap_test = Val(false)
    )

    haskey(kwargs, :callback) && (callback = CallbackSet(callback, kwargs[:callback]))
    haskey(odesolve_kwargs, :callback) &&
        (callback = CallbackSet(callback, odesolve_kwargs[:callback]))
    run_kwargs = pairs(__without_verbose(kwargs))
    odeprob = ODEProblem{isinplace(prob), true}(f, prob.u0, tspan, prob.p)
    odealias = SciMLBase.ODEAliasSpecifier(;
        alias_p = alias.alias_p, alias_f = alias.alias_f, alias_u0 = alias.alias_u0
    )
    return (; odeprob, tc_cache, abstol, reltol, callback, run_kwargs, odealias)
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::DynamicSS,
        args...; abstol = 1.0e-8, reltol = 1.0e-6, odesolve_kwargs = (;),
        save_idxs = nothing, termination_condition = NonlinearSolveBase.NormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...
    )
    lp = __scc_lowering(prob)
    lp !== nothing && return __solve_scc_lowering(
        prob, lp, alg, args...; abstol, reltol, odesolve_kwargs,
        termination_condition, alias, save_idxs, kwargs...
    )

    setup = __dynamicss_ode_setup(
        prob, alg; abstol, reltol, odesolve_kwargs, termination_condition, alias, kwargs...
    )
    odesol = solve(
        setup.odeprob, alg.alg, args...; setup.abstol, setup.reltol,
        setup.run_kwargs..., odesolve_kwargs..., setup.callback, save_end = true,
        alias = setup.odealias
    )
    return __steady_state_solution(prob, alg, setup, odesol, save_idxs)
end

function __steady_state_solution(prob, alg::DynamicSS, setup, odesol, save_idxs)
    resid, u, retcode = __get_result_from_sol(setup.tc_cache, odesol)

    if save_idxs !== nothing
        u = u[save_idxs]
        resid = resid[save_idxs]
    end

    return SciMLBase.build_solution(
        prob, DynamicSS(odesol.alg, alg.tspan), u, resid;
        retcode, odesol.stats, original = odesol
    )
end

function SciMLBase.__init(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::DynamicSS,
        args...; abstol = 1.0e-8, reltol = 1.0e-6, odesolve_kwargs = (;),
        save_idxs = nothing, termination_condition = NonlinearSolveBase.NormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...
    )
    lp = __scc_lowering(prob)
    lp !== nothing && return SteadyStateSCCCache(
        prob, lp, alg, args, (;
            abstol, reltol, odesolve_kwargs, termination_condition, alias,
            save_idxs, kwargs...,
        )
    )

    setup = __dynamicss_ode_setup(
        prob, alg; abstol, reltol, odesolve_kwargs, termination_condition, alias, kwargs...
    )
    integrator = init(
        setup.odeprob, alg.alg, args...; setup.abstol, setup.reltol,
        setup.run_kwargs..., odesolve_kwargs..., setup.callback, save_end = true,
        alias = setup.odealias
    )
    return SteadyStateODECache(prob, alg, integrator, setup, save_idxs)
end

# SICNM: Semi-Implicit Continuous Newton Method
# Solves 0 = g(y) by integrating the DAE  ẏ = z, 0 = J(y)z + g(y)  to steady state,
# where J is the Jacobian of g. See the SICNM docstring for details and references.

struct SICNMJacVecTag end

# Evaluate the residual g and the Jacobian-vector product J(y)z simultaneously from a
# single dual-number evaluation of g at y + ε z.
function __sicnm_dual_seed(y, z)
    T = eltype(y)
    TagType = typeof(ForwardDiff.Tag(SICNMJacVecTag(), T))
    td = ForwardDiff.Dual{TagType}(zero(T), one(T))
    return @. y + td * z
end

function __sicnm_g_and_jvp(g::G, y, z) where {G}
    resd = g(__sicnm_dual_seed(y, z))
    return map(ForwardDiff.value, resd), map(d -> first(ForwardDiff.partials(d)), resd)
end

function __sicnm_g_and_jvp!(gval, jvp, g!::G, y, z) where {G}
    yd = __sicnm_dual_seed(y, z)
    resd = similar(yd)
    g!(resd, yd)
    @. gval = ForwardDiff.value(resd)
    @. jvp = first(ForwardDiff.partials(resd))
    return nothing
end

# Shared `SICNM` setup for `__solve` and `__init`: builds the extended DAE ODE
# problem whose continuous-Newton flow drives `g(y) = 0`, along with the
# termination callback based on the residual `g`. Mirrors `__dynamicss_ode_setup`.
function __sicnm_ode_setup(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::SICNM;
        abstol, reltol, odesolve_kwargs, termination_condition, kwargs...
    )
    prob.u0 isa AbstractVector ||
        throw(ArgumentError("SICNM currently only supports `AbstractVector` initial conditions"))
    tspan = __get_tspan(prob.u0, alg)
    iip = isinplace(prob)
    p = prob.p
    t0 = first(tspan)

    g = if prob isa SteadyStateProblem
        iip ? ((res, y) -> prob.f(res, y, p, t0)) : (y -> prob.f(y, p, t0))
    elseif prob isa SciMLBase.HomotopyProblem
        # The `λ = λspan[2]` end of the homotopy is the actual system.
        λ1 = last(prob.λspan)
        fnl = NonlinearSolveBase.get_raw_f(SciMLBase.unwrapped_f(prob.f.f))
        iip ? ((res, y) -> fnl(res, y, p, λ1)) : (y -> fnl(y, p, λ1))
    elseif prob isa NonlinearProblem
        # AutoSpecialize wraps `prob.f` in FunctionWrappers compiled only for the
        # standard solver dual types, which cannot accept the SICNM JVP duals, so
        # unwrap down to the raw user function
        fnl = NonlinearSolveBase.get_raw_f(SciMLBase.unwrapped_f(prob.f.f))
        iip ? ((res, y) -> fnl(res, y, p)) : (y -> fnl(y, p))
    end

    # consistent initialization: z₀ = -J(y₀)⁻¹ g(y₀), solved with LinearSolve.jl
    y0 = float.(prob.u0)
    n = length(y0)
    if iip
        g0 = similar(y0)
        g(g0, y0)
        J0 = ForwardDiff.jacobian((res, y) -> g(res, y), similar(g0), y0)
    else
        g0 = g(y0)
        J0 = ForwardDiff.jacobian(g, y0)
    end
    z0 = solve(LinearProblem(J0, -g0), alg.linsolve).u
    u0 = vcat(y0, z0)
    T = eltype(u0)

    # extended DAE:  M [ẏ; ż] = [z; J(y)z + g(y)],  M = diag(I, 0)
    mass_matrix = Diagonal(vcat(fill(one(T), n), fill(zero(T), n)))
    fext = if iip
        (du, u, p_, t) -> begin
            y = view(u, 1:n)
            z = view(u, (n + 1):(2n))
            copyto!(view(du, 1:n), z)
            gval = similar(u, n)
            jvp = view(du, (n + 1):(2n))
            __sicnm_g_and_jvp!(gval, jvp, g, y, z)
            jvp .+= gval
            return nothing
        end
    else
        (u, p_, t) -> begin
            y = view(u, 1:n)
            z = view(u, (n + 1):(2n))
            gval, jvp = __sicnm_g_and_jvp(g, y, z)
            return vcat(z, jvp .+ gval)
        end
    end

    # termination is based on the nonlinear residual g(y), not on du of the DAE
    tc_cache = init(prob, termination_condition, g0, y0; abstol, reltol)
    abstol = NonlinearSolveBase.get_abstol(tc_cache)
    reltol = NonlinearSolveBase.get_reltol(tc_cache)

    gbuf = iip ? similar(g0) : nothing
    function terminate_function(u, t, integrator)
        y = view(u, 1:n)
        gval = if iip
            g(gbuf, y)
            gbuf
        else
            g(y)
        end
        return tc_cache(gval, y, view(integrator.uprev, 1:n), t)
    end

    callback = TerminateSteadyState(
        abstol, reltol, terminate_function;
        wrap_test = Val(false)
    )

    haskey(kwargs, :callback) && (callback = CallbackSet(callback, kwargs[:callback]))
    haskey(odesolve_kwargs, :callback) &&
        (callback = CallbackSet(callback, odesolve_kwargs[:callback]))
    kwargs = pairs(__without_verbose(kwargs))

    # The transient trajectory of the continuous-Newton flow is irrelevant — only
    # the steady state (where g(y) = 0) matters, and it is pinned down by the
    # residual-based termination callback, not by the accuracy of the ODE solve.
    # So the ODE integration uses a loose default tolerance (as in the reference
    # SICNM implementation, which steps at Atol = Rtol = 0.1), which lets the
    # stiffly accurate solver take large damping steps toward equilibrium. Tying
    # the ODE tolerance to the tight residual tolerance instead would force an
    # accurate transient and defeat the purpose (an order of magnitude more work).
    # `odesolve_kwargs` still overrides these when a user wants finer control.
    Tt = real(eltype(u0))
    ode_abstol = convert(Tt, 1 // 10)
    ode_reltol = convert(Tt, 1 // 10)

    odefun = SciMLBase.ODEFunction{iip, SciMLBase.FullSpecialize}(fext; mass_matrix)
    odeprob = ODEProblem{iip}(odefun, u0, tspan, p)
    run_kwargs = pairs(__without_verbose(kwargs))

    return (;
        odeprob, tc_cache, n, g, gbuf, iip,
        ode_abstol, ode_reltol, callback, run_kwargs,
    )
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::SICNM,
        args...; abstol = 1.0e-8, reltol = 1.0e-6, odesolve_kwargs = (;),
        save_idxs = nothing,
        termination_condition = NonlinearSolveBase.AbsNormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...
    )
    lp = __scc_lowering(prob)
    lp !== nothing && return __solve_scc_lowering(
        prob, lp, alg, args...; abstol, reltol, odesolve_kwargs,
        termination_condition, alias, save_idxs, kwargs...
    )

    setup = __sicnm_ode_setup(
        prob, alg; abstol, reltol, odesolve_kwargs, termination_condition, kwargs...
    )
    odesol = solve(
        setup.odeprob, alg.alg, args...; abstol = setup.ode_abstol,
        reltol = setup.ode_reltol, setup.run_kwargs..., odesolve_kwargs...,
        setup.callback, save_end = true
    )
    return __steady_state_solution(prob, alg, setup, odesol, save_idxs)
end

function __steady_state_solution(prob, alg::SICNM, setup, odesol, save_idxs)
    u, retcode = __sicnm_result(setup.tc_cache, odesol, setup.n)
    resid = if setup.iip
        setup.g(setup.gbuf, u)
        setup.gbuf
    else
        setup.g(u)
    end

    if save_idxs !== nothing
        u = u[save_idxs]
        resid = resid[save_idxs]
    end

    return SciMLBase.build_solution(
        prob, SICNM(odesol.alg, alg.tspan, alg.linsolve), u, resid;
        retcode, odesol.stats, original = odesol
    )
end

function SciMLBase.__init(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::SICNM,
        args...; abstol = 1.0e-8, reltol = 1.0e-6, odesolve_kwargs = (;),
        save_idxs = nothing,
        termination_condition = NonlinearSolveBase.AbsNormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...
    )
    lp = __scc_lowering(prob)
    lp !== nothing && return SteadyStateSCCCache(
        prob, lp, alg, args, (;
            abstol, reltol, odesolve_kwargs, termination_condition, alias,
            save_idxs, kwargs...,
        )
    )

    setup = __sicnm_ode_setup(
        prob, alg; abstol, reltol, odesolve_kwargs, termination_condition, kwargs...
    )
    integrator = init(
        setup.odeprob, alg.alg, args...; abstol = setup.ode_abstol,
        reltol = setup.ode_reltol, setup.run_kwargs..., odesolve_kwargs...,
        setup.callback, save_end = true
    )
    return SteadyStateODECache(prob, alg, integrator, setup, save_idxs)
end

function __sicnm_result(tc_cache, odesol, n)
    u, _, retcode = termination_condition_result(
        tc_cache, last(odesol.u)[1:n], last(odesol.t), odesol.retcode
    )
    return u, retcode
end

function __get_result_from_sol(tc_cache, odesol)
    u, t, retcode = termination_condition_result(
        tc_cache, last(odesol.u), last(odesol.t), odesol.retcode
    )
    du = odesol(t, Val{1})
    return du, u, retcode
end
