function SciMLBase.__solve(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::SSRootfind,
        args...; kwargs...
    )
    nlprob = NonlinearProblem(prob)
    nlsol = solve(nlprob, alg.alg, args...; kwargs...)
    return SciMLBase.build_solution(
        prob, SSRootfind(nlsol.alg), nlsol.u, nlsol.resid;
        nlsol.retcode, nlsol.stats, nlsol.left, nlsol.right, original = nlsol
    )
end

__get_tspan(u0, alg::Union{DynamicSS, SICNM}) = __get_tspan(u0, alg.tspan)
__get_tspan(u0, tspan::Tuple) = tspan
function __get_tspan(u0, tspan::Number)
    return convert.(
        SciMLBase.value(real(eltype(u0))), (SciMLBase.value(zero(tspan)), tspan)
    )
end

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::DynamicSS,
        args...; abstol = 1.0e-8, reltol = 1.0e-6, odesolve_kwargs = (;),
        save_idxs = nothing, termination_condition = NonlinearSolveBase.NormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...
    )
    tspan = __get_tspan(prob.u0, alg)

    f = if prob isa SteadyStateProblem
        prob.f
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
    kwargs = pairs(Base.structdiff((; kwargs...), (; verbose = nothing)))
    # Construct and solve the ODEProblem
    odeprob = ODEProblem{isinplace(prob), true}(f, prob.u0, tspan, prob.p)
    odesol = solve(
        odeprob, alg.alg, args...; abstol, reltol, kwargs...,
        odesolve_kwargs..., callback, save_end = true,
        alias = SciMLBase.ODEAliasSpecifier(;
            alias_p = alias.alias_p,
            alias_f = alias.alias_f, alias_u0 = alias.alias_u0
        )
    )

    resid, u, retcode = __get_result_from_sol(termination_condition, tc_cache, odesol)

    if save_idxs !== nothing
        u = u[save_idxs]
        resid = resid[save_idxs]
    end

    return SciMLBase.build_solution(
        prob, DynamicSS(odesol.alg, alg.tspan), u, resid;
        retcode, odesol.stats, original = odesol
    )
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

function SciMLBase.__solve(
        prob::SciMLBase.AbstractSteadyStateProblem, alg::SICNM,
        args...; abstol = 1.0e-8, reltol = 1.0e-6, odesolve_kwargs = (;),
        save_idxs = nothing,
        termination_condition = NonlinearSolveBase.AbsNormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...
    )
    prob.u0 isa AbstractVector ||
        throw(ArgumentError("SICNM currently only supports `AbstractVector` initial conditions"))
    tspan = __get_tspan(prob.u0, alg)
    iip = isinplace(prob)
    p = prob.p
    t0 = first(tspan)

    g = if prob isa SteadyStateProblem
        iip ? ((res, y) -> prob.f(res, y, p, t0)) : (y -> prob.f(y, p, t0))
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
    kwargs = pairs(Base.structdiff((; kwargs...), (; verbose = nothing)))

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
    odesol = solve(
        odeprob, alg.alg, args...; abstol = ode_abstol, reltol = ode_reltol,
        kwargs..., odesolve_kwargs..., callback, save_end = true
    )

    u, retcode = __sicnm_result(termination_condition, tc_cache, odesol, n)
    resid = if iip
        g(gbuf, u)
        gbuf
    else
        g(u)
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

function __sicnm_result(::AbstractNonlinearTerminationMode, tc_cache, odesol, n)
    u = last(odesol.u)[1:n]
    retcode = ifelse(
        odesol.retcode == ReturnCode.Terminated, ReturnCode.Success,
        ReturnCode.Failure
    )
    return u, retcode
end

function __sicnm_result(::AbstractSafeNonlinearTerminationMode, tc_cache, odesol, n)
    u = last(odesol.u)[1:n]
    retcode_tc = tc_cache.retcode
    retcode = if odesol.retcode == ReturnCode.Terminated
        ifelse(retcode_tc != ReturnCode.Default, retcode_tc, ReturnCode.Success)
    elseif odesol.retcode == ReturnCode.Success
        ReturnCode.Failure
    else
        odesol.retcode
    end
    return u, retcode
end

function __sicnm_result(::AbstractSafeBestNonlinearTerminationMode, tc_cache, odesol, n)
    u = copy(tc_cache.u)
    retcode_tc = tc_cache.retcode
    retcode = if odesol.retcode == ReturnCode.Terminated
        ifelse(retcode_tc != ReturnCode.Default, retcode_tc, ReturnCode.Success)
    elseif odesol.retcode == ReturnCode.Success
        ReturnCode.Failure
    else
        odesol.retcode
    end
    return u, retcode
end

function __get_result_from_sol(::AbstractNonlinearTerminationMode, tc_cache, odesol)
    u, t = last(odesol.u), last(odesol.t)
    du = odesol(t, Val{1})
    return (
        du, u,
        ifelse(
            odesol.retcode == ReturnCode.Terminated, ReturnCode.Success,
            ReturnCode.Failure
        ),
    )
end

function __get_result_from_sol(::AbstractSafeNonlinearTerminationMode, tc_cache, odesol)
    u, t = last(odesol.u), last(odesol.t)
    du = odesol(t, Val{1})
    retcode_tc = tc_cache.retcode

    retcode = if odesol.retcode == ReturnCode.Terminated
        ifelse(retcode_tc != ReturnCode.Default, retcode_tc, ReturnCode.Success)
    elseif odesol.retcode == ReturnCode.Success
        ReturnCode.Failure
    else
        odesol.retcode
    end

    return du, u, retcode
end

function __get_result_from_sol(::AbstractSafeBestNonlinearTerminationMode, tc_cache, odesol)
    u, t = tc_cache.u, only(tc_cache.saved_values)
    du = odesol(t, Val{1})
    retcode_tc = tc_cache.retcode

    retcode = if odesol.retcode == ReturnCode.Terminated
        ifelse(retcode_tc != ReturnCode.Default, retcode_tc, ReturnCode.Success)
    elseif odesol.retcode == ReturnCode.Success
        ReturnCode.Failure
    else
        odesol.retcode
    end

    return du, u, retcode
end
