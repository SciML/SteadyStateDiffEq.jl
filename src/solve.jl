function SciMLBase.__solve(prob::SciMLBase.AbstractSteadyStateProblem, alg::SSRootfind,
        args...; kwargs...)
    nlprob = NonlinearProblem(prob)
    nlsol = solve(nlprob, alg.alg, args...; kwargs...)
    return SciMLBase.build_solution(prob, SSRootfind(nlsol.alg), nlsol.u, nlsol.resid;
        nlsol.retcode, nlsol.stats, nlsol.left, nlsol.right, original = nlsol)
end

__get_tspan(u0, alg::DynamicSS) = __get_tspan(u0, alg.tspan)
__get_tspan(u0, tspan::Tuple) = tspan
function __get_tspan(u0, tspan::Number)
    return convert.(
        DiffEqBase.value(real(eltype(u0))), (DiffEqBase.value(zero(tspan)), tspan))
end

function SciMLBase.__solve(prob::SciMLBase.AbstractSteadyStateProblem, alg::DynamicSS,
        args...; abstol = 1e-8, reltol = 1e-6, odesolve_kwargs = (;),
        save_idxs = nothing, termination_condition = NonlinearSolveBase.NormTerminationMode(infnorm),
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...)
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

    callback = TerminateSteadyState(abstol, reltol, terminate_function;
        wrap_test = Val(false))

    haskey(kwargs, :callback) && (callback = CallbackSet(callback, kwargs[:callback]))
    haskey(odesolve_kwargs, :callback) &&
        (callback = CallbackSet(callback, odesolve_kwargs[:callback]))
    kwargs = pairs(merge((; kwargs...), haskey(kwargs, :verbose) ? (verbose = true,) : (;)))
    # Construct and solve the ODEProblem
    odeprob = ODEProblem{isinplace(prob), true}(f, prob.u0, tspan, prob.p)
    odesol = solve(odeprob, alg.alg, args...; abstol, reltol, kwargs...,
        odesolve_kwargs..., callback, save_end = true,
        alias = SciMLBase.ODEAliasSpecifier(; alias_p = alias.alias_p,
            alias_f = alias.alias_f, alias_u0 = alias.alias_u0))

    resid, u, retcode = __get_result_from_sol(termination_condition, tc_cache, odesol)

    if save_idxs !== nothing
        u = u[save_idxs]
        resid = resid[save_idxs]
    end

    return SciMLBase.build_solution(prob, DynamicSS(odesol.alg, alg.tspan), u, resid;
        retcode, original = odesol)
end

function __get_result_from_sol(::AbstractNonlinearTerminationMode, tc_cache, odesol)
    u, t = last(odesol.u), last(odesol.t)
    du = odesol(t, Val{1})
    return (du, u,
        ifelse(odesol.retcode == ReturnCode.Terminated, ReturnCode.Success,
            ReturnCode.Failure))
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
    u, t = tc_cache.u, only(DiffEqBase.get_saved_values(tc_cache))
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
