function DiffEqBase.prepare_alg(alg::DynamicSS)
    DynamicSS(DiffEqBase.prepare_alg(alg.alg), alg.abstol, alg.reltol, alg.tspan)
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem,
                            alg::SteadyStateDiffEqAlgorithm, args...;
                            abstol = 1e-8, kwargs...)
    @warn """
    This method is deprecated in favor of using NonlinearSolve.jl. Note that an ODEProblem
    can be converted into a steady state NonlinearProblem via
    `NonlinearProblem(prob::ODEProblem)`. The algorithm `NLSolveJL` as part of the
    SciMLNLSolve.jl set of nonlinear solvers for NonlinearSolve.jl is equivalent to
    SteadyStateDiffEq.jl's default `SSRootfind` (with a few improvements).

    See [the documentation of NonlinearSolve.jl](https://docs.sciml.ai/NonlinearSolve/stable/)
    for more details.
    """

    if typeof(prob.u0) <: Number
        u0 = [prob.u0]
    else
        u0 = vec(deepcopy(prob.u0))
    end

    sizeu = size(prob.u0)
    p = prob.p

    if typeof(prob) <: SteadyStateProblem
        if !isinplace(prob) &&
           (typeof(prob.u0) <: AbstractVector || typeof(prob.u0) <: Number)
            f! = (du, u) -> (du[:] = prob.f(u, p, Inf); nothing)
        elseif !isinplace(prob) && typeof(prob.u0) <: AbstractArray
            f! = (du, u) -> (du[:] = vec(prob.f(reshape(u, sizeu), p, Inf)); nothing)
        elseif typeof(prob.u0) <: AbstractVector
            f! = (du, u) -> (prob.f(du, u, p, Inf); nothing)
        else # Then it's an in-place function on an abstract array
            f! = (du, u) -> (prob.f(reshape(du, sizeu),
                                    reshape(u, sizeu), p, Inf);
                             du = vec(du);
                             nothing)
        end
    elseif typeof(prob) <: NonlinearProblem
        if !isinplace(prob) &&
           (typeof(prob.u0) <: AbstractVector || typeof(prob.u0) <: Number)
            f! = (du, u) -> (du[:] = prob.f(u, p); nothing)
        elseif !isinplace(prob) && typeof(prob.u0) <: AbstractArray
            f! = (du, u) -> (du[:] = vec(prob.f(reshape(u, sizeu), p)); nothing)
        elseif typeof(prob.u0) <: AbstractVector
            f! = (du, u) -> (prob.f(du, u, p); nothing)
        else # Then it's an in-place function on an abstract array
            f! = (du, u) -> (prob.f(reshape(du, sizeu),
                                    reshape(u, sizeu), p);
                             du = vec(du);
                             nothing)
        end
    end

    # du = similar(u)
    # f = (u) -> (f!(du,u); du) # out-of-place version

    if typeof(alg) <: SSRootfind
        original = alg.nlsolve(f!, u0, abstol)
        if typeof(original) <: NLsolve.SolverResults
            u = reshape(original.zero, sizeu)
            resid = similar(u)
            f!(resid, u)
            DiffEqBase.build_solution(prob, alg, u, resid; retcode = ReturnCode.Success,
                                      original = original)
        else
            u = reshape(original, sizeu)
            resid = similar(u)
            f!(resid, u)
            DiffEqBase.build_solution(prob, alg, u, resid; retcode = ReturnCode.Success)
        end
    else
        error("Algorithm not recognized")
    end
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem,
                            alg::DynamicSS, args...; save_everystep = false,
                            save_start = false, save_idxs = nothing, kwargs...)
    tspan = alg.tspan isa Tuple ? alg.tspan :
            convert.(DiffEqBase.value(real(eltype(prob.u0))),
                     (DiffEqBase.value(zero(alg.tspan)), alg.tspan))
    if typeof(prob) <: SteadyStateProblem
        f = prob.f
    elseif typeof(prob) <: NonlinearProblem
        if isinplace(prob)
            f = (du, u, p, t) -> prob.f(du, u, p)
        else
            f = (u, p, t) -> prob.f(u, p)
        end
    end

    mode = SciMLBase._unwrap_val(_get_termination_mode(alg.termination_condition))

    if mode ∈ SAFE_BEST_TERMINATION_MODES && !save_everystep
        throw(ArgumentError("`save_everystep` must be `true` if using `$(mode)` termination condition."))
    end

    storage = mode ∈ SAFE_TERMINATION_MODES ? Dict() : nothing
    callback = TerminateSteadyState(alg.termination_condition.abstol,
                                    alg.termination_condition.reltol,
                                    _get_termination_condition(alg.termination_condition,
                                                               storage))

    _prob = ODEProblem(f, prob.u0, tspan, prob.p)
    sol = solve(_prob, alg.alg, args...; kwargs..., save_everystep, save_start, callback)

    idx = if storage === nothing || !haskey(storage, :best_objective_value_iteration)
        length(sol.u)
    else
        storage[:best_objective_value_iteration]
    end
    u, t = sol.u[idx], sol.t[idx]

    if isinplace(prob)
        du = similar(sol.u[end])
        f(du, u, prob.p, t)
    else
        du = f(u, prob.p, t)
    end

    retcode = sol.retcode == ReturnCode.Terminated &&
              _has_converged(du, u, alg.termination_condition) ? ReturnCode.Success :
              ReturnCode.Failure

    if save_idxs !== nothing
        u = sol.u[idx][save_idxs]
        du = du[save_idxs]
    else
        u = sol.u[idx]
    end

    return DiffEqBase.build_solution(prob, alg, u, du; retcode)
end
