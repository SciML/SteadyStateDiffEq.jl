function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem,
                            alg::SteadyStateDiffEqAlgorithm, args...;
                            abstol = 1e-8, kwargs...)
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
        u = reshape(alg.nlsolve(f!, u0, abstol), sizeu)
        resid = similar(u)
        f!(resid, u)
        DiffEqBase.build_solution(prob, alg, u, resid; retcode = :Success)
    else
        error("Algorithm not recognized")
    end
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem,
                            alg::DynamicSS, args...; save_everystep = false,
                            save_start = false, save_idxs = nothing, kwargs...)
    tspan = alg.tspan isa Tuple ? alg.tspan :
            convert.(real(eltype(prob.u0)), (zero(alg.tspan), alg.tspan))
    if typeof(prob) <: SteadyStateProblem
        f = prob.f
    elseif typeof(prob) <: NonlinearProblem
        if isinplace(prob)
            f = (du, u, p, t) -> prob.f(du, u, p)
        else
            f = (u, p, t) -> prob.f(u, p)
        end
    end

    _prob = ODEProblem(f, prob.u0, tspan, prob.p)
    sol = solve(_prob, alg.alg, args...; kwargs...,
                callback = TerminateSteadyState(alg.abstol, alg.reltol),
                save_everystep = save_everystep, save_start = save_start)
    if isinplace(prob)
        du = similar(sol.u[end])
        f(du, sol.u[end], prob.p, sol.t[end])
    else
        du = f(sol.u[end], prob.p, sol.t[end])
    end
    function array_condition()
        all(abs(d) <= abstol || abs(d) <= reltol * abs(u)
            for (d, abstol, reltol, u)
                in zip(du, Iterators.cycle(alg.abstol), Iterators.cycle(alg.reltol),
                       sol.u[end]))
    end
    function broadcast_condition()
        all((abs.(du) .<= alg.abstol) .| (abs.(du) .<= alg.reltol .* abs.(sol.u[end])))
    end

    if save_idxs !== nothing
        u = sol.u[end][save_idxs]
        du = du[save_idxs]
    else
        u = sol.u[end]
    end

    if sol.retcode == :Terminated &&
       (typeof(sol.u[end]) <: Array ? array_condition() : broadcast_condition())
        _sol = DiffEqBase.build_solution(prob, alg, u, du; retcode = :Success)
    else
        _sol = DiffEqBase.build_solution(prob, alg, u, du; retcode = :Failure)
    end
    _sol
end
