"""
    SCC(; nlalg=NonlinearSolvePolyAlgorithm((NewtonRaphson(), TrustRegion())),
        linalg=nothing, polish=NewtonRaphson(),
        polish_abstol=1e-12, polish_maxiters=10)

Solve steady nonlinear equations in strongly connected component order. ModelingToolkit
problems use their symbolic decomposition, including linear blocks. A numerical problem
without structural information is treated as one block. An `SCCNonlinearProblem` can
also be supplied directly. Parameters and initial states are copied before solving.

`nlalg` solves each nonlinear block, by default trying Newton then trust-region fallback
from the same initial guess; `linalg` selects its linear-block solver. Successful
nonlinear blocks are refined with `polish`, or refinement can be disabled with `nothing`.
`polish_abstol` and `polish_maxiters` control refinement. `maxiters` applies per block.
If refinement fails, retain the first solution that met the requested solve tolerance.
Statistics sum counters reported by the nonlinear solves; linear-block work is not
included in `NLStats`.

Scaling is optional and belongs to `scale(prob)`. If scaled, the same fixed variable
and residual scales are used throughout the solve, including refinement. Inter-block
dependencies are evaluated in original coordinates. Solve tolerances apply to scaled
residuals; use `unscale` to recover original units and symbolic indexing.
"""
@concrete struct SCC <: SteadyStateDiffEqAlgorithm
    nlalg
    linalg
    polish
    polish_abstol
    polish_maxiters::Int
end

SCC(; nlalg = NonlinearSolveBase.NonlinearSolvePolyAlgorithm((NewtonRaphson(), TrustRegion())),
    linalg = nothing, polish = NewtonRaphson(),
    polish_abstol = 1e-12, polish_maxiters = 10) =
    SCC(nlalg, linalg, polish, polish_abstol, polish_maxiters)

SciMLBase.isadaptive(::SCC) = false
SciMLBase.allowscomplex(::SCC) = false

# A steady solution has no time series, but its original ODE model's observed
# functions still take a time argument, including derived-parameter expressions.
const SCCSolution = NonlinearSolution{T, N, U, R, P, A} where {T, N, U, R, P, A <: SCC}
SII.is_time_dependent(sol::SCCSolution) = SII.is_time_dependent(sol.prob)
SII.current_time(sol::SCCSolution) = SII.current_time(sol.prob)

function SciMLBase.solve(prob::SteadyStateProblem, alg::SCC; kwargs...)
    return SciMLBase.__solve(prob, alg; kwargs...)
end

function SciMLBase.__solve(prob::SteadyStateProblem, alg::SCC; kwargs...)
    nlsol = solve(_steady_nonlinear(prob, prob.f.sys), alg; kwargs...)
    original, u = _steady_recover(prob, nlsol, prob.f.sys)
    resid = isinplace(prob) ? similar(u) : prob.f(u, original.p, Inf)
    isinplace(prob) && prob.f(resid, u, original.p, Inf)
    return SciMLBase.build_solution(original, alg, u, resid;
        nlsol.retcode, nlsol.stats, original = nlsol)
end

_scc_decompose(prob, sys) = (prob, nothing, nothing)

function SciMLBase.__solve(prob::NonlinearProblem, alg::SCC;
        abstol = 1e-10, reltol = 0.0, maxiters = 10000,
        alias = SciMLBase.NonlinearAliasSpecifier(), kwargs...)
    f = NonlinearSolveBase.get_raw_f(SciMLBase.unwrapped_f(prob.f.f))
    scaled = f isa ScaledResidual
    base = scaled ? remake(f.prob; u0 = f.variable_scales .* prob.u0, p = prob.p) : prob
    copied = deepcopy(base)
    work, variables, equations = _scc_decompose(copied, copied.f.sys)
    mapping = scaled ? NonlinearScaling(base, f.variable_scales, f.residual_scales) : nothing
    inner = if work isa SciMLBase.SCCNonlinearProblem
        su = scaled ? f.variable_scales[variables] : nothing
        sf = scaled ? f.residual_scales[equations] : nothing
        _scc_sweep(work, alg, su, sf; abstol, reltol, maxiters, kwargs...)
    else
        _scc_nonlinear_block(work, alg, mapping; abstol, reltol, maxiters, kwargs...)
    end
    u = if variables === nothing
        inner.u
    else
        values = similar(base.u0)
        values[variables] = inner.u
        values
    end
    scaled && (u = u ./ f.variable_scales)
    resid = _scaling_residual(prob, u, prob.p)
    return SciMLBase.build_solution(prob, alg, u, resid;
        inner.retcode, inner.stats, original = inner)
end

function _scc_nonlinear_block(prob, alg, mapping;
        abstol, reltol, maxiters, kwargs...)
    work = mapping === nothing ? prob : first(scale(prob;
        variable_scales = mapping.variable_scales, residual_scales = mapping.residual_scales))
    sol = solve(work, alg.nlalg; abstol, reltol, maxiters, kwargs...)
    stats = sol.stats
    if successful_retcode(sol) && alg.polish !== nothing
        refined = solve(remake(work; u0 = sol.u), alg.polish;
            abstol = min(abstol, alg.polish_abstol), reltol = 0.0,
            maxiters = alg.polish_maxiters, kwargs...)
        stats = stats === nothing ? refined.stats :
            refined.stats === nothing ? stats : merge(stats, refined.stats)
        successful_retcode(refined) && (sol = refined)
    end
    mapping === nothing || (sol = unscale(sol, mapping))
    return SciMLBase.build_solution(prob, alg, sol.u, sol.resid;
        sol.retcode, stats, original = sol)
end

function SciMLBase.solve(prob::SciMLBase.SCCNonlinearProblem, alg::SCC;
        abstol = 1e-10, reltol = 0.0, maxiters = 10000, kwargs...)
    sol = _scc_sweep(deepcopy(prob), alg, nothing, nothing;
        abstol, reltol, maxiters, kwargs...)
    return SciMLBase.build_solution(prob, alg, sol.u, sol.resid;
        sol.retcode, sol.stats, original = sol.original)
end

function _scc_sweep(prob, alg, su, sf; abstol, reltol, maxiters, kwargs...)
    sols = []
    u = reduce(vcat, (_scc_initial_state(block) for block in prob.probs))
    resid = fill(NaN, length(u))
    stats = SciMLBase.NLStats(0, 0, 0, 0, 0)
    retcode = ReturnCode.Success
    offset = 0
    for (block, update!) in zip(prob.probs, prob.explicitfuns!)
        indices = (offset + 1):(offset + length(_scc_initial_state(block)))
        update!(block.p, sols)
        sol = if block isa LinearProblem
            linear = remake(block; A = block.A, b = block.b)
            work = su === nothing ? linear : LinearProblem(
                (linear.A .* transpose(su[indices])) ./ sf[indices], linear.b ./ sf[indices])
            result = solve(work, alg.linalg; abstol, reltol, __without_verbose(kwargs)...)
            values = su === nothing ? result.u : result.u .* su[indices]
            residual = linear.A * values - linear.b
            SciMLBase.build_solution(NonlinearProblem(Returns(nothing), values, block.p),
                alg, values, residual; result.retcode)
        else
            mapping = su === nothing ? nothing : NonlinearScaling(block, su[indices], sf[indices])
            _scc_nonlinear_block(block, alg, mapping; abstol, reltol, maxiters, kwargs...)
        end
        push!(sols, sol)
        u[indices] = sol.u
        resid[indices] = sol.resid
        sol.stats isa SciMLBase.NLStats && (stats = merge(stats, sol.stats))
        if !successful_retcode(sol)
            retcode = sol.retcode
            break
        end
        offset += length(indices)
    end
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, stats, original = sols)
end

_scc_initial_state(prob) = prob.u0
_scc_initial_state(prob::LinearProblem) = prob.u0 === nothing ? zero.(prob.b) : prob.u0
