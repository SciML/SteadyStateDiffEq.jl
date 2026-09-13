"""
    NonlinearScaling

Map returned by [`scale`](@ref). `variable_scales` converts scaled states to original
states; `residual_scales` converts scaled residuals to original residuals. `prob` is
the original problem. Scales are fixed for a solve and must not be mutated.
"""
struct NonlinearScaling{P, U, F}
    prob::P
    variable_scales::U
    residual_scales::F
end

# Retain the transformation independently of symbolic indexing: SCC decomposition
# needs the original equations, while the scaled problem has numerical coordinates.
struct ScaledResidual{IIP, P, U, F}
    prob::P
    variable_scales::U
    residual_scales::F
end

function (f::ScaledResidual{true})(res, z, p)
    f.prob.f(res, f.variable_scales .* z, p)
    res ./= f.residual_scales
    return nothing
end

(f::ScaledResidual{false})(z, p) =
    f.prob.f(f.variable_scales .* z, p) ./ f.residual_scales

"""
    scaled_prob, scaling = scale(prob::NonlinearProblem;
        variable_scales=nothing, residual_scales=nothing)

Prototype diagonal scaling for real scalar or vector nonlinear problems. Return an
ordinary `NonlinearProblem` with `z₀ = u₀ ./ variable_scales` and residual
`G(z, p) = F(variable_scales .* z, p) ./ residual_scales`, without changing `prob`.

By default, variable scales are `abs.(u₀)`, replacing zeros with one. Residual scale
`i` is the maximum of `abs(Fᵢ(u₀))`, `maximum(abs.(J[i, :]) .* variable_scales)`,
and `sqrt(eps(T))`. Use a supplied analytic Jacobian or ForwardDiff to compute `J`.
Explicit positive, finite scales override these rules; explicit residual scales
avoid computing a Jacobian for scale selection. Scale arrays must match their
corresponding state or residual arrays.

Both in-place and out-of-place residuals and analytic Jacobians are supported.
The scaled problem uses numerical coordinates; original symbolic indexing is
restored by `unscale`. Other custom derivative hooks are not carried over. This
prototype rejects bounds, callbacks, and initialization systems, whose coordinate
transformations require separate handling. Solve tolerances apply to `G`, not `F`.
"""
function scale(prob::NonlinearProblem; variable_scales = nothing, residual_scales = nothing)
    _scaling_check_value(prob.u0, "Initial state")
    for kw in (:lb, :ub, :callback)
        haskey(prob.kwargs, kw) && throw(ArgumentError("scale does not yet transform `$kw`"))
    end
    for bound in (:lb, :ub)
        if hasproperty(prob, bound) && getproperty(prob, bound) !== nothing
            throw(ArgumentError("scale does not yet transform `$bound`"))
        end
    end
    prob.f.initialization_data === nothing ||
        throw(ArgumentError("scale requires a problem without an initialization system"))
    u0 = float.(prob.u0)
    f0 = _scaling_residual(prob, u0, prob.p)
    _scaling_check_value(f0, "Initial residual")
    (u0 isa Number) == (f0 isa Number) ||
        throw(ArgumentError("State and residual must both be scalars or both be vectors"))
    su = variable_scales === nothing ? map(x -> iszero(x) ? one(x) : abs(x), u0) :
        _scaling_check_scales(variable_scales, u0, "variable_scales")
    sf = if residual_scales === nothing
        J = _scaling_jacobian(prob, u0, f0)
        sensitivity = u0 isa Number ? abs(J) * su :
            vec(maximum(abs.(J) .* transpose(su); dims = 2))
        max.(abs.(f0), sensitivity, sqrt(eps(eltype(float.(f0)))))
    else
        _scaling_check_scales(residual_scales, f0, "residual_scales")
    end
    _scaling_check_scales(sf, f0, "residual_scales")
    scaled_f = ScaledResidual{isinplace(prob), typeof(prob), typeof(su), typeof(sf)}(prob, su, sf)
    jac = if prob.f.jac === nothing
        nothing
    elseif isinplace(prob)
        function (J, z, p)
            prob.f.jac(J, su .* z, p)
            J .*= transpose(su)
            J ./= sf
            return nothing
        end
    else
        (z, p) -> prob.f.jac(su .* z, p) .* transpose(su) ./ sf
    end
    f = SciMLBase.NonlinearFunction{isinplace(prob)}(
        scaled_f; jac, jac_prototype = deepcopy(prob.f.jac_prototype),
        sparsity = prob.f.sparsity, colorvec = prob.f.colorvec,
        resid_prototype = prob.f.resid_prototype === nothing ? nothing :
            prob.f.resid_prototype ./ sf
    )
    scaled = NonlinearProblem{isinplace(prob)}(
        f, u0 ./ su, prob.p, prob.problem_type; prob.kwargs...
    )
    return scaled, NonlinearScaling(prob, su, sf)
end

struct SteadyStateScaling{P, S}
    prob::P
    nonlinear_scaling::S
end

"""
    scaled_prob, scaling = scale(prob::SteadyStateProblem; kwargs...)

Form the steady nonlinear equations and apply [`scale`](@ref). With ModelingToolkit
loaded, compile the retained source model at steady state and transfer the problem's
current states and parameters by symbolic identity. This preserves the SCC structure
and reconstructs eliminated states when calling `unscale`. No initialization solve
or time integration is performed. Conservation constraints must be part of the model.

The returned problem is an ordinary `NonlinearProblem`. Scale overrides refer to
its steady nonlinear coordinates, which can differ from the dynamic state vector.
"""
function scale(prob::SteadyStateProblem; kwargs...)
    nlprob = _steady_nonlinear(prob, prob.f.sys)
    scaled, mapping = scale(nlprob; kwargs...)
    return scaled, SteadyStateScaling(prob, mapping)
end

_steady_nonlinear(prob, sys) = NonlinearProblem(prob)

function unscale(sol::NonlinearSolution, mapping::SteadyStateScaling)
    nlsol = unscale(sol, mapping.nonlinear_scaling)
    prob, u = _steady_recover(mapping.prob, nlsol, mapping.prob.f.sys)
    resid = if isinplace(prob)
        r = similar(u)
        prob.f(r, u, prob.p, Inf)
        r
    else
        prob.f(u, prob.p, Inf)
    end
    return SciMLBase.build_solution(prob, sol.alg, u, resid;
        sol.retcode, sol.stats, original = sol)
end

function _steady_recover(prob, nlsol, sys)
    original = prob.p === nlsol.prob.p ? prob : remake(prob; p = nlsol.prob.p)
    return original, nlsol.u
end

"""
    unscale(sol::NonlinearSolution, scaling::NonlinearScaling)

Recover the original state coordinates and recompute the original residual. Return
a `NonlinearSolution` with the original problem's symbolic information, the inner
solver's return code and statistics, and `original = sol`. If the scaled problem's
parameters were changed with `remake`, carry those parameters into the returned
problem. Success still refers to the scaled solver's convergence criterion.
"""
function unscale(sol::NonlinearSolution, scaling::NonlinearScaling)
    prob = sol.prob.p === scaling.prob.p ? scaling.prob :
        remake(scaling.prob; p = sol.prob.p)
    u = scaling.variable_scales .* sol.u
    resid = _scaling_residual(prob, u, sol.prob.p)
    return SciMLBase.build_solution(
        prob, sol.alg, u, resid; sol.retcode, sol.stats, original = sol
    )
end

function _scaling_residual(prob, u, p)
    isinplace(prob) || return prob.f(u, p)
    prototype = prob.f.resid_prototype
    res = prototype === nothing ? similar(u) : similar(prototype, eltype(u))
    prob.f(res, u, p)
    return res
end

function _scaling_jacobian(prob, u, f0)
    if prob.f.jac !== nothing
        isinplace(prob) || return prob.f.jac(u, prob.p)
        J = prob.f.jac_prototype === nothing ? zeros(eltype(u), length(f0), length(u)) :
            copy(prob.f.jac_prototype)
        prob.f.jac(J, u, prob.p)
        return J
    end
    f = u -> _scaling_residual(prob, u, prob.p)
    return u isa Number ? ForwardDiff.derivative(f, u) : ForwardDiff.jacobian(f, u)
end

function _scaling_check_value(x, name)
    (x isa Real || x isa AbstractVector{<:Real}) ||
        throw(ArgumentError("$name must be a real scalar or vector"))
    isempty(x) && throw(ArgumentError("$name must not be empty"))
    all(isfinite, x) || throw(ArgumentError("$name must be finite"))
    return nothing
end

function _scaling_check_scales(s, ref, name)
    _scaling_check_value(s, name)
    size(s) == size(ref) || throw(DimensionMismatch("$name must match its state or residual"))
    all(x -> x > 0, s) || throw(ArgumentError("$name must be positive"))
    return float.(copy(s))
end
