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
    nlsol = solve(nlprob, alg.alg, args...; kwargs...)
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

# `explicitfuns![i]` receives the upstream blocks' trial values as solution-like
# objects: ModelingToolkit's generated functions index `sols[j][k]` while its
# cache copier reads `sols[j].u`.
struct SCCTrialSolution{U}
    u::U
end
Base.getindex(sol::SCCTrialSolution, i) = sol.u[i]
Base.length(sol::SCCTrialSolution) = length(sol.u)

function __scc_block_length(block)
    u_i = state_values(block)
    return u_i === nothing ? length(block.b) : length(u_i)
end

# The vector field for `LinearProblem` blocks is `b - A * u`: `calculate_A_b` in
# ModelingToolkit produces `A * u - b = -expr`, and the same convention gives
# decaying dynamics for directly constructed positive-definite `A * u = b`.
function __scc_block_residual!(resid, u, block::LinearProblem)
    # `remake` recomputes `A` and `b` from the parameter cache that `explicitfun`
    # just updated; for a plain `LinearProblem` it is an identity copy.
    block = remake(block; A = block.A, b = block.b)
    mul!(resid, block.A, u)
    resid .= block.b .- resid
    return resid
end

# The `λ = 1` end of a `HomotopyProblem` block is the actual system.
function __scc_block_residual!(resid, u, block::SciMLBase.HomotopyProblem)
    if isinplace(block)
        block.f(resid, u, block.p, last(block.λspan))
    else
        resid .= block.f(u, block.p, last(block.λspan))
    end
    return resid
end

function __scc_block_residual!(resid, u, block)
    if isinplace(block)
        block.f(resid, u, block.p)
    else
        resid .= block.f(u, block.p)
    end
    return resid
end

function __scc_residual!(du, u, prob, sols)
    base = 0
    for i in eachindex(prob.probs)
        block = prob.probs[i]
        idxs = (base + 1):(base + __scc_block_length(block))
        SciMLBase.invoke_with_despecialized_parameters(
            prob.explicitfuns![i], (block.p, view(sols, 1:(i - 1)))
        )
        __scc_block_residual!(view(du, idxs), view(u, idxs), block)
        sols[i] = SCCTrialSolution(view(u, idxs))
        base = last(idxs)
    end
    return du
end

# ODE right-hand side `du = resid(u)` for `DynamicSS`: the concatenated block
# residuals in SCC order, evaluated against the trial state rather than solved
# block-wise. Upstream trial values reach each block's parameter cache through
# `explicitfuns!` exactly as in the SCC solve.
struct SCCResidualRHS{P, S}
    prob::P
    sols::S
end
function (f::SCCResidualRHS)(du, u, p, t)
    __scc_residual!(du, u, f.prob, f.sols)
    return nothing
end

function __scc_dynamicss_u0(prob)
    u0 = state_values(prob)
    u0 === nothing || return float.(u0)
    # An all-`LinearProblem` SCC problem carries no states; start at zero.
    n = sum(__scc_block_length, prob.probs)
    T = mapreduce(block -> eltype(block.b), promote_type, prob.probs)
    return zeros(T, n)
end

# `SCCNonlinearProblem` has no top-level `u0`/`kwargs`/`f` to feed the generic
# nonlinear solve path, so it cannot reach `__solve` directly. Wrap its residual
# as a `SteadyStateProblem` for the ODE integration and wrap the result back on
# the original problem. The problem is copied since `explicitfuns!` mutate the
# blocks' parameter caches on every residual evaluation.
function SciMLBase.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::DynamicSS,
        args...; kwargs...
    )
    work = deepcopy(prob)
    sols = Vector{SCCTrialSolution}(undef, length(work.probs))
    f = SciMLBase.ODEFunction{true}(SCCResidualRHS(work, sols))
    p = something(parameter_values(work), SciMLBase.NullParameters())
    sssol = SciMLBase.__solve(
        SteadyStateProblem(f, __scc_dynamicss_u0(work), p), alg, args...;
        kwargs...
    )
    return SciMLBase.build_solution(
        prob, sssol.alg, sssol.u, sssol.resid;
        sssol.retcode, sssol.stats, original = sssol
    )
end

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
    kwargs = pairs(__without_verbose(kwargs))
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

    resid, u, retcode = __get_result_from_sol(tc_cache, odesol)

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
    odesol = solve(
        odeprob, alg.alg, args...; abstol = ode_abstol, reltol = ode_reltol,
        kwargs..., odesolve_kwargs..., callback, save_end = true
    )

    u, retcode = __sicnm_result(tc_cache, odesol, n)
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
