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
    # A stored `lowered_problem` (e.g. an `SCCNonlinearProblem`) solves in the
    # lowering's own state ordering, so its solution is expressed on the
    # lowering rather than the steady-state problem.
    solprob = __lowered_problem(prob) === nothing ? prob : nlprob
    return __build_ssrootfind_solution(solprob, nlsol)
end


# Solve-time `u0`/`p` go through `remake`, which chunks `u0` into the blocks and
# enforces `parameters_alias`; the solve then runs on the updated problem.
function __scc_apply_overrides(prob; u0 = nothing, p = nothing, kwargs...)
    if u0 !== nothing || p !== nothing
        prob = remake(prob; u0 = u0 === nothing ? missing : u0, p = p === nothing ? missing : p)
    end
    return prob, kwargs
end

# An SCCNonlinearProblem has no top-level `u0`/`kwargs` fields, so it cannot go
# through the generic AbstractNonlinearProblem solve preprocessing. Forward it
# directly to the wrapped algorithm instead, which dispatches to the SCC solver
# loaded downstream (SCCNonlinearSolve.jl).
function SciMLBase.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::SSRootfind,
        args...; kwargs...
    )
    prob, kwargs = __scc_apply_overrides(prob; kwargs...)
    nlsol = solve(prob, alg.alg, args...; kwargs...)
    return __build_ssrootfind_solution(prob, nlsol)
end

# Evaluate a stateless block's residual once: out-of-place blocks return it,
# in-place blocks write into a `resid_prototype`-sized buffer. `HomotopyProblem`
# residuals take the continuation parameter `λ`, evaluated at its end.
function __scc_stateless_resid(
        block::Union{NonlinearProblem, SciMLBase.HomotopyProblem}, u
    )
    p = parameter_values(block)
    f = block.f
    args = block isa SciMLBase.HomotopyProblem ? (u, p, last(block.λspan)) : (u, p)
    if isinplace(block)
        fbase = f isa SciMLBase.HomotopyNonlinearFunction ? f.f : f
        proto = hasproperty(fbase, :resid_prototype) ? fbase.resid_prototype : nothing
        resid = proto === nothing ? similar(u) : similar(proto)
        f(resid, args...)
        return resid
    end
    return f(args...)
end

# `DynamicSS`/`SICNM` on an SCC problem solve the blocks sequentially
# in SCC order: each block's parameter cache is updated from the upstream
# solutions through `explicitfuns!`, `LinearProblem` blocks are solved directly,
# and the remaining blocks are integrated to steady state on the block
# residual. `explicitfuns!` mutate the block caches, matching the
# `SCCNonlinearSolve` protocol; each upstream solution is stripped to a plain
# solution so generated explicit functions can index `sols[j][k]`/`sols[j].u`.
function __solve_scc_blocks(
        prob::Union{SciMLBase.SCCNonlinearProblem, SciMLBase.SCCSteadyStateProblem},
        alg::Union{DynamicSS, SICNM}, args...; kwargs...
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
        elseif state_values(block) !== nothing && isempty(state_values(block))
            # A stateless block carries no integration; its residual must be
            # identically empty, which is verified by evaluating it once.
            u = copy(state_values(block))
            resid = __scc_stateless_resid(block, u)
            isempty(resid) || throw(
                DimensionMismatch(
                    "SCC block $i has no states but a residual of length $(length(resid))"
                )
            )
            SciMLBase.build_solution(
                block, nothing, u, resid; retcode = ReturnCode.Success
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

    u = isempty(sols) ? Float64[] : vcat(map(sol -> sol.u, sols)...)
    resid = isempty(sols) ? Float64[] : vcat(map(sol -> sol.resid, sols)...)
    retcode = SciMLBase.ReturnCode.Success
    for s in sols
        if !SciMLBase.successful_retcode(s)
            retcode = s.retcode
            break
        end
    end
    return SciMLBase.build_solution(prob, alg, u, resid; retcode, original = sols)
end

function SciMLBase.solve(
        prob::SciMLBase.SCCNonlinearProblem, alg::Union{DynamicSS, SICNM},
        args...; kwargs...
    )
    prob, kwargs = __scc_apply_overrides(prob; kwargs...)
    return __solve_scc_blocks(prob, alg, args...; kwargs...)
end

function SciMLBase.solve(
        prob::SciMLBase.SCCSteadyStateProblem, args...;
        save_idxs = nothing, kwargs...
    )
    prob, kwargs = __scc_apply_overrides(prob; kwargs...)
    alg = isempty(args) ? nothing : first(args)
    alg === nothing && (alg = get(kwargs, :alg, nothing))
    kwargs = (; (n => v for (n, v) in pairs(kwargs) if n !== :alg)...)
    rest = isempty(args) ? () : Base.tail(args)
    sol = __solve_scc_steady_state(prob, alg, rest...; kwargs...)
    return __scc_save_idxs(prob, sol, save_idxs)
end

function __solve_scc_steady_state(
        prob::SciMLBase.SCCSteadyStateProblem, alg::Union{DynamicSS, SICNM},
        args...; kwargs...
    )
    return __solve_scc_blocks(prob, alg, args...; kwargs...)
end

function __solve_scc_steady_state(
        prob::SciMLBase.SCCSteadyStateProblem, alg::SSRootfind,
        args...; kwargs...
    )
    nlsol = solve(SCCNonlinearProblem(prob), alg.alg, args...; kwargs...)
    return SciMLBase.build_solution(
        prob, SSRootfind(nlsol.alg), nlsol.u, nlsol.resid;
        nlsol.retcode, nlsol.stats, nlsol.left, nlsol.right, original = nlsol
    )
end

function __solve_scc_steady_state(
        prob::SciMLBase.SCCSteadyStateProblem, ::Nothing,
        args...; kwargs...
    )
    nlsol = solve(SCCNonlinearProblem(prob), args...; kwargs...)
    return SciMLBase.build_solution(
        prob, nlsol.alg, nlsol.u, nlsol.resid;
        nlsol.retcode, nlsol.stats, nlsol.left, nlsol.right, original = nlsol
    )
end

function __solve_scc_steady_state(
        prob::SciMLBase.SCCSteadyStateProblem, alg,
        args...; kwargs...
    )
    nlsol = solve(SCCNonlinearProblem(prob), alg, args...; kwargs...)
    return SciMLBase.build_solution(
        prob, nlsol.alg, nlsol.u, nlsol.resid;
        nlsol.retcode, nlsol.stats, nlsol.left, nlsol.right, original = nlsol
    )
end

# Index provider for the saved-state layout of an `SCCSteadyStateProblem`:
# `idxs` are the positions of the saved states in the full compressed layout.
# Parameters and independent variables forward to the wrapped `sys`; observed
# quantities are unavailable since they need the full state.
struct SCCSavedStates{S}
    sys::S
    idxs::Vector{Int}
end

SymbolicIndexingInterface.symbolic_container(w::SCCSavedStates) = w.sys

function SymbolicIndexingInterface.variable_index(w::SCCSavedStates, sym)
    idx = variable_index(w.sys, sym)
    idx === nothing && return nothing
    if idx isa AbstractArray
        out = map(i -> something(findfirst(==(i), w.idxs), 0), idx)
        any(iszero, out) && return nothing
        return out
    end
    return findfirst(==(idx), w.idxs)
end

function SymbolicIndexingInterface.is_variable(w::SCCSavedStates, sym)
    return variable_index(w, sym) !== nothing
end

function SymbolicIndexingInterface.variable_symbols(w::SCCSavedStates)
    return variable_symbols(w.sys)[w.idxs]
end

SymbolicIndexingInterface.all_variable_symbols(w::SCCSavedStates) =
    variable_symbols(w)
SymbolicIndexingInterface.is_observed(::SCCSavedStates, sym) = false

function SymbolicIndexingInterface.all_symbols(w::SCCSavedStates)
    return union(
        variable_symbols(w), SymbolicIndexingInterface.parameter_symbols(w.sys),
        SymbolicIndexingInterface.independent_variable_symbols(w.sys)
    )
end

# A live view of an `SCCSteadyStateProblem` whose state layout is the
# saved-state selection: `idxs` are the saved positions in `prob`'s compressed
# layout. `state_values`, `solve` and `remake` all act in the saved layout;
# the view shares the wrapped problem's blocks, parameters and index provider,
# narrowed to the saved states through `SCCSavedStates`.
struct SCCSavedProblem{
        uType, P <: SciMLBase.SCCSteadyStateProblem, I, F,
    } <: SciMLBase.AbstractSteadyStateProblem{uType, false}
    prob::P
    idxs::I
    f::F
end

function SCCSavedProblem(prob::SciMLBase.SCCSteadyStateProblem, idxs)
    f = if prob.f.sys === nothing
        SciMLBase.NonlinearFunction{false}(Returns(nothing))
    else
        SciMLBase.NonlinearFunction{false}(
            Returns(nothing); sys = SCCSavedStates(prob.f.sys, vec(collect(idxs)))
        )
    end
    uType = state_values(prob) === nothing ? Nothing : typeof(state_values(prob)[idxs])
    return SCCSavedProblem{uType, typeof(prob), typeof(idxs), typeof(f)}(
        prob, idxs, f
    )
end

__saved_positions(w::SCCSavedProblem) = w.idxs isa Integer ? (w.idxs,) : w.idxs

function SymbolicIndexingInterface.state_values(w::SCCSavedProblem)
    states = state_values(w.prob)
    return states === nothing ? nothing : states[w.idxs]
end
SymbolicIndexingInterface.parameter_values(w::SCCSavedProblem) =
    parameter_values(w.prob)
SymbolicIndexingInterface.symbolic_container(w::SCCSavedProblem) = w.f
function SymbolicIndexingInterface.set_state!(w::SCCSavedProblem, val, idx)
    return set_state!(w.prob, val, __saved_positions(w)[idx])
end
function SymbolicIndexingInterface.set_parameter!(w::SCCSavedProblem, val, idx)
    return set_parameter!(w.prob, val, idx)
end
function SymbolicIndexingInterface.finalize_parameters_hook!(w::SCCSavedProblem, sym)
    return finalize_parameters_hook!(w.prob, sym)
end

function SciMLBase.solve(
        w::SCCSavedProblem, args...; save_idxs = nothing, u0 = nothing, p = nothing,
        kwargs...
    )
    if u0 !== nothing || p !== nothing
        w = remake(w; u0 = u0 === nothing ? missing : u0, p = p === nothing ? missing : p)
    end
    composed = if save_idxs === nothing
        w.idxs
    else
        __saved_positions(w)[__resolve_save_idxs(w, save_idxs)]
    end
    return solve(w.prob, args...; save_idxs = composed, kwargs...)
end

function SciMLBase.remake(
        w::SCCSavedProblem; u0 = missing, p = missing, kwargs...
    )
    if u0 isa Number || (u0 isa AbstractArray && eltype(u0) <: Number)
        old = state_values(w.prob)
        T = promote_type(
            old === nothing ? Union{} : eltype(old),
            u0 isa Number ? typeof(u0) : eltype(u0)
        )
        full = if old === nothing
            # A guess-less problem has no values at the unsaved positions;
            # zeros match the guess `LinearSolve` uses when none is given.
            zeros(T, sum(size(b.A, 2) for b in w.prob.probs))
        else
            copyto!(Vector{T}(undef, length(old)), old)
        end
        positions = collect(__saved_positions(w))
        if u0 isa Number
            full[positions] .= u0
        else
            full[positions] = u0
        end
        u0 = full
    end
    return SCCSavedProblem(remake(w.prob; u0, p, kwargs...), w.idxs)
end

# `save_idxs` is a selection in the compressed state layout: integers pass
# through, symbolic entries are resolved through `prob`'s index provider and
# must be states of it. A non-array `save_idxs` is a single entry and keeps the
# scalar shape of `u`/`resid`, matching `DynamicSS` on a `SteadyStateProblem`.
function __resolve_save_idxs(prob, save_idxs)
    function resolve(s)
        symbolic_type(s) === NotSymbolic() && return s
        idx = variable_index(prob, s)
        idx === nothing && throw(
            ArgumentError("`save_idxs` entry $s is not a state of the problem")
        )
        return idx
    end
    return save_idxs isa AbstractArray ? map(resolve, save_idxs) : resolve(save_idxs)
end

function __scc_save_idxs(
        prob::SciMLBase.SCCSteadyStateProblem, sol, save_idxs
    )
    save_idxs === nothing && return sol
    idxs = __resolve_save_idxs(prob, save_idxs)
    return SciMLBase.build_solution(
        SCCSavedProblem(prob, idxs), sol.alg, sol.u[idxs], sol.resid[idxs];
        retcode = sol.retcode, stats = sol.stats, left = sol.left,
        right = sol.right, original = sol
    )
end

# `SteadyStateProblem.lowered_problem` is a SciMLBase 3.x field; reads go
# through `hasfield` so this package also loads where the field is gone.
function __lowered_problem(prob)
    return prob isa SteadyStateProblem && hasfield(typeof(prob), :lowered_problem) ?
        prob.lowered_problem : nothing
end

# A `SteadyStateProblem` that records an `SCCNonlinearProblem` lowering is
# solved block-sequentially in the lowering's ordering instead of one
# monolithic integration. Returns `nothing` when there is no SCC lowering.
function __solve_scc_lowering(
        prob, alg, args...; save_idxs = nothing, kwargs...
    )
    lp = __lowered_problem(prob)
    lp === nothing && return nothing
    lp isa SciMLBase.AbstractSciMLProblem || (lp = lp(prob))
    lp isa SciMLBase.SCCNonlinearProblem || return nothing
    sccsol = solve(lp, alg, args...; kwargs...)
    save_idxs === nothing && return sccsol
    return SciMLBase.build_solution(
        lp, sccsol.alg, sccsol.u[save_idxs], sccsol.resid[save_idxs];
        retcode = sccsol.retcode, original = sccsol
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
    sccsol = __solve_scc_lowering(
        prob, alg, args...; abstol, reltol, odesolve_kwargs,
        termination_condition, alias, save_idxs, kwargs...
    )
    sccsol !== nothing && return sccsol

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
    sccsol = __solve_scc_lowering(
        prob, alg, args...; abstol, reltol, odesolve_kwargs,
        termination_condition, alias, save_idxs, kwargs...
    )
    sccsol !== nothing && return sccsol

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
