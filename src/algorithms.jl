abstract type SteadyStateDiffEqAlgorithm <: SciMLBase.AbstractSteadyStateAlgorithm end

"""
    SSRootfind(alg = nothing)

Use a Nonlinear Solver to find the steady state. Requires that a nonlinear solver is
given as the first argument.

!!! note

    The default `alg` of `nothing` works only if `NonlinearSolve.jl` is installed and
    loaded.
"""
@concrete struct SSRootfind <: SteadyStateDiffEqAlgorithm
    alg
end

SSRootfind() = SSRootfind(nothing)

"""
    DynamicSS(alg = nothing; tspan = Inf)

Requires that an ODE algorithm is given as the first argument.  The absolute and
relative tolerances specify the termination conditions on the derivative's closeness to
zero.  This internally uses the `TerminateSteadyState` callback from the Callback Library.
The simulated time for which given ODE is solved can be limited by `tspan`.  If `tspan` is
a number, it is equivalent to passing `(zero(tspan), tspan)`.

Example usage:

```julia
using SteadyStateDiffEq, OrdinaryDiffEq
sol = solve(prob, DynamicSS(Tsit5()))

using Sundials
sol = solve(prob, DynamicSS(CVODE_BDF()); dt = 1.0)
```

!!! note

    The default `alg` of `nothing` works only if `DifferentialEquations.jl` is installed and
    loaded.

!!! note

    If you use `CVODE_BDF` you may need to give a starting `dt` via `dt = ....`.
"""
@concrete struct DynamicSS <: SteadyStateDiffEqAlgorithm
    alg
    tspan
end

DynamicSS(alg = nothing; tspan = Inf) = DynamicSS(alg, tspan)

function DiffEqBase.prepare_alg(alg::DynamicSS, u0, p, f)
    return DynamicSS(DiffEqBase.prepare_alg(alg.alg, u0, p, f), alg.tspan)
end

## SciMLBase Trait Definitions
SciMLBase.isadaptive(::SSRootfind) = false

for aType in (:SSRootfind, :DynamicSS),
        op in (
            :isadaptive, :isautodifferentiable, :allows_arbitrary_number_types,
            :allowscomplex,
        )

    op == :isadaptive && aType == :SSRootfind && continue

    @eval function SciMLBase.$(op)(alg::$aType)
        internal_alg = alg.alg
        # Internal Alg is nothing means we will handle everything correctly downstream
        internal_alg === nothing && return true
        !hasmethod(SciMLBase.$(op), Tuple{typeof(internal_alg)}) && return false
        return SciMLBase.$(op)(internal_alg)
    end
end
