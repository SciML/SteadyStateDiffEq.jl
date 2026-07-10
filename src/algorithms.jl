abstract type SteadyStateDiffEqAlgorithm <: SciMLBase.AbstractSteadyStateAlgorithm end

"""
    SSRootfind(alg = nothing)

Solve a steady-state problem by converting it to a `NonlinearProblem` and calling a
nonlinear solver.

## Arguments

  - `alg`: the nonlinear solver algorithm passed to `solve`. When `alg === nothing`,
    the default nonlinear solver is selected by the downstream solver package.

## Example

```julia
using SteadyStateDiffEq, NonlinearSolve

sol = solve(prob, SSRootfind(NewtonRaphson()))
```

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

Solve a steady-state problem by evolving the corresponding ODE until the derivative is
close to zero.

`DynamicSS` internally adds a `TerminateSteadyState` callback. The `abstol` and
`reltol` keywords passed to `solve` control the steady-state termination condition. Use
`odesolve_kwargs` to pass separate keyword arguments to the ODE solve.

## Arguments

  - `alg`: the ODE solver algorithm passed to `solve`. When `alg === nothing`, the
    default ODE solver is selected by the downstream solver package.

## Keywords

  - `tspan`: the time span used for the ODE solve. If `tspan` is a number, it is
    equivalent to `(zero(tspan), tspan)`.

## Example

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
