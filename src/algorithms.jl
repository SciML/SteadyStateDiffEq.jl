abstract type SteadyStateDiffEqAlgorithm <: SciMLBase.AbstractSteadyStateAlgorithm end

"""
    SSRootfind(alg = nothing)

Solve a steady-state problem by converting it to a `NonlinearProblem` and calling a
nonlinear solver.

## Arguments

  - `alg`: the nonlinear solver algorithm passed to `solve`. When `alg === nothing`,
    the default nonlinear solver is selected by the downstream solver package.

## Fields

  - `alg`: nonlinear solver algorithm, or `nothing` to request downstream default
    selection.

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

## Keyword Arguments

  - `tspan`: the time span used for the ODE solve. If `tspan` is a number, it is
    equivalent to `(zero(tspan), tspan)`.

## Fields

  - `alg`: ODE solver algorithm, or `nothing` to request downstream default selection.
  - `tspan`: time span passed to the internal ODE solve.

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

"""
    SICNM(alg; tspan = Inf)

The Semi-Implicit Continuous Newton Method for solving a steady-state (or nonlinear)
problem `0 = g(y)`. The problem is converted into the differential-algebraic system

```math
\\begin{aligned}
\\dot{y} &= z\\\\
0 &= J(y) z + g(y)
\\end{aligned}
```

with the consistent initialization ``z(0) = -J(y_0)^{-1} g(y_0)``, where ``J`` is the
Jacobian of ``g``. The exact solution of this DAE is the continuous Newton flow, along
which ``g(y(t)) = g(y_0) e^{-t}``, so integrating to steady state solves the nonlinear
system. Because the DAE is integrated with an implicitly stable method with step size
control, this is significantly more robust than Newton's method (which corresponds to
integrating the same flow with explicit Euler steps), at the cost of more work per
step. It is particularly effective on ill-conditioned problems such as power flow
equations where Newton's method diverges.

`SICNM` internally adds a callback that terminates the integration when the nonlinear
residual `g(y)` satisfies the termination condition. The `abstol` and `reltol` keywords
passed to `solve` control this termination condition; use `odesolve_kwargs` to pass
separate keyword arguments to the ODE solve.

The Jacobian-vector products `J(y) z` required by the DAE are computed via ForwardDiff
dual numbers together with the residual evaluation, so no full Jacobian of `g` is ever
materialized inside the right-hand side (a single dense Jacobian is used for the
initialization of `z(0)`).

## Arguments

  - `alg`: the ODE solver algorithm used to integrate the DAE. It must support mass
    matrices and should be stiffly accurate and L-stable. [`Rodas3d`](https://docs.sciml.ai/DiffEqDocs/stable/solvers/dae_solve/)
    from OrdinaryDiffEqRosenbrock.jl was constructed specifically for this method and is
    the recommended choice; other Rosenbrock methods such as `Rodas4` or `Rodas5P` also
    work well.

## Keywords

  - `tspan`: the time span used for the ODE solve. If `tspan` is a number, it is
    equivalent to `(zero(tspan), tspan)`. Since the residual decays like ``e^{-t}``
    along the exact flow, the default `Inf` combined with the termination callback is
    normally appropriate.

## Example

```julia
using SteadyStateDiffEq, OrdinaryDiffEqRosenbrock

sol = solve(prob, SICNM(Rodas3d()))
```

## References

Yu, R., Gu, W., Xu, Y., Lu, S. (2024). Semi-implicit Continuous Newton Method for Power
Flow Analysis. arXiv:2312.02809. https://arxiv.org/abs/2312.02809
"""
@concrete struct SICNM <: SteadyStateDiffEqAlgorithm
    alg
    tspan
end

SICNM(alg = nothing; tspan = Inf) = SICNM(alg, tspan)

function DiffEqBase.prepare_alg(alg::SICNM, u0, p, f)
    return SICNM(DiffEqBase.prepare_alg(alg.alg, u0, p, f), alg.tspan)
end

## SciMLBase Trait Definitions
SciMLBase.isadaptive(::SSRootfind) = false

# SICNM computes Jacobian-vector products with ForwardDiff, which requires real numbers
SciMLBase.allowscomplex(::SICNM) = false

for aType in (:SSRootfind, :DynamicSS, :SICNM),
        op in (
            :isadaptive, :isautodifferentiable, :allows_arbitrary_number_types,
            :allowscomplex,
        )

    op == :isadaptive && aType == :SSRootfind && continue
    op == :allowscomplex && aType == :SICNM && continue

    @eval function SciMLBase.$(op)(alg::$aType)
        internal_alg = alg.alg
        # Internal Alg is nothing means we will handle everything correctly downstream
        internal_alg === nothing && return true
        !hasmethod(SciMLBase.$(op), Tuple{typeof(internal_alg)}) && return false
        return SciMLBase.$(op)(internal_alg)
    end
end
