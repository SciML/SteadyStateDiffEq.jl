# SteadyStateDiffEq.jl

SteadyStateDiffEq.jl provides algorithms for solving steady-state problems in the
SciML ecosystem.

## Installation

```julia
using Pkg
Pkg.add("SteadyStateDiffEq")
```

## Usage

Use `SSRootfind` to solve the steady-state residual equation with a nonlinear solver:

```julia
using SciMLBase: SteadyStateProblem, solve
using SteadyStateDiffEq
using NonlinearSolve

prob = SteadyStateProblem((u, p, t) -> 1 .- u, [0.0])
sol = solve(prob, SSRootfind())
```

Use `DynamicSS` to integrate the system until its derivative is close to zero:

```julia
using SciMLBase: SteadyStateProblem, solve
using SteadyStateDiffEq
using Sundials: CVODE_BDF

prob = SteadyStateProblem((u, p, t) -> 1 .- u, [0.0])
sol = solve(prob, DynamicSS(CVODE_BDF()); dt = 1.0)
```

Use `SICNM` (the semi-implicit continuous Newton method) to solve the steady-state
residual equation by integrating the continuous Newton flow, written as a
differential-algebraic equation, until the residual is close to zero. This is much more
robust than Newton's method on ill-conditioned problems such as power flow equations:

```julia
using SciMLBase: SteadyStateProblem, solve
using SteadyStateDiffEq
using OrdinaryDiffEqRosenbrock: Rodas3d

prob = SteadyStateProblem((u, p, t) -> 1 .- u, [0.0])
sol = solve(prob, SICNM(Rodas3d()))
```

## Initialization and stepping

`DynamicSS` and `SICNM` also support the `init`/`solve!` interface. `init(prob, alg;
kwargs...)` accepts the same keyword arguments as `solve` and returns a cache, and
`solve!(cache)` returns the same solution as `solve(prob, alg; kwargs...)`: its return
code comes from the termination condition (for example `ReturnCode.Unstable` from a
safe termination mode, or a failure when the time span ends before steady state is
reached), and `save_idxs` selects components of the final state.

For a problem that is integrated as a whole, `cache.integrator` is the underlying ODE
integrator, and `step!(cache)` advances it. For `SICNM` that integrator solves the
extended continuous-Newton system, so its state is `[y; z]`: the first
`length(prob.u0)` components are the steady-state variables `y` and the rest are the
Newton direction `z`. Its intermediate states are therefore not states of `prob`;
`solve!` returns only `y`.

The cache supports symbolic state indexing (`cache[x]` and `getsym(cache, x)`),
`state_values(cache)`, and `parameter_values(cache)` through
SymbolicIndexingInterface. State access exposes the current steady-state variables,
including only `y` for SICNM, even when `save_idxs` selects a subset in the solution.
`SciMLBase.reinit!(cache, u0; p)` rebuilds the integrator and termination condition
at the new operating point with the initialization options. SICNM recomputes the
consistent Newton direction from `u0` and `p`. Keyword arguments override the
initialization options; omitting `u0` or `p` uses the latest initial state or current
parameters. Access `cache.integrator` after reinitialization to use the new integrator.

When `prob` carries an `SCCNonlinearProblem` lowering, the blocks are solved one after
another and there is no single integrator to step. `solve!` runs that sequential solve.

```julia
using SciMLBase: SteadyStateProblem, init, solve!, step!
using SteadyStateDiffEq
using OrdinaryDiffEqRosenbrock: Rodas5P

prob = SteadyStateProblem((u, p, t) -> 1 .- u, [0.0])
cache = init(prob, SICNM(Rodas5P()))
step!(cache)
sol = solve!(cache)
```

## API

```@docs
SSRootfind
DynamicSS
SICNM
SciMLBase.reinit!(::SteadyStateDiffEq.SteadyStateODECache)
```
