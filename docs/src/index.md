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

## API

```@docs
SSRootfind
DynamicSS
SICNM
```
