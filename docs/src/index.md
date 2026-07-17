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
using SteadyStateDiffEq, NonlinearSolve

sol = solve(prob, SSRootfind(NewtonRaphson()))
```

Use `DynamicSS` to integrate the system until its derivative is close to zero:

```julia
using SteadyStateDiffEq, OrdinaryDiffEq

sol = solve(prob, DynamicSS(Tsit5()))
```

Use `SICNM` (the semi-implicit continuous Newton method) to solve the steady-state
residual equation by integrating the continuous Newton flow, written as a
differential-algebraic equation, until the residual is close to zero. This is much more
robust than Newton's method on ill-conditioned problems such as power flow equations:

```julia
using SteadyStateDiffEq, OrdinaryDiffEqRosenbrock

sol = solve(prob, SICNM(Rodas3d()))
```

## API

```@docs
SSRootfind
DynamicSS
SICNM
```
