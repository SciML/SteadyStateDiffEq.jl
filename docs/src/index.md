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

## API

```@docs
SSRootfind
DynamicSS
```
