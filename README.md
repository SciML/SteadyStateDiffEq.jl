# SteadyStateDiffEq.jl

[![Join the chat at https://julialang.zulipchat.com #sciml-bridged](https://img.shields.io/static/v1?label=Zulip&message=chat&color=9558b2&labelColor=389826)](https://julialang.zulipchat.com/#narrow/stream/279055-sciml-bridged)
[![Build Status](https://github.com/SciML/SteadyStateDiffEq.jl/workflows/Tests/badge.svg)](https://github.com/SciML/SteadyStateDiffEq.jl/actions?query=workflow%3ATests)
[![Coverage Status](https://coveralls.io/repos/SciML/SteadyStateDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/SciML/SteadyStateDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/SciML/SteadyStateDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/SciML/SteadyStateDiffEq.jl?branch=master)

SteadyStateDiffEq.jl is a component package in the DifferentialEquations ecosystem.
It holds the steady state solvers for differential equations.
While completely independent and usable on its own, users interested in using this
functionality should check out [DifferentialEquations.jl](https://github.com/SciML/DifferentialEquations.jl).

## Usage

SteadyStateDiffEq.jl provides two main algorithms for finding steady states:

### SSRootfind - Nonlinear Solver Approach

Use a nonlinear solver to directly find the steady state:

```julia
using SteadyStateDiffEq, NonlinearSolve

function f!(du, u, p, t)
    du[1] = 2 - 2u[1]
    du[2] = u[1] - 4u[2]
end

u0 = zeros(2)
prob = SteadyStateProblem(f!, u0)
sol = solve(prob, SSRootfind())
```

### DynamicSS - Time Evolution Approach

Evolve the system forward in time until derivatives approach zero:

```julia
using SteadyStateDiffEq, OrdinaryDiffEq

sol = solve(prob, DynamicSS(Tsit5()))
```

For more details, see the [SciML documentation](https://docs.sciml.ai/DiffEqDocs/stable/).

## Breaking Changes in v2

 1. `NLsolve.jl` dependency has been dropped. `SSRootfind` requires a nonlinear solver to be
    specified.
 2. `DynamicSS` no longer stores `abstol` and `reltol`. To use separate tolerances for the
    odesolve and the termination, specify `odesolve_kwargs` in `solve`.
 3. The deprecated termination conditions are dropped, see [NonlinearSolve.jl Docs](https://docs.sciml.ai/NonlinearSolve/stable/basics/TerminationCondition/)
    for details on this.
