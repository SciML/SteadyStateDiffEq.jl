# SteadyStateDiffEq.jl

[![Join the chat at https://gitter.im/JuliaDiffEq/Lobby](https://badges.gitter.im/JuliaDiffEq/Lobby.svg)](https://gitter.im/JuliaDiffEq/Lobby?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge)
[![Build Status](https://github.com/SciML/SteadyStateDiffEq.jl/workflows/CI/badge.svg)](https://github.com/SciML/SteadyStateDiffEq.jl/actions?query=workflow%3ACI)
[![Coverage Status](https://coveralls.io/repos/JuliaDiffEq/SteadyStateDiffEq.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/JuliaDiffEq/SteadyStateDiffEq.jl?branch=master)
[![codecov.io](http://codecov.io/github/JuliaDiffEq/SteadyStateDiffEq.jl/coverage.svg?branch=master)](http://codecov.io/github/JuliaDiffEq/SteadyStateDiffEq.jl?branch=master)

SteadyStateDiffEq.jl is a component package in the DifferentialEquations ecosystem.
It holds the steady state solvers for differential equations.
While completely independent and usable on its own, users interested in using this
functionality should check out [DifferentialEquations.jl](https://github.com/JuliaDiffEq/DifferentialEquations.jl).

## Breaking Changes in v2

 1. `NLsolve.jl` dependency has been dropped. `SSRootfind` requires a nonlinear solver to be
    specified.
 2. `DynamicSS` no longer stores `abstol` and `reltol`. To use separate tolerances for the
    odesolve and the termination, specify `odesolve_kwargs` in `solve`.
 3. The deprecated termination conditions are dropped, see [NonlinearSolve.jl Docs](https://docs.sciml.ai/NonlinearSolve/stable/basics/TerminationCondition/)
    for details on this.
