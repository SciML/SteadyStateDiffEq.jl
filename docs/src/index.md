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

## Initialization with SCC lowerings

For a `SteadyStateProblem` whose `lowered_problem` is an `SCCNonlinearProblem`,
`init(prob, alg; kwargs...)` returns a deferred solve object. Initialization
materializes a callable lowering once and records it along with the algorithm
and options. Each `solve!` performs a fresh solve on that lowering, with the same
algorithm and solution state ordering as `solve(prob, alg; kwargs...)`.
This interface supports the default solver, `SSRootfind`, `DynamicSS`, and `SICNM`.

The deferred object's `state_values` and `parameter_values` expose the original
problem's initial state and parameters by reference. They remain initial data
after `solve!`; they are not an iterate or a copy for editing. The object does not
support `step!`, `reinit!`, or symbolic indexing. Read the returned solution for
solved state values and symbolic indexing, including variables eliminated during
lowering. Repeated solves allocate fresh solver caches and reuse the materialized
lowering; call `init` on a remade problem to change the operating point.

Problems without an SCC lowering use the regular nonlinear solver caches.

## API

```@docs
SSRootfind
DynamicSS
SICNM
```
