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

## SCC-split steady states

`SCCSteadyStateProblem` is the SCC-split form of a steady-state problem: its
residual decomposes into a block-lower-triangular sequence of `LinearProblem`/
`NonlinearProblem`/`HomotopyProblem` blocks. Build it from a block list with
`SCCSteadyStateProblem(probs, explicitfuns!, p)`, or from a model with
`ModelingToolkit.SCCSteadyStateProblem(sys, op)`.

`DynamicSS` and `SICNM` solve the blocks sequentially in SCC order — each
nonlinear block is integrated to steady state on its own residual after
`explicitfuns!` updates its caches from the upstream solutions. `SSRootfind`, a
bare `solve`, or any nonlinear algorithm solve the problem through the SCC
solver (`SCCNonlinearSolve.jl`). The solution's `u` and `resid` are in the
compressed SCC state layout; symbolic indexing (`sol[x]`, `sol.ps[p]`) goes
through `prob.f.sys`, and `save_idxs` is a selection in that layout — integers,
or states resolved through `variable_index`.

A solve with `save_idxs` returns a solution whose `prob` is a saved-state view
of the original problem: it shares the original's blocks and parameters, but
its states are the saved entries in selection order. Solving or remaking that
problem works in the saved layout — `solve(saved.prob, alg)` applies the same
selection and returns a solution over the saved states, and a `save_idxs`
passed there selects among them. `sol[s]` resolves saved states in any order;
states that were not saved and observed expressions (which need the full
state) throw. Parameters and independent variables index as usual, and
`sol.original` is the full solution over all states.

Solve-time `u0` and `p` keywords are applied to the problem with `remake`
before dispatch: `u0` is given in the compressed state layout — in the saved
layout when solving a saved-state view — and `p` is subject to
`parameters_alias`, so an unaliased problem throws since its blocks hold
separate parameter objects. The returned solution's `prob` is the updated
problem.

## API

```@docs
SSRootfind
DynamicSS
SICNM
```
