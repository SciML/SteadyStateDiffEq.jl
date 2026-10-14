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

A solve with `save_idxs` returns a solution whose `prob` shares the original
problem's blocks and parameters but whose index provider lists the saved states
in selection order. `sol[s]` resolves saved states in any order; states that
were not saved and observed expressions (which need the full state) throw.
Parameters and independent variables index as usual, and `sol.original` is the
full solution over all states.

## API

```@docs
SSRootfind
DynamicSS
SICNM
```
