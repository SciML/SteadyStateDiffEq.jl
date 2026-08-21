module SteadyStateDiffEq

using ConcreteStructs: @concrete
import DiffEqBase
using NonlinearSolveBase: NonlinearSolveBase, termination_condition_result
using DiffEqCallbacks: TerminateSteadyState
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, norm
using LinearSolve: LinearSolve
using SciMLPublic: @public
using SciMLBase: SciMLBase, CallbackSet, LinearProblem, NonlinearProblem, ODEProblem,
    SteadyStateProblem, get_du, init, isinplace, solve

const infnorm = Base.Fix2(norm, Inf)

include("algorithms.jl")
include("solve.jl")
include("precompilation.jl")

export SSRootfind, DynamicSS, SICNM
@public SteadyStateDiffEqAlgorithm

end
