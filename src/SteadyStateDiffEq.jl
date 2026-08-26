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
    NonlinearSolution, ReturnCode, SteadyStateProblem, SteadyStateSolution, get_du, init,
    isinplace, remake, solve, successful_retcode

const infnorm = Base.Fix2(norm, Inf)

include("algorithms.jl")
include("solve.jl")
include("precompilation.jl")

export SSRootfind, DynamicSS, SICNM
export NonlinearProblem, NonlinearSolution, ReturnCode, SteadyStateProblem,
    SteadyStateSolution, remake, solve, successful_retcode
@public SteadyStateDiffEqAlgorithm

end
