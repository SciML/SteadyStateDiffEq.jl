module SteadyStateDiffEq

using Reexport: @reexport
@reexport using SciMLBase

using ConcreteStructs: @concrete
import DiffEqBase
using NonlinearSolveBase: NonlinearSolveBase, AbstractNonlinearTerminationMode,
    AbstractSafeNonlinearTerminationMode,
    AbstractSafeBestNonlinearTerminationMode
using DiffEqCallbacks: TerminateSteadyState
using ForwardDiff: ForwardDiff
using LinearAlgebra: Diagonal, norm
using SciMLBase: SciMLBase, CallbackSet, NonlinearProblem, ODEProblem,
    ReturnCode, SteadyStateProblem, get_du, init, isinplace, solve

const infnorm = Base.Fix2(norm, Inf)

include("algorithms.jl")
include("solve.jl")

export SSRootfind, DynamicSS, SICNM

end
