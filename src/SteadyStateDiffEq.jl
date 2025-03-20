module SteadyStateDiffEq

using Reexport: @reexport
@reexport using SciMLBase

using ConcreteStructs: @concrete
using NonlinearSolveBase
using DiffEqCallbacks: TerminateSteadyState
using LinearAlgebra: norm
using SciMLBase: SciMLBase, CallbackSet, NonlinearProblem, ODEProblem,
                 ReturnCode, SteadyStateProblem, get_du, init, isinplace

const infnorm = Base.Fix2(norm, Inf)

include("algorithms.jl")
include("solve.jl")

export SSRootfind, DynamicSS

end
