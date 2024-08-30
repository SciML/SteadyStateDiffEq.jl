module SteadyStateDiffEq

using Reexport: @reexport
@reexport using DiffEqBase

using ConcreteStructs: @concrete
using DiffEqBase: DiffEqBase, AbstractNonlinearTerminationMode,
                  AbstractSafeNonlinearTerminationMode,
                  AbstractSafeBestNonlinearTerminationMode,
                  NonlinearSafeTerminationReturnCode, NormTerminationMode
using DiffEqCallbacks: TerminateSteadyState
using LinearAlgebra: norm
using SciMLBase: SciMLBase, CallbackSet, NonlinearProblem, ODEProblem,
                 ReturnCode, SteadyStateProblem, get_du, init, isinplace

const infnorm = Base.Fix2(norm, Inf)

include("algorithms.jl")
include("solve.jl")

export SSRootfind, DynamicSS

end
