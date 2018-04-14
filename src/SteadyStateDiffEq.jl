__precompile__()

module SteadyStateDiffEq

using Reexport
@reexport using DiffEqBase

using NLsolve, DiffEqCallbacks

using Compat

import DiffEqBase: solve

include("algorithms.jl")
include("solve.jl")

export SSRootfind, DynamicSS

end # module
