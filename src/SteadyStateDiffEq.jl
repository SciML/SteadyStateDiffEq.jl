__precompile__()

module SteadyStateDiffEq

using Reexport
@reexport using DiffEqBase
  
using NLsolve

using Compat

import DiffEqBase: solve

include("algorithms.jl")
include("solve.jl")

export SSRootfind

end # module
