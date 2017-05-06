__precompile__()

module SteadyStateDiffEq

using DiffEqBase, NLsolve

using Compat

import DiffEqBase: solve

include("algorithms.jl")
include("solve.jl")

export SSRootfind

end # module
