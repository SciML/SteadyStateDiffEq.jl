module DiffEqSteadyState

using DiffEqBase, NLsolve

import DiffEqBase: solve

include("algorithms.jl")
include("solve.jl")

export SSRootfind

end # module
