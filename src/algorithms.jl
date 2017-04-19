abstract SteadyStateDiffEqAlgorithm <: AbstractSteadyStateAlgorithm

immutable SSRootfind{F} <: SteadyStateDiffEqAlgorithm
  nlsolve::F
end
SSRootfind(;nlsolve=NLsolve.nlsolve) = SSRootfind(nlsolve)
