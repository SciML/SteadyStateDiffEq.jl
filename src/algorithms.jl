abstract SteadyStateDiffEqAlgorithm <: AbstractSteadyStateAlgorithm

immutable SSRootfind{F} <: SteadyStateDiffEqAlgorithm
  nlsolve::F
end
SSRootfind(;nlsolve=(f,u0) -> (res=NLsolve.nlsolve(f,u0);res.zero)) = SSRootfind(nlsolve)
