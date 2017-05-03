@compat abstract type SteadyStateDiffEqAlgorithm <: AbstractSteadyStateAlgorithm end

immutable SSRootfind{F} <: SteadyStateDiffEqAlgorithm
  nlsolve::F
end
SSRootfind(;nlsolve=(f,u0) -> (res=NLsolve.nlsolve(f,u0);res.zero)) = SSRootfind(nlsolve)
