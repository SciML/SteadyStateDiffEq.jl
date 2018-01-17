@compat abstract type SteadyStateDiffEqAlgorithm <: AbstractSteadyStateAlgorithm end

immutable SSRootfind{F} <: SteadyStateDiffEqAlgorithm
  nlsolve::F
end
SSRootfind(;nlsolve=(f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,ftol = abstol);res.zero)) = SSRootfind(nlsolve)
