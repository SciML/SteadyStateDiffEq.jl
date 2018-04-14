abstract type SteadyStateDiffEqAlgorithm <: AbstractSteadyStateAlgorithm end

struct SSRootfind{F} <: SteadyStateDiffEqAlgorithm
  nlsolve::F
end
SSRootfind(;nlsolve=(f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,ftol = abstol);res.zero)) = SSRootfind(nlsolve)

struct DynamicSS{A,AT,RT} <: SteadyStateDiffEqAlgorithm
  alg::A
  abstol::AT
  reltol::RT
end
DynamicSS(alg;abstol = 1e-8, reltol = 1e-6) = DynamicSS(alg,abstol,reltol)
