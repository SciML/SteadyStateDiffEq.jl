abstract type SteadyStateDiffEqAlgorithm <: DiffEqBase.AbstractSteadyStateAlgorithm end

struct SSRootfind{F} <: SteadyStateDiffEqAlgorithm
  nlsolve::F
end
SSRootfind(;nlsolve=(f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,ftol = abstol);res.zero)) = SSRootfind(nlsolve)

struct DynamicSS{A,AT,RT,TS} <: SteadyStateDiffEqAlgorithm
  alg::A
  abstol::AT
  reltol::RT
  tspan::TS
end
DynamicSS(alg; abstol = 1e-8, reltol = 1e-6, tspan = Inf) =
  DynamicSS(alg, abstol, reltol, tspan)

# Backward compatibility:
DynamicSS(alg, abstol, reltol) = DynamicSS(alg; abstol=abstol, reltol=reltol)
