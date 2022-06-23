abstract type SteadyStateDiffEqAlgorithm <: DiffEqBase.AbstractSteadyStateAlgorithm end

struct SSRootfind{F} <: SteadyStateDiffEqAlgorithm
    nlsolve::F
end
function SSRootfind(;
                    nlsolve = (f, u0, abstol) -> (res = NLsolve.nlsolve(f, u0,
                                                                        ftol = abstol);
                                                  res.zero))
    SSRootfind(nlsolve)
end

struct DynamicSS{A, AT, RT, TS} <: SteadyStateDiffEqAlgorithm
    alg::A
    abstol::AT
    reltol::RT
    tspan::TS
end
function DynamicSS(alg; abstol = 1e-8, reltol = 1e-6, tspan = Inf)
    DynamicSS(alg, abstol, reltol, tspan)
end

# Backward compatibility:
DynamicSS(alg, abstol, reltol) = DynamicSS(alg; abstol = abstol, reltol = reltol)

## SciMLBase Trait Definitions

SciMLBase.isadaptive(alg::SteadyStateDiffEqAlgorithm) = true

SciMLBase.isautodifferentiable(alg::SSRootfind) = true
SciMLBase.allows_arbitrary_number_types(alg::SSRootfind) = true
SciMLBase.allowscomplex(alg::SSRootfind) = true

SciMLBase.isautodifferentiable(alg::DynamicSS) = SciMLBase.isautodifferentiable(alg.alg)
function SciMLBase.allows_arbitrary_number_types(alg::DynamicSS)
    SciMLBase.allows_arbitrary_number_types(alg.alg)
end
SciMLBase.allowscomplex(alg::DynamicSS) = SciMLBase.allowscomplex(alg.alg)
