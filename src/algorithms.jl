abstract type SteadyStateDiffEqAlgorithm <: DiffEqBase.AbstractSteadyStateAlgorithm end

struct SSRootfind{F} <: SteadyStateDiffEqAlgorithm
    nlsolve::F
end
function SSRootfind(;
        nlsolve = (f, u0, abstol) -> (NLsolve.nlsolve(f, u0,
            ftol = abstol)))
    SSRootfind(nlsolve)
end

"""
    DynamicSS(alg; abstol = 1e-8, reltol = 1e-6, tspan = Inf,
              termination_condition = SteadyStateTerminationCriteria(:default; abstol,
                                                                     reltol))

Requires that an ODE algorithm is given as the first argument.  The absolute and
relative tolerances specify the termination conditions on the derivative's closeness to
zero.  This internally uses the `TerminateSteadyState` callback from the Callback Library.
The simulated time for which given ODE is solved can be limited by `tspan`.  If `tspan` is
a number, it is equivalent to passing `(zero(tspan), tspan)`.

Example usage:

```julia
using SteadyStateDiffEq, OrdinaryDiffEq
sol = solve(prob,DynamicSS(Tsit5()))

using Sundials
sol = solve(prob,DynamicSS(CVODE_BDF()),dt=1.0)
```

!!! note

    If you use `CVODE_BDF` you may need to give a starting `dt` via `dt=....`.*
"""
struct DynamicSS{A, AT, RT, TS, TC <: NLSolveTerminationCondition} <:
       SteadyStateDiffEqAlgorithm
    alg::A
    abstol::AT
    reltol::RT
    tspan::TS
    termination_condition::TC
end

function DynamicSS(alg; abstol = 1e-8, reltol = 1e-6, tspan = Inf,
        termination_condition = NLSolveTerminationCondition(NLSolveTerminationMode.SteadyStateDefault;
            abstol,
            reltol))
    DynamicSS(alg, abstol, reltol, tspan, termination_condition)
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
