# Developer API

!!! warning "Developer API, not user API"
    This page documents the extension contract for solver authors. Application
    code should use the concrete [`SSRootfind`](@ref SteadyStateDiffEq.SSRootfind),
    [`DynamicSS`](@ref SteadyStateDiffEq.DynamicSS), and
    [`SICNM`](@ref SteadyStateDiffEq.SICNM) algorithms instead of defining new
    subtypes or calling `SciMLBase.__solve` directly.

The contract is defined by
[`SteadyStateDiffEqAlgorithm`](@ref SteadyStateDiffEq.SteadyStateDiffEqAlgorithm).
New solver packages should also read the [SciMLBase algorithm interface](https://docs.sciml.ai/SciMLBase/stable/interfaces/Algorithms/)
and follow its rules for common `solve` keywords, algorithm-specific fields, and
algorithm capability traits.

```@docs
SteadyStateDiffEq.SteadyStateDiffEqAlgorithm
```
