using SciMLTesting, SteadyStateDiffEq, Test
using JET

include("public_api_docs.jl")

run_qa(
    SteadyStateDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    # Aqua deps_compat: missing [compat] for stdlib LinearAlgebra
    # https://github.com/SciML/SteadyStateDiffEq.jl/issues/134
    aqua_broken = (:deps_compat,),
    ei_kwargs = (
        # Names still not declared public by their owning package (or Base); drop each
        # as it is made public upstream.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractSteadyStateProblem, :__solve, :value,  # SciMLBase
                :prepare_alg,  # DiffEqBase
                :structdiff,  # Base
            ),
        ),
        # Non-public NonlinearSolveBase termination-mode abstract type imported explicitly.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractSafeBestNonlinearTerminationMode,  # NonlinearSolveBase
            ),
        ),
    ),
)
