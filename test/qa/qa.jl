using SciMLTesting, SteadyStateDiffEq, Test
using JET

dependency_reexports(pkg) = Tuple(
    name for name in public_api_names(pkg)
        if isdefined(pkg, name) && parentmodule(getfield(pkg, name)) !== pkg
)

const DEPENDENCY_REEXPORTS = dependency_reexports(SteadyStateDiffEq)

run_qa(
    SteadyStateDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    api_docs_kwargs = (;
        rendered = true,
        ignore = DEPENDENCY_REEXPORTS,
        rendered_ignore = DEPENDENCY_REEXPORTS,
    ),
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
