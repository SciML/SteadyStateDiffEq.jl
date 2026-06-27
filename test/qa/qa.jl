using SciMLTesting, SteadyStateDiffEq, Test
using JET

run_qa(
    SteadyStateDiffEq;
    explicit_imports = true,
    jet_kwargs = (; target_defined_modules = true),
    # Aqua deps_compat: missing [compat] for stdlib LinearAlgebra
    # https://github.com/SciML/SteadyStateDiffEq.jl/issues/134
    aqua_broken = (:deps_compat,),
    ei_kwargs = (
        # `value` is owned by SciMLBase but re-exported and accessed via DiffEqBase.
        all_qualified_accesses_via_owners = (; ignore = (:value,)),
        # Names still not declared public by their owning package (or Base) as of
        # SciMLBase 3.27, NonlinearSolveBase 2.31, DiffEqBase 7.6; drop each as it
        # is made public upstream.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractSteadyStateProblem, :__solve,  # SciMLBase
                :get_abstol, :get_reltol,  # NonlinearSolveBase
                :prepare_alg, :value,  # DiffEqBase
                :structdiff,  # Base
            ),
        ),
        # Non-public NonlinearSolveBase termination-mode abstract types imported explicitly.
        all_explicit_imports_are_public = (;
            ignore = (
                :AbstractNonlinearTerminationMode,
                :AbstractSafeBestNonlinearTerminationMode,
                :AbstractSafeNonlinearTerminationMode,  # NonlinearSolveBase
            ),
        ),
    ),
)
