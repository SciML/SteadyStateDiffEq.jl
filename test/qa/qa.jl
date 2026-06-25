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
        # Non-public names of upstream SciML packages (SciMLBase, NonlinearSolveBase,
        # DiffEqBase) and Base accessed by qualification; they go public as those
        # libraries declare them.
        all_qualified_accesses_are_public = (;
            ignore = (
                :AbstractSteadyStateAlgorithm, :AbstractSteadyStateProblem,  # SciMLBase
                :NonlinearAliasSpecifier, :__solve, :build_solution, :isadaptive,  # SciMLBase
                :Default, :Failure, :Success, :Terminated,  # SciMLBase.ReturnCode
                :Fix2, :structdiff,  # Base
                :get_abstol, :get_reltol,  # NonlinearSolveBase
                :prepare_alg,  # DiffEqBase
                :value,  # SciMLBase (accessed via DiffEqBase)
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
