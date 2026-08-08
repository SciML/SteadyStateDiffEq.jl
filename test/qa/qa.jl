using SciMLTesting, SteadyStateDiffEq, Test
using JET

const QUALIFIED_PUBLIC_IGNORE = (
    # Required to unwrap NonlinearSolve's AutoSpecialize callable; documented by
    # NonlinearSolveBase but not declared public by its owner.
    :get_raw_f,
    # Documented ForwardDiff API that ForwardDiff does not declare public.
    :Dual, :Tag, :jacobian, :partials, :value,
)

run_qa(
    SteadyStateDiffEq;
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = QUALIFIED_PUBLIC_IGNORE,
        ),
    ),
)
