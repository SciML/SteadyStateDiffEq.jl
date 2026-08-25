using SciMLTesting, SteadyStateDiffEq, Test
using JET

const REEXPORTS = (
    :NonlinearProblem, :NonlinearSolution, :ReturnCode, :SteadyStateProblem,
    :SteadyStateSolution, :remake, :solve, :successful_retcode,
)

const QUALIFIED_PUBLIC_IGNORE = (
    # Required to unwrap NonlinearSolve's AutoSpecialize callable; documented by
    # NonlinearSolveBase but not declared public by its owner.
    :get_raw_f,
    # Documented ForwardDiff API that ForwardDiff does not declare public.
    :Dual, :Tag, :jacobian, :partials, :value,
)

run_qa(
    SteadyStateDiffEq;
    reexports_allow = REEXPORTS,
    ei_kwargs = (;
        all_qualified_accesses_are_public = (;
            ignore = QUALIFIED_PUBLIC_IGNORE,
        ),
    ),
)

@testset "Reexport surface" begin
    @testset "$name" for name in REEXPORTS
        @test name in names(SteadyStateDiffEq)
        @test isdefined(@__MODULE__, name)
    end
end
