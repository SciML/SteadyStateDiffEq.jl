using Test, SciMLBase, SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq
using SymbolicIndexingInterface: state_values, SymbolCache, getsym

@testset "non-SCC cache compatibility" begin
    @testset "$lowering, $(isempty(algs) ? :default : :DynamicSS)" for lowering in (:none, :stored, :callable),
            algs in ((), (DynamicSS(Tsit5()),))

        sys = SymbolCache([:x], [:k])
        nlprob = NonlinearProblem(NonlinearFunction((u, p) -> p .- u; sys), [0.0], [2.0])
        lowered_problem = lowering === :none ? nothing :
            lowering === :stored ? nlprob : (_ -> nlprob)
        prob = SteadyStateProblem(
            ODEFunction((u, p, t) -> p .- u; sys), [0.0], [2.0]; lowered_problem
        )
        cache = init(prob, algs...; abstol = 1.0e-10, reltol = 1.0e-10)
        @test cache[:x] == 0.0
        @test state_values(cache) !== prob.u0
        @test state_values(cache) !== nlprob.u0
        sol = solve!(cache)
        @test sol.u ≈ [2.0] atol = 1.0e-8
        @test state_values(cache) ≈ [2.0] atol = 1.0e-8
        @test getsym(cache, :x)(cache) ≈ 2.0 atol = 1.0e-8
        SciMLBase.reinit!(cache, [0.0]; p = [3.0])
        @test solve!(cache).u ≈ [3.0] atol = 1.0e-8
        SciMLBase.reinit!(cache, [0.0]; p = [4.0])
        SciMLBase.step!(cache)
        @test solve!(cache).u ≈ [4.0] atol = 1.0e-8
    end
end

@testset "non-SCC lowering solver keywords" for builder in (false, true)
    nlprob = NonlinearProblem((u, p) -> u .^ 2 .- 2, [10.0])
    prob = SteadyStateProblem(
        (u, p, t) -> u .^ 2 .- 2, [10.0];
        lowered_problem = builder ? (_ -> nlprob) : nlprob, maxiters = 0
    )
    @test solve(prob).retcode == ReturnCode.MaxIters
    @test solve(prob; maxiters = 0).retcode == ReturnCode.MaxIters
    sol = solve(prob; maxiters = 100, abstol = 1.0e-10, reltol = 1.0e-10)
    @test successful_retcode(sol)
    @test sol.u ≈ [sqrt(2)] atol = 1.0e-8
end
