using Test, SciMLBase, SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq
using NonlinearSolveBase: AbsNormSafeBestTerminationMode
using LinearAlgebra: norm
using SymbolicIndexingInterface: SymbolCache, state_values, parameter_values, getsym,
    current_time

function cache_interface_problem(iip, lowering = :none)
    sys = SymbolCache([:x], [:source, :rate])
    f = iip ? ((du, u, p, t) -> (du .= p[1] .- p[2] .* u)) :
        ((u, p, t) -> p[1] .- p[2] .* u)
    nlprob = NonlinearProblem((u, p) -> p[1] .- p[2] .* u, [0.0], [4.0, 2.0])
    lowered_problem = lowering === :none ? nothing :
        lowering === :stored ? nlprob : (_ -> nlprob)
    return SteadyStateProblem(
        ODEFunction{iip}(f; sys), [0.0], [4.0, 2.0]; lowered_problem
    )
end

@testset "initialized cache interfaces and reset" begin
    @testset "non-SCC initialized cache interface" begin
        @testset "$(nameof(typeof(alg))), iip=$iip, lowering=$lowering" for alg in (
                    DynamicSS(Tsit5()), SICNM(Rodas5P()),
                ), iip in (false, true), lowering in (:none, :stored, :callable)

            prob = cache_interface_problem(iip, lowering)
            cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
            @test cache[:x] == 0.0
            @test state_values(cache) == [0.0]
            @test state_values(cache) !== prob.u0
            @test parameter_values(cache) == [4.0, 2.0]
            @test current_time(cache) == Inf
            @test getsym(cache, :source)(cache) == 4.0
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test sol.u ≈ [2.0] atol = 1.0e-8
            @test state_values(cache) ≈ [2.0] atol = 1.0e-8
            @test getsym(cache, :x)(cache) ≈ 2.0 atol = 1.0e-8
        end
    end

    @testset "steady-state cache reinitialization" begin
        @testset "$(nameof(typeof(alg))), iip=$iip, safe=$safe" for safe in (false, true),
                alg in (safe ? DynamicSS(Rodas5P()) : DynamicSS(Tsit5()), SICNM(Rodas5P())),
                iip in (false, true)

            prob = cache_interface_problem(iip)
            kwargs = safe ? (; termination_condition = AbsNormSafeBestTerminationMode(Base.Fix2(norm, Inf))) : (;)
            cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10, kwargs...)
            first_sol = solve!(cache)
            @test successful_retcode(first_sol)
            @test first_sol.u ≈ [2.0] atol = 1.0e-8
            for (u0, p, equilibrium) in (([7.0], [9.0, 3.0], 3.0), ([-5.0], [-2.0, 2.0], -1.0))
                SciMLBase.reinit!(cache, u0; p)
                @test state_values(cache) == u0
                @test state_values(cache) !== u0
                @test parameter_values(cache) == p
                @test cache[:x] == only(u0)
                if alg isa SICNM
                    @test cache.integrator.u ≈ [only(u0), equilibrium - only(u0)]
                end
                step!(cache)
                sol = solve!(cache)
                @test successful_retcode(sol)
                @test sol.u ≈ [equilibrium] atol = 1.0e-8
                @test p[1] .- p[2] .* sol.u ≈ [0.0] atol = 1.0e-8
                @test sol.prob.u0 == u0
                @test sol.prob.p == p
                @test state_values(cache) ≈ [equilibrium] atol = 1.0e-8
            end
            SciMLBase.reinit!(cache)
            @test state_values(cache) == [-5.0]
            @test parameter_values(cache) == [-2.0, 2.0]
            @test solve!(cache).u ≈ [-1.0] atol = 1.0e-8
            @test first_sol.u ≈ [2.0] atol = 1.0e-8
            @test prob.u0 == [0.0]
            @test prob.p == [4.0, 2.0]
        end
    end

    @testset "keyword u0 cache reset" begin
        @testset "$(nameof(typeof(alg))), iip=$iip" for alg in (
                    DynamicSS(Tsit5()), SICNM(Rodas5P()),
                ), iip in (false, true)

            f = iip ? ((du, u, p, t) -> (du .= p .- u)) : ((u, p, t) -> p .- u)
            prob = SteadyStateProblem(ODEFunction{iip}(f), [0.0], [2.0])
            cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
            @test SciMLBase.reinit!(cache; u0 = [-3.0], p = [4.0]) === cache
            @test state_values(cache) == [-3.0]
            @test parameter_values(cache) == [4.0]
            @test cache.prob.u0 == [-3.0]
            @test cache.prob.p == [4.0]
            @test !haskey(cache.kwargs, :u0)
            if alg isa SICNM
                @test cache.integrator.u ≈ [-3.0, 7.0]
            end
            @test solve!(cache).u ≈ [4.0] atol = 1.0e-8
            @test SciMLBase.reinit!(cache, [7.0]; p = [8.0]) === cache
            @test state_values(cache) == [7.0]
            @test parameter_values(cache) == [8.0]
            @test !haskey(cache.kwargs, :u0)
            @test solve!(cache).u ≈ [8.0] atol = 1.0e-8
            @test SciMLBase.reinit!(cache; u0 = [-1.0]) === cache
            @test state_values(cache) == [-1.0]
            @test parameter_values(cache) == [8.0]
            @test solve!(cache).u ≈ [8.0] atol = 1.0e-8
            @test prob.u0 == [0.0]
            @test prob.p == [2.0]
        end
    end
    @testset "cache state with solution slicing" for alg in (DynamicSS(Tsit5()), SICNM(Rodas5P()))
        sys = SymbolCache([:x, :y], [:a, :b])
        prob = SteadyStateProblem(ODEFunction((u, p, t) -> p .- u; sys), zeros(2), [2.0, 3.0])
        cache = init(prob, alg; save_idxs = [2], abstol = 1.0e-10, reltol = 1.0e-10)
        @test length(state_values(cache)) == 2
        @test solve!(cache).u ≈ [3.0] atol = 1.0e-8
        SciMLBase.reinit!(cache, [-3.0, -4.0]; p = [6.0, 9.0], abstol = 1.0e-11, reltol = 1.0e-11)
        @test state_values(cache) == [-3.0, -4.0]
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test sol.u ≈ [9.0] atol = 1.0e-8
        @test sol.resid ≈ [0.0] atol = 1.0e-8
        @test state_values(cache) ≈ [6.0, 9.0] atol = 1.0e-8
        @test cache[:x] ≈ 6.0 atol = 1.0e-8
        @test getsym(cache, [:x, :y])(cache) ≈ [6.0, 9.0] atol = 1.0e-8
        SciMLBase.reinit!(cache; save_idxs = [1])
        @test solve!(cache).u ≈ [6.0] atol = 1.0e-8
    end

    @testset "symbolic parameter proxy and state assignment" begin
        @testset "$(nameof(typeof(alg))), iip=$iip" for alg in (
                    DynamicSS(Tsit5()), SICNM(Rodas5P()),
                ), iip in (false, true)

            sys = SymbolCache([:x], [:a])
            f = iip ? ((du, u, p, t) -> (du .= p[1] .- u)) : ((u, p, t) -> p[1] .- u)
            prob = SteadyStateProblem(ODEFunction{iip}(f; sys), [0.0], [2.0])
            cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
            @test :ps in propertynames(cache)
            @test cache.ps[:a] == 2.0
            @test cache[:x] == 0.0
            cache.ps[:a] = 6.0
            @test cache.ps[:a] == 6.0
            @test parameter_values(cache) == [6.0]
            cache[:x] = 1.0
            @test cache[:x] == 1.0
            @test state_values(cache) == [1.0]
            if alg isa SICNM
                @test cache.integrator.u ≈ [1.0, 5.0]
            end
            step!(cache)
            sol = solve!(cache)
            @test successful_retcode(sol)
            @test sol.u ≈ [6.0] atol = 1.0e-8
            @test cache[:x] ≈ 6.0 atol = 1.0e-8
            @test cache.ps[:a] == 6.0

            cache2 = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
            cache2.ps[:a] = 8.0
            cache2[:x] = -1.0
            sol2 = solve!(cache2)
            @test successful_retcode(sol2)
            @test sol2.u ≈ [8.0] atol = 1.0e-8
            @test cache2[:x] ≈ 8.0 atol = 1.0e-8
            @test cache2.ps[:a] == 8.0
        end
    end
end
