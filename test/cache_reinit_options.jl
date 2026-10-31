using Test, SciMLBase, SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq
using NonlinearSolveBase: AbsNormSafeBestTerminationMode, AbsNormTerminationMode
using SymbolicIndexingInterface: SymbolCache, state_values, parameter_values, getsym
using LinearAlgebra: norm

@testset "type-changing steady-state cache reset options" begin
    @testset "$option, $(nameof(typeof(alg)))" for option in (
                :add_slice, :remove_slice, :add_dt, :termination_mode,
            ), alg in (
                option === :termination_mode ? DynamicSS(Rodas5P()) : DynamicSS(Tsit5()),
                SICNM(Rodas5P()),
            )

        sys = SymbolCache([:x, :y], [:a, :b])
        prob = SteadyStateProblem(ODEFunction((u, p, t) -> p .- u; sys), zeros(2), [2.0, 3.0])
        initial_options = option === :remove_slice ? (; save_idxs = [2]) : (;)
        cache = init(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10, initial_options...)
        retained = solve!(cache)
        initial_result = option === :remove_slice ? [3.0] : [2.0, 3.0]
        @test successful_retcode(retained)
        @test retained.u ≈ initial_result atol = 1.0e-8
        old_integrator = cache.integrator
        u0, p = [-2.0, -3.0], [4.0, 5.0]
        overrides = if option === :add_slice
            (; save_idxs = [2])
        elseif option === :remove_slice
            (; save_idxs = nothing)
        elseif option === :add_dt
            (; dt = 0.01)
        else
            (; termination_condition = AbsNormSafeBestTerminationMode(Base.Fix2(norm, Inf)))
        end
        @test SciMLBase.reinit!(cache, u0; p, overrides...) === cache
        @test cache.integrator !== old_integrator
        @test state_values(cache) == u0
        @test parameter_values(cache) == p
        @test getsym(cache, [:x, :y])(cache) == u0
        if option === :add_dt
            @test cache.integrator.dt == 0.01
        end
        if alg isa SICNM
            @test cache.integrator.u ≈ vcat(u0, p - u0)
        end
        SciMLBase.step!(cache)
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test sol.u ≈ (option === :add_slice ? [5.0] : p) atol = 1.0e-8
        @test sol.resid ≈ zero(sol.u) atol = 1.0e-8
        @test state_values(cache) ≈ p atol = 1.0e-8
        @test getsym(cache, [:x, :y])(cache) ≈ p atol = 1.0e-8
        @test sol.prob.u0 == u0
        @test sol.prob.p == p
        @test retained.u ≈ initial_result atol = 1.0e-8
        @test u0 == [-2.0, -3.0]
        @test p == [4.0, 5.0]
        @test prob.u0 == zeros(2)
        @test prob.p == [2.0, 3.0]

        @test SciMLBase.reinit!(cache; p = [6.0, 7.0]) === cache
        if option === :add_dt
            @test cache.integrator.dt == 0.01
        end
        again = solve!(cache)
        reference = solve(
            remake(prob; u0, p = [6.0, 7.0]), alg;
            abstol = 1.0e-10, reltol = 1.0e-10, initial_options..., overrides...
        )
        @test again.retcode == reference.retcode
        @test reference.u ≈ (option === :add_slice ? [7.0] : [6.0, 7.0]) atol = 1.0e-8
        @test again.u ≈ (option === :add_slice ? [7.0] : [6.0, 7.0]) atol = 1.0e-8
        @test sol.u ≈ (option === :add_slice ? [5.0] : p) atol = 1.0e-8
        @test retained.u ≈ initial_result atol = 1.0e-8

        previous_prob, previous_integrator = cache.prob, cache.integrator
        previous_u, previous_p = copy(state_values(cache)), copy(parameter_values(cache))
        @test_throws DimensionMismatch SciMLBase.reinit!(
            cache, zeros(3); p = [8.0, 9.0], save_idxs = 1, dt = 0.02
        )
        @test cache.prob === previous_prob
        @test cache.integrator === previous_integrator
        @test state_values(cache) == previous_u
        @test parameter_values(cache) == previous_p
        @test getsym(cache, [:x, :y])(cache) == previous_u
        @test again.u ≈ (option === :add_slice ? [7.0] : [6.0, 7.0]) atol = 1.0e-8

        @test SciMLBase.reinit!(cache; p = [8.0, 9.0], save_idxs = nothing) === cache
        @test solve!(cache).u ≈ [8.0, 9.0] atol = 1.0e-8
        if option === :termination_mode
            @test SciMLBase.reinit!(
                cache; termination_condition = AbsNormTerminationMode(Base.Fix2(norm, Inf))
            ) === cache
            @test solve!(cache).u ≈ [8.0, 9.0] atol = 1.0e-8
        end
    end
end
