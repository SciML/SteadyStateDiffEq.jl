using Test, SciMLBase, SteadyStateDiffEq, OrdinaryDiffEq
using SymbolicIndexingInterface: SymbolCache, state_values, setu

@testset "SICNM assignment with a coupled domain constraint" begin
    @testset "iip=$iip, direct_setter=$direct_setter, synchronize=$synchronize" for
        iip in (false, true), direct_setter in (false, true),
            synchronize in (:state, :integrator, :step, :solve)

        evaluations = Ref(0)
        function g(u, p, t)
            evaluations[] += 1
            return [p[1] - u[1], sqrt(u[1] - u[2]) - p[2]]
        end
        f = iip ? ((du, u, p, t) -> (du .= g(u, p, t))) : g
        sys = SymbolCache([:x, :y], [:a, :b])
        prob = SteadyStateProblem(ODEFunction{iip}(f; sys), [4.0, 3.0], [3.0, 1.0])
        cache = init(prob, SICNM(Rodas5P()); abstol = 1.0e-10, reltol = 1.0e-10)
        before = evaluations[]
        if direct_setter
            setu(cache, [:x, :y])(cache, [2.0, 1.0])
        else
            cache[[:x, :y]] = [2.0, 1.0]
        end
        @test evaluations[] == before
        if synchronize === :state
            @test state_values(cache) == [2.0, 1.0]
            @test cache.integrator.u ≈ [2.0, 1.0, 1.0, 1.0]
            after = evaluations[]
            @test cache[[:x, :y]] == [2.0, 1.0]
            @test evaluations[] == after
        elseif synchronize === :integrator
            @test cache.integrator.u ≈ [2.0, 1.0, 1.0, 1.0]
        elseif synchronize === :step
            step!(cache)
        end
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test sol.u ≈ [3.0, 2.0] atol = 1.0e-8
        @test g(sol.u, prob.p, 0.0) ≈ zeros(2) atol = 1.0e-8
        @test prob.u0 == [4.0, 3.0]
    end
end

@testset "SICNM mutation after synchronization and stepping" begin
    @testset "iip=$iip, direct_setter=$direct_setter" for iip in (false, true),
            direct_setter in (false, true)

        sys = SymbolCache([:x], [:a])
        f = iip ? ((du, u, p, t) -> (du .= p .- u)) : ((u, p, t) -> p .- u)
        prob = SteadyStateProblem(ODEFunction{iip}(f; sys), [0.0], [2.0])
        cache = init(prob, SICNM(Rodas5P()); abstol = 1.0e-10, reltol = 1.0e-10)
        for (u, p) in ((1.0, 3.0), (-2.0, 4.0))
            step!(cache)
            cache.ps[:a] = p
            if direct_setter
                setu(cache, :x)(cache, u)
            else
                cache[:x] = u
            end
            @test cache[:x] == u
            @test cache.integrator.u ≈ [u, p - u]
        end
        sol = solve!(cache)
        @test successful_retcode(sol)
        @test sol.u ≈ [4.0] atol = 1.0e-8
    end
end
