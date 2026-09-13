using SteadyStateDiffEq: scale, unscale, successful_retcode
using NonlinearSolve, SciMLBase, ForwardDiff, Test
using SymbolicIndexingInterface: SymbolCache

@testset "Nonlinear problem scaling" begin
    f(u, p) = [u[1] - 2000u[2] - p, u[2]^2 - 350^2]
    j(u, p) = [1.0 -2000.0; 0.0 2u[2]]
    u0 = [1_000_000.0, 300.0]
    p = 500_000.0

    @testset "Automatic scaling, inplace=$iip analytic=$analytic" for
            iip in (false, true), analytic in (false, true)
        fn = if iip
            f! = (res, u, p) -> (res .= f(u, p))
            jac = analytic ? ((J, u, p) -> (J .= j(u, p))) : nothing
            NonlinearFunction{true}(f!; jac)
        else
            NonlinearFunction{false}(f; jac = analytic ? j : nothing)
        end
        prob = NonlinearProblem(fn, copy(u0), p)
        scaled, mapping = scale(prob)
        @test scaled isa NonlinearProblem
        @test isinplace(scaled) == iip
        @test mapping.variable_scales == [1e6, 300]
        @test mapping.residual_scales == [1e6, 180_000]
        @test scaled.u0 == [1, 1]
        z = [1.1, 0.9]
        actual = if iip
            res = similar(z)
            scaled.f(res, z, p)
            res
        else
            scaled.f(z, p)
        end
        @test actual ≈ f(mapping.variable_scales .* z, p) ./ mapping.residual_scales
        if analytic
            J = if iip
                mat = zeros(2, 2)
                scaled.f.jac(mat, z, p)
                mat
            else
                scaled.f.jac(z, p)
            end
            @test J ≈ [1.0 -0.6; 0.0 z[2]]
        end
        for alg in (NewtonRaphson(), TrustRegion())
            scaled_sol = solve(scaled, alg; abstol = 1e-12, reltol = 0.0)
            sol = unscale(scaled_sol, mapping)
            @test successful_retcode(sol)
            @test sol.u ≈ [1_200_000, 350] rtol = 1e-10
            @test sol.resid == f(sol.u, p)
            @test maximum(abs, sol.resid) < 1e-5
            @test sol.prob === prob
            @test sol.original === scaled_sol
            @test sol.stats === scaled_sol.stats
        end
        @test prob.u0 == u0
        @test prob.f === fn
    end

    @testset "Explicit scales and symbolic solution" begin
        symbols = SymbolCache([:pressure, :temperature], [:offset])
        fn = NonlinearFunction{false}((u, p) -> f(u, p[1]); sys = symbols)
        prob = NonlinearProblem(fn, copy(u0), [p])
        scaled, mapping = scale(prob;
            variable_scales = [2e5, 100.0], residual_scales = [1e5, 1e4])
        @test scaled.u0 == [5, 3]
        @test scaled.f.sys === nothing
        sol = unscale(solve(scaled, NewtonRaphson(); abstol = 1e-12), mapping)
        @test sol[:pressure] ≈ 1_200_000
        @test sol[:temperature] ≈ 350
        changed = remake(scaled; p = [400_000.0])
        changed_sol = unscale(solve(changed, NewtonRaphson(); abstol = 1e-12), mapping)
        @test changed_sol[:pressure] ≈ 1_100_000
        @test changed_sol.prob.p == [400_000]
        @test maximum(abs, changed_sol.resid) < 1e-5
        @test prob.p == [p]
    end

    @testset "Scalar, zero guess, and failure" begin
        prob = NonlinearProblem((u, p) -> u^2 - 2, 1.0f0)
        scaled, mapping = scale(prob)
        @test mapping.variable_scales isa Float32
        @test mapping.residual_scales isa Float32
        sol = unscale(solve(scaled, NewtonRaphson(); abstol = 1f-6), mapping)
        @test sol.u ≈ sqrt(2.0f0)
        @test sol.resid == prob.f(sol.u, prob.p)
        zero_prob, zero_map = scale(NonlinearProblem((u, p) -> u - 2, 0.0))
        @test zero_map.variable_scales == 1
        @test unscale(solve(zero_prob, NewtonRaphson()), zero_map).u ≈ 2
        hard, hard_map = scale(NonlinearProblem((u, p) -> u^2 - 2, 10.0))
        failed = solve(hard, NewtonRaphson(); maxiters = 1, abstol = 1e-14)
        @test !successful_retcode(failed)
        recovered = unscale(failed, hard_map)
        @test recovered.retcode == failed.retcode
        @test recovered.resid == recovered.u^2 - 2
    end

    @testset "Explicit residual scales avoid Jacobian evaluation" begin
        fn = NonlinearFunction{false}(f; jac = (u, p) -> error("Jacobian evaluated"))
        prob = NonlinearProblem(fn, copy(u0), p)
        scaled, mapping = scale(prob; residual_scales = [1e6, 180_000.0])
        @test scaled.f(scaled.u0, p) ≈ f(u0, p) ./ mapping.residual_scales
    end

    @testset "Validation" begin
        prob = NonlinearProblem(f, copy(u0), p)
        @test_throws ArgumentError scale(prob; variable_scales = [0.0, 1.0])
        @test_throws ArgumentError scale(prob; residual_scales = [Inf, 1.0])
        @test_throws DimensionMismatch scale(prob; variable_scales = [1.0])
        @test_throws ArgumentError scale(remake(prob; u0 = [NaN, 300.0]))
        @test_throws ArgumentError scale(NonlinearProblem(f, u0, p; lb = [0.0, 0.0]))
        @test_throws ArgumentError scale(NonlinearProblem(f, u0, p; callback = identity))
    end
end
