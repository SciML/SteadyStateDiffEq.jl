using SteadyStateDiffEq, ModelingToolkit, SciMLBase, Test
using SteadyStateDiffEq: scale, unscale
using ModelingToolkit: t_nounits as t, D_nounits as D

@testset "Scaled steady-state SCC interface" begin
    @variables a(t) b(t) x(t) y(t) c(t) d(t) [irreducible = true]
    @parameters k = 2.0
    @parameters q
    @named model = System([
        D(a) ~ 2.5q - 3a - b,
        D(b) ~ 5k - a - 2b,
        D(x) ~ a^2 + b - x^2 - y,
        D(y) ~ a + b^2 - x - y^2,
        D(c) ~ x + y + 7k - 2c - d,
        D(d) ~ 2x + y + 11k - c - 3d,
    ], t; bindings = Dict(q => 2k))
    sys = mtkcompile(model)
    prob = SteadyStateProblem(sys,
        [a => 0.8, b => 1.8, x => 0.8, y => 1.8, c => 2.8, d => 3.8, k => 1.0])
    initial = copy(prob.u0)
    scaled, mapping = scale(prob)
    nonlinear = mapping.nonlinear_scaling.prob
    @test nonlinear.u0 == prob[unknowns(nonlinear.f.sys)]
    scaled_sol = solve(scaled, SCC())
    sol = unscale(scaled_sol, mapping)
    @test sol isa SteadyStateSolution
    @test sol.prob === prob
    @test successful_retcode(sol)
    @test sol[[a, b, x, y, c, d]] ≈ [1, 2, 1, 2, 3, 4] atol = 1e-9
    @test maximum(abs, sol.resid) < 1e-9
    @test prob.u0 == initial
    @test prob.ps[k] == sol.ps[k] == 1.0
    @test sol.ps[q] == 2.0
    blocks = scaled_sol.original.prob.probs
    @test length(blocks) == 3
    @test blocks[1] isa LinearProblem
    @test blocks[2] isa NonlinearProblem
    @test blocks[3] isa LinearProblem
    @test sol.stats.nsteps > 0

    direct = solve(prob, SCC())
    @test successful_retcode(direct)
    @test direct.u ≈ sol.u atol = 1e-9
    @test direct.prob === prob

    changed, changed_mapping = scale(remake(prob; p = [k => 1.1]))
    changed_sol = unscale(solve(changed, SCC()), changed_mapping)
    @test successful_retcode(changed_sol)
    @test changed_sol.ps[k] == 1.1
    @test changed_sol[[a, b, x, y, c, d]] ≈ 1.1 .* [1, 2, 1, 2, 3, 4] atol = 1e-8
    @test prob.ps[k] == 1.0
end

@testset "Numerical steady-state scaling" for iip in (false, true)
    f = iip ? ((r, u, p, t) -> (r .= [u[1]^2 - p[1], u[2] - p[2]])) :
        ((u, p, t) -> [u[1]^2 - p[1], u[2] - p[2]])
    prob = SteadyStateProblem(f, [1e3, 1e-3], [2e6, 3e-3])
    scaled, mapping = scale(prob)
    sol = unscale(solve(scaled, SCC()), mapping)
    @test successful_retcode(sol)
    @test sol.prob === prob
    @test sol.u ≈ [sqrt(2e6), 3e-3]
    @test maximum(abs, sol.resid) < 1e-8
    @test prob.u0 == [1e3, 1e-3]
end

@testset "SCC failure stops dependent blocks" begin
    first = NonlinearProblem((u, p) -> u .^ 2 .- 2, [10.0])
    second = NonlinearProblem((u, p) -> u, [1.0])
    prob = SCCNonlinearProblem((first, second),
        (Returns(nothing), (p, sols) -> error("Failed upstream block must stop the sweep")))
    sol = solve(prob, SCC(); maxiters = 1, abstol = 1e-14)
    @test !successful_retcode(sol)
    @test length(sol.original) == 1
    @test isnan(sol.resid[2])
    @test first.u0 == [10.0]
end

@testset "SCC without symbolic equations" begin
    first = NonlinearProblem((u, p) -> [1e6 * (u[1]^2 - 2), 1e-6 * (u[2] - 3)], [1.0, 2.0])
    second = NonlinearProblem((u, p) -> [u[1]^2 - sum(p)], [2.0], zeros(2))
    prob = SCCNonlinearProblem((first, second),
        (Returns(nothing), (p, sols) -> (p .= sols[1].u)))
    sol = solve(prob, SCC(); abstol = 1e-8)
    @test successful_retcode(sol)
    @test sol.u ≈ [sqrt(2), 3, sqrt(sqrt(2) + 3)] atol = 1e-9
    @test first.u0 == [1.0, 2.0]
    @test second.u0 == [2.0]
    @test second.p == zeros(2)

    linear = LinearProblem(reshape([2.0], 1, 1), [4.0])
    linear_scc = SCCNonlinearProblem((linear,), (Returns(nothing),))
    linear_sol = solve(linear_scc, SCC())
    @test successful_retcode(linear_sol)
    @test linear_sol.u ≈ [2.0]
end
