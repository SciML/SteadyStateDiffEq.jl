using SteadyStateDiffEq, NonlinearSolve, OrdinaryDiffEq, Test
using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using SciMLBase: SCCNonlinearProblem
using SymbolicIndexingInterface: state_values, variable_symbols

# Solving a `SteadyStateProblem` through its stored `lowered_problem` is an
# implementation detail: the solution is one of the `SteadyStateProblem`, with
# `u` in `prob.u0` order and the time-dependent system (SciML/SteadyStateDiffEq.jl#171).
function test_original_coordinates(sol, prob, sys, expected; atol)
    @test successful_retcode(sol)
    @test sol.prob isa SteadyStateProblem
    @test sol.prob.f.sys === prob.f.sys
    @test ModelingToolkit.is_time_dependent(sol.prob.f.sys)
    @test length(sol.u) == length(prob.u0)
    @test sol.u ≈ [expected[s] for s in unknowns(sys)] atol = atol
    @test all(isequal.(variable_symbols(sol), unknowns(sys)))
    @test maximum(abs, sol.resid) < 10atol
    for (s, v) in expected
        @test sol[s] ≈ v atol = atol
    end
    return nothing
end

@testset "Lowering eliminates every unknown" begin
    @parameters a = 1.0 b = 2.0
    @variables x(t) = 1.0 y(t) = 1.0 z(t) = 1.0
    eqs = [D(x) ~ a - x, D(y) ~ x - b * y, D(z) ~ y - z]
    sys = mtkcompile(System(eqs, t; name = :chain))
    op = [x => 0.1, y => 0.2, z => 0.3]
    prob = SteadyStateProblem(sys, op)
    @test isempty(state_values(NonlinearProblem(prob)))
    expected = Dict(x => 1.0, y => 0.5, z => 0.5)

    @testset "alg=$alg" for alg in (
            nothing, NewtonRaphson(), SSRootfind(), SSRootfind(NewtonRaphson()),
        )
        sol = alg === nothing ? solve(prob) : solve(prob, alg)
        test_original_coordinates(sol, prob, sys, expected; atol = 1.0e-12)
        @test SteadyStateProblem(sol.prob.f.sys, op) isa SteadyStateProblem
    end
end

@testset "Lowering reorders the unknowns" begin
    @parameters α = 1.5 β = 1.0 γ = 3.0 δ = 1.0
    @variables u(t) = 1.0 v(t) = 1.0
    eqs = [D(u) ~ α * u - β * u * v, D(v) ~ -γ * v + δ * u * v]
    sys = mtkcompile(System(eqs, t; name = :lv))
    prob = SteadyStateProblem(sys, [u => 2.9, v => 1.6])
    lowered = NonlinearProblem(prob)
    @test !all(isequal.(variable_symbols(lowered), unknowns(sys)))
    expected = Dict(u => 3.0, v => 1.5)

    @testset "alg=$alg" for alg in (
            nothing, NewtonRaphson(), SSRootfind(NewtonRaphson()),
        )
        sol = alg === nothing ? solve(prob) : solve(prob, alg)
        test_original_coordinates(sol, prob, sys, expected; atol = 1.0e-10)

        uidx = findfirst(isequal(u), unknowns(sys))
        ssol = alg === nothing ? solve(prob; save_idxs = [uidx]) :
            solve(prob, alg; save_idxs = [uidx])
        @test ssol.u ≈ [3.0] atol = 1.0e-10
    end

    # A `u0` override is given in `prob`'s coordinates.
    sol = solve(prob, NewtonRaphson(); u0 = [1.4, 2.8])
    @test sol.u ≈ [expected[s] for s in unknowns(sys)] atol = 1.0e-10
end

@testset "SCC lowering" begin
    @variables a(t) b(t) x(t) [irreducible = true]
    @named model = System(
        [D(a) ~ 5 - 3a - b, D(b) ~ 5 - a - 2b, D(x) ~ a + b - x^3], t
    )
    sys = mtkcompile(model)
    prob = SteadyStateProblem(sys, [a => 0.8, b => 1.8, x => 0.8])
    @test NonlinearProblem(prob) isa SCCNonlinearProblem
    expected = Dict(a => 1.0, b => 2.0, x => cbrt(3))

    @testset "alg=$alg" for alg in (
            NewtonRaphson(), SSRootfind(NewtonRaphson()),
            DynamicSS(Tsit5()), SICNM(Rodas5P()),
        )
        sol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10)
        test_original_coordinates(sol, prob, sys, expected; atol = 1.0e-8)
        @test sol.original.prob isa SCCNonlinearProblem

        xidx = findfirst(isequal(x), unknowns(sys))
        ssol = solve(prob, alg; abstol = 1.0e-10, reltol = 1.0e-10, save_idxs = [xidx])
        @test ssol.u ≈ [cbrt(3)] atol = 1.0e-8
    end
end
