using ModelingToolkit
using ModelingToolkit: t_nounits as t, D_nounits as D
using NonlinearSolve
using SteadyStateDiffEq
using SciMLBase: LinearProblem, NonlinearProblem, SCCNonlinearProblem, successful_retcode
using Test

@variables begin
    a(t), [irreducible = true]
    b(t), [irreducible = true]
    x(t), [irreducible = true]
    y(t), [irreducible = true]
    c(t), [irreducible = true]
    d(t), [irreducible = true]
end

@named model = System([
    D(a) ~ 5 - 3a - b,
    D(b) ~ 5 - a - 2b,
    D(x) ~ a + b - x^2 - y,
    D(y) ~ 3a + b - x - y^2,
    D(c) ~ x + y + 7 - 2c - d,
    D(d) ~ 2x + y + 11 - c - 3d,
], t)

# Retain the linear blocks so this example exercises both SCC solver paths.
reassemble_alg = ModelingToolkit.StructuralTransformations.DefaultReassembleAlgorithm(
    inline_linear_sccs = false
)
sssys = mtkcompile(NonlinearSystem(model); reassemble_alg)
guesses = [a => 0.8, b => 1.8, x => 0.8, y => 1.8, c => 2.8, d => 3.8]
sccprob = SCCNonlinearProblem(sssys, guesses; combine_sccs = false)

@test length(sccprob.probs) == 3
@test sccprob.probs[1] isa LinearProblem
@test sccprob.probs[2] isa NonlinearProblem
@test sccprob.probs[3] isa LinearProblem
@test size(sccprob.probs[1].A) == (2, 2)
@test length(sccprob.probs[2].u0) == 2
@test size(sccprob.probs[3].A) == (2, 2)

sol = solve(sccprob, NewtonRaphson(); abstol = 1e-12, reltol = 1e-12)
states = [a, b, x, y, c, d]
@test successful_retcode(sol)
@test sol[states] ≈ [1, 2, 1, 2, 3, 4] atol = 1e-9
@test maximum(abs, sol.resid) < 1e-9

fullprob = NonlinearProblem(sssys, guesses)
fullsol = solve(fullprob, NewtonRaphson(); abstol = 1e-12, reltol = 1e-12)
@test successful_retcode(fullsol)
@test sol[states] ≈ fullsol[states] atol = 1e-9

@testset "SSRootfind with linear and nonlinear SCCs" for alg in (SSRootfind(), SSRootfind(NewtonRaphson()))
    wrapped = solve(sccprob, alg; abstol = 1e-12, reltol = 1e-12)
    @test successful_retcode(wrapped)
    @test wrapped[states] ≈ sol[states] atol = 1e-9
    @test maximum(abs, wrapped.resid) < 1e-9
    @test wrapped.prob === sccprob
    @test wrapped.original.prob isa SCCNonlinearProblem
end

println("Block types: ", nameof.(typeof.(sccprob.probs)))
println("Steady state [a, b, x, y, c, d]: ", sol[states])
println("Maximum residual: ", maximum(abs, sol.resid))
