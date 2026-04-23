using SteadyStateDiffEq, NonlinearSolve, Sundials, OrdinaryDiffEq, DiffEqCallbacks, Test
using DiffEqBase: DEVerbosity
using NonlinearSolve.NonlinearSolveBase
using NonlinearSolve.NonlinearSolveBase: NormTerminationMode, RelTerminationMode,
    RelNormTerminationMode,
    AbsTerminationMode, AbsNormTerminationMode

function f(du, u, p, t)
    du[1] = 2 - 2u[1]
    return du[2] = u[1] - 4u[2]
end

u0 = zeros(2)
prob = SteadyStateProblem(f, u0)

@testset "NonlinearSolve: $(nameof(typeof(alg)))" for alg in (
        nothing,
        NewtonRaphson(; autodiff = AutoFiniteDiff()), KINSOL(),
    )
    sol = solve(prob, SSRootfind(alg))
    @test SciMLBase.successful_retcode(sol.retcode)

    du = zeros(2)
    f(du, sol.u, nothing, 0)
    @test maximum(du) < 1.0e-11
end

@testset "OrdinaryDiffEq" begin
    du = zeros(2)
    p = nothing

    sol = solve(prob, DynamicSS(Tsit5()); abstol = 1.0e-9, reltol = 1.0e-9)
    @test SciMLBase.successful_retcode(sol.retcode)
    @test sol.stats !== nothing
    @test sol.stats === sol.original.stats

    f(du, sol.u, p, 0)
    @test du ≈ [0, 0] atol = 1.0e-7

    sol = solve(prob, DynamicSS(Tsit5(), tspan = 1.0e-3))
    @test sol.retcode != ReturnCode.Success

    sol = solve(prob, DynamicSS(CVODE_BDF()), dt = 1.0)
    @test SciMLBase.successful_retcode(sol.retcode)

    # test autodiff
    sol = solve(prob, DynamicSS(Rodas5P()))

    @test SciMLBase.successful_retcode(sol.retcode)

    # scalar save_idxs
    scalar_sol = solve(prob, DynamicSS(CVODE_BDF()), dt = 1.0, save_idxs = 1)
    @test scalar_sol[1] ≈ sol[1] atol = 1.0e-6

    f(du, sol.u, p, 0)
    @test du ≈ [0, 0] atol = 1.0e-6
end

# Float32
u0 = [0.0f0, 0.0f0]

function foop(u, p, t)
    @test eltype(u) == eltype(u0)
    dx = 2 - 2u[1]
    dy = u[1] - 4u[2]
    return [dx, dy]
end

function fiip(du, u, p, t)
    @test eltype(u) == eltype(u0)
    du[1] = 2 - 2u[1]
    return du[2] = u[1] - 4u[2]
end

tspan = (0.0f0, 1.0f0)
proboop = SteadyStateProblem(foop, u0)
prob = SteadyStateProblem(fiip, u0)

sol = solve(proboop, DynamicSS(Tsit5(), tspan = 1.0f-3))
@test typeof(u0) == typeof(sol.u)
proboop = SteadyStateProblem(ODEProblem(foop, u0, tspan))
sol2 = solve(proboop, DynamicSS(Tsit5()); abstol = 1.0e-4)
@test typeof(u0) == typeof(sol2.u)

sol = solve(prob, DynamicSS(Tsit5(), tspan = 1.0f-3))
@test typeof(u0) == typeof(sol.u)
prob = SteadyStateProblem(ODEProblem(fiip, u0, tspan))
sol2 = solve(prob, DynamicSS(Tsit5()); abstol = 1.0e-4)
@test typeof(u0) == typeof(sol2.u)

for termination_condition in [
        NormTerminationMode(SteadyStateDiffEq.infnorm), RelTerminationMode(), RelNormTerminationMode(SteadyStateDiffEq.infnorm),
        AbsTerminationMode(), AbsNormTerminationMode(SteadyStateDiffEq.infnorm),
    ]
    sol_tc = solve(prob, DynamicSS(Tsit5()); termination_condition)
    @show sol_tc.retcode, termination_condition
    @test SciMLBase.successful_retcode(sol_tc.retcode)
    @test sol_tc.u ≈ sol2.u
end

# Complex u
u0 = [1.0im]

function fcomplex(du, u, p, t)
    return du[1] = (0.1im - 1) * u[1]
end

prob = SteadyStateProblem(ODEProblem(fcomplex, u0, (0.0, 1.0)))
sol = solve(prob, DynamicSS(Tsit5()))
@test SciMLBase.successful_retcode(sol.retcode)
@test abs(sol.u[end]) < 1.0e-8

# Callbacks
u0 = zeros(2)
prob = SteadyStateProblem(f, u0)
saved_values = SavedValues(Float64, Vector{Float64})
cb = SavingCallback(
    (u, t, integrator) -> copy(u), saved_values, save_everystep = true,
    save_start = true
)
sol = solve(
    prob, DynamicSS(Tsit5()); callback = cb, save_everystep = true,
    save_start = true
)
@test SciMLBase.successful_retcode(sol.retcode)
@test isapprox(saved_values.saveval[end], sol.u)

# Verbose kwarg contract for DynamicSS:
# The outer `solve` is reached via either `NonlinearSolve.solve` (which forwards
# a `NonlinearVerbosity`) or directly. The inner ODE solve only understands
# `Val{true/false}`, `Bool`, or `DEVerbosity`, so `DynamicSS.__solve` strips the
# top-level `verbose` kwarg unconditionally and expects users to thread an
# ODE-layer verbosity through `odesolve_kwargs = (verbose = ..., )` if they
# want to override the ODE layer's `DEFAULT_VERBOSE`.
@testset "DynamicSS verbose kwarg handling" begin
    u0 = zeros(2)
    prob = SteadyStateProblem(f, u0)

    # Default path: no verbose in either kwargs or odesolve_kwargs.
    sol = solve(prob, DynamicSS(Tsit5()))
    @test SciMLBase.successful_retcode(sol.retcode)

    # NonlinearSolve-layer verbosity in outer kwargs gets stripped before the
    # inner ODE solve (otherwise the ODE's `_process_verbose_param` would
    # MethodError on `NonlinearVerbosity`).
    sol = solve(
        prob, DynamicSS(Tsit5());
        verbose = NonlinearSolveBase.NonlinearVerbosity()
    )
    @test SciMLBase.successful_retcode(sol.retcode)

    # ODE-layer verbosity: threaded through `odesolve_kwargs` as a proper
    # `DEVerbosity`. DiffEqBase v7 intentionally rejects a bare `Bool` here
    # (`_process_verbose_param(::Bool)` throws), so callers must pass a
    # `DEVerbosity` or one of the `SciMLLogging` presets.
    sol = solve(
        prob, DynamicSS(Tsit5());
        odesolve_kwargs = (verbose = DEVerbosity(),)
    )
    @test SciMLBase.successful_retcode(sol.retcode)
end
