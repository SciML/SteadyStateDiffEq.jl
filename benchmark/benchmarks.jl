using SteadyStateDiffEq, OrdinaryDiffEq, NonlinearSolve, BenchmarkTools

const SUITE = BenchmarkGroup()

# Steady state of du = p[1] - p[2]*u - u^2
function f_ss(du, u, p, t)
    du[1] = p[1] - p[2] * u[1] - u[1]^2
    return nothing
end
prob_scalar = SteadyStateProblem(ODEFunction(f_ss), [0.5], [1.0, 0.5])

function f_vec(du, u, p, t)
    du[1] = u[2]
    du[2] = -p[1] * u[1] - p[2] * u[2]
    return nothing
end
prob_vec = SteadyStateProblem(ODEFunction(f_vec), [1.0, 0.5], [2.0, 0.3])

# =============================================================================
# Steady-state solves
# =============================================================================

SUITE["solve"] = BenchmarkGroup()

# Rootfind approach (Newton on f(u) = 0)
SUITE["solve"]["ssrootfind_scalar"] = @benchmarkable solve(
    $prob_scalar, SSRootfind()
)
SUITE["solve"]["ssrootfind_vec"] = @benchmarkable solve($prob_vec, SSRootfind())

# Dynamic steady-state (integrate to convergence)
SUITE["solve"]["dynamicss_scalar"] = @benchmarkable solve(
    $prob_scalar, DynamicSS(Tsit5())
)
SUITE["solve"]["dynamicss_vec"] = @benchmarkable solve(
    $prob_vec, DynamicSS(Tsit5())
)

# =============================================================================
# Problem construction
# =============================================================================

SUITE["construct"] = BenchmarkGroup()
SUITE["construct"]["steadystateproblem"] = @benchmarkable SteadyStateProblem(
    $(ODEFunction(f_ss)), [0.5], [1.0, 0.5]
)
