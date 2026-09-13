using SteadyStateDiffEq: NonlinearProblem, scale, unscale, solve, successful_retcode
using NonlinearSolve: NewtonRaphson

# Unknowns are pressure in Pa and temperature in K.
function residual(u, p)
    pressure, temperature = u
    return [pressure - 2000temperature - 500_000, temperature^2 - 350^2]
end

prob = NonlinearProblem(residual, [1_000_000.0, 300.0])
scaled_prob, scaling = scale(prob)
scaled_sol = solve(scaled_prob, NewtonRaphson(); abstol = 1e-12, reltol = 0.0)
sol = unscale(scaled_sol, scaling)

println("Variable scales: ", scaling.variable_scales)
println("Residual scales: ", scaling.residual_scales)
println("Scaled initial guess: ", scaled_prob.u0)
println("Scaled solution: ", scaled_sol.u)
println("Pressure [Pa], temperature [K]: ", sol.u)
println("Original residual: ", sol.resid)
@assert successful_retcode(sol)
@assert sol.u ≈ [1_200_000.0, 350.0]
