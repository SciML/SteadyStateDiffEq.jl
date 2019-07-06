using SteadyStateDiffEq, DiffEqBase, NLsolve, Sundials
using Test

function f(du,u,p,t)
  du[1] = 2 - 2u[1]
  du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob = SteadyStateProblem(f,u0)
abstol = 1e-8
sol = solve(prob,SSRootfind())
@test sol.retcode == :Success
p = nothing

du = zeros(2)
f(du,sol.u,nothing,0)
@test maximum(du) < 1e-11

prob = ODEProblem(f,u0,(0.0,1.0))
prob = SteadyStateProblem(prob)
sol = solve(prob,SSRootfind(nlsolve = (f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,autodiff=:forward,method=:newton,iterations=Int(1e6),ftol=abstol);res.zero) ))
@test sol.retcode == :Success

f(du,sol.u,nothing,0)
@test du == [0,0]

# Use Sundials
sol = solve(prob,SSRootfind(nlsolve = (f,u0,abstol) -> (res=Sundials.kinsol(f,u0)) ))
@test sol.retcode == :Success
f(du,sol.u,nothing,0)
@test du == [0,0]

using OrdinaryDiffEq
sol = solve(prob,DynamicSS(Rodas5()))
@test sol.retcode == :Success

f(du,sol.u,p,0)
@test du ≈ [0,0] atol = 1e-7

sol = solve(prob,DynamicSS(Rodas5(),tspan=1e-3))
@test sol.retcode != :Success

sol = solve(prob,DynamicSS(CVODE_BDF()),dt=1.0)
@test sol.retcode == :Success

f(du,sol.u,p,0)
@test du ≈ [0,0] atol = 1e-6
