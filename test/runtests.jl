using SteadyStateDiffEq, DiffEqBase, NLsolve, Sundials
using Base.Test

function f(du,u,p,t)
  du[1] = 2 - 2u[1]
  du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob = SteadyStateProblem(f,u0)
abstol = 1e-8
sol = solve(prob,SSRootfind())

du = zeros(2)
f(du,sol.u,nothing,0)
@test maximum(du) < 1e-11

prob = ODEProblem(f,u0,(0.0,1.0))
prob = SteadyStateProblem(prob)
sol = solve(prob,SSRootfind(nlsolve = (f,u0,abstol) -> (res=NLsolve.nlsolve(f,u0,autodiff=true,method=:newton,iterations=Int(1e6),ftol=abstol);res.zero) ))

f(du,sol.u,nothing,0)
@test du == [0,0]

# Use Sundials
sol = solve(prob,SSRootfind(nlsolve = (f,u0,abstol) -> (res=Sundials.kinsol(f,u0)) ))
f(du,sol.u,nothing,0)
@test du == [0,0]
