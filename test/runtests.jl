using SteadyStateDiffEq, DiffEqBase, NLsolve, Sundials
using Base.Test

function f(t,u,du)
  du[1] = 2 - 2u[1]
  du[2] = u[1] - 4u[2]
end
u0 = zeros(2)
prob = SteadyStateProblem(f,u0)

sol = solve(prob,SSRootfind())

du = zeros(2)
f(0,sol.u,du)
@test maximum(du) < 1e-11

prob = ODEProblem(f,u0,(0.0,1.0))
prob = SteadyStateProblem(prob)
sol = solve(prob,SSRootfind(nlsolve = (f,u0) -> (res=NLsolve.nlsolve(f,u0,autodiff=true,method=:newton,iterations=Int(1e6));res.zero) ))

f(0,sol.u,du)
@ test du == [0,0]

# Use Sundials
sol = solve(prob,SSRootfind(nlsolve = Sundials.kinsol))

f(0,sol.u,du)
@ test du == [0,0]
