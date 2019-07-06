function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem,
                            alg::SteadyStateDiffEqAlgorithm,args...;
                            abstol=1e-8,kwargs...)

  if prob.f.mass_matrix != I
    error("This solver is not able to use mass matrices.")
  end

  if typeof(prob.u0) <: Number
    u0 = [prob.u0]
  else
    u0 = vec(deepcopy(prob.u0))
  end

  sizeu = size(prob.u0)
  p = prob.p

  if !isinplace(prob) && (typeof(prob.u0)<:AbstractVector || typeof(prob.u0)<:Number)
    f! = (du,u) -> (du[:] = prob.f(u,p,Inf); nothing)
  elseif !isinplace(prob) && typeof(prob.u0)<:AbstractArray
    f! = (du,u) -> (du[:] = vec(prob.f(reshape(u, sizeu),p,Inf)); nothing)
  elseif typeof(prob.u0)<:AbstractVector
    f! = (du,u) -> (prob.f(du,u,p,Inf); nothing)
  else # Then it's an in-place function on an abstract array
    f! = (du,u) -> (prob.f(reshape(du, sizeu),
                    reshape(u, sizeu),p,Inf);
                    du=vec(du); nothing)
  end

  # du = similar(u)
  # f = (u) -> (f!(du,u); du) # out-of-place version

  if typeof(alg) <: SSRootfind
    u = alg.nlsolve(f!,u0,abstol)
    resid = similar(u)
    f!(resid,u)
    DiffEqBase.build_solution(prob,alg,u,resid;retcode = :Success)
  else
    error("Algorithm not recognized")
  end
end

function DiffEqBase.__solve(prob::DiffEqBase.AbstractSteadyStateProblem,
                            alg::DynamicSS,args...;save_everystep=false,
                            save_start=false,kwargs...)

  tspan = alg.tspan isa Tuple ? alg.tspan : (zero(alg.tspan), alg.tspan)
  _prob = ODEProblem(prob.f,prob.u0,tspan,prob.p)
  sol = solve(_prob,alg.alg,args...;kwargs...,
              callback=TerminateSteadyState(alg.abstol,alg.reltol),
              save_everystep=save_everystep,save_start=save_start)
  if isinplace(prob)
    du = similar(sol.u[end])
    prob.f(du, sol.u[end], prob.p, sol.t[end])
  else
    du = prob.f(sol.u[end], prob.p, sol.t[end])
  end
  if sol.retcode == :Terminated && all(abs(d) <= abstol ||
      abs(d) <= reltol*abs(u) for (d,abstol, reltol, u) in
      zip(du, Iterators.cycle(alg.abstol), Iterators.cycle(alg.reltol), sol.u[end]))
    _sol = DiffEqBase.build_solution(prob,alg,sol.u[end],du;retcode = :Success)
  else
    _sol = DiffEqBase.build_solution(prob,alg,sol.u[end],du;retcode = :Failure)
  end
  _sol
end
