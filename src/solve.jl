function solve(prob::AbstractSteadyStateProblem,alg::SteadyStateDiffEqAlgorithm)

  if prob.mass_matrix != I
    error("This solver is not able to use mass matrices.")
  end

  if typeof(prob.u0) <: Number
    u0 = [prob.u0]
  else
    u0 = vec(deepcopy(prob.u0))
  end

  sizeu = size(prob.u0)

  if !isinplace(prob) && (typeof(prob.u0)<:AbstractVector || typeof(prob.u0)<:Number)
    f! = (u,du) -> (du[:] = prob.f(0, u); nothing)
  elseif !isinplace(prob) && typeof(prob.u0)<:AbstractArray
    f! = (u,du) -> (du[:] = vec(prob.f(0, reshape(u, sizeu))); nothing)
  elseif typeof(prob.u0)<:AbstractVector
    f! = (u,du) -> (prob.f(0, u, du); nothing)
  else # Then it's an in-place function on an abstract array
    f! = (u,du) -> (prob.f(0, reshape(u, sizeu),reshape(du, sizeu));
                    u = vec(u); du=vec(du); nothing)
  end

  # du = similar(u)
  # f = (u) -> (f!(du,u); du) # out-of-place version

  if typeof(alg) <: SSRootfind
    res = alg.nlsolve(f!,u0)
    build_solution(prob,alg,res.zero,res.residual_norm;retcode = :Success)
  else
    error("Algorithm not recognized")
  end
end
