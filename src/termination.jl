struct SteadyStateTerminationCriteria{mode, T}
    abstol::T
    reltol::T
end

function Base.show(io::IO, s::SteadyStateTerminationCriteria{mode}) where {mode}
    print(io,
          "SteadyStateTerminationCriteria(mode = $(mode), abstol = $(s.abstol), reltol = $(s.reltol))")
end

function SteadyStateTerminationCriteria(mode = :default; abstol::T = 1e-8,
                                        reltol::T = 1e-8) where {T}
    SteadyStateTerminationCriteria{mode, T}(abstol, reltol)
end

_get_termination_mode(::SteadyStateTerminationCriteria{mode}) where {mode} = Val(mode)

for mode in (:default, :norm, :rel, :rel_norm, :abs, :abs_norm)
    T = SteadyStateTerminationCriteria{mode}
    mode_val = Val(mode)
    @eval function _get_termination_condition(::$(T), storage = nothing)
        function _termination_condition_closure(integrator, abstol, reltol, min_t)
            return _has_converged(get_du(integrator), integrator.u, $(mode_val), abstol,
                                  reltol)
        end
    end
end

for mode in (:rel_safe, :rel_safe_best, :abs_safe, :abs_safe_best)
    T = SteadyStateTerminationCriteria{mode}
    mode_val = Val(mode)
    @eval function _get_termination_condition(cond::$(T), storage = nothing)
        nstep, protective_threshold, objective_values = 0, T(1e3), T[]

        aType = typeof(cond.abstol)

        if $(mode ∈ (:rel_safe_best, :abs_safe_best))
            @assert storage!==nothing "Storage must be provided for $(mode) termination criteria"

            storage[:best_objective_value] = aType(Inf)
            storage[:best_objective_value_iteration] = 0
        end

        function _termination_condition_closure(integrator, abstol, reltol, min_t)
            du = get_du(integrator)
            u = integrator.u

            if $(mode ∈ (:abs_safe, :abs_safe_best))
                objective = norm(du)
                criteria = abstol
            else
                objective = norm(du) / (norm(du .+ u) + eps(aType))
                criteria = reltol
            end

            if $(mode ∈ (:rel_safe_best, :abs_safe_best))
                if objective < storage[:best_objective_value]
                    storage[:best_objective_value] = objective
                    storage[:best_objective_value_iteration] = nstep + 1
                end
            end

            # Main Termination Criteria
            objective <= criteria && return true

            # Terminate if there has been no improvement for the last 30 steps
            nstep += 1
            push!(objective_values, objective)
            last_30_values = objective_values[max(1, length(objective_values) - nstep):end]

            objective <= 3 * criteria && nstep >= 30 &&
                maximum(last_30_values) < 1.3 * minimum(last_30_values) && return true

            # Protective break
            objective >= objective_values[1] * protective_threshold * length(du) &&
                return true

            return false
        end

        return _termination_condition_closure
    end
end

# Convergence Criterions
@inline function _has_converged(du, u,
                                cond::SteadyStateTerminationCriteria,
                                abstol = cond.abstol,
                                reltol = cond.reltol)
    return _has_converged(du, u, _get_termination_mode(cond), abstol, reltol)
end

for mode in (:default, :norm, :rel, :rel_norm, :rel_safe, :rel_safe_best, :abs, :abs_norm,
             :abs_safe, :abs_safe_best)
    mode_val = Val(mode)
    @eval begin @inline @inbounds function _has_converged(du, u, ::$(typeof(mode_val)),
                                                          abstol, reltol)
        if $(mode == :norm)
            du_norm = norm(du)
            return du_norm <= abstol && du_norm <= reltol * norm(du + u)
        elseif $(mode == :rel)
            return all(abs.(du) .<= reltol .* abs.(u))
        elseif $(mode ∈ (:rel_norm, :rel_safe, :rel_safe_best))
            return norm(du) <= reltol * norm(du .+ u)
        elseif $(mode == :abs)
            return all(abs.(du) .<= abstol)
        elseif $(mode ∈ (:abs_norm, :abs_safe, :abs_safe_best))
            return norm(du) <= abstol
        elseif $(mode == :default)
            return all((abs.(du) .<= abstol) .& (abs.(du) .<= reltol .* abs.(u)))
        end
    end end
end
