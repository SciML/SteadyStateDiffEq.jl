struct SafeTerminationOptions{T1, T2, T3}
    protective_threshold::T1
    patience_steps::Int
    patience_objective_multiplier::T2
    min_max_factor::T3
end

function Base.show(io::IO, s::SafeTerminationOptions)
    print(io,
          "SafeTerminationOptions(protective_threshold = $(s.protective_threshold), patience_steps = $(s.patience_steps), patience_objective_multiplier = $(s.patience_objective_multiplier), min_max_factor = $(s.min_max_factor))")
end

@enumx SafeTerminationReturnCode begin
    Success
    PatienceTermination
    ProtectiveTermination
    Failure
end

@doc doc"""
    SteadyStateTerminationCriteria(mode = :default; abstol::T = 1e-8,
                                   reltol::T = 1e-6, protective_threshold = 1e3,
                                   patience_steps::Int = 30,
                                   patience_objective_multiplier = 3,
                                   min_max_factor = 1.3)

Define the termination criteria for the SteadyStateProblem.

## Termination Conditions

#### Termination on Absolute Tolerance

  * `:abs`: Terminates if ``all \left( | \frac{\partial u}{\partial t} | \leq abstol \right)``
  * `:abs_norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq abstol``
  * `:abs_safe`: Essentially `abs_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
  * `:abs_safe_best`: Same as `:abs_safe` but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination on Relative Tolerance

  * `:rel`: Terminates if ``all \left(| \frac{\partial u}{\partial t} | \leq reltol \times | u | \right)``
  * `:rel_norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|``
  * `:rel_safe`: Essentially `rel_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
  * `:rel_safe_best`: Same as `:rel_safe` but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination using both Absolute and Relative Tolerances

  * `:norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|`` or ``\| \frac{\partial u}{\partial t} \| \leq abstol``
  * `:default`: Check if all values of the derivative is close to zero wrt both relative and absolute tolerance. This is usable for small problems but doesn't scale well for neural networks.

## General Arguments

  * `abstol`: Absolute Tolerance
  * `reltol`: Relative Tolerance

## Arguments specific to `*_safe_*` modes

  * `protective_threshold`: If the objective value increased by this factor wrt initial objective terminate immediately.
  * `patience_steps`: If objective is within `patience_objective_multiplier` factor of the criteria and no improvement within `min_max_factor` has happened then terminate.

"""
struct SteadyStateTerminationCriteria{mode, T, S <:
                                               Union{<:SafeTerminationOptions, Nothing}}
    abstol::T
    reltol::T

    safe_termination_options::S
end

function Base.show(io::IO, s::SteadyStateTerminationCriteria{mode}) where {mode}
    print(io,
          "SteadyStateTerminationCriteria(mode = $(mode), abstol = $(s.abstol), reltol = $(s.reltol)")
    if mode ∈ (:rel_safe, :rel_safe_best, :abs_safe, :abs_safe_best)
        print(io, ", safe_termination_options = ", s.safe_termination_options, ")")
    else
        print(io, ")")
    end
end

function SteadyStateTerminationCriteria(mode = :default; abstol::T = 1e-8,
                                        reltol::T = 1e-6, protective_threshold = 1e3,
                                        patience_steps::Int = 30,
                                        patience_objective_multiplier = 3,
                                        min_max_factor = 1.3) where {T}
    safe_termination_options = if mode ∈
                                  (:rel_safe, :rel_safe_best, :abs_safe, :abs_safe_best)
        SafeTerminationOptions(protective_threshold, patience_steps,
                               patience_objective_multiplier, min_max_factor)
    else
        nothing
    end
    return SteadyStateTerminationCriteria{mode, T, typeof(safe_termination_options)}(abstol,
                                                                                     reltol,
                                                                                     safe_termination_options)
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
    @eval function _get_termination_condition(cond::$(T), storage)
        aType = typeof(cond.abstol)
        nstep = 0
        protective_threshold = aType(cond.safe_termination_options.protective_threshold)
        objective_values = aType[]
        patience_objective_multiplier = cond.safe_termination_options.patience_objective_multiplier

        if $(mode ∈ (:rel_safe_best, :abs_safe_best))
            storage[:best_objective_value] = aType(Inf)
            storage[:best_objective_value_iteration] = 0
        end

        @inbounds function _termination_condition_closure(integrator, abstol, reltol, min_t)
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
            if objective <= criteria
                storage[:return_code] = SafeTerminationReturnCode.Success
                return true
            end

            # Terminate if there has been no improvement for the last 30 steps
            nstep += 1
            push!(objective_values, objective)

            if objective <= typeof(criteria)(patience_objective_multiplier) * criteria
                if nstep >= cond.safe_termination_options.patience_steps
                    last_k_values = objective_values[max(1,
                                                         length(objective_values) -
                                                         cond.safe_termination_options.patience_steps):end]
                    if maximum(last_k_values) <
                       typeof(criteria)(cond.safe_termination_options.min_max_factor) *
                       minimum(last_k_values)
                        storage[:return_code] = SafeTerminationReturnCode.PatienceTermination
                        return true
                    end
                end
            end

            # Protective break
            if objective >= objective_values[1] * protective_threshold * length(du)
                storage[:return_code] = SafeTerminationReturnCode.ProtectiveTermination
                return true
            end

            storage[:return_code] = SafeTerminationReturnCode.Failure
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
    @eval @inline @inbounds function _has_converged(du, u, ::$(typeof(mode_val)),
                                                    abstol, reltol)
        if $(mode == :norm)
            du_norm = norm(du)
            return du_norm <= abstol || du_norm <= reltol * norm(du + u)
        elseif $(mode == :rel)
            return all(abs.(du) .<= reltol .* abs.(u))
        elseif $(mode ∈ (:rel_norm, :rel_safe, :rel_safe_best))
            return norm(du) <= reltol * norm(du .+ u)
        elseif $(mode == :abs)
            return all(abs.(du) .<= abstol)
        elseif $(mode ∈ (:abs_norm, :abs_safe, :abs_safe_best))
            return norm(du) <= abstol
        elseif $(mode == :default)
            return all((abs.(du) .<= abstol) .| (abs.(du) .<= reltol .* abs.(u)))
        end
    end
end
