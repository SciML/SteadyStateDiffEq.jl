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

@enumx SteadyStateTerminationMode begin
    Default
    Norm
    Rel
    RelNorm
    Abs
    AbsNorm
    RelSafe
    RelSafeBest
    AbsSafe
    AbsSafeBest
end

const BASIC_TERMINATION_MODES = (SteadyStateTerminationMode.Default,
                                 SteadyStateTerminationMode.Norm,
                                 SteadyStateTerminationMode.Rel,
                                 SteadyStateTerminationMode.RelNorm,
                                 SteadyStateTerminationMode.Abs,
                                 SteadyStateTerminationMode.AbsNorm)

const SAFE_TERMINATION_MODES = (SteadyStateTerminationMode.RelSafe,
                                SteadyStateTerminationMode.RelSafeBest,
                                SteadyStateTerminationMode.AbsSafe,
                                SteadyStateTerminationMode.AbsSafeBest)

const SAFE_BEST_TERMINATION_MODES = (SteadyStateTerminationMode.RelSafeBest,
                                     SteadyStateTerminationMode.AbsSafeBest)

@doc doc"""
    SteadyStateTerminationCriteria(mode = SteadyStateTerminationMode.Default; abstol::T = 1e-8,
                                   reltol::T = 1e-6, protective_threshold = 1e3,
                                   patience_steps::Int = 30,
                                   patience_objective_multiplier = 3,
                                   min_max_factor = 1.3)

Define the termination criteria for the SteadyStateProblem.

## Termination Conditions

#### Termination on Absolute Tolerance

  * `SteadyStateTerminationMode.Abs`: Terminates if ``all \left( | \frac{\partial u}{\partial t} | \leq abstol \right)``
  * `SteadyStateTerminationMode.AbsNorm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq abstol``
  * `SteadyStateTerminationMode.AbsSafe`: Essentially `abs_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
  * `SteadyStateTerminationMode.AbsSafeBest`: Same as `SteadyStateTerminationMode.AbsSafe` but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination on Relative Tolerance

  * `SteadyStateTerminationMode.Rel`: Terminates if ``all \left(| \frac{\partial u}{\partial t} | \leq reltol \times | u | \right)``
  * `SteadyStateTerminationMode.RelNorm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|``
  * `SteadyStateTerminationMode.RelSafe`: Essentially `rel_norm` + terminate if there has been no improvement for the last 30 steps + terminate if the solution blows up (diverges)
  * `SteadyStateTerminationMode.RelSafeBest`: Same as `SteadyStateTerminationMode.RelSafe` but uses the best solution found so far, i.e. deviates only if the solution has not converged

#### Termination using both Absolute and Relative Tolerances

  * `SteadyStateTerminationMode.Norm`: Terminates if ``\| \frac{\partial u}{\partial t} \| \leq reltol \times \| \frac{\partial u}{\partial t} + u \|`` or ``\| \frac{\partial u}{\partial t} \| \leq abstol``
  * `SteadyStateTerminationMode.Default`: Check if all values of the derivative is close to zero wrt both relative and absolute tolerance. This is usable for small problems but doesn't scale well for neural networks.

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
    if mode ∈ SAFE_TERMINATION_MODES
        print(io, ", safe_termination_options = ", s.safe_termination_options, ")")
    else
        print(io, ")")
    end
end

function SteadyStateTerminationCriteria(mode = SteadyStateTerminationMode.Default;
                                        abstol::T = 1e-8,
                                        reltol::T = 1e-6, protective_threshold = 1e3,
                                        patience_steps::Int = 30,
                                        patience_objective_multiplier = 3,
                                        min_max_factor = 1.3) where {T}
    @assert mode ∈ instances(SteadyStateTerminationMode.T)
    safe_termination_options = if mode ∈
                                  (SteadyStateTerminationMode.RelSafe,
                                   SteadyStateTerminationMode.RelSafeBest,
                                   SteadyStateTerminationMode.AbsSafe,
                                   SteadyStateTerminationMode.AbsSafeBest)
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

for mode in BASIC_TERMINATION_MODES
    T = SteadyStateTerminationCriteria{mode}
    mode_val = Val(mode)
    @eval function _get_termination_condition(::$(T), storage = nothing)
        function _termination_condition_closure(integrator, abstol, reltol, min_t)
            return _has_converged(get_du(integrator), integrator.u, $(mode_val), abstol,
                                  reltol)
        end
    end
end

for mode in SAFE_TERMINATION_MODES
    T = SteadyStateTerminationCriteria{mode}
    mode_val = Val(mode)
    @eval function _get_termination_condition(cond::$(T), storage)
        aType = typeof(cond.abstol)
        nstep = 0
        protective_threshold = aType(cond.safe_termination_options.protective_threshold)
        objective_values = aType[]
        patience_objective_multiplier = cond.safe_termination_options.patience_objective_multiplier

        if $(mode ∈ SAFE_BEST_TERMINATION_MODES)
            storage[:best_objective_value] = aType(Inf)
            storage[:best_objective_value_iteration] = 0
        end

        @inbounds function _termination_condition_closure(integrator, abstol, reltol, min_t)
            du = get_du(integrator)
            u = integrator.u

            if $(mode ∈ SAFE_BEST_TERMINATION_MODES)
                objective = norm(du)
                criteria = abstol
            else
                objective = norm(du) / (norm(du .+ u) + eps(aType))
                criteria = reltol
            end

            if $(mode ∈ SAFE_BEST_TERMINATION_MODES)
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

for mode in instances(SteadyStateTerminationMode.T)
    mode_val = Val(mode)
    @eval @inline @inbounds function _has_converged(du, u, ::$(typeof(mode_val)),
                                                    abstol, reltol)
        if $(mode == SteadyStateTerminationMode.Norm)
            du_norm = norm(du)
            return du_norm <= abstol || du_norm <= reltol * norm(du + u)
        elseif $(mode == SteadyStateTerminationMode.Rel)
            return all(abs.(du) .<= reltol .* abs.(u))
        elseif $(mode ∈
                 (SteadyStateTerminationMode.RelNorm, SteadyStateTerminationMode.RelSafe,
                  SteadyStateTerminationMode.RelSafeBest))
            return norm(du) <= reltol * norm(du .+ u)
        elseif $(mode == SteadyStateTerminationMode.Abs)
            return all(abs.(du) .<= abstol)
        elseif $(mode ∈
                 (SteadyStateTerminationMode.AbsNorm, SteadyStateTerminationMode.AbsSafe,
                  SteadyStateTerminationMode.AbsSafeBest))
            return norm(du) <= abstol
        elseif $(mode == SteadyStateTerminationMode.Default)
            return all((abs.(du) .<= abstol) .| (abs.(du) .<= reltol .* abs.(u)))
        end
    end
end
