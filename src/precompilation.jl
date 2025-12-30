using PrecompileTools

@setup_workload begin
    # Minimal setup for precompilation
    function _f_oop(u, p, t)
        return 1.0 - u
    end

    function _f_iip(du, u, p, t)
        du[1] = 2.0 - 2.0 * u[1]
        du[2] = u[1] - 4.0 * u[2]
        return nothing
    end

    @compile_workload begin
        # Precompile algorithm constructors
        SSRootfind()
        DynamicSS()
        DynamicSS(nothing; tspan = 1.0)
        DynamicSS(nothing; tspan = (0.0, 1.0))

        # Precompile SteadyStateProblem creation for common types
        # Scalar case (out-of-place)
        prob_scalar = SteadyStateProblem(_f_oop, 0.5)

        # Vector case (in-place)
        u0 = zeros(2)
        prob_vector = SteadyStateProblem(_f_iip, u0)

        # NonlinearProblem conversion (used internally in SSRootfind)
        nlprob = NonlinearProblem(prob_vector)

        # ODEProblem conversion (used internally in DynamicSS)
        tspan = (0.0, Inf)
        odeprob = ODEProblem{true, true}(_f_iip, u0, tspan, nothing)
    end
end
