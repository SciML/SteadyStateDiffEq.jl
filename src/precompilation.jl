using PrecompileTools: @compile_workload, @setup_workload

@setup_workload begin
    function _precompile_f_oop(u, p, t)
        return 1.0 .- u
    end

    function _precompile_f_iip(du, u, p, t)
        du[1] = 2.0 - 2.0 * u[1]
        du[2] = u[1] - 4.0 * u[2]
        return nothing
    end

    @compile_workload begin
        SSRootfind()
        DynamicSS()
        DynamicSS(nothing; tspan = 1.0)
        DynamicSS(nothing; tspan = (0.0, 1.0))
        SICNM()

        prob_scalar = SteadyStateProblem(_precompile_f_oop, 0.5)
        u0 = zeros(2)
        prob_vector = SteadyStateProblem(_precompile_f_iip, u0)

        du = similar(u0)
        _precompile_f_iip(du, u0, nothing, 0.0)
        init(
            prob_vector,
            NonlinearSolveBase.NormTerminationMode(infnorm),
            du,
            u0;
            abstol = 1.0e-8,
            reltol = 1.0e-6
        )

        NonlinearProblem(prob_vector)
        ODEProblem{true, true}(_precompile_f_iip, u0, (0.0, Inf), nothing)
    end
end
