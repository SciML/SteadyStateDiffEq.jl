module SteadyStateDiffEqModelingToolkitExt

using SteadyStateDiffEq, ModelingToolkit
import SciMLBase
import SteadyStateDiffEq: _steady_nonlinear, _steady_recover, _scc_decompose

function _steady_nonlinear(prob, sys::ModelingToolkit.System)
    source = sys
    while ModelingToolkit.get_parent(source) !== nothing
        source = ModelingToolkit.get_parent(source)
    end
    steady = mtkcompile(NonlinearSystem(ModelingToolkit.expand_connections(source));
        reassemble_alg = ModelingToolkit.StructuralTransformations.DefaultReassembleAlgorithm(
            inline_linear_sccs = false))
    op = Dict{Any, Any}(v => prob[v] for v in unknowns(steady))
    for p in parameters(steady)
        ModelingToolkit.is_parameter(sys, p) && (op[p] = prob.ps[p])
    end
    return NonlinearProblem(steady, op; build_initializeprob = false)
end

function _steady_recover(prob, sol, sys::ModelingToolkit.System)
    op = Dict(p => sol.ps[p] for p in parameters(sys)
        if ModelingToolkit.is_parameter(sol.prob.f.sys, p))
    changed = any(!isequal(prob.ps[p], value) for (p, value) in op)
    original = changed ? remake(prob; p = op) : prob
    return original, sol[unknowns(sys)]
end

function _scc_decompose(prob, sys::ModelingToolkit.System)
    op = Dict{Any, Any}(v => prob[v] for v in unknowns(sys))
    for p in parameters(sys)
        op[p] = prob.ps[p]
    end
    work = SciMLBase.SCCNonlinearProblem(sys, op;
        combine_sccs = false, build_initializeprob = false)
    work isa SciMLBase.SCCNonlinearProblem || return (work, nothing, nothing)
    variables = Dict(v => i for (i, v) in enumerate(unknowns(sys)))
    equations_map = Dict(eq => i for (i, eq) in enumerate(equations(sys)))
    return work, [variables[v] for v in unknowns(work.f.sys)],
        [equations_map[eq] for eq in equations(work.f.sys)]
end

end
