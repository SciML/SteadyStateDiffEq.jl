using JET, SteadyStateDiffEq, Test

@testset "JET static analysis" begin
    rep = JET.report_package(SteadyStateDiffEq; target_modules = (SteadyStateDiffEq,))
    reports = JET.get_reports(rep)
    @test length(reports) == 0
end
