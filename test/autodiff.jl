using ModelingToolkit
using OrdinaryDiffEq
using ForwardDiff
using ForwardDiff: Dual

@variables begin
    t
    y1(t) = 1.0
    y2(t) = 0.0
    y3(t) = 0.0
    y4(t) = 0.0
    y5(t) = 0.0
    y6(t) = 0.0
    y7(t) = 0.0
    y8(t) = 0.0057
end

@parameters begin
    k1 = 1.71
    k2 = 280.0
    k3 = 8.32
    k4 = 0.69
    k5 = 0.43
    k6 = 1.81
end

D = Differential(t)
eqs = [
    D(y1) ~ (-k1*y1 + k5*y2 + k3*y3 + 0.0007),
    D(y2) ~ (k1*y1 - 8.75*y2),
    D(y3) ~ (-10.03*y3 + k5*y4 + 0.035*y5),
    D(y4) ~ (k3*y2 + k1*y3 - 1.12*y4),
    D(y5) ~ (-1.745*y5 + k5*y6 + k5*y7),
    D(y6) ~ (-k2*y6*y8 + k4*y4 + k1*y5 - k5*y6 + k4*y7),
    D(y7) ~ (k2*y6*y8 - k6*y7),
    D(y8) ~ (-k2*y6*y8 + k6*y7)
]

@named model = ODESystem(eqs)

struct Tag end
T = typeof(ForwardDiff.Tag(Tag(),Float64))

u0 = [
    Dual{T, Float64, 7}(1.8983788068509213,ForwardDiff.Partials((0.6970981087506788,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.0057,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0)))
]

p = [
    Dual{T, Float64, 7}(1.5174072564237708,ForwardDiff.Partials((0.0,0.38355212192378396,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.43,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(8.32,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(268.1197063467946,ForwardDiff.Partials((0.0,0.0,44.91823438292696,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(0.69,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0))),
    Dual{T, Float64, 7}(20.0,ForwardDiff.Partials((0.0,0.0,0.0,0.0,0.0,0.0,0.0)))
]

prob = SteadyStateProblem(model, u0, p)
alg = DynamicSS(QNDF())
sol = solve(prob, alg)