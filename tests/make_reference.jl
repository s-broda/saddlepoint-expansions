# Reference values for the Python tests, computed with the Julia package SaddlepointExpansions.jl
# (same algorithms, same quadrature grids and quasi-Monte Carlo points), written to
# reference_julia.json. Run from the Julia package directory:
#   julia --project=. /path/to/tests/make_reference.jl
using SaddlepointExpansions, LinearAlgebra

# a minimal JSON writer (dicts, vectors, numbers, strings, bools)
function tojson(io, x::AbstractDict)
    print(io, "{")
    first = true
    for (k, v) in x
        first || print(io, ",")
        first = false
        print(io, "\"", k, "\":")
        tojson(io, v)
    end
    print(io, "}")
end
function tojson(io, x::AbstractArray)
    print(io, "[")
    for (i, v) in enumerate(x)
        i > 1 && print(io, ",")
        tojson(io, v)
    end
    print(io, "]")
end
tojson(io, x::AbstractString) = print(io, "\"", x, "\"")
tojson(io, x::Bool) = print(io, x ? "true" : "false")
tojson(io, x::Integer) = print(io, x)
tojson(io, x::AbstractFloat) = isnan(x) ? print(io, "\"nan\"") : isinf(x) ? print(io, x > 0 ? "\"inf\"" : "\"-inf\"") : print(io, repr(Float64(x)))
tojson(io, x::Tuple) = tojson(io, collect(x))
tojson(io, ::Nothing) = print(io, "null")

dkey(nu) = join(nu, ",")
derivdict(D) = Dict(dkey(nu) => v for (nu, v) in D)

R = Dict{String,Any}()

# Wang's example: X = U0+U1-2, Y = U0+U2-2, means of n = 10
A = [1.0 0; 1 1; 0 1]; n = 10; l = [-2.0, -2.0]
K = exponential_sums_cgf(A; n, shift=l)
R["wang"] = Dict(
    "deriv_t" => [0.3, -0.2], "deriv_N" => 7, "deriv" => derivdict(derivatives(K, [0.3, -0.2], 7)),
    "lower" => let r = tailprob(K, [-0.5, -0.5]; order=5, lower=true); Dict("x" => [-0.5, -0.5], "order" => 5, "F" => r.F, "tau" => r.tau, "kappa" => r.info.kappa) end,
    "lower2" => let r = tailprob(K, [-1.0, 2.0]; order=5, lower=true); Dict("x" => [-1.0, 2.0], "order" => 5, "F" => r.F, "tau" => r.tau) end,
    "upper_ie" => let r = tailprob(K, [-0.8, 0.5]; order=3); Dict("x" => [-0.8, 0.5], "order" => 3, "F" => r.F, "tau" => r.tau) end,
    "density" => let r = density(K, [-0.5, -0.5]; order=4); Dict("x" => [-0.5, -0.5], "order" => 4, "f" => r.f) end,
    "saddle" => let (t, _) = saddlepoint(K, [-0.5, -0.5]); t end)

# Kolassa and Li's example, n = 5
K5 = exponential_sums_cgf(A; n=5)
R["kl"] = Dict(
    "upper" => let r = tailprob(K5, [3.0, 3.0]; order=5); Dict("x" => [3.0, 3.0], "order" => 5, "F" => r.F, "tau" => r.tau) end,
    "boundary" => let r = tailprob(K5, [2.5, 3.0]; order=5); Dict("x" => [2.5, 3.0], "order" => 5, "F" => r.F, "tau" => r.tau) end,
    "far" => let r = tailprob(K5, [6.0, 7.0]; order=4); Dict("x" => [6.0, 7.0], "order" => 4, "F" => r.F, "tau" => r.tau) end)

# pairs of quadratic forms with a common component
mm = [15, 20, 10]; ll = [5, 10, 8]
Kc = common_component_cgf(mm, ll)
R["bqf"] = Dict("m" => mm, "lam" => ll,
    "upper" => let r = tailprob(Kc, [55.0, 70.0]; order=4); Dict("x" => [55.0, 70.0], "order" => 4, "F" => r.F, "kappa" => r.info.kappa, "exact" => bqfcc_exact(55.0, 70.0, mm, ll).Fup) end,
    "density" => let r = density(Kc, [55.0, 70.0]; order=4); Dict("x" => [55.0, 70.0], "order" => 4, "f" => r.f, "exact" => bqfcc_exact(55.0, 70.0, mm, ll).pdf) end,
    "lower" => let r = tailprob(Kc, [40.0, 60.0]; order=3, lower=true); Dict("x" => [40.0, 60.0], "order" => 3, "F" => r.F, "exact" => bqfcc_exact(40.0, 60.0, mm, ll).Flo) end)

# trivariate
m3 = [15, 20, 25, 10]; l3 = [5, 10, 0, 8]
K3 = common_component_cgf(m3, l3)
R["tri"] = Dict("m" => m3, "lam" => l3,
    "upper" => let r = tailprob(K3, [80.0, 95.0, 100.0]; order=4); Dict("x" => [80.0, 95.0, 100.0], "order" => 4, "F" => r.F, "err" => r.info.err, "exact" => mvcc_exact([80.0, 95.0, 100.0], m3, l3).Fup) end,
    "density" => let r = density(K3, [80.0, 95.0, 100.0]; order=4); Dict("x" => [80.0, 95.0, 100.0], "order" => 4, "f" => r.f, "refined" => r.info.refined, "exact" => mvcc_exact([80.0, 95.0, 100.0], m3, l3).pdf) end,
    "ie" => let r = tailprob(K3, [55.0, 70.0, 75.0]; order=3); Dict("x" => [55.0, 70.0, 75.0], "order" => 3, "F" => r.F, "tau" => r.tau, "exact" => mvcc_exact([55.0, 70.0, 75.0], m3, l3).Fup) end)

# Wishart diagonal
V = 0.75 .* Matrix(1.0I, 3, 3) .+ 0.25
Kw = wishart_diag_cgf(V, 5)
R["wishart"] = Dict("V" => [V[i, :] for i in 1:3], "n" => 5,
    "deriv_t" => [0.05, -0.02, 0.03], "deriv_N" => 6, "deriv" => derivdict(derivatives(Kw, [0.05, -0.02, 0.03], 6)),
    "upper" => let r = tailprob(Kw, [6.5, 6.5, 7.5]; order=3); Dict("x" => [6.5, 6.5, 7.5], "order" => 3, "F" => r.F) end)

# four dimensions
m4 = [15, 20, 25, 30, 10]; l4 = [5, 10, 0, 3, 8]
K4 = common_component_cgf(m4, l4); x4 = [70.0, 85.0, 95.0, 105.0]
R["d4"] = Dict("m" => m4, "lam" => l4, "x" => x4,
    "density" => let r = density(K4, x4; order=3); Dict("order" => 3, "f" => r.f, "refined" => r.info.refined) end,
    "qmc" => let r = tailprob(K4, x4; order=3); Dict("order" => 3, "F" => r.F, "err" => r.info.err, "exact" => mvcc_exact(x4, m4, l4).Fup) end)

# univariate
Kg = gamma_cgf(3.0)
Kx = ncx2_cgf(5, 10)
R["uni"] = Dict(
    "gamma" => let r = tailprob(Kg, 6.0; order=4), rl = tailprob(Kg, 6.0; order=4, lower=true), f = density(Kg, 6.0; order=4)
        Dict("a" => 3.0, "x" => 6.0, "order" => 4, "upp" => r.F, "cdf" => rl.F, "pdf" => f.f, "tau" => r.tau) end,
    "ncx2" => let r = tailprob(Kx, 30.0; order=6), f = density(Kx, 30.0; order=6)
        Dict("m" => 5, "lam" => 10, "x" => 30.0, "order" => 6, "upp" => r.F, "pdf" => f.f) end,
    "gamma_far" => let r = tailprob(Kg, 20.0; order=5); Dict("x" => 20.0, "order" => 5, "upp" => r.F) end)

# the uniform distribution: the mean of 10 U(-1,1) variables (sinhc at the origin)
Ku = sample_mean(uniform_cgf(), 10)
R["uniform"] = Dict("n" => 10, "x" => 0.3, "order" => 4,
    "upp" => tailprob(Ku, 0.3; order=4).F, "pdf" => density(Ku, 0.3; order=4).f,
    "deriv0" => derivdict(derivatives(uniform_cgf(), [0.0], 6)),
    "bivar" => let Kb = CGF(t -> log(sinhc(t[1])) + log(sinhc(t[1] + t[2])) + log(sinhc(t[2])), 2)
        r = tailprob(sample_mean(Kb, 8), [0.4, 0.3]; order=3); Dict("x" => [0.4, 0.3], "order" => 3, "F" => r.F, "tau" => r.tau) end)

# the noncentral F as a ratio
m_, n_, lam = 4, 12, 3; fq = 2.5; c = fq*m_/n_
KF = CGF(t -> ncx2_cgf(m_, lam)([t[1]]) + ncx2_cgf(n_)([t[2]]), 2; domain=t -> t[1] < 0.5 && t[2] < 0.5)
R["ratio"] = Dict("m" => m_, "n" => n_, "lam" => lam, "f" => fq, "c" => c, "order" => 4,
    "density" => ratio_density(KF, c; order=4).f, "cdf" => ratio_tailprob(KF, c; order=4, lower=true).F,
    "upp" => ratio_tailprob(KF, c; order=4).F, "shat" => ratio_density(KF, c; order=4).shat)

# mixed corner of a bivariate normal
S = [1.0 0.6; 0.6 2.0]
R["normal_mixed"] = Dict("S" => [S[i, :] for i in 1:2], "x" => [0.5, -0.3], "signs" => [1, -1],
    "F" => tailprob(normal_cgf([0.0, 0.0], S), [0.5, -0.3]; order=2, signs=[1, -1]).F,
    "exact" => orthant2([0.5, 0.3], [1.0 -0.6; -0.6 2.0]))

out = joinpath(@__DIR__, "reference_julia.json")
open(out, "w") do io
    tojson(io, R)
end
println("written ", out)
