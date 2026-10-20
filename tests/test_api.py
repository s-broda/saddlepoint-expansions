"""Transformations, argument checking and consistency of the high-level interface."""
import numpy as np
import pytest

import saddlepoint_expansions as se

A = np.array([[1.0, 0], [1, 1], [0, 1]])
K = se.exponential_sums_cgf(A, n=10, shift=[-2.0, -2.0])


def D(Kc, t, N):
    return se.derivatives(Kc, np.atleast_1d(np.asarray(t, dtype=float)), N)


def test_reflect_marginal_linear():
    t = np.array([0.2, 0.1])
    Dr = D(se.reflect(K, [1, -1]), t, 4)
    D0 = D(K, np.array([0.2, -0.1]), 4)
    for nu in Dr:
        assert Dr[nu] == pytest.approx((-1) ** nu[1] * D0[nu], rel=1e-13, abs=1e-13)
    Dm = D(se.marginal(K, [1]), 0.4, 5)
    D1 = D(K, np.array([0.0, 0.4]), 5)
    for k in range(6):
        assert Dm[(k,)] == pytest.approx(D1[(0, k)], rel=1e-13)
    B = np.array([[1.0, 2.0], [0.5, -1.0]])
    Dl = D(se.linear_combination(K, B), [0.1, -0.05], 1)
    Db = D(K, B @ np.array([0.1, -0.05]), 1)
    assert Dl[(0, 0)] == pytest.approx(Db[(0, 0)])
    assert Dl[(1, 0)] == pytest.approx(B[0, 0] * Db[(1, 0)] + B[1, 0] * Db[(0, 1)])
    assert se.marginal(K, [0]).domain(np.array([15.0])) is False and se.marginal(K, [0]).domain(np.array([5.0])) is True


def test_sample_mean_shift_sum():
    Kg = se.gamma_cgf(3.0, 2.0)
    Ks = se.sample_mean(Kg, 4)
    assert D(Ks, 0.3, 3)[(3,)] == pytest.approx(D(Kg, 0.3 / 4, 3)[(3,)] / 16)
    Ksh = se.shift(Kg, 1.5)
    assert D(Ksh, 0.2, 2)[(1,)] == pytest.approx(D(Kg, 0.2, 2)[(1,)] + 1.5)
    assert D(Ksh, 0.2, 2)[(2,)] == pytest.approx(D(Kg, 0.2, 2)[(2,)])
    Kc = se.common_component_cgf([15, 20, 10], [5, 10, 8])
    Ksum = se.marginal(Kc, [0]) + se.ncx2_cgf(3, 1)
    assert D(Ksum, 0.1, 2)[(2,)] == pytest.approx(D(se.marginal(Kc, [0]), 0.1, 2)[(2,)] + D(se.ncx2_cgf(3, 1), 0.1, 2)[(2,)])
    assert not Ksum.domain(np.array([0.6])) and Ksum.domain(np.array([0.4]))
    assert se.derivatives(Kg, 2.0, 3) is None               # outside the domain
    assert se.derivatives(se.CGF(lambda t: -3 * np.log(1 - t[0]), 1), 2.0, 3) is None   # DomainError


def test_normal_cumulants_vanish():
    Dn = se.derivatives(se.normal_cgf([1.0, -1.0], [[2.0, 0.5], [0.5, 1.0]]), np.array([0.3, 0.1]), 5)
    assert Dn[(1, 0)] == pytest.approx(1 + 2 * 0.3 + 0.5 * 0.1) and Dn[(1, 1)] == pytest.approx(0.5)
    assert all(abs(Dn[nu]) < 1e-15 for nu in Dn if sum(nu) >= 3)


def test_quadratic_forms_with_mean():
    e = np.eye(3)
    Kq = se.quadratic_forms_cgf([e[:, :2] @ e[:2, :], np.outer(e[:, 2], e[:, 2])], mu=[1.0, 2.0, 0.5])
    Kx = se.CGF(lambda t: se.ncx2_cgf(2, 5.0)(t[:1]) + se.ncx2_cgf(1, 0.25)(t[1:]), 2)
    Dq = D(Kq, [0.1, -0.2], 4)
    Dx = D(Kx, [0.1, -0.2], 4)
    for nu in Dq:
        assert Dq[nu] == pytest.approx(Dx[nu], rel=1e-11, abs=1e-11)
    assert se.derivatives(se.wishart_diag_cgf(np.eye(3), 5), np.array([0.5, 0.5, 0.5]), 2) is None


def test_argument_errors():
    with pytest.raises(ValueError):
        se.tailprob(K, [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        se.tailprob(K, [0.1, 0.2], order=0)
    with pytest.raises(ValueError):
        se.tailprob(K, [0.1, 0.2], signs=[1, 0])
    with pytest.raises(ValueError):
        se.reflect(K, [1, 2])
    with pytest.raises(ValueError):
        se.marginal(K, [0, 0])
    with pytest.raises(ValueError):
        K + se.gamma_cgf(1.0)
    with pytest.raises(ValueError):
        se.ratio_density(se.gamma_cgf(3.0), 1.0)
    with pytest.raises(ValueError):
        se.saddlepoint(K, 1.0)


def test_univariate_against_exact_values():
    from scipy.stats import gamma
    Kg = se.gamma_cgf(3.0)
    r = se.tailprob(Kg, 6.0, order=4)
    f = se.density(Kg, 6.0, order=4)
    assert abs(r["F"][3] - gamma.sf(6.0, 3)) / gamma.sf(6.0, 3) < 1e-4
    assert abs(f["f"][3] - gamma.pdf(6.0, 3)) / gamma.pdf(6.0, 3) < 1e-5
    rl = se.tailprob(Kg, 6.0, order=4, lower=True)
    assert rl["F"][3] == pytest.approx(1 - r["F"][3])


def ih_cdf(x, n):
    from math import comb, factorial, floor
    if x <= 0:
        return 0.0
    if x >= n:
        return 1.0
    return sum((-1) ** k * comb(n, k) * (x - k) ** n for k in range(floor(x) + 1)) / factorial(n)


def ih_pdf(x, n):
    from math import comb, factorial, floor
    if x <= 0 or x >= n:
        return 0.0
    return sum((-1) ** k * comb(n, k) * (x - k) ** (n - 1) for k in range(floor(x) + 1)) / factorial(n - 1)


def test_uniform_against_irwin_hall():
    # the mean of n U(-1,1) variables is (2S - n)/n with S Irwin-Hall; sinhc at the origin
    D = se.derivatives(se.uniform_cgf(), 0.0, 6)
    assert D[(2,)] == pytest.approx(1 / 3) and abs(D[(3,)]) < 1e-15
    assert D[(4,)] == pytest.approx(-2 / 15) and D[(6,)] == pytest.approx(16 / 63)
    n, x = 10, 0.3
    Ku = se.sample_mean(se.uniform_cgf(), n)
    ex = 1 - ih_cdf((n * x + n) / 2, n)
    fex = n / 2 * ih_pdf((n * x + n) / 2, n)
    r = se.tailprob(Ku, x, order=4)
    f = se.density(Ku, x, order=4)
    ec = np.abs((r["F"] - ex) / ex)
    ep = np.abs((f["f"] - fex) / fex)
    assert ec[0] < 2e-2 and ec[1] < 5e-4 and ec[2] < 5e-5 and ec[3] < 5e-6
    assert ep[0] < 3e-2 and ep[1] < 6e-4 and ep[2] < 5e-6 and ep[3] < 1e-5
    Kab = se.uniform_cgf(2.0, 5.0)
    assert se.derivatives(Kab, 0.0, 2)[(1,)] == pytest.approx(3.5)
    assert se.derivatives(Kab, 0.0, 2)[(2,)] == pytest.approx(9 / 12)


def test_saddlepoint_stays_on_the_strip_without_domain():
    # finite and convex beyond the pole: the convexity test along the step keeps the iteration on the strip
    Kabs = se.CGF(lambda t: -3 * np.log(np.abs(1 - t[0])), 1)
    for x in (6.0, 30.0, 300.0):
        s, _ = se.saddlepoint(Kabs, x)
        assert s == pytest.approx(1 - 3 / x, rel=1e-10)
    Kabs2 = se.CGF(lambda t: -np.sum(np.log(np.abs(1 - A @ t))), 2)
    sa, _ = se.saddlepoint(Kabs2, [30.0, 40.0])
    st, _ = se.saddlepoint(se.exponential_sums_cgf(A), [30.0, 40.0])
    assert np.allclose(sa, st, rtol=1e-10)


def test_closed_form_taylor_option_and_vector_ratios():
    from scipy.special import betaln, betainc, gammaln
    from scipy.integrate import quad
    toep = lambda r, n: r ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))
    n = 8
    Kq = se.quadratic_forms_cgf([toep(0.5, n), np.eye(n)], Sigma=toep(0.3, n), mu=0.5 * np.ones(n))
    Kg = se.quadratic_forms_cgf([toep(0.5, n), np.eye(n)], Sigma=toep(0.3, n), mu=0.5 * np.ones(n), closedform=False)
    assert Kq.taylor is not None and Kg.taylor is None
    D1 = se.derivatives(Kq, [0.05, -0.02], 7)
    D2 = se.derivatives(Kg, [0.05, -0.02], 7)
    assert all(abs(D1[k] - D2[k]) <= 1e-12 * max(1.0, abs(D2[k])) for k in D1)
    assert se.derivatives(Kq, [0.5, 0.5], 2) is None
    r1 = se.tailprob(Kq, [14.0, 13.0], order=4)
    r2 = se.tailprob(Kg, [14.0, 13.0], order=4)
    assert np.max(np.abs(r1["F"] - r2["F"]) / r2["F"]) < 1e-12
    # the option survives the transformations
    K3q = se.quadratic_forms_cgf([toep(0.5, n), np.eye(n), toep(0.2, n)])
    K3g = se.quadratic_forms_cgf([toep(0.5, n), np.eye(n), toep(0.2, n)], closedform=False)
    Ks = se.shift(se.sample_mean(se.reflect(Kq, [1, -1]), 3), [0.1, 0.2]) + se.marginal(K3q, [0, 2])
    Kt = se.shift(se.sample_mean(se.reflect(Kg, [1, -1]), 3), [0.1, 0.2]) + se.marginal(K3g, [0, 2])
    assert Ks.taylor is not None
    Ds = se.derivatives(Ks, [0.03, -0.04], 5)
    Dt = se.derivatives(Kt, [0.03, -0.04], 5)
    assert all(abs(Ds[k] - Dt[k]) <= 1e-11 * max(1.0, abs(Dt[k])) for k in Ds)
    # one ratio: the Rice code and the general code agree
    f1 = se.ratio_density(Kq, 1.2, order=4)["f"]
    f2 = se.ratio_density(Kq, 1.2, order=4, general=True)["f"]
    assert np.max(np.abs(f1 - f2) / f2) < 1e-12
    assert np.max(np.abs(se.ratio_density(Kq, [1.2], order=4)["f"] - f1) / f1) < 1e-12
    # two independent ratios of gamma variables: products of beta-prime densities and beta cdfs
    a = np.array([3.0, 4.0, 5.0, 6.0])
    K4 = se.CGF(lambda t: -np.sum(a * np.log(1 - t)), 4, domain=lambda t: bool(np.all(t < 1)))
    r = np.array([0.7, 0.9])
    bp = lambda x, p, q: x ** (p - 1) * (1 + x) ** (-p - q) / np.exp(betaln(p, q))
    f = se.ratio_density(K4, r, order=5)["f"]
    ex = bp(r[0], a[0], a[2]) * bp(r[1], a[1], a[3])
    ef = np.abs((f - ex) / ex)
    assert ef[0] < 0.1 and ef[1] < 3e-3 and ef[2] < 2e-4 and ef[4] < 1e-5
    F = se.ratio_tailprob(K4, r, order=5, lower=True)["F"]
    exF = betainc(a[0], a[2], r[0] / (1 + r[0])) * betainc(a[1], a[3], r[1] / (1 + r[1]))
    assert abs(F[4] - exF) / exF < 1e-6 and abs(F[0] - exF) / exF < 2e-2
    # the rate n^-4 of order four as n doubles
    e4 = [abs(se.ratio_density(se.sample_mean(K4, nn), r, order=4)["f"][3] / (bp(r[0], nn * a[0], nn * a[2]) * bp(r[1], nn * a[1], nn * a[3])) - 1)
          for nn in (20, 40)]
    assert 10 < e4[0] / e4[1] < 22
    # a common denominator
    K3 = se.CGF(lambda t: -3 * np.log(1 - t[0]) - 4 * np.log(1 - t[1]) - 5 * np.log(1 - t[2]), 3,
                domain=lambda t: bool(np.all(t < 1)))
    gp = lambda x, p: x ** (p - 1) * np.exp(-x - gammaln(p))
    exc = quad(lambda y: y ** 2 * gp(0.5 * y, 3) * gp(0.8 * y, 4) * gp(y, 5), 0, np.inf, epsrel=1e-12)[0]
    assert abs(se.ratio_density(K3, [0.5, 0.8], order=4)["f"][3] - exc) / exc < 3e-5
    with pytest.raises(ValueError):
        se.ratio_density(K4, [0.5, 0.5, 0.5, 0.5])
    # comparisons on the constant term
    u = se.variables(1, 3)
    assert (u[0] + 0.5) < 1.0 and not ((u[0] + 0.5) > 1.0) and (u[0] - 0.5) < 0.0
    Kif = se.CGF(lambda t: -3 * np.log(1 - t[0]) if t[0] < 1 else np.nan, 1)
    assert se.derivatives(Kif, 0.3, 3)[(3,)] == pytest.approx(3 * 2 / 0.7 ** 3)
    assert se.derivatives(Kif, 2.0, 3) is None
