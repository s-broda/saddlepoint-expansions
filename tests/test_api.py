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
