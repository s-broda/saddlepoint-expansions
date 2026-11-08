"""Conditional expansions (Broda and Kan, "Conditional saddlepoint expansions of arbitrary order"):
the printed rows of the tables of the paper (conditional_rows.txt, shared with the Julia package)
recomputed from the cumulant generating function and exact values, and consistency checks (B empty,
Gaussian vectors, the Dirichlet distribution, the derivative interface, the leading term).

The relative errors are printed with two significant digits; a computed value reproduces a printed
one when it rounds to it, and values below 1e-10, at the rounding level of the exact values, are
compared to 2e-12 absolute. The rows are those of the tables regenerated with the corrected program
spa_cond.m of the paper (closed-form derivatives in the canonical coordinates in cond_beta.m, the
truncation of the orthant integrals with the degree of the Hermite polynomials, Kolassa's truncation
of the lattice amplitude about the origin, relative errors of lattice tails to min(P, 1 - P))."""
import math
import os
from collections import defaultdict

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import betainc, betaincinv, betaln, erfc
from scipy.stats import ncx2, chi2

import saddlepoint_expansions as se

ROWS = defaultdict(list)
with open(os.path.join(os.path.dirname(__file__), "conditional_rows.txt")) as fh:
    for ln in fh:
        if ln.startswith("#"):
            continue
        f = ln.split()
        ROWS[int(f[0])].append((int(f[1]), f[2:]))


def pval(s):
    return math.nan if s == "-" else float(s)


def printed(c, p):
    if abs(p) < 1e-10:
        return abs(c - p) <= 2e-12
    e = math.floor(math.log10(abs(p)) + 1e-9)
    return abs(c - p) <= 0.05 * 10.0 ** e * (1 + 1e-9)


def rowcheck(comp, cells, fixes):
    ok = []
    for k, c in enumerate(comp):
        p = fixes.get(k, pval(cells[k]))
        if math.isnan(p):
            ok.append(True)
            continue
        good = printed(c, p)
        if k in fixes:
            good = good and not printed(c, pval(cells[k]))
        ok.append(good)
    return ok


def betapdf(w, p, q):
    return math.exp((p - 1) * math.log(w) + (q - 1) * math.log1p(-w) - betaln(p, q))


def Phi(x):
    return erfc(-x / math.sqrt(2)) / 2




def beta_K(block, N):
    if block <= 2:
        a, b = (1.0, 1.0) if block == 1 else (0.5, 3.0)
        return se.CGF(lambda t: -N * a * np.log(1 - t[0] - t[1]) - N * b * np.log(1 - t[1]), 2,
                      domain=lambda t: t[0] + t[1] < 1 and t[1] < 1), (a, b, None)
    a, b, c = 1.0, 2.0, 0.5
    return se.CGF(lambda t: -N * a * np.log(1 - t[0] - t[1]) - N * b * np.log(1 - t[1] - t[2]) - N * c * np.log(1 - t[1]), 3,
                  domain=lambda t: t[0] + t[1] < 1 and t[1] + t[2] < 1 and t[1] < 1), (a, b, c)


@pytest.mark.parametrize("i", range(45))
def test_table_gamma_sum(i):
    block, cells = ROWS[1][i]
    N = int(cells[0])
    q = float(cells[1])
    K, (a, b, c) = beta_K(block, N)
    if block <= 2:
        s2 = N * (a + b)
        w = betaincinv(N * a, N * b, q)
        x = [w * s2, s2]
        Pl = betainc(N * a, N * b, w)
        fex = betapdf(w, N * a, N * b) / s2
    else:
        s2 = N * (a + b + c)
        s3 = N * b
        w = betaincinv(N * a, N * c, q)
        x = [w * (s2 - s3), s2, s3]
        Pl = betainc(N * a, N * c, w)
        fex = betapdf(w, N * a, N * c) / (s2 - s3)
    Pu = 1 - Pl
    r = se.conditional_tailprob(K, x, 0, order=5)
    f = se.conditional_density(K, x, 0, order=5)
    assert r["status"] == "ok" and f["status"] == "ok"
    assert printed(Pu, pval(cells[2])) and round(r["tau"][0], 2) == pval(cells[3])
    comp = np.concatenate([(r["F"] - Pu) / min(Pu, Pl), f["f"] / fex - 1])
    assert all(rowcheck(comp, cells[4:14], {}))


@pytest.mark.parametrize("i", range(30))
def test_table_quadratic_forms(i):
    block, cells = ROWS[2][i]
    m, lam = ([15, 20, 10], [5, 10, 8]) if block == 1 else ([40, 60, 30], [20, 10, 30])
    c2 = float(cells[0])
    q = float(cells[1])
    mu2 = m[1] + lam[1] + m[2] + lam[2]
    sd2 = math.sqrt(2 * (m[1] + 2 * lam[1]) + 2 * (m[2] + 2 * lam[2]))
    x2 = mu2 + c2 * sd2
    f0 = lambda u: ncx2.pdf(u, m[2], lam[2])
    f2 = lambda u: ncx2.pdf(x2 - u, m[1], lam[1])
    opts = dict(epsabs=0.0, epsrel=1e-12, limit=200)
    den = quad(lambda u: f0(u) * f2(u), 0, x2, **opts)[0]

    def ex(x1, g):
        pts = [x1] if x1 < x2 else None
        return quad(lambda u: f0(u) * f2(u) * g(x1 - u), 0, x2, points=pts, **opts)[0] / den

    cdf1 = lambda y: ncx2.cdf(y, m[0], lam[0]) if y > 0 else 0.0
    sf1 = lambda y: ncx2.sf(y, m[0], lam[0]) if y > 0 else 1.0
    x1 = brentq(lambda x: ex(x, cdf1) - q, 1e-6, mu2 + m[0] + lam[0] + 20 * sd2, xtol=1e-13, rtol=1e-14)
    Pu, Pl = ex(x1, sf1), ex(x1, cdf1)
    fex = quad(lambda u: f0(u) * f2(u) * ncx2.pdf(x1 - u, m[0], lam[0]), 0, min(x1, x2), **opts)[0] / den
    K = se.common_component_cgf(m, lam)
    r = se.conditional_tailprob(K, [x1, x2], 0, order=5)
    f = se.conditional_density(K, [x1, x2], 0, order=5)
    assert printed(Pu, pval(cells[2])) and round(r["tau"][0], 2) == pval(cells[3])
    comp = np.concatenate([(r["F"] - Pu) / min(Pu, Pl), f["f"] / fex - 1])
    assert all(rowcheck(comp, cells[4:14], {}))


# the exact upper quantiles of S = N Rbar (Gil-Pelaez inversion, computed by the Julia package)
QEX = {1: (-0.649984406337777, 0.173640308684513, 0.404657822568259, 0.830136764358454, 1.28888641542161),
       2: (-0.28101745873196, 0.342338885972127, 0.518020283322166, 0.844515706483764, 1.20359268966255),
       3: (-0.179727297726595, 0.335098902241048, 0.480370305892609, 0.751043128437577, 1.05038179960608),
       4: (-1.17188025603077, -0.0622176078581697, 0.251186562311314, 0.834501869250704, 1.47747984786112),
       5: (-0.525777371167559, 0.329433323857853, 0.571415079075747, 1.02356949383106, 1.52620380057527)}
DESIGN = {1: (5, 5), 2: (5, 15), 3: (5, 25), 4: (10, 4), 5: (10, 14)}


def lams(N, T0):
    out = []
    for T in range(T0 + 1, T0 + N + 1):
        A = np.diag(np.ones(T - 1) / 2, 1)
        A = A + A.T
        Q = np.linalg.svd(np.ones((1, T)))[2][1:].T          # orthonormal basis of the complement of 1
        out.append(np.linalg.eigvalsh(Q.T @ A @ Q))
    return out


@pytest.mark.parametrize("block", [2, 3, 5])
def test_table_sums_of_ratios(block):
    """The mean of N ratios of quadratic forms, P(Rbar >= x/N) = P(X >= x | D = 1): the blocks with
    T_i = 16..20 and 26..30 (N = 5, order 4) and N = 10, T_i = 15..24 (order 2); Skovgaard's formula
    at the median, where 1/w - 1/z cancels in double precision, is not compared."""
    N, T0 = DESIGN[block]
    lam = lams(N, T0)
    d = N + 1
    K = se.CGF(lambda t: -sum(sum(np.log(1 - 2 * (t[0] * l + t[i + 1])) for l in lam[i]) for i in range(N)) / 2, d)
    mu = sum(np.mean(a) for a in lam)
    va = sum((np.sum(a) ** 2 + 2 * np.sum(a ** 2)) / (len(a) * (len(a) + 2)) - np.mean(a) ** 2 for a in lam)
    cells_b = [c for (b, c) in ROWS[3] if b == block]
    for k, p in enumerate((0.5, 0.1, 0.05, 0.01, 0.001)):
        cells = cells_b[k]
        x = QEX[block][k]
        xv = [x] + [1.0] * N
        r = se.conditional_tailprob(K, xv, 0, order=(4 if N == 5 else 2))
        assert r["status"] == "ok"
        assert round(x / N, 4) == pval(cells[1]) and round(r["tau"][0], 3) == pval(cells[3])
        tau, taut = r["tau"], r["taut"]
        D = se.derivatives(K, tau, 2)
        Dt = se.derivatives(K, taut, 2)
        e = lambda i: tuple(1 if j == i else 0 for j in range(d))
        ee = lambda i, j: tuple(a + b for a, b in zip(e(i), e(j)))
        K2h = np.array([[D[ee(i, j)] for j in range(d)] for i in range(d)])
        K2t = np.array([[Dt[ee(i + 1, j + 1)] for j in range(N)] for i in range(N)])
        z0 = (0,) * d
        w = np.sign(tau[0]) * math.sqrt(2 * ((Dt[z0] - taut @ xv) - (D[z0] - tau @ xv)))
        z = tau[0] * math.sqrt(np.linalg.det(K2h) / np.linalg.det(K2t))
        u = tau[0] * math.sqrt(K2h[0, 0] - K2h[0, 1:] @ np.linalg.solve(K2h[1:, 1:], K2h[1:, 0]))
        Plr = Phi(-w) - math.exp(-w * w / 2) / math.sqrt(2 * math.pi) * (1 / w - 1 / z)
        Phr = u / z * math.exp((u * u - w * w) / 2) * Phi(-u)
        if p < 0.5:
            assert abs(r["info"]["Fp"][0] / Phr - 1) < 1e-12
        Pn = Phi(-(x - mu) / math.sqrt(va))
        comp = np.concatenate([[Pn - p, Plr - p, Phr - p], r["F"] - p]) / p
        ok = rowcheck(comp, cells[4:4 + len(comp)], {})
        if p == 0.5:
            ok[1] = True
        assert all(ok)


def test_table_logistic_regression():
    n = 30
    X = np.column_stack([np.ones(n, dtype=int), np.arange(n) % 5, np.arange(n) % 3])
    mx = X.sum(axis=0)
    pm = np.zeros(mx + 1)
    pm[0, 0, 0] = 1.0
    for j in range(n):                       # the exact joint pmf of S under beta = 0
        pm = (pm + np.roll(pm, tuple(X[j]), axis=(0, 1, 2))) / 2
    K = se.logistic_regression_cgf(X)
    i = 0
    for k0, k2 in ((15, 15), (10, 12), (12, 9)):
        q = pm[k0, :, k2] / pm[k0, :, k2].sum()
        Qu = np.cumsum(q[::-1])[::-1]
        Ql = np.cumsum(q)
        for k1 in sorted(set(int(np.argmax(Ql >= tg)) for tg in (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))):
            cells = ROWS[4][i][1]
            i += 1
            Pu, Pl, pk = Qu[k1], Ql[k1], q[k1]
            x = [k0, k1, k2]
            r = se.conditional_tailprob(K, x, 1, order=4, lattice=True)
            f = se.conditional_density(K, x, 1, order=4, lattice=True)
            # the last column: the amplitude expanded about the origin, kept to s^2 at order 2
            ro = se.conditional_tailprob(K, x, 1, order=4, lattice=True, sinh="origin")
            assert int(cells[0]) == k1 and printed(Pu, pval(cells[1])) and round(r["tau"][1], 2) == pval(cells[2])
            assert r["status"] == "ok"
            comp = np.concatenate([(r["F"] - Pu) / min(Pu, 1 - Pu), f["f"] / pk - 1, [(ro["F"][1] - Pu) / min(Pu, 1 - Pu)]])
            assert all(rowcheck(comp, cells[3:12], {}))
            rl = se.conditional_tailprob(K, [k0, k1 - 1, k2], 1, order=4, lattice=True, lower=True)
            assert np.max(np.abs(rl["F"] + r["F"] - 1)) < 1e-14
    assert i == 21


def rd(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / np.abs(np.asarray(b))))


def test_b_empty():
    Kg = se.gamma_cgf(3.0)
    for xx in (6.0, 1.5):
        assert rd(se.conditional_tailprob(Kg, [xx], 0, order=4)["F"], se.tailprob(Kg, xx, order=4)["F"]) < 1e-13
        assert rd(se.conditional_density(Kg, [xx], 0, order=4)["f"], se.density(Kg, xx, order=4)["f"]) < 1e-14
    K5 = se.exponential_sums_cgf(np.array([[1.0, 0], [1, 1], [0, 1]]), n=5)
    for xx in ([3.0, 3.0], [2.5, 3.0], [1.2, 3.5], [0.8, 0.9]):
        assert rd(se.conditional_tailprob(K5, xx, [0, 1], order=4)["F"], se.tailprob(K5, xx, order=4)["F"]) < 1e-12
        assert rd(se.conditional_density(K5, xx, [0, 1], order=4)["f"], se.density(K5, xx, order=4)["f"]) < 1e-14


def test_gaussian_exact():
    Sig = np.array([[2.0, 0.6, 0.5, 0.3], [0.6, 1.5, -0.4, 0.2], [0.5, -0.4, 1.0, 0.1], [0.3, 0.2, 0.1, 0.8]])
    mu = np.array([0.5, -1.0, 0.2, 0.3])
    Kn = se.normal_cgf(mu, Sig)

    def condnorm(x, A):
        B = [j for j in range(4) if j not in A]
        C = Sig[np.ix_(A, B)] @ np.linalg.inv(Sig[np.ix_(B, B)])
        V = Sig[np.ix_(A, A)] - C @ Sig[np.ix_(B, A)]
        return mu[A] + C @ (np.asarray(x)[B] - mu[B]), (V + V.T) / 2

    for x, A in (([1.0, 0.0, 0.5, 0.0], [0]), ([-1.0, 0.0, 0.5, 0.0], [0]), ([1.5, 0.0, 0.5, 0.3], [0, 1]),
                 ([0.2, -2.5, 0.5, 0.3], [0, 1]), ([-1.0, -2.5, 0.5, 0.3], [1, 0])):
        Av = sorted(A)
        m, V = condnorm(x, Av)
        xa = np.asarray(x)[Av]
        ex = Phi(-(xa[0] - m[0]) / math.sqrt(V[0, 0])) if len(A) == 1 else se.orthant2(xa - m, V)
        r = se.conditional_tailprob(Kn, x, A, order=3)
        assert np.max(np.abs(r["F"] / ex - 1)) < 1e-12 and r["status"] == "ok"
        fex = math.exp(-(xa - m) @ np.linalg.solve(V, xa - m) / 2) / math.sqrt((2 * math.pi) ** len(A) * np.linalg.det(V))
        assert np.max(np.abs(se.conditional_density(Kn, x, A, order=3)["f"] / fex - 1)) < 1e-13
    x = [0.2, -2.5, 0.5, 0.3]
    m, V = condnorm(x, [0, 1])
    rm = se.conditional_tailprob(Kn, x, [0, 1], order=2, signs=[-1, 1])
    assert abs(rm["F"][1] / se.orthant2([-(x[0] - m[0]), x[1] - m[1]], [[V[0, 0], -V[0, 1]], [-V[0, 1], V[1, 1]]]) - 1) < 1e-12


def test_dirichlet():
    al = (5.0, 7.5, 10.0)
    Kd = se.CGF(lambda t: -al[0] * np.log(1 - t[0] - t[2]) - al[1] * np.log(1 - t[1] - t[2]) - al[2] * np.log(1 - t[2]), 3,
                domain=lambda t: t[0] + t[2] < 1 and t[1] + t[2] < 1 and t[2] < 1)
    s = sum(al)
    for w1, w2 in ((0.35, 0.45), (0.15, 0.40), (0.30, 0.25)):
        ex = quad(lambda v: betapdf(v, al[0], al[1] + al[2]) * (1 - betainc(al[1], al[2], w2 / (1 - v))), w1, 1 - w2,
                  epsabs=0, epsrel=1e-13)[0]
        r = se.conditional_tailprob(Kd, [w1 * s, w2 * s, s], [0, 1], order=4)
        e = np.abs(r["F"] / ex - 1)
        assert e[0] < 2e-2 and e[1] < 1e-3 and e[2] < 2e-5 and e[3] < 1e-5
        assert r["status"] == "ok" and np.max(r["info"]["kappa"]) < 1e3


def test_derivative_interface_and_leading_term():
    N, a, b, c = 16, 1.0, 2.0, 0.5

    def Kf3(nu, t):
        n = sum(nu)
        p1, p2, p3 = 1 - t[0] - t[1], 1 - t[1] - t[2], 1 - t[1]
        if p1 <= 0 or p2 <= 0 or p3 <= 0:
            return math.nan
        if n == 0:
            return -N * (a * math.log(p1) + b * math.log(p2) + c * math.log(p3))
        v = 0.0
        if nu[2] == 0:
            v += a * math.factorial(n - 1) / p1 ** n
        if nu[0] == 0:
            v += b * math.factorial(n - 1) / p2 ** n
        if nu[0] == 0 and nu[2] == 0:
            v += c * math.factorial(n - 1) / p3 ** n
        return N * v

    K3 = se.CGF(lambda t: -N * a * np.log(1 - t[0] - t[1]) - N * b * np.log(1 - t[1] - t[2]) - N * c * np.log(1 - t[1]), 3)
    x3 = [0.7 * 14, 14 * N / 4, 8 * N / 4]
    r1 = se.spa_cond(Kf3, x3, 0, 4)
    r2 = se.conditional_tailprob(K3, x3, 0, order=4)
    f2 = se.conditional_density(K3, x3, 0, order=4)
    assert rd(r1["P"], r2["F"]) < 1e-10 and rd(r1["f"], f2["f"]) < 1e-10
    assert np.array_equal(r1["Pp"][1::2], r1["P"]) and len(r1["Pp"]) == 8 and r1["taut"][0] == 0
    with pytest.raises(ValueError):
        se.conditional_tailprob(K3, x3, 3)
    with pytest.raises(ValueError):
        se.conditional_tailprob(K3, x3, [0, 0])
    with pytest.raises(ValueError):
        se.conditional_tailprob(K3, x3, [0, 1], signs=[1])
