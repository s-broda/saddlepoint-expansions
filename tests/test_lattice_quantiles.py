"""Lattice variables without conditioning components (tailprob and density with lattice=True, the
case B empty of Theorem 3 of the conditional paper) against exact probabilities and the rows of
its table of lattice sums; quantiles by Newton's method in saddlepoint space; normalized
densities."""
import math

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import betainc, erfcinv, gammainc, gammaln
from scipy.stats import gamma as gammad, ncf

import saddlepoint_expansions as se
from test_conditional import ROWS, pval, rowcheck


def ppmf(k, l):
    return 0.0 if k < 0 else math.exp(k * math.log(l) - l - gammaln(k + 1))


def pupp(k, l):
    return 1.0 if k <= 0 else gammainc(k, l)          # P(S >= k)


def test_table_lattice_sums():
    l1, l2, l0 = 2.0, 2.0, 1.0
    K1 = se.poisson_cgf(5.0)
    K2 = se.CGF(lambda t: l1 * (np.exp(t[0]) - 1) + l2 * (np.exp(t[1]) - 1) + l0 * (np.exp(t[0] + t[1]) - 1), 2)
    n = 0
    for block, cells in ROWS[0]:
        if block == 1:
            k = int(cells[0])
            K = K1
            Pex, pex = pupp(k, 5.0), ppmf(k, 5.0)
            rc = se.conditional_tailprob(K, [k], 0, order=4, lattice=True)
            fc = se.conditional_density(K, [k], 0, order=4, lattice=True)
            r = se.tailprob(K, k, order=4, lattice=True)
            f = se.density(K, k, order=4, lattice=True)
            assert round(r["tau"], 2) == pval(cells[2])
        else:
            k = [int(v) for v in cells[0].split(",")]
            K = K2
            Pex = sum(ppmf(y, l0) * pupp(k[0] - y, l1) * pupp(k[1] - y, l2) for y in range(61))
            pex = sum(ppmf(y, l0) * ppmf(k[0] - y, l1) * ppmf(k[1] - y, l2) for y in range(min(k) + 1))
            r = se.tailprob(K, k, order=4, lattice=True)
            f = se.density(K, k, order=4, lattice=True)
            rc = se.conditional_tailprob(K, k, [0, 1], order=4, lattice=True)
            fc = se.conditional_density(K, k, [0, 1], order=4, lattice=True)
            assert list(np.round(r["tau"], 2)) == [pval(v) for v in cells[2].split(",")]
        assert np.array_equal(r["F"], rc["F"]) and np.array_equal(f["f"], fc["f"]) and r["status"] == "ok"
        assert round(Pex, 4) == pval(cells[1]) and round(pex, 4) == pval(cells[7])
        comp = np.concatenate([(r["F"] - Pex) / min(Pex, 1 - Pex), f["f"] / pex - 1])
        assert all(rowcheck(comp, cells[3:7] + cells[8:12], {}))
        n += 1
    assert n == 11


def test_lattice_counts():
    K1 = se.poisson_cgf(5.0)
    for k in (3, 8):
        assert np.max(np.abs(se.tailprob(K1, k, order=4, lattice=True)["F"] + se.tailprob(K1, k - 1, order=4, lattice=True, lower=True)["F"] - 1)) < 1e-14
    with pytest.raises(se.DomainError):
        se.tailprob(K1, 0, lattice=True)
    with pytest.raises(ValueError):
        se.density(K1, 3, lattice=True, normalize=True)
    Kb = se.binomial_cgf(40, 0.3)
    for k in (6, 18, 25):
        Pu = betainc(k, 41 - k, 0.3)
        pk = math.exp(gammaln(41) - gammaln(k + 1) - gammaln(41 - k) + k * math.log(0.3) + (40 - k) * math.log(0.7))
        e = np.abs(se.tailprob(Kb, k, order=4, lattice=True)["F"] / Pu - 1)
        ef = np.abs(se.density(Kb, k, order=4, lattice=True)["f"] / pk - 1)
        assert e[3] < 5e-4 and e[3] < e[0] and ef[3] < 1e-4 and ef[3] < ef[0] / 10
    Knb = se.negative_binomial_cgf(5.0, 0.4)
    for k in (2, 7, 20):
        Pu = betainc(k, 5.0, 0.6)
        pk = math.exp(gammaln(k + 5.0) - gammaln(5.0) - gammaln(k + 1) + 5 * math.log(0.4) + k * math.log(0.6))
        e = np.abs(se.tailprob(Knb, k, order=4, lattice=True)["F"] / Pu - 1)
        ef = np.abs(se.density(Knb, k, order=4, lattice=True)["f"] / pk - 1)
        assert e[3] < 1e-3 and ef[3] < 1e-4 and ef[3] < ef[0] / 10
    ro = se.tailprob(K1, 7, order=3, lattice=True, sinh="origin")["F"]
    rs = se.tailprob(K1, 7, order=3, lattice=True)["F"]
    assert abs(ro[2] / rs[2] - 1) < 1e-4


def test_quantiles():
    Kg = se.gamma_cgf(3.0)
    for p in (1e-6, 0.05, 0.5, 0.95):
        for lower in (True, False):
            q = se.quantile(Kg, p, order=4, lower=lower)
            F = se.tailprob(Kg, q["q"], order=4, lower=lower)["F"][-1]
            assert abs(F - p) <= 1e-12 * min(p, 1 - p)
            ex = gammad.cdf(q["q"], 3) if lower else gammad.sf(q["q"], 3)
            assert abs(ex / p - 1) < 1e-4
    q = se.quantile(se.normal_cgf(1.0, 4.0), 0.025, order=2)
    assert abs(q["q"] - (1 - 2 * math.sqrt(2) * erfcinv(0.05))) < 1e-12
    with pytest.raises(ValueError):
        se.quantile(Kg, 1.5)
    m, n, lam = 4, 12, 3
    KF = se.CGF(lambda t: se.ncx2_cgf(m, lam)(t[:1]) + se.ncx2_cgf(n)(t[1:]), 2, domain=lambda t: t[0] < 0.5 and t[1] < 0.5)
    for p in (0.01, 0.5, 0.99):
        q = se.ratio_quantile(KF, p, order=4)
        assert abs(se.ratio_tailprob(KF, q["r"], order=4, lower=True)["F"][-1] - p) <= 1e-12 * min(p, 1 - p)
        assert abs(ncf.cdf(q["r"] * n / m, m, n, lam) - p) < 3e-4 * min(p, 1 - p)
    q = se.ratio_quantile(KF, 1e-4, order=4, lower=False)
    assert abs(se.ratio_tailprob(KF, q["r"], order=4)["F"][-1] / 1e-4 - 1) < 1e-11


def test_normalized_densities():
    def cg(a):
        return math.exp(gammaln(a) + a + (0.5 - a) * math.log(a)) / math.sqrt(2 * math.pi)

    for a in (0.5, 3.0):
        r = se.density(se.gamma_cgf(a), 2.0, order=4, normalize=True)
        tol = 1e-8 if a < 1 else 1e-12
        assert abs(r["info"]["mass"][0] / cg(a) - 1) < tol
        assert np.max(np.abs(r["f"] / gammad.pdf(2.0, a) - 1)) < tol
    assert np.max(np.abs(se.density(se.normal_cgf(1.0, 2.0), 0.3, order=3, normalize=True)["info"]["mass"] - 1)) < 1e-12
    Kx = se.ncx2_cgf(4, 3)
    r = se.density(Kx, 7.0, order=3, normalize=True)
    tot = sum(quad(lambda x: se.density(Kx, x, order=3, normalize=r["info"]["mass"])["f"][0], a, b, epsrel=1e-11, epsabs=0, limit=200)[0]
              for a, b in ((1e-9, 10), (10, 40), (40, np.inf)))
    assert abs(tot - 1) < 1e-9
    # two dimensions, first order: a linear transformation of two independent gamma variables
    Ki = se.CGF(lambda t: -2 * np.log(1 - t[0]) - 3 * np.log(1 - t[1]), 2, domain=lambda t: t[0] < 1 and t[1] < 1)
    Kl = se.linear_combination(Ki, np.array([[1.0, 0.5], [0.3, 1.0]]))
    m1 = se.density(Kl, [1.0, 2.0], order=1, normalize=True)["info"]["mass"]
    assert abs(m1[0] / (cg(2.0) * cg(3.0)) - 1) < 1e-8
    m, n, lam = 4, 12, 3
    KF = se.CGF(lambda t: se.ncx2_cgf(m, lam)(t[:1]) + se.ncx2_cgf(n)(t[1:]), 2, domain=lambda t: t[0] < 0.5 and t[1] < 0.5)
    rr = se.ratio_density(KF, 1.0, order=3, normalize=True)
    tot = sum(quad(lambda r: se.ratio_density(KF, r, order=3, normalize=rr["info"]["mass"])["f"][2], a, b, epsrel=1e-10, epsabs=0, limit=200)[0]
              for a, b in ((1e-6, 1), (1, 5), (5, 50), (50, np.inf)))
    assert abs(tot - 1) < 1e-8
