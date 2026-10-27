"""Studentized means Q = X/sqrt(Y): the expansions of Bao, Broda and Kan against exact values
(noncentral t, scipy) and against the reference values of the Dickey-Fuller application (the
Julia package and the programs of the paper agree on them to 1e-10), and the complex-coefficient
arithmetic of the Taylor algebra that they use."""
import math

import numpy as np
import pytest
from scipy.stats import nct

import saddlepoint_expansions as se
from saddlepoint_expansions.taylor import get_algebra


def test_complex_taylor_algebra():
    alg = get_algebra(2, 6)
    x, y = alg.variables()
    p = (1 + 2j) + x - 0.5j * y
    q = p * p
    assert q.constant_term() == pytest.approx((1 + 2j) ** 2)
    assert q.coeff((1, 1)) == pytest.approx(2 * (-0.5j))
    r = (p * p).sqrt()                       # principal branch: constant term 1 + 2j
    assert r.constant_term() == pytest.approx(1 + 2j)
    assert np.allclose(r.c, p.c)
    e = p.exp().log()
    assert np.allclose(e.c, p.c)
    w = (2.0 + x) ** (-2.5)
    wc = ((2.0 + 0j) + x) ** (-2.5)
    assert np.allclose(wc.c, w.c)
    assert (1 / p * p).constant_term() == pytest.approx(1.0)


def test_noncentral_t():
    # X ~ N(mu, 1), Y ~ chi2_nu/nu independent, n pairs: Xbar/sqrt(Ybar) = T/sqrt(n), T noncentral t
    mu, nu, n = 1.0, 3.0, 16
    K1 = se.CGF(lambda t: mu * t[0] + t[0] ** 2 / 2 - nu / 2 * np.log(1 - 2 * t[1] / nu), 2,
                domain=lambda t: t[1] < nu / 2)
    Kn = se.sample_mean(K1, n)
    for q in (0.5, 1.5, 2.5):
        sd = se.studentized_saddlepoint(Kn, q)
        assert sd["v"] > 0 and (q > sd["q0"]) == (sd["s"] > 0)
        r = se.studentized_tailprob(Kn, q, order=3, lower=True)
        ex = nct.cdf(q * math.sqrt(n), n * nu, math.sqrt(n) * mu)
        assert r["status"] == "ok"
        assert abs(r["info"]["Fp"][0] - ex) < 1e-3
        assert abs(r["F"][-1] - ex) < 2e-9
        assert abs(r["info"]["Fp"][-1] - ex) < abs(r["info"]["Fp"][0] - ex) / 100
        ru = se.studentized_tailprob(Kn, q, order=3)
        assert ru["F"][-1] == pytest.approx(1 - r["F"][-1])
        d = se.studentized_density(Kn, q, order=3)
        fex = nct.pdf(q * math.sqrt(n), n * nu, math.sqrt(n) * mu) * math.sqrt(n)
        assert abs(d["f"][0] / fex - 1) < 1e-2
        assert abs(d["f"][2] / fex - 1) < 1e-6
    Q = se.studentized_quantile(Kn, 0.05, order=3)
    assert abs(se.studentized_tailprob(Kn, Q["q"], order=3, lower=True)["F"][-1] - 0.05) < 1e-9
    assert abs(nct.cdf(Q["q"] * math.sqrt(n), n * nu, math.sqrt(n) * mu) - 0.05) < 1e-8
    sd = se.studentized_saddlepoint(Kn, 1.5)
    loc = se.studentized_locus(Kn, sd["s"], 0.9 * sd["t"])
    assert loc["t"] == pytest.approx(sd["t"]) and loc["v"] == pytest.approx(sd["v"]) and loc["q"] == pytest.approx(1.5)
    with pytest.raises(ValueError):
        se.studentized_density(se.marginal(K1, [0]), 1.0)


def test_dickey_fuller():
    # reference values at the 5% asymptotic critical values (c = 0): distribution function with
    # the terms up to n^(-j/2), j = 0..5, and density with M = 1..4 terms (the terms beyond the
    # fourth need more than double precision at these points)
    ref = {1: (-1.941, [0.04131561, 0.04932910, 0.05021031, 0.05004336, 0.04996497, 0.04997437],
               [1.137468e-01, 1.145105e-01, 1.142462e-01, 1.143557e-01]),
           2: (-2.862, [0.04748127, 0.05196366, 0.04837446, 0.04492099, 0.04534934, 0.04962684],
               [1.288676e-01, 1.235403e-01, 1.262806e-01, 1.274765e-01]),
           3: (-3.411, [0.04466519, 0.04979224, 0.04990885, 0.04967211, 0.04978553, 0.04999147],
               [1.307302e-01, 1.305121e-01, 1.304833e-01, 1.306511e-01])}
    for m, (q, Fref, fref) in ref.items():
        K = se.dickey_fuller_cgf(m)
        r = se.studentized_tailprob(K, q, order=3, lower=True)
        assert np.max(np.abs(r["info"]["Fp"] - np.array(Fref))) < 1e-8
        d = se.studentized_density(K, q, order=4)
        assert np.max(np.abs(d["f"][:3] / np.array(fref[:3]) - 1)) < 2e-6
        assert abs(d["f"][3] / fref[3] - 1) < 1e-3
        # the cancellation indicator reports where double precision runs out (M2 at order 4)
        assert d["status"] == ("coefficients inaccurate" if d["info"]["kappa"][-1] > 1e10 else "ok")
        assert (d["status"] == "coefficients inaccurate") == (m == 2)
    # the saddlepoint and the distribution function at the 0.75 quantile of M3 under c = -5
    K = se.dickey_fuller_cgf(3, c=-5.0)
    q = -2.025180818400757
    r = se.studentized_tailprob(K, q, order=3, lower=True)
    assert r["saddle"]["s"] == pytest.approx(2.391156286, abs=1e-8)
    assert r["saddle"]["t"] == pytest.approx(10.320589682, abs=1e-8)
    assert abs(r["info"]["Fp"][-1] - 0.739908646) < 1e-8
    # the 5% quantile of the six-term expansion for M2 (exact limit: -2.861569)
    Q = se.studentized_quantile(se.dickey_fuller_cgf(2), 0.05, order=3)
    assert Q["q"] == pytest.approx(-2.859305, abs=1e-5)
    # the series window of D: the same value on both sides of it
    K2 = se.dickey_fuller_cgf(2, c=-5.0)
    K2w = se.dickey_fuller_cgf(2, c=-5.0, window=1e-3)
    assert K2(np.array([0.3, 11.0])) == pytest.approx(K2w(np.array([0.3, 11.0])), rel=1e-10)
    with pytest.raises(ValueError):
        se.dickey_fuller_cgf(4)
