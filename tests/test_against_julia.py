"""The Python package against reference values computed with the Julia package
SaddlepointExpansions.jl (tests/make_reference.jl); the two use the same algorithms, quadrature
grids and quasi-Monte Carlo points, so they should agree to rounding, apart from the cases the
Julia package recomputes in higher precision."""
import json
import os

import numpy as np
import pytest

import saddlepoint_expansions as se

with open(os.path.join(os.path.dirname(__file__), "reference_julia.json")) as fh:
    REF = json.load(fh)


def arr(v):
    return np.array([np.nan if x == "nan" else x for x in v], dtype=float)


def rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def check_derivs(K, entry, tol=1e-12):
    D = se.derivatives(K, np.array(entry["deriv_t"]), entry["deriv_N"])
    for key, val in entry["deriv"].items():
        nu = tuple(int(s) for s in key.split(","))
        assert abs(D[nu] - val) <= tol * max(1.0, abs(val))


A = np.array([[1.0, 0], [1, 1], [0, 1]])


class TestWang:
    K = se.exponential_sums_cgf(A, n=10, shift=[-2.0, -2.0])
    R = REF["wang"]

    def test_derivatives(self):
        check_derivs(self.K, self.R)

    def test_saddlepoint(self):
        tau, info = se.saddlepoint(self.K, [-0.5, -0.5])
        assert rel(tau, self.R["saddle"]) < 1e-13 and info["decrement"] < 1e-10

    @pytest.mark.parametrize("key", ["lower", "lower2"])
    def test_lower_tail(self, key):
        e = self.R[key]
        r = se.tailprob(self.K, e["x"], order=e["order"], lower=True)
        assert rel(r["F"], e["F"]) < 1e-11
        assert rel(r["tau"], e["tau"]) < 1e-12
        assert r["status"] == "ok"

    def test_inclusion_exclusion(self):
        e = self.R["upper_ie"]
        r = se.tailprob(self.K, e["x"], order=e["order"])
        assert np.any(r["tau"] < 0)
        assert rel(r["F"], e["F"]) < 1e-11

    def test_density(self):
        e = self.R["density"]
        r = se.density(self.K, e["x"], order=e["order"])
        assert rel(r["f"], e["f"]) < 1e-12


class TestKolassaLi:
    K = se.exponential_sums_cgf(A, n=5)
    R = REF["kl"]

    @pytest.mark.parametrize("key", ["upper", "boundary", "far"])
    def test_upper_tail(self, key):
        e = self.R[key]
        r = se.tailprob(self.K, e["x"], order=e["order"])
        assert rel(r["F"], e["F"]) < 1e-11
        assert rel(r["tau"], e["tau"]) < 1e-12 or np.allclose(r["tau"], e["tau"], atol=1e-13)

    def test_boundary_saddlepoint(self):
        r = se.tailprob(self.K, [2.5, 3.0], order=2)
        assert r["tau"][0] == 0.0


class TestQuadraticForms:
    R = REF["bqf"]
    K = se.common_component_cgf(R["m"], R["lam"])

    def test_upper(self):
        e = self.R["upper"]
        r = se.tailprob(self.K, e["x"], order=e["order"])
        assert rel(r["F"], e["F"]) < 1e-11
        assert rel(r["info"]["kappa"], e["kappa"]) < 1e-8
        err = np.abs(np.array(r["F"]) - e["exact"]) / e["exact"]
        assert err[0] < 2e-2 and err[3] < 3e-6

    def test_density(self):
        e = self.R["density"]
        r = se.density(self.K, e["x"], order=e["order"])
        assert rel(r["f"], e["f"]) < 1e-12
        assert abs(r["f"][3] - e["exact"]) / e["exact"] < 2e-5

    def test_lower(self):
        e = self.R["lower"]
        r = se.tailprob(self.K, e["x"], order=e["order"], lower=True)
        assert rel(r["F"], e["F"]) < 1e-11
        assert abs(r["F"][2] - e["exact"]) / e["exact"] < 1e-4


class TestTrivariate:
    R = REF["tri"]
    K = se.common_component_cgf(R["m"], R["lam"])

    def test_upper(self):
        e = self.R["upper"]
        r = se.tailprob(self.K, e["x"], order=e["order"])
        assert rel(r["F"], e["F"]) < 1e-10
        # the change with half the nodes is at rounding level in both implementations
        assert np.all(r["info"]["err"] <= np.maximum(2 * arr(e["err"]), 1e-12 * np.abs(arr(e["F"]))))
        err = np.abs(np.array(r["F"]) - e["exact"]) / e["exact"]
        assert err[0] < 2e-2 and err[3] < 2e-6

    def test_density(self):
        e = self.R["density"]
        r = se.density(self.K, e["x"], order=e["order"])
        tol = 1e-6 if e["refined"] else 1e-11
        assert rel(r["f"], e["f"]) < tol

    def test_inclusion_exclusion(self):
        e = self.R["ie"]
        r = se.tailprob(self.K, e["x"], order=e["order"])
        assert np.any(r["tau"] < 0)
        assert rel(r["F"], e["F"]) < 1e-10
        assert abs(r["F"][2] - e["exact"]) / e["exact"] < 3e-5


class TestWishart:
    R = REF["wishart"]
    K = se.wishart_diag_cgf(np.array(R["V"]), R["n"])

    def test_derivatives(self):
        check_derivs(self.K, self.R)

    def test_upper(self):
        e = self.R["upper"]
        r = se.tailprob(self.K, e["x"], order=e["order"])
        assert rel(r["F"], e["F"]) < 1e-10


class TestFourDimensions:
    R = REF["d4"]
    K = se.common_component_cgf(R["m"], R["lam"])

    def test_density(self):
        e = self.R["density"]
        r = se.density(self.K, self.R["x"], order=e["order"])
        tol = 1e-6 if e["refined"] else 1e-11
        assert rel(r["f"], e["f"]) < tol

    def test_qmc(self):
        e = self.R["qmc"]
        r = se.tailprob(self.K, self.R["x"], order=e["order"])
        # same lattice, shifts and conditioning: agreement to rounding
        assert rel(r["F"], e["F"]) < 1e-8
        assert rel(r["info"]["err"], e["err"]) < 1e-6
        assert abs(r["F"][2] - e["exact"]) <= max(1e-4 * e["exact"], 4 * r["info"]["err"][2])


class TestUnivariate:
    R = REF["uni"]

    def test_gamma(self):
        e = self.R["gamma"]
        K = se.gamma_cgf(e["a"])
        r = se.tailprob(K, e["x"], order=e["order"])
        rl = se.tailprob(K, e["x"], order=e["order"], lower=True)
        f = se.density(K, e["x"], order=e["order"])
        assert rel(r["F"], e["upp"]) < 1e-13 and rel(rl["F"], e["cdf"]) < 1e-13 and rel(f["f"], e["pdf"]) < 1e-13
        assert abs(r["tau"] - e["tau"]) < 1e-14
        e2 = self.R["gamma_far"]                       # Gauss-Hermite branch, |u| > 2.5
        assert rel(se.tailprob(K, e2["x"], order=e2["order"])["F"], e2["upp"]) < 1e-12

    def test_ncx2(self):
        e = self.R["ncx2"]
        K = se.ncx2_cgf(e["m"], e["lam"])
        assert rel(se.tailprob(K, e["x"], order=e["order"])["F"], e["upp"]) < 1e-13
        assert rel(se.density(K, e["x"], order=e["order"])["f"], e["pdf"]) < 1e-13


class TestRatio:
    R = REF["ratio"]

    def test_noncentral_f(self):
        e = self.R
        m, n, lam = e["m"], e["n"], e["lam"]
        K = se.CGF(lambda t: se.ncx2_cgf(m, lam)(t[:1]) + se.ncx2_cgf(n)(t[1:]), 2,
                   domain=lambda t: t[0] < 0.5 and t[1] < 0.5)
        rd = se.ratio_density(K, e["c"], order=e["order"])
        assert rel(rd["f"], e["density"]) < 1e-12 and abs(rd["shat"] - e["shat"]) < 1e-13
        assert rel(se.ratio_tailprob(K, e["c"], order=e["order"], lower=True)["F"], e["cdf"]) < 1e-12
        assert rel(se.ratio_tailprob(K, e["c"], order=e["order"])["F"], e["upp"]) < 1e-12


def test_normal_mixed_corner():
    e = REF["normal_mixed"]
    K = se.normal_cgf([0.0, 0.0], np.array(e["S"]))
    r = se.tailprob(K, e["x"], order=2, signs=e["signs"])
    assert rel(r["F"], e["F"]) < 1e-10
    assert abs(r["F"][0] / e["exact"] - 1) < 1e-9
    assert abs(se.orthant2([0.5, 0.3], [[1.0, -0.6], [-0.6, 2.0]]) / e["exact"] - 1) < 1e-12


def test_uniform_against_julia():
    e = REF["uniform"]
    Ku = se.sample_mean(se.uniform_cgf(), e["n"])
    assert rel(se.tailprob(Ku, e["x"], order=e["order"])["F"], e["upp"]) < 1e-12
    assert rel(se.density(Ku, e["x"], order=e["order"])["f"], e["pdf"]) < 1e-12
    D = se.derivatives(se.uniform_cgf(), 0.0, 6)
    for key, val in e["deriv0"].items():
        assert abs(D[(int(key),)] - val) <= 1e-14 * max(1.0, abs(val))
    b = e["bivar"]
    Kb = se.CGF(lambda t: np.log(se.sinhc(t[0])) + np.log(se.sinhc(t[0] + t[1])) + np.log(se.sinhc(t[1])), 2)
    r = se.tailprob(se.sample_mean(Kb, 8), b["x"], order=b["order"])
    assert rel(r["F"], b["F"]) < 1e-11 and rel(r["tau"], b["tau"]) < 1e-12
