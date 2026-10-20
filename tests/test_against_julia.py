"""The Python package against reference values computed with the Julia package
SaddlepointExpansions.jl (reference_values.csv, written by test/reference_values.jl of that
package and shared with the MATLAB counterpart); the two use the same algorithms, quadrature grids
and quasi-Monte Carlo points, so they should agree to rounding, apart from the cases the Julia
package recomputes in higher precision."""
import csv
import os
from collections import defaultdict

import numpy as np
import pytest

import saddlepoint_expansions as se

REF = defaultdict(dict)
with open(os.path.join(os.path.dirname(__file__), "reference_values.csv")) as fh:
    for row in csv.DictReader(fh):
        REF[row["case"]].setdefault(row["quantity"], {})[int(row["order"])] = float(row["value"])


def get(case, q):
    """The values of a quantity as an array over the orders (or components)."""
    d = REF[case][q]
    return np.array([d[k] for k in sorted(d)])


def get1(case, q):
    return REF[case][q][1]


def derivs(case):
    out = {}
    for q, d in REF[case].items():
        if q.startswith("deriv_"):
            out[tuple(int(s) for s in q[6:].split("_"))] = d[1]
    return out


def rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


def check_derivs(K, t, case, tol=1e-12):
    D = se.derivatives(K, np.array(t), max(sum(nu) for nu in derivs(case)))
    for nu, val in derivs(case).items():
        assert abs(D[nu] - val) <= tol * max(1.0, abs(val))


A = np.array([[1.0, 0], [1, 1], [0, 1]])
toep = lambda r, n: r ** np.abs(np.subtract.outer(np.arange(n), np.arange(n)))


class TestMatlabCases:
    """The cases of the MATLAB test suite (the first part of the CSV)."""

    def test_expsum2(self):
        K = se.CGF(lambda t: -np.log(1 - t[0]) - np.log(1 - t[1]) - np.log(1 - t[0] - t[1]), 2,
                   domain=lambda t: t[0] < 1 and t[1] < 1 and t[0] + t[1] < 1)
        Kn = se.sample_mean(K, 5)
        assert rel(se.tailprob(Kn, [3, 3], order=4)["F"], get("expsum2", "F")) < 1e-12
        assert rel(se.density(Kn, [3, 3], order=4)["f"], get("expsum2", "f")) < 1e-12
        assert rel(se.tailprob(Kn, [1.5, 1.8], order=4, lower=True)["F"], get("expsum2", "Flow")) < 1e-12
        assert rel(se.tailprob(Kn, [2.5, 1.5], order=4, signs=[1, -1])["F"], get("expsum2", "Fmix")) < 1e-12

    def test_univariate(self):
        assert rel(se.tailprob(se.gamma_cgf(5), 8, order=5)["F"], get("gamma", "F")) < 1e-13
        assert rel(se.tailprob(se.gamma_cgf(5), 3, order=5, lower=True)["F"], get("gamma", "Flow")) < 1e-13
        assert rel(se.density(se.gamma_cgf(5), 8, order=5)["f"], get("gamma", "f")) < 1e-13
        assert rel(se.tailprob(se.ncx2_cgf(10, 5), 25, order=5)["F"], get("ncx2", "F")) < 1e-13
        assert rel(se.density(se.ncx2_cgf(10, 5), 25, order=5)["f"], get("ncx2", "f")) < 1e-13
        Ku = se.sample_mean(se.uniform_cgf(), 4)
        assert rel(se.tailprob(Ku, 0.4, order=4)["F"], get("uniform", "F")) < 1e-13
        assert rel(se.density(Ku, 0.4, order=4)["f"], get("uniform", "f")) < 1e-13

    def test_common_component(self):
        Kc = se.common_component_cgf([15, 20, 10], [5, 10, 8])
        x = [48.09950493836207795336339, 65.23368793961408597955183]
        assert rel(se.tailprob(Kc, x, order=5)["F"], get("cc2", "F")) < 1e-12
        assert rel(se.density(Kc, x, order=5)["f"], get("cc2", "f")) < 1e-12
        Kc3 = se.common_component_cgf([4, 5, 6, 7])
        assert rel(se.tailprob(Kc3, [14, 16, 17], order=3)["F"], get("cc3", "F")) < 1e-8
        assert rel(se.density(Kc3, [14, 16, 17], order=3)["f"], get("cc3", "f")) < 1e-11

    def test_quadratic_forms_and_ratios(self):
        n = 8
        Kq = se.quadratic_forms_cgf([toep(0.5, n), np.eye(n)], Sigma=toep(0.3, n), mu=0.5 * np.ones(n))
        assert rel(se.tailprob(Kq, [14, 13], order=4)["F"], get("qf2", "F")) < 1e-12
        assert rel(se.density(Kq, [14, 13], order=4)["f"], get("qf2", "f")) < 1e-12
        check_derivs(Kq, [0.05, -0.02], "qf2")
        Kg = se.quadratic_forms_cgf([toep(0.5, n), np.eye(n)], Sigma=toep(0.3, n), mu=0.5 * np.ones(n), closedform=False)
        check_derivs(Kg, [0.05, -0.02], "qf2", tol=1e-11)         # by automatic differentiation of the generic formula
        assert rel(se.ratio_density(Kq, 1.2, order=4)["f"], get("ratio_qf", "f")) < 1e-12
        assert rel(se.ratio_tailprob(Kq, 1.2, order=4)["F"], get("ratio_qf", "F")) < 1e-12
        assert rel(se.ratio_tailprob(Kq, 0.9, order=4, lower=True)["F"], get("ratio_qf", "Flow")) < 1e-12
        assert rel(se.ratio_density(Kq, 1.2, order=4, general=True)["f"], get("ratio2", "f_qf_general")) < 1e-11

    def test_wishart_normal_expsum3(self):
        Kw = se.wishart_diag_cgf(np.array([[1.0, 0.4], [0.4, 2.0]]), 10)
        assert rel(se.tailprob(Kw, [12, 25], order=4)["F"], get("wishart", "F")) < 1e-11
        Kz = se.normal_cgf([0.0, 1.0], [[1.0, 0.5], [0.5, 2.0]])
        assert rel(se.tailprob(Kz, [0.5, 1.5], order=2)["F"], get("normal", "F")) < 1e-11
        Ae = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1]])
        Ke = se.exponential_sums_cgf(Ae, n=6)
        assert rel(se.tailprob(Ke, [2.2, 2.6, 1.9], order=3)["F"], get("expsum3", "F")) < 1e-8
        assert rel(se.density(Ke, [2.2, 2.6, 1.9], order=3)["f"], get("expsum3", "f")) < 1e-11


class TestWang:
    K = se.exponential_sums_cgf(A, n=10, shift=[-2.0, -2.0])

    def test_derivatives(self):
        check_derivs(self.K, [0.3, -0.2], "wang")

    def test_saddlepoint(self):
        tau, info = se.saddlepoint(self.K, [-0.5, -0.5])
        assert rel(tau, get("wang", "tau")) < 1e-13 and info["decrement"] < 1e-10

    @pytest.mark.parametrize("q,x", [("Flow", [-0.5, -0.5]), ("Flow2", [-1.0, 2.0])])
    def test_lower_tail(self, q, x):
        r = se.tailprob(self.K, x, order=5, lower=True)
        assert rel(r["F"], get("wang", q)) < 1e-11
        assert rel(r["tau"], get("wang", q + "_tau")) < 1e-12
        assert r["status"] == "ok"

    def test_inclusion_exclusion(self):
        r = se.tailprob(self.K, [-0.8, 0.5], order=3)
        assert np.any(r["tau"] < 0)
        assert rel(r["F"], get("wang", "Fie")) < 1e-11

    def test_density(self):
        assert rel(se.density(self.K, [-0.5, -0.5], order=4)["f"], get("wang", "f")) < 1e-12


class TestKolassaLi:
    K = se.exponential_sums_cgf(A, n=5)

    @pytest.mark.parametrize("q,x,order", [("F33", [3.0, 3.0], 5), ("Fbdry", [2.5, 3.0], 5), ("Ffar", [6.0, 7.0], 4)])
    def test_upper_tail(self, q, x, order):
        r = se.tailprob(self.K, x, order=order)
        assert rel(r["F"], get("kl", q)) < 1e-11
        assert np.allclose(r["tau"], get("kl", q + "_tau"), rtol=1e-12, atol=1e-13)

    def test_boundary_saddlepoint(self):
        assert se.tailprob(self.K, [2.5, 3.0], order=2)["tau"][0] == 0.0


class TestQuadraticFormPairs:
    K = se.common_component_cgf([15, 20, 10], [5, 10, 8])

    def test_upper(self):
        r = se.tailprob(self.K, [55.0, 70.0], order=4)
        assert rel(r["F"], get("bqf", "F")) < 1e-11
        assert rel(r["info"]["kappa"], get("bqf", "F_kappa")) < 1e-8
        ex = get1("bqf", "F_exact")
        err = np.abs(r["F"] - ex) / ex
        assert err[0] < 2e-2 and err[3] < 3e-6

    def test_density(self):
        r = se.density(self.K, [55.0, 70.0], order=4)
        assert rel(r["f"], get("bqf", "f")) < 1e-12
        assert abs(r["f"][3] - get1("bqf", "f_exact")) / get1("bqf", "f_exact") < 2e-5

    def test_lower(self):
        r = se.tailprob(self.K, [40.0, 60.0], order=3, lower=True)
        assert rel(r["F"], get("bqf", "Flow")) < 1e-11
        assert abs(r["F"][2] - get1("bqf", "Flow_exact")) / get1("bqf", "Flow_exact") < 1e-4


class TestTrivariate:
    K = se.common_component_cgf([15, 20, 25, 10], [5, 10, 0, 8])

    def test_upper(self):
        r = se.tailprob(self.K, [80.0, 95.0, 100.0], order=4)
        assert rel(r["F"], get("tri", "F")) < 1e-10
        # the change with half the nodes is at rounding level in both implementations
        assert np.all(r["info"]["err"] <= np.maximum(2 * get("tri", "F_err"), 1e-12 * np.abs(get("tri", "F"))))
        ex = get1("tri", "F_exact")
        err = np.abs(r["F"] - ex) / ex
        assert err[0] < 2e-2 and err[3] < 2e-6

    def test_density(self):
        r = se.density(self.K, [80.0, 95.0, 100.0], order=4)
        tol = 1e-6 if get1("tri", "f_refined") else 1e-11
        assert rel(r["f"], get("tri", "f")) < tol

    def test_inclusion_exclusion(self):
        r = se.tailprob(self.K, [55.0, 70.0, 75.0], order=3)
        assert np.any(r["tau"] < 0)
        assert rel(r["F"], get("tri", "Fie")) < 1e-10
        assert abs(r["F"][2] - get1("tri", "Fie_exact")) / get1("tri", "Fie_exact") < 3e-5


class TestWishart3:
    K = se.wishart_diag_cgf(0.75 * np.eye(3) + 0.25, 5)

    def test_derivatives(self):
        check_derivs(self.K, [0.05, -0.02, 0.03], "wishart3")

    def test_upper(self):
        assert rel(se.tailprob(self.K, [6.5, 6.5, 7.5], order=3)["F"], get("wishart3", "F")) < 1e-10


class TestFourDimensions:
    K = se.common_component_cgf([15, 20, 25, 30, 10], [5, 10, 0, 3, 8])
    x = [70.0, 85.0, 95.0, 105.0]

    def test_density(self):
        r = se.density(self.K, self.x, order=3)
        tol = 1e-6 if get1("d4", "f_refined") else 1e-11
        assert rel(r["f"], get("d4", "f")) < tol

    def test_qmc(self):
        r = se.tailprob(self.K, self.x, order=3)
        assert rel(r["F"], get("d4", "F")) < 1e-8          # same lattice, shifts and conditioning
        assert rel(r["info"]["err"], get("d4", "F_err")) < 1e-6
        ex = get1("d4", "F_exact")
        assert abs(r["F"][2] - ex) <= max(1e-4 * ex, 4 * r["info"]["err"][2])


class TestUnivariateAndRatios:
    def test_gamma3(self):
        K = se.gamma_cgf(3.0)
        assert rel(se.tailprob(K, 6.0, order=4)["F"], get("gamma3", "F")) < 1e-13
        assert rel(se.tailprob(K, 6.0, order=4, lower=True)["F"], get("gamma3", "Flow")) < 1e-13
        assert rel(se.density(K, 6.0, order=4)["f"], get("gamma3", "f")) < 1e-13
        assert abs(se.tailprob(K, 6.0, order=4)["tau"] - get1("gamma3", "tau")) < 1e-14
        assert rel(se.tailprob(K, 20.0, order=5)["F"], get("gamma3", "Ffar")) < 1e-12   # Gauss-Hermite branch

    def test_ncx2b(self):
        K = se.ncx2_cgf(5, 10)
        assert rel(se.tailprob(K, 30.0, order=6)["F"], get("ncx2b", "F")) < 1e-13
        assert rel(se.density(K, 30.0, order=6)["f"], get("ncx2b", "f")) < 1e-13

    def test_uniform10(self):
        Ku = se.sample_mean(se.uniform_cgf(), 10)
        assert rel(se.tailprob(Ku, 0.3, order=4)["F"], get("uniform10", "F")) < 1e-12
        assert rel(se.density(Ku, 0.3, order=4)["f"], get("uniform10", "f")) < 1e-12
        D = se.derivatives(se.uniform_cgf(), 0.0, 6)
        for nu, val in derivs("uniform10").items():
            assert abs(D[nu] - val) <= 1e-14 * max(1.0, abs(val))
        Kb = se.CGF(lambda t: np.log(se.sinhc(t[0])) + np.log(se.sinhc(t[0] + t[1])) + np.log(se.sinhc(t[1])), 2)
        r = se.tailprob(se.sample_mean(Kb, 8), [0.4, 0.3], order=3)
        assert rel(r["F"], get("uniform10", "Fbi")) < 1e-11 and rel(r["tau"], get("uniform10", "Fbi_tau")) < 1e-12

    def test_noncentral_f(self):
        m, n, lam = 4, 12, 3
        c = 2.5 * m / n
        K = se.CGF(lambda t: se.ncx2_cgf(m, lam)(t[:1]) + se.ncx2_cgf(n)(t[1:]), 2,
                   domain=lambda t: t[0] < 0.5 and t[1] < 0.5)
        rd = se.ratio_density(K, c, order=4)
        assert rel(rd["f"], get("ncf", "f")) < 1e-12 and abs(rd["shat"] - get1("ncf", "shat")) < 1e-13
        assert rel(se.ratio_tailprob(K, c, order=4, lower=True)["F"], get("ncf", "Flow")) < 1e-12
        assert rel(se.ratio_tailprob(K, c, order=4)["F"], get("ncf", "F")) < 1e-12

    def test_two_ratios(self):
        a = np.array([3.0, 4.0, 5.0, 6.0])
        K4 = se.CGF(lambda t: -np.sum(a * np.log(1 - t)), 4, domain=lambda t: bool(np.all(t < 1)))
        assert rel(se.ratio_density(K4, [0.7, 0.9], order=5)["f"], get("ratio2", "f")) < 1e-11
        assert rel(se.ratio_tailprob(K4, [0.7, 0.9], order=5, lower=True)["F"], get("ratio2", "Flow")) < 1e-11
        assert rel(se.ratio_tailprob(K4, [0.7, 0.9], order=5)["F"], get("ratio2", "F")) < 1e-11
        K3 = se.CGF(lambda t: -3 * np.log(1 - t[0]) - 4 * np.log(1 - t[1]) - 5 * np.log(1 - t[2]), 3,
                    domain=lambda t: bool(np.all(t < 1)))
        assert rel(se.ratio_density(K3, [0.5, 0.8], order=4)["f"], get("ratio2", "f_common")) < 1e-11


def test_normal_mixed_corner():
    K = se.normal_cgf([0.0, 0.0], np.array([[1.0, 0.6], [0.6, 2.0]]))
    r = se.tailprob(K, [0.5, -0.3], order=2, signs=[1, -1])
    assert rel(r["F"], get("normal_mixed", "F")) < 1e-10
    ex = get1("normal_mixed", "exact")
    assert abs(r["F"][0] / ex - 1) < 1e-9
    assert abs(se.orthant2([0.5, 0.3], [[1.0, -0.6], [-0.6, 2.0]]) / ex - 1) < 1e-12
