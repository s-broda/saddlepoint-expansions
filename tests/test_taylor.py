"""The Taylor algebra against closed-form derivatives."""
import math

import numpy as np
import pytest

from saddlepoint_expansions import DomainError, get_algebra, variables
from saddlepoint_expansions.taylor import TPoly


A = np.array([[1.0, 0], [1, 1], [0, 1]])
n = 10
l = np.array([-2.0, -2.0])


def K(t):
    return -n * np.sum(np.log(1 - A @ t / n)) + l @ t


def lin_exp(nu, t):
    k = sum(nu)
    z = 1 - A @ t / n
    if k == 0:
        return -n * np.sum(np.log(z)) + l @ t
    v = 0.0
    for i in range(A.shape[0]):
        an = np.prod([A[i, j] ** nu[j] for j in range(2)])
        if an == 0:
            continue
        v += n * math.factorial(k - 1) * an / n ** k / z[i] ** k
    if k == 1:
        v += l[list(nu).index(1)]
    return v


def test_derivatives_match_closed_form():
    t = np.array([0.3, -0.2])
    u = variables(2, 11)
    p = K(t + u)
    alg = get_algebra(2, 11)
    assert len(alg.exps) == math.comb(11 + 2, 2)
    for nu in alg.exps:
        ref = lin_exp(nu, t)
        assert abs(p.derivative(nu) - ref) <= 1e-13 * max(1.0, abs(ref))


def test_domain_error():
    u = variables(2, 3)
    with pytest.raises(DomainError):
        K(np.array([20.0, 0.0]) + u)
    with pytest.raises(DomainError):
        np.sqrt(u[0] - 1.0)
    with pytest.raises(DomainError):
        1.0 / u[0]


def test_elementary_functions():
    u = variables(1, 8)
    x = u[0]
    c = 0.7
    p = np.exp(np.log(1 + x) * 2.5)          # (1 + x)^2.5
    q = (1 + x) ** 2.5
    assert np.allclose(p.c, q.c, atol=1e-14)
    r = np.sqrt(c + x) * np.sqrt(c + x)
    assert np.allclose(r.c, (c + x).c, atol=1e-14)
    s = (c + x) / (c + x)
    assert abs(s.coeff((0,)) - 1) < 1e-15 and np.all(np.abs(s.c[1:]) < 1e-14)
    e = np.exp(x)
    assert np.allclose(e.c, [1 / math.factorial(k) for k in range(9)])
    assert (2.0 ** x).coeff((3,)) == pytest.approx(math.log(2) ** 3 / 6)


def test_numpy_interoperability():
    u = variables(3, 4)
    M = np.array([[1.0, 2.0, 0.0], [0.5, 0.0, 1.0]])
    v = M @ u                                # object array of polynomials
    assert v.dtype == object and isinstance(v[0], TPoly)
    assert v[0].coeff((0, 1, 0)) == 2.0
    w = np.float64(2.0) * u[0] + u[1] * np.float64(3.0) - 1
    assert isinstance(w, TPoly) and w.coeff((1, 0, 0)) == 2.0 and w.coeff((0,) * 3) == -1.0
    s = np.sum(u)
    assert isinstance(s, TPoly) and s.coeff((0, 0, 1)) == 1.0
    e = np.exp(u)                            # elementwise
    assert e.dtype == object and e[2].coeff((0, 0, 2)) == 0.5
    outer = u[0] * np.eye(2)                 # polynomial times array
    assert outer.dtype == object and outer[0, 0].coeff((1, 0, 0)) == 1.0 and outer[0, 1].coeff((1, 0, 0)) == 0.0
    assert np.exp(np.eye(2) - 2 * (u[0] * np.eye(2)))[1, 1].coeff((1, 0, 0)) == pytest.approx(-2 * math.e)


def test_integer_powers_and_bool():
    u = variables(2, 6)
    p = (u[0] + 2.0) ** 3
    assert p.coeff((3, 0)) == 1.0 and p.coeff((0, 0)) == 8.0 and p.coeff((2, 0)) == 6.0
    assert bool(u[0]) and not bool(u[0] * 0.0)
    with pytest.raises(TypeError):
        float(u[0])


def test_trigonometric_and_hyperbolic():
    from saddlepoint_expansions import sincu, sinhc
    u = variables(1, 9)
    x = u[0]
    c = 0.4
    p = c + x
    for f, fm in ((np.sin, math.sin), (np.cos, math.cos), (np.sinh, math.sinh), (np.cosh, math.cosh)):
        q = f(p)
        assert q.coeff((0,)) == pytest.approx(fm(c))
    assert (np.sin(p) ** 2 + np.cos(p) ** 2 - 1).c == pytest.approx(np.zeros(10), abs=1e-14)
    assert (np.cosh(p) ** 2 - np.sinh(p) ** 2 - 1).c == pytest.approx(np.zeros(10), abs=1e-14)
    assert (np.tan(p) - np.sin(p) / np.cos(p)).c == pytest.approx(np.zeros(10), abs=1e-14)
    assert (np.tanh(p) ** 2 + 1 / np.cosh(p) ** 2 - 1).c == pytest.approx(np.zeros(10), abs=1e-14)
    a = np.arctan(p)
    assert a.derivative((1,)) == pytest.approx(1 / (1 + c * c))
    assert a.derivative((2,)) == pytest.approx(-2 * c / (1 + c * c) ** 2)
    assert (np.arctan(np.tan(p)) - p).c == pytest.approx(np.zeros(10), abs=1e-12)
    assert np.sin(p).derivative((5,)) == pytest.approx(math.cos(c))
    # removable singularities
    s0 = sinhc(x)
    assert s0.coeff((0,)) == 1.0 and s0.coeff((2,)) == pytest.approx(1 / 6) and s0.coeff((4,)) == pytest.approx(1 / 120)
    assert s0.coeff((1,)) == 0.0 and s0.coeff((3,)) == 0.0
    c0 = sincu(x)
    assert c0.coeff((2,)) == pytest.approx(-1 / 6) and c0.coeff((4,)) == pytest.approx(1 / 120)
    assert (sinhc(p) - np.sinh(p) / p).c == pytest.approx(np.zeros(10), abs=1e-14)
    assert (sincu(p) - np.sin(p) / p).c == pytest.approx(np.zeros(10), abs=1e-14)
    assert sinhc(0.0) == 1.0 and sincu(0.0) == 1.0 and sinhc(0.5) == pytest.approx(math.sinh(0.5) / 0.5)
    assert np.allclose(sincu(np.array([0.0, 1.0])), [1.0, math.sin(1.0)])
