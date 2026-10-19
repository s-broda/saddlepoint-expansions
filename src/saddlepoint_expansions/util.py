"""Shared helpers: normal functions, coefficient recursions, Gauss quadrature nodes and the
saddlepoint solver (ports of the Julia package)."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc

SQRT2PI = math.sqrt(2 * math.pi)


def normpdf(x):
    return np.exp(-np.square(x) / 2) / SQRT2PI


def normcdf(x):
    return erfc(-np.asarray(x, dtype=float) / math.sqrt(2)) / 2


def fact(k):
    return float(math.factorial(k))


def hcoef(n):
    """h[i] = (-2)^i (1/2)_i = He_{2i}(0), i = 0..n."""
    h = np.empty(n + 1)
    v = 1.0
    h[0] = 1.0
    for i in range(1, n + 1):
        v *= -(2 * i - 1)
        h[i] = v
    return h


def dcoef(theta, J):
    """d[r, j] = [s^j] Theta(s)^r / r!, Theta(s) = sum_{k>=1} theta[k-1] s^k, r, j = 1..J
    (1-based storage, row and column 0 unused); d(0,0) = 1 is implicit."""
    d = np.zeros((J + 1, J + 1))
    for j in range(1, J + 1):
        d[1, j] = theta[j - 1]
    for r in range(2, J + 1):
        for j in range(r, J + 1):
            s = 0.0
            for i in range(1, j - r + 2):
                s += theta[i - 1] * d[r - 1, j - i]
            d[r, j] = s / r
    return d


def acoef(d, h, i, j):
    """a_{i,j} = sum_{k=0}^{j} d(k,j) h_{i+k}, with d(0,0) = 1 and d(0,j) = 0 for j > 0."""
    if j == 0:
        return h[i]
    s = 0.0
    for k in range(1, j + 1):
        s += d[k, j] * h[i + k]
    return s


_GH = {}
_GL = {}


def gauher(n):
    """Nodes and weights of Gauss-Hermite quadrature for the weight exp(-x^2/2)/sqrt(2 pi)."""
    if n not in _GH:
        x, w = np.polynomial.hermite.hermgauss(n)
        _GH[n] = (x * math.sqrt(2), w / math.sqrt(math.pi))
    return _GH[n]


def gauleg(n):
    """Nodes and weights of Gauss-Legendre quadrature on [-1, 1]."""
    if n not in _GL:
        _GL[n] = np.polynomial.legendre.leggauss(n)
    return _GL[n]


class SaddlepointError(RuntimeError):
    """The saddlepoint iteration failed."""


def _newton(Kfun, tau, x):
    d = len(x)

    def e(j):
        return tuple(1 if i == j else 0 for i in range(d))

    g = np.array([Kfun(e(j), tau) for j in range(d)]) - x
    H = np.array([[Kfun(tuple(a + b for a, b in zip(e(i), e(j))), tau) for j in range(d)]
                  for i in range(d)])
    dg = np.diag(H)
    if not np.all(np.isfinite(dg) & (dg > 0)):
        raise SaddlepointError("spa_saddle: the Hessian of K is not positive definite at the current iterate")
    Ds = 1.0 / np.sqrt(dg)
    Hs = Ds[:, None] * H * Ds[None, :]
    if not np.all(np.isfinite(Hs)):
        raise SaddlepointError("spa_saddle: the Hessian of K is not positive definite at the current iterate")
    try:
        np.linalg.cholesky((Hs + Hs.T) / 2)
    except np.linalg.LinAlgError:
        raise SaddlepointError("spa_saddle: the Hessian of K is not positive definite at the current iterate")
    step = -(Ds * np.linalg.solve(Hs, Ds * g))
    delta = math.sqrt(max(-float(g @ step), 0.0))
    return g, H, step, delta


def spa_saddle(Kfun, x, tau0=None):
    """Saddlepoint ``K'(tau) = x`` in any dimension: Newton's method with backtracking on the
    convex objective ``K(tau) - tau'x``, convergence measured by the Newton decrement (see the
    Julia package for the stopping rules). ``Kfun(nu, t)`` returns the derivative of order
    ``nu`` (a tuple) at ``t``, NaN outside the domain. Returns ``(tau, info)``."""
    x = np.asarray(x, dtype=float)
    d = len(x)
    tau = np.zeros(d) if tau0 is None else np.asarray(tau0, dtype=float).copy()
    if len(tau) != d:
        raise ValueError("spa_saddle: tau0 must have the length of x")
    tol, nqmax = 1e-12, 3
    z = (0,) * d

    def obj(t):
        return Kfun(z, t) - float(t @ x)

    f = obj(tau)
    if not np.isfinite(f):
        raise SaddlepointError("spa_saddle: the starting value is outside the domain of K")
    nq = 0
    it = 0
    while it < 200:
        it += 1
        g, _, step, delta = _newton(Kfun, tau, x)
        if delta <= 1e-6:                                # quadratic phase: full steps, no test
            tn = tau + step
            if np.isfinite(obj(tn)):
                tau = tn
                f = obj(tau)
            nq += 1
            if delta <= tol or nq >= nqmax:
                break
            continue
        if np.array_equal(tau + step, tau):
            raise SaddlepointError(f"spa_saddle: the iteration stalled at rounding level with decrement {delta:.2g}")
        lam = 1.0
        while True:
            tn = tau + lam * step
            fn = obj(tn)
            if np.isfinite(fn) and fn - f <= 1e-4 * lam * float(g @ step):
                break
            lam /= 2
            if lam < 1e-14 or np.array_equal(tn, tau):
                raise SaddlepointError(f"spa_saddle: saddlepoint iteration failed to find a descent step (decrement {delta:.2g})")
        tau = tn
        f = fn
    g, H, _, delta = _newton(Kfun, tau, x)
    snap = (np.abs(tau) * np.sqrt(np.abs(np.diag(H))) < tol) & (tau != 0)
    if snap.any():
        ts = tau.copy()
        ts[snap] = 0.0
        gs, _, _, ds = _newton(Kfun, ts, x)
        if ds <= max(tol, 2 * delta):
            tau, g, delta = ts, gs, ds
        else:
            snap[:] = False
    if delta > 1e-6:
        raise SaddlepointError(f"spa_saddle: saddlepoint iteration did not converge (Newton decrement {delta:.2g})")
    info = dict(iterations=it, decrement=delta, residual=g, tol=tol, snapped=snap)
    return tau, info


def addstatus(s, msg):
    return msg if s == "ok" else s + "; " + msg
