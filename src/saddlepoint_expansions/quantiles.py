"""Quantiles of the saddlepoint distribution functions and normalized density expansions, both
computed in saddlepoint space (port of quantiles.jl of the Julia package): the point is
parametrized by its saddlepoint, ``x = K'(s)``, so that every trial point of the root search and
every node of the quadrature over ``x`` comes with its saddlepoint and no saddlepoint equation is
solved there. For a ratio the parameter is the saddlepoint of ``W = X_1 - r X_2`` along the locus
``s K_1(s, t) + t K_2(s, t) = 0``, ``t = -r s``, on which ``r = K_1/K_2``."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad_vec

from .cgf import _taylor, kfun, kwhite
from .univariate import spa_ratio_pdf, spa_uni


def _uni_derivs(K, s, N):
    P = _taylor(K, np.array([s]), N)
    if P is None:
        return None
    Kd = np.array([P.coeff((k,)) * math.factorial(k) for k in range(N + 1)])
    if not (np.all(np.isfinite(Kd)) and Kd[2] > 0):
        return None
    return Kd


def _snewton(ev, s0, scale, tol, maxit=200, name="quantile"):
    """Safeguarded Newton for an increasing function of s: ev(s) -> dict(g, dg, ...) or None
    outside the domain; steps limited to 4 scale(e), pulled back into the domain, bisection once
    bracketed."""
    s = s0
    e = ev(s)
    if e is None:
        raise RuntimeError(f"{name}: the starting point is outside the domain of K")
    lo, hi = -np.inf, np.inf
    eps = np.finfo(float).eps
    for it in range(1, maxit + 1):
        if abs(e["g"]) <= tol:
            return s, e, it
        if e["g"] < 0:
            lo = s
        else:
            hi = s
        smax = 4 * scale(e)
        step = -e["g"] / e["dg"] if e["dg"] > 0 else (smax if e["g"] < 0 else -smax)
        sn = s + min(max(step, -smax), smax)
        if np.isfinite(lo) and np.isfinite(hi) and not (lo < sn < hi):
            sn = (lo + hi) / 2
        en = ev(sn)
        while en is None:
            sn = (s + sn) / 2
            if abs(sn - s) <= 4 * eps * max(abs(s), 1.0):
                raise RuntimeError(f"{name}: the iteration reached the boundary of the domain of K")
            en = ev(sn)
        if np.isfinite(lo) and np.isfinite(hi) and hi - lo <= 8 * eps * max(abs(lo), abs(hi), 1.0):
            return sn, en, it
        s, e = sn, en
    raise RuntimeError(f"{name}: the iteration did not converge")


def quantile(K, p, order=3, lower=True, tol=1e-12):
    """The quantile of the saddlepoint distribution function of order ``order`` of a univariate
    variable with cumulant generating function ``K`` (Hauschildt-Robinson expansion): ``q`` with
    ``P(X <= q) = p`` under the expansion (``lower=False``: ``P(X >= q) = p``, computed directly for
    small upper probabilities). The root search runs in saddlepoint space, ``x = K'(s)``: each
    iterate gives the point and its saddlepoint, no saddlepoint equation is solved, and the
    derivative in ``s`` is the density expansion of the same order times ``K''(s)``; safeguarded
    Newton (bracketing, bisection, steps of at most a few standard deviations, pulled back into the
    domain). Converged when the distribution function is within ``tol * min(p, 1 - p)`` of ``p``.
    Returns a dict with ``q``, ``tau`` (its saddlepoint), ``F`` (the expansion at ``q``) and
    ``iterations``. For continuous variables."""
    if K.d != 1:
        raise ValueError("quantile: K must be univariate (see ratio_quantile and studentized_quantile)")
    if order < 1:
        raise ValueError("quantile: order must be a positive integer")
    if not 0 < p < 1:
        raise ValueError("quantile: p must be in (0, 1)")
    N = 2 * order + 1

    def ev(s):
        Kd = _uni_derivs(K, s, N)
        if Kd is None:
            return None
        r = spa_uni(Kd, s, Kd[1], order)
        F = r["cdf"][-1] if lower else r["upp"][-1]
        return dict(g=(F - p if lower else p - F), dg=r["pdf"][-1] * Kd[2], x=Kd[1], F=F, K2=Kd[2])

    s, e, it = _snewton(ev, 0.0, lambda e: 1 / math.sqrt(e["K2"]), tol * min(p, 1 - p))
    return dict(q=e["x"], tau=s, F=e["F"], iterations=it)


def _ratio_locus(K, s, tguess):
    """t with s K_1 + t K_2 = 0 by Newton's method in t from tguess, with r = K_1/K_2, dr/ds and
    dt/ds along the locus; None outside the domain or without convergence."""
    t = 0.0 if s == 0 else tguess
    eps = np.finfo(float).eps
    for _ in range(60):
        P = _taylor(K, np.array([s, t]), 2)
        if P is None:
            return None
        K1, K2 = P.coeff((1, 0)), P.coeff((0, 1))
        K11, K12, K22 = 2 * P.coeff((2, 0)), P.coeff((1, 1)), 2 * P.coeff((0, 2))
        G = s * K1 + t * K2
        Gt = s * K12 + K2 + t * K22
        if not (K2 > 0 and Gt > 0):
            return None
        dt = G / Gt
        if abs(dt) <= 50 * eps * (1 + abs(t)) or G == 0:
            Gs = K1 + s * K11 + t * K12
            tp = -Gs / Gt
            r = K1 / K2
            rp = ((K11 + K12 * tp) * K2 - K1 * (K12 + K22 * tp)) / K2 ** 2
            return dict(t=t, r=r, rp=rp, tp=tp)
        t -= dt
    return None


def _ratio_at(K, s, loc, order):
    r = loc["r"]
    N = 2 * order + 1
    P = _taylor(K, np.array([s, -r * s]), N, np.array([[1.0, 0.0], [-r, 1.0]]))
    if P is None:
        return None
    KWd = np.array([P.coeff((k, 0)) * math.factorial(k) for k in range(N + 1)])
    if not (np.all(np.isfinite(KWd)) and KWd[2] > 0):
        return None
    gd = np.array([P.coeff((l, 1)) * math.factorial(l) for l in range(2 * order - 1)])
    u = spa_uni(KWd, s, 0.0, order)
    f = spa_ratio_pdf(KWd[:2 * order + 1], gd, s, order)
    return dict(u=u, f=np.asarray(f), KW2=KWd[2])


def ratio_quantile(K, p, order=3, lower=True, tol=1e-12):
    """The quantile of the saddlepoint distribution function of order ``order`` of the ratio
    ``R = X_1/X_2``, ``X_2 > 0``, ``K`` the joint cumulant generating function of ``(X_1, X_2)``:
    ``r`` with ``P(R <= r) = p`` (``lower=False``: ``P(R >= r) = p``). The search runs over the
    saddlepoint ``shat`` of ``W = X_1 - r X_2`` along the curve ``s K_1 + t K_2 = 0``,
    ``t = -r shat``, on which ``r = K_1/K_2``: each trial ``shat`` gives its ``t`` (a one-dimensional
    Newton iteration on the curve, continued from the previous iterate), its ``r`` and its
    distribution function directly, with the derivative ``f_R(r) dr/dshat`` from the ratio density
    expansion of the same order; safeguarded as :func:`quantile`. Returns a dict with ``r``,
    ``shat``, ``F`` and ``iterations``."""
    if K.d != 2:
        raise ValueError("ratio_quantile: K must be the joint cumulant generating function of the numerator and the denominator (d = 2)")
    if order < 1:
        raise ValueError("ratio_quantile: order must be a positive integer")
    if not 0 < p < 1:
        raise ValueError("ratio_quantile: p must be in (0, 1)")
    last = [0.0, 0.0, 0.0]

    def ev(s):
        s0, t0, tp0 = last
        loc = _ratio_locus(K, s, t0 + tp0 * (s - s0))
        if loc is None:
            return None
        a = _ratio_at(K, s, loc, order)
        if a is None:
            return None
        last[:] = [s, loc["t"], loc["tp"]]
        F = a["u"]["cdf"][-1] if lower else a["u"]["upp"][-1]
        return dict(g=(F - p if lower else p - F), dg=a["f"][-1] * loc["rp"], r=loc["r"], F=F, KW2=a["KW2"])

    s, e, it = _snewton(ev, 0.0, lambda e: 1 / math.sqrt(e["KW2"]), tol * min(p, 1 - p), name="ratio_quantile")
    return dict(r=e["r"], shat=s, F=e["F"], iterations=it)


# ----- the total mass of density expansions, by quadrature in saddlepoint space

def _srange(logw, h, c):
    """The interval of v around 0 where logw (None outside the domain) has not fallen c below its
    largest value: doubling steps from h, bisection toward the boundary; infinite for an end at
    which it has not fallen after 4096 h (an algebraic tail)."""
    l0 = logw(0.0)
    if l0 is None:
        raise RuntimeError("normalize: the starting point is outside the domain of K")
    ends = []
    for dr in (-1, 1):
        inside = 0.0
        v = h
        lmax = l0
        while True:
            l = logw(dr * v)
            if l is None:
                out = v
                for _ in range(60):
                    mid = (inside + out) / 2
                    if logw(dr * mid) is None:
                        out = mid
                    else:
                        inside = mid
                    if out - inside <= 1e-12 * max(out, 1.0):
                        break
                ends.append(dr * inside)
                break
            lmax = max(lmax, l)
            if l < lmax - c:
                ends.append(dr * v)
                break
            inside = v
            v *= 2
            if v > 4096 * h:
                ends.append(dr * np.inf)
                break
    return ends[0], ends[1]


def _integrate(f, lo, hi, rtol):
    """The integral over [lo, 0] and [0, hi] of the vector function f."""
    a = quad_vec(f, lo, 0.0, epsrel=rtol, epsabs=0.0, limit=500)[0]
    b = quad_vec(f, 0.0, hi, epsrel=rtol, epsabs=0.0, limit=500)[0]
    return a + b


def _density_mass(K, order, rtol=None):
    """The masses of the density expansions of orders 1..order: the integral over x of f_M(x) as
    the integral over s of f_M(K'(s)) det K''(s), nested over the coordinates (see the Julia
    package)."""
    from .conditional import _cond_dens
    d = K.d
    rtol = (1e-12 if d == 1 else 1e-9) if rtol is None else rtol
    maxdeg = 2 * order + 1
    Kf = kfun(K, maxdeg)
    Kw = kwhite(K, maxdeg)

    def e(j):
        return tuple(1 if i == j else 0 for i in range(d))

    def ee(i, j):
        return tuple(a + b for a, b in zip(e(i), e(j)))

    H0 = np.array([[Kf(ee(i, j), np.zeros(d)) for j in range(d)] for i in range(d)])

    def jet2(s):
        P = _taylor(K, np.asarray(s, dtype=float), 2)
        if P is None:
            return None
        g = np.array([P.coeff(e(j)) for j in range(d)])
        H = np.array([[P.coeff(ee(i, j)) * (2 if i == j else 1) for j in range(d)] for i in range(d)])
        return P.constant_term(), g, H

    def lead(s):
        jt = jet2(s)
        if jt is None:
            return None
        K0, g, H = jt
        if not np.all(np.isfinite(H)):
            return None
        try:
            Lc = np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return None
        return K0 - float(np.dot(s, g)) - d / 2 * math.log(2 * math.pi) + float(np.sum(np.log(np.diag(Lc))))

    def full(s):
        if d == 1:
            Kd = _uni_derivs(K, float(s[0]), maxdeg)
            if Kd is None:
                return np.zeros(order)
            return np.asarray(spa_uni(Kd, float(s[0]), Kd[1], order)["pdf"]) * Kd[2]
        jt = jet2(s)
        if jt is None:
            return np.zeros(order)
        _, g, H = jt
        # where K'' is numerically singular (at the boundary of the domain, where the density is
        # negligible) the node counts as outside
        if not np.all(np.isfinite(H)):
            return np.zeros(order)
        try:
            np.linalg.cholesky(H)
        except np.linalg.LinAlgError:
            return np.zeros(order)
        logf0, sb, _ = _cond_dens(Kf, Kw, np.asarray(s, dtype=float), g, order)
        return math.exp(logf0) * sb * np.linalg.det(H)

    def inside(s):
        return jet2(s) is not None

    def level(k, outer):
        ck = 0.0 if k == d - 1 else -float(H0[k, k + 1:] @ outer) / H0[k, k]
        h = 1 / math.sqrt(H0[k, k])

        def pt(v):
            return np.concatenate([np.zeros(k), [v], outer])

        if not inside(pt(ck)):
            found = False
            for j in range(41):
                for dr in (-1, 1):
                    v = ck + dr * h * 2.0 ** (j / 2)
                    if inside(pt(v)):
                        ck = v
                        found = True
                        break
                if found:
                    break
            if not found:
                return np.zeros(order)
        if k == 0:
            lw = lambda v: lead(pt(ck + v))
            f = lambda v: full(pt(ck + v))
        else:
            memo = {}

            def inner(v):
                if v not in memo:
                    memo[v] = level(k - 1, np.concatenate([[ck + v], outer]))
                return memo[v]

            lw = lambda v: (math.log(inner(v)[0]) if inner(v)[0] > 0 else None)
            f = inner
        lo, hi = _srange(lw, h, 50.0)
        return _integrate(f, lo, hi, rtol)

    return level(d - 1, np.zeros(0))


def _ratio_mass(K, order, rtol=1e-10):
    """The masses of the ratio density expansions: the integral over r of f_R(r) as the integral
    over the saddlepoint along the locus of f_R(r(s)) dr/ds."""
    cache = {0.0: (0.0, 0.0)}

    def locus(s):
        sn = min(cache, key=lambda v: abs(v - s))
        t0, tp0 = cache[sn]
        loc = _ratio_locus(K, s, t0 + tp0 * (s - sn))
        if loc is None:
            tt, tp = t0, tp0
            m = 8
            for j in range(1, m + 1):
                sj = sn + (s - sn) * j / m
                loc = _ratio_locus(K, sj, tt + tp * (s - sn) / m)
                if loc is None:
                    return None
                tt, tp = loc["t"], loc["tp"]
        cache[s] = (loc["t"], loc["tp"])
        return loc

    def vals(s):
        loc = locus(s)
        if loc is None:
            return None
        a = _ratio_at(K, s, loc, order)
        if a is None:
            return None
        return a["f"] * loc["rp"], a

    def lw(s):
        v = vals(s)
        if v is None or not v[0][0] > 0:
            return None
        return math.log(v[0][0])

    a0 = vals(0.0)
    if a0 is None:
        raise RuntimeError("normalize: the ratio expansion cannot be evaluated at the centre")
    lo, hi = _srange(lw, 1 / math.sqrt(a0[1]["KW2"]), 50.0)

    def f(s):
        v = vals(s)
        return np.zeros(order) if v is None else v[0]

    return _integrate(f, lo, hi, rtol)
