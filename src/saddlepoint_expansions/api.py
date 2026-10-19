"""The high-level interface: saddlepoint, tail probabilities, densities and ratios from a CGF."""
from __future__ import annotations

import numpy as np

from .cgf import CGF, _fact, _taylor, kfun, kwhite, reflect
from .multivariate import spa_mv, spa_orthant, spa_orthant2
from .taylor import DomainError, get_algebra
from .univariate import spa_ratio_pdf, spa_uni
from .util import spa_saddle


def saddlepoint(K, x, tau0=None):
    """The saddlepoint ``K'(tau) = x`` by Newton's method with backtracking (see
    :func:`~saddlepoint_expansions.util.spa_saddle`). Returns ``(tau, info)``; for ``d = 1``,
    ``x`` may be a scalar and ``tau`` is then a scalar."""
    scalar = np.ndim(x) == 0
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    if len(xx) != K.d:
        raise ValueError(f"saddlepoint: x must have length {K.d}")
    tau, info = spa_saddle(kfun(K, 2), xx, None if tau0 is None else np.atleast_1d(tau0))
    return (tau[0] if scalar else tau), info


def _signs(name, d, lower, signs):
    if signs is None:
        return np.full(d, -1.0) if lower else np.ones(d)
    s = np.asarray(signs, dtype=float)
    if len(s) != d:
        raise ValueError(f"{name}: signs must have length {d}")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError(f"{name}: signs must be +-1")
    return s


def tailprob(K, x, order=3, lower=False, signs=None, **kwargs):
    """Saddlepoint expansions of orders ``1..order`` of the orthant probability ``P(X >= x)``
    (componentwise) for the random vector with cumulant generating function ``K``.

    Returns a dict with ``F`` (array over the orders; ``F[M-1]`` keeps the terms of orders
    ``n^(-m/2)``, ``m <= 2M-1``, and has relative error ``O(n^-M)`` uniformly for saddlepoints
    in compact subsets of the closed positive orthant, Theorem 1 of the paper; ``F[0]`` is the
    approximation of Kolassa (2003) in two or more dimensions), ``tau`` (the saddlepoint),
    ``status`` and ``info`` (the full result of the underlying routine).

    ``lower=True`` gives ``P(X <= x)``, and ``signs`` (a vector of ``+-1``) any mixed corner,
    e.g. ``signs=[1, -1]`` for ``P(X1 >= x1, X2 <= x2)``. When the saddlepoint has negative
    components the expansion is applied through the inclusion-exclusion identity of the paper.
    In three dimensions the orthant integrals are computed by Gauss quadrature and in four or
    more by randomized quasi-Monte Carlo (``info['err']`` is then a standard error); keyword
    arguments are passed to :func:`~saddlepoint_expansions.multivariate.spa_mv` (``nG``,
    ``nqmc``, ...) and, for ``d = 2``, to
    :func:`~saddlepoint_expansions.multivariate.spa_orthant2` (``nG``, ``logscale``, ...).
    For ``d = 1``, ``x`` may be a scalar."""
    if order < 1:
        raise ValueError("tailprob: order must be a positive integer")
    d = K.d
    scalar = np.ndim(x) == 0
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    if len(xx) != d:
        raise ValueError(f"tailprob: x must have length {d}")
    if not np.all(np.isfinite(xx)):
        raise ValueError("tailprob: x must be finite")
    sg = _signs("tailprob", d, lower, signs)
    Kr = K if np.all(sg == 1) else reflect(K, sg)
    xr = sg * xx
    maxdeg = 2 * order + 1
    Kf = kfun(Kr, maxdeg)
    if d == 1:
        s, sinfo = spa_saddle(Kf, xr, None)
        Kd = [Kf((k,), s) for k in range(maxdeg + 1)]
        if not np.all(np.isfinite(Kd)):
            raise RuntimeError("tailprob: the saddlepoint is outside the domain of K")
        r = spa_uni(Kd, s[0], xr[0], order)
        r["saddle"] = sinfo
        tau = sg * s
        return dict(F=r["upp"], tau=(tau[0] if scalar else tau), status="ok", info=r)
    if d == 2:
        r = spa_orthant2(Kf, xr, order, Kwhite=kwhite(Kr, maxdeg), **kwargs)
        return dict(F=r["Fup"], tau=sg * r["tau"], status=r["status"], info=r)
    r = spa_orthant(Kf, xr, order, Kwhite=kwhite(Kr, maxdeg), **kwargs)
    return dict(F=r["Fup"], tau=sg * r["tau"], status="ok", info=r)


def density(K, x, order=3, **kwargs):
    """Saddlepoint expansions of orders ``1..order`` of the density at ``x`` (Theorem 3 of the
    paper; Daniels's expansion when ``d = 1``): ``f[M-1]`` keeps the terms ``m <= M-1`` and
    has relative error ``O(n^-M)``. A finite sum of Hermite polynomials, no integration.
    Returns a dict with ``f``, ``tau``, ``status`` and ``info``. In three or more dimensions
    the cancellation indicator ``info['kappa_pdf']`` should be checked at high orders (the
    Julia package repeats the computation in higher precision when it is large; this package
    works in double precision)."""
    if order < 1:
        raise ValueError("density: order must be a positive integer")
    d = K.d
    scalar = np.ndim(x) == 0
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    if len(xx) != d:
        raise ValueError(f"density: x must have length {d}")
    if not np.all(np.isfinite(xx)):
        raise ValueError("density: x must be finite")
    maxdeg = 2 * order + 1
    Kf = kfun(K, maxdeg)
    if d == 1:
        s, sinfo = spa_saddle(Kf, xx, None)
        Kd = [Kf((k,), s) for k in range(maxdeg + 1)]
        if not np.all(np.isfinite(Kd)):
            raise RuntimeError("density: the saddlepoint is outside the domain of K")
        r = spa_uni(Kd, s[0], xx[0], order)
        r["saddle"] = sinfo
        return dict(f=r["pdf"], tau=(s[0] if scalar else s), status="ok", info=r)
    if d == 2:
        from .bivariate import spa_bi
        r = spa_bi(Kf, xx, order, Kwhite=kwhite(K, maxdeg), **kwargs)
        return dict(f=r["pdf"], tau=r["tau"], status="ok", info=r)
    r = spa_mv(Kf, xx, order, method="none", Kwhite=kwhite(K, maxdeg), **kwargs)
    st = "ok" if np.all(r["kappa_pdf"] <= 1e6) else "large cancellation in the density sum"
    return dict(f=r["pdf"], tau=r["tau"], status=st, info=r)


def _ratio_cgf(K, r):
    if K.d != 2:
        raise ValueError("ratio: K must be the joint cumulant generating function of the numerator and the denominator (d = 2)")
    Kf = K.K
    dom = None if K.domain is None else (lambda s, D=K.domain: D(np.array([s[0], -r * s[0]])))

    def f(s):
        t = np.empty(2, dtype=object)
        t[0] = s[0]
        t[1] = -r * s[0]
        return Kf(t)

    return CGF(f, 1, domain=dom)


def ratio_density(K, r, order=3):
    """Saddlepoint expansions of orders ``1..order`` of the density at ``r`` of the ratio
    ``X1/X2`` of the two components of a vector with cumulant generating function ``K``
    (``d = 2``), the denominator being positive: the Rice expansion through the saddlepoint of
    ``K_W(s) = K(s, -r s)`` (Theorem 4 of the paper). Returns a dict with ``f``, ``shat``,
    ``status`` and ``info``."""
    if order < 1:
        raise ValueError("ratio_density: order must be a positive integer")
    KW = _ratio_cgf(K, r)
    s, sinfo = spa_saddle(kfun(KW, 2), np.array([0.0]), None)
    sh = float(s[0])
    N = 2 * order + 1
    alg = get_algebra(2, N)
    v = alg.variables()
    t = np.empty(2, dtype=object)
    t[0] = sh + v[0]
    t[1] = -r * (sh + v[0]) + v[1]
    try:
        P = K.K(t)
    except DomainError:
        raise RuntimeError("ratio_density: the saddlepoint is outside the domain of K")
    if not np.isfinite(P.constant_term()):
        raise RuntimeError("ratio_density: the saddlepoint is outside the domain of K")
    KWd = [P.coeff((k, 0)) * _fact((k,)) for k in range(2 * order + 1)]
    gd = [P.coeff((l, 1)) * _fact((l,)) for l in range(2 * order - 1)]
    f = spa_ratio_pdf(KWd, gd, sh, order)
    return dict(f=f, shat=sh, status="ok", info=dict(KWd=np.array(KWd), gd=np.array(gd), saddle=sinfo))


def ratio_tailprob(K, r, order=3, lower=False):
    """Saddlepoint expansions of orders ``1..order`` of ``P(X1/X2 >= r)`` (``P(X1/X2 <= r)``
    with ``lower=True``) for a positive denominator: the univariate expansion of
    :func:`tailprob` applied to ``W = X1 - r X2`` at ``0``. Returns a dict with ``F``,
    ``shat``, ``status`` and ``info``."""
    if order < 1:
        raise ValueError("ratio_tailprob: order must be a positive integer")
    KW = _ratio_cgf(K, r)
    t = tailprob(KW, 0.0, order=order, lower=lower)
    return dict(F=t["F"], shat=t["tau"], status=t["status"], info=t["info"])
