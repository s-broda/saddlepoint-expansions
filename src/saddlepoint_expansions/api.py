"""The high-level interface: saddlepoint, tail probabilities, densities and ratios from a CGF."""
from __future__ import annotations

import numpy as np

from .cgf import CGF, _fact, _taylor, kfun, kwhite, linear_combination, reflect
from .ratios import ratio_cgf2, ratio_mv
from .multivariate import spa_mv, spa_orthant, spa_orthant2
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


def tailprob(K, x, order=3, lower=False, signs=None, lattice=False, sinh="saddle", **kwargs):
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
    For ``d = 1``, ``x`` may be a scalar.

    ``lattice=True``: ``X = S`` is integer-valued and so is ``x = k``; ``F`` is ``P(S >= k)`` with
    the continuity correction and the amplitude ``prod_j s_j/(2 sinh(s_j/2))``, by Theorem 3 of
    Broda and Kan, "Conditional saddlepoint expansions of arbitrary order", with no conditioning
    components (the computation of :func:`~saddlepoint_expansions.conditional_tailprob` with all
    components in ``A``, with the same numbers): the saddlepoint at ``k - 1/2``, ``sinh='saddle'``
    expanding the amplitude about the saddlepoint and ``'origin'`` about the origin (Kolassa 2003).
    In one dimension ``F[0]`` has the relative error ``O(1/n)`` of the continuity-corrected
    approximations of Daniels (1987). Negative saddlepoint components are handled by the
    inclusion-exclusion with ``P(S_j <= k_j - 1) = P(-S_j >= -k_j + 1)``; where the corrected point
    is outside the convex hull of the support (``P(S >= 0)`` of a count) there is no saddlepoint
    and a DomainError is raised. ``info`` then holds ``Fp``, ``kappa`` and ``err``."""
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
    if lattice:
        from .conditional import _spa_cond
        nG = kwargs.get("nG", 64)
        r = _spa_cond(Kf, kwhite(Kr, maxdeg), xr, list(range(d)), order, density=False, lattice=True,
                      sinh=sinh, nG=nG)
        tau = sg * r["tau"]
        return dict(F=r["P"], tau=(tau[0] if scalar else tau), status=r["status"],
                    info=dict(Fp=r["Pp"], kappa=r["kappaP"], err=r["err"], nG=nG))
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


def density(K, x, order=3, lattice=False, normalize=False, **kwargs):
    """Saddlepoint expansions of orders ``1..order`` of the density at ``x`` (Theorem 3 of the
    paper; Daniels's expansion when ``d = 1``): ``f[M-1]`` keeps the terms ``m <= M-1`` and
    has relative error ``O(n^-M)``. A finite sum of Hermite polynomials, no integration.
    Returns a dict with ``f``, ``tau``, ``status`` and ``info``. In three or more dimensions
    the cancellation indicator ``info['kappa_pdf']`` should be checked at high orders (the
    Julia package repeats the computation in higher precision when it is large; this package
    works in double precision).

    ``lattice=True`` (integer-valued ``S``, integer ``x = k``): ``f`` is ``P(S = k)``, the same
    expansion at ``k`` (Daniels 1987 at order one; Theorem 3 of the conditional paper with no
    conditioning components), computed as :func:`~saddlepoint_expansions.conditional_density`
    with all components in ``A``, with the same numbers.

    ``normalize=True`` divides the expansion of every order by its total mass, returned in
    ``info['mass']`` (pass it back as ``normalize`` to reuse it): the integral over ``x`` is computed
    in saddlepoint space, ``x = K'(s)``, ``dx = det K''(s) ds``, so that the saddlepoint is known at
    every node and no saddlepoint equation is solved, by ``scipy.integrate.quad_vec`` over the
    domain of ``K`` (nested over the coordinates in more dimensions, which is slow), the tails
    truncated where the leading term has fallen by ``e^-50``. For the normal the masses are one;
    for the gamma with shape ``a``, ``c_1 = Gamma(a) e^a a^(1/2 - a)/sqrt(2 pi)``, and every
    normalized order is exact."""
    if normalize is not False and normalize is not None:
        if lattice:
            raise ValueError("density: normalize is for continuous variables")
        from .quantiles import _density_mass
        if normalize is True:
            mass = _density_mass(K, order)
        else:
            mass = np.asarray(normalize, dtype=float)
            if len(mass) < order:
                raise ValueError(f"density: normalize must be True or a vector of {order} masses")
        r = density(K, x, order=order, **kwargs)
        info = dict(r["info"])
        info["mass"] = mass[:order]
        return dict(f=np.asarray(r["f"]) / mass[:order], tau=r["tau"], status=r["status"], info=info)
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
    if lattice:
        from .conditional import _spa_cond
        r = _spa_cond(Kf, kwhite(K, maxdeg), xx, list(range(d)), order, tail=False)
        return dict(f=r["f"], tau=(r["tauf"][0] if scalar else r["tauf"]), status=r["status"],
                    info=dict(kappa=r["kappaf"]))
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
    return linear_combination(K, np.array([[1.0], [-float(r)]]))


def ratio_density(K, r, order=3, general=False, normalize=False):
    """Saddlepoint expansions of orders ``1..order`` of the joint density at ``r`` of the
    vector of ratios ``R = (X_1/Y_1, ..., X_d/Y_d)``, ``d = len(r)``, with positive denominators.
    ``K`` is the cgf of ``(X_1..X_d, Y_1..Y_d)`` (dimension ``2d``) or, for a common denominator
    ``Y``, of ``(X_1..X_d, Y)`` (dimension ``d + 1``). For a scalar ``r`` (one ratio, ``K`` of
    dimension two) the expansion is the Rice expansion through the saddlepoint of
    ``K_W(s) = K(s, -r s)`` (Theorem 4 of the paper); for a vector it is Theorem 4 with the
    amplitude expansion of Corollary 1 in whitened coordinates, every Taylor coefficient taken
    from one Taylor polynomial of ``K(s, -r*s + tau)`` (:func:`~saddlepoint_expansions.ratios.ratio_mv`).
    ``general=True`` uses the vector code also for ``d = 1``. Returns a dict with ``f``,
    ``shat``, ``status`` and ``info``.

    For one ratio, ``normalize=True`` divides the expansion of every order by its total mass
    (``info['mass']``, reusable as ``normalize``), the integral over ``r`` computed over the
    saddlepoint ``shat`` along the curve ``s K_1 + t K_2 = 0``, ``t = -r shat``, on which
    ``r = K_1/K_2``, with ``dr = (dr/dshat) dshat`` (see
    :func:`~saddlepoint_expansions.ratio_quantile`)."""
    if normalize is not False and normalize is not None:
        from .quantiles import _ratio_mass
        Kr = ratio_cgf2(K, 1, "ratio_density") if np.ndim(r) == 1 else K
        if normalize is True:
            mass = _ratio_mass(Kr, order)
        else:
            mass = np.asarray(normalize, dtype=float)
        rr = ratio_density(K, r, order=order, general=general)
        info = dict(rr["info"])
        info["mass"] = mass[:order]
        return dict(f=np.asarray(rr["f"]) / mass[:order], shat=rr["shat"], status=rr["status"], info=info)
    if order < 1:
        raise ValueError("ratio_density: order must be a positive integer")
    if np.ndim(r) == 1 and (len(r) > 1 or general):
        rr = np.asarray(r, dtype=float)
        f, info = ratio_mv(ratio_cgf2(K, len(rr), "ratio_density"), rr, order)
        return dict(f=f, shat=info["shat"], status="ok", info=info)
    if np.ndim(r) == 1:
        K = ratio_cgf2(K, 1, "ratio_density")
        r = float(r[0])
    if general:
        f, info = ratio_mv(ratio_cgf2(K, 1, "ratio_density"), [r], order)
        return dict(f=f, shat=float(info["shat"][0]), status="ok", info=info)
    KW = _ratio_cgf(K, r)
    s, sinfo = spa_saddle(kfun(KW, 2), np.array([0.0]), None)
    sh = float(s[0])
    N = 2 * order + 1
    P = _taylor(K, np.array([sh, -r * sh]), N, np.array([[1.0, 0.0], [-float(r), 1.0]]))
    if P is None:
        raise RuntimeError("ratio_density: the saddlepoint is outside the domain of K")
    KWd = [P.coeff((k, 0)) * _fact((k,)) for k in range(2 * order + 1)]
    gd = [P.coeff((l, 1)) * _fact((l,)) for l in range(2 * order - 1)]
    f = spa_ratio_pdf(KWd, gd, sh, order)
    return dict(f=f, shat=sh, status="ok", info=dict(KWd=np.array(KWd), gd=np.array(gd), saddle=sinfo))


def ratio_tailprob(K, r, order=3, lower=False, signs=None, **kwargs):
    """Saddlepoint expansions of orders ``1..order`` of ``P(R >= r)`` for the vector of ratios
    ``R = (X_1/Y_1, ..., X_d/Y_d)`` with positive denominators (``K`` as in
    :func:`ratio_density`; a scalar ``r`` for one ratio): the probability equals ``P(W >= 0)``
    for ``W = X - r*Y``, and :func:`tailprob` is applied to ``W`` at ``0`` (Theorem 4 of the
    paper). ``lower=True`` gives ``P(R <= r)``, ``signs`` a mixed corner, and the other keyword
    arguments are passed to ``tailprob``. Returns a dict with ``F``, ``shat``, ``status`` and
    ``info``."""
    if order < 1:
        raise ValueError("ratio_tailprob: order must be a positive integer")
    if np.ndim(r) == 1 and len(r) > 1:
        rr = np.asarray(r, dtype=float)
        d = len(rr)
        K2 = ratio_cgf2(K, d, "ratio_tailprob")
        KW = linear_combination(K2, np.vstack([np.eye(d), -np.diag(rr)]))
        t = tailprob(KW, np.zeros(d), order=order, lower=lower, signs=signs, **kwargs)
        return dict(F=t["F"], shat=t["tau"], status=t["status"], info=t["info"])
    if np.ndim(r) == 1:
        K = ratio_cgf2(K, 1, "ratio_tailprob")
        r = float(r[0])
    KW = _ratio_cgf(K, r)
    t = tailprob(KW, 0.0, order=order, lower=lower, **kwargs)
    return dict(F=t["F"], shat=t["tau"], status=t["status"], info=t["info"])
