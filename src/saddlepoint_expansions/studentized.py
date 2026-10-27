"""Studentized means: the density and distribution function of ``Q = X/sqrt(Y)``, ``Y > 0``, from
the joint cumulant generating function ``K(s, t)`` of ``(X, Y)``: the expansions of Bao, Broda and
Kan, "Inversion formulas and saddlepoint expansions for Studentized means" (port of the Julia
package). The density is the bivariate amplitude expansion of Theorem 2 about the saddlepoint,
evaluated in the coordinates of a complex factorization of the Hessian (indefinite where
``that < 0``); the distribution function is the Hauschildt-Robinson type expansion of Theorem 3
after the reduction to one dimension, with every Taylor coefficient obtained from the Taylor
polynomial of ``K`` by Newton's method and composition on power series. The saddlepoint equations
are ``K_s = q v``, ``K_t = v^2``, ``q s + 2 t v = 0``; the saddlepoints for all ``q`` lie on the
locus ``s K_s + 2 t K_t = 0``, on which ``v = sqrt(K_t)`` and ``q = K_s/v``, which is what the
quantile search uses.
"""
from __future__ import annotations

import math

import numpy as np

from .cgf import _taylor
from .multivariate import mi_list
from .taylor import DomainError, TPoly, from_coefficients, get_algebra
from .univariate import _Iherm
from .util import hcoef, normcdf, normpdf


def _check(K, name):
    if K.d != 2:
        raise ValueError(f"{name}: K must be the joint cumulant generating function of (X, Y), of dimension 2")


def _jet(K, s, t, N):
    P = _taylor(K, np.array([s, t], dtype=float), N)
    if P is None:
        raise RuntimeError(f"studentized: the point (s, t) = ({s}, {t}) is outside the domain of K")
    return P


# ----- the saddlepoint

def _newton3(K, q, s, t, v, tol=1e-14, maxit=60):
    """Newton's method on the saddlepoint equations from (s, t, v), with the Jacobian from the
    second-order Taylor polynomial of K."""
    for _ in range(maxit):
        P = _taylor(K, np.array([s, t]), 2)
        if P is None:
            raise RuntimeError("outside the domain")
        Ks, Kt = P.derivative((1, 0)), P.derivative((0, 1))
        Kss, Kst, Ktt = P.derivative((2, 0)), P.derivative((1, 1)), P.derivative((0, 2))
        F = np.array([Ks - q * v, Kt - v * v, q * s + 2 * t * v])
        A = np.array([[Kss, Kst, -q], [Kst, Ktt, -2 * v], [q, 2 * v, 2 * t]])
        d = np.linalg.solve(A, F)
        s, t, v = s - d[0], t - d[1], v - d[2]
        if np.sum(np.abs(d)) < tol * (1 + abs(s) + abs(t) + abs(v)):
            return s, t, v
    raise RuntimeError(f"the saddlepoint iteration did not converge at q = {q}")


def _continue(K, qa, qb, s, t, v, depth=0):
    try:
        return _newton3(K, qb, s, t, v)
    except (RuntimeError, np.linalg.LinAlgError, DomainError):
        if depth >= 12:
            raise
        qm = (qa + qb) / 2
        s, t, v = _continue(K, qa, qm, s, t, v, depth + 1)
        return _continue(K, qm, qb, s, t, v, depth + 1)


def studentized_saddlepoint(K, q):
    """The saddlepoint ``(shat, that, vhat)`` of the expansions for ``Q = X/sqrt(Y)`` at ``q``:
    the root of ``K_s = q v``, ``K_t = v^2``, ``q s + 2 t v = 0`` with ``vhat > 0``, for the
    joint cumulant generating function ``K`` of ``(X, Y)``, ``Y > 0``, by Newton's method with
    continuation in ``q`` from ``q0 = E(X)/sqrt(E(Y))``, where ``shat = that = 0``. Returns a
    dict with ``s``, ``t``, ``v`` and ``q0``."""
    _check(K, "studentized_saddlepoint")
    q = float(q)
    P0 = _jet(K, 0.0, 0.0, 2)
    EU, EV = P0.derivative((1, 0)), P0.derivative((0, 1))
    if not EV > 0:
        raise RuntimeError("studentized_saddlepoint: E(Y) must be positive")
    q0 = EU / math.sqrt(EV)
    s, t, v = 0.0, 0.0, math.sqrt(EV)
    steps = max(4, int(abs(q - q0) / 0.25) + 1)
    for k in range(1, steps + 1):
        s, t, v = _continue(K, q0 + (q - q0) * (k - 1) / steps, q0 + (q - q0) * k / steps, s, t, v)
    return dict(s=s, t=t, v=v, q0=q0)


def studentized_locus(K, s, tguess, tol=1e-14, maxit=60):
    """The point of the saddlepoint locus ``s K_s + 2 t K_t = 0`` with abscissa ``s``, by
    Newton's method in ``t`` from ``tguess``; returns a dict with ``t``, ``v = sqrt(K_t)`` and
    the value ``q = K_s/v`` whose saddlepoint it is. The saddlepoints for all ``q`` lie on this
    curve through the origin, so that a value of ``shat`` determines the saddlepoint and ``q``
    without solving the three equations; :func:`studentized_quantile` searches along it."""
    _check(K, "studentized_locus")
    s = float(s)
    t = float(tguess)
    for _ in range(maxit):
        P = _jet(K, s, t, 2)
        Ks, Kt = P.derivative((1, 0)), P.derivative((0, 1))
        Kst, Ktt = P.derivative((1, 1)), P.derivative((0, 2))
        g = s * Ks + 2 * t * Kt
        gp = s * Kst + 2 * Kt + 2 * t * Ktt
        t -= g / gp
        if abs(g / gp) < tol * (1 + abs(t)):
            P = _jet(K, s, t, 2)
            Ks, Kt = P.derivative((1, 0)), P.derivative((0, 1))
            if not Kt > 0:
                raise RuntimeError("studentized_locus: K_t is not positive at the point found")
            v = math.sqrt(Kt)
            return dict(t=t, v=v, q=Ks / v)
    raise RuntimeError(f"studentized_locus: the iteration did not converge at s = {s}")


# ----- the density: Theorem 2, in whitened coordinates with a complex factorization of psi''

def _complex_factor(H):
    """A complex L with H = L L' from the eigen-decomposition of the symmetric 2 x 2 matrix H
    (indefinite where that < 0; its (1, 1) entry vanishes in symmetric cases). Returns L^-T."""
    a, b, c = H[0, 0], H[1, 0], H[1, 1]
    th = math.atan2(2 * b, a - c) / 2
    cs, sn = math.cos(th), math.sin(th)
    V = np.array([[cs, -sn], [sn, cs]])
    l1 = a * cs * cs + 2 * b * sn * cs + c * sn * sn
    l2 = a * sn * sn - 2 * b * sn * cs + c * cs * cs
    if l1 == 0 or l2 == 0:
        raise RuntimeError("studentized_density: psi'' is singular at the saddlepoint")
    r1, r2 = np.sqrt(complex(l1)), np.sqrt(complex(l2))
    return V @ np.diag([1 / r1, 1 / r2])                 # inv(L)' = V diag(1/sqrt(lambda))


def _subs(P, w):
    """P evaluated at the vector w of polynomials of the same algebra (composition)."""
    alg = P.alg
    d = alg.d
    pw = []
    for i in range(d):
        pows = [alg.constant(1.0)]
        for k in range(1, alg.N + 1):
            pows.append(pows[-1] * w[i])
        pw.append(pows)
    out = alg.constant(0.0) + 0j
    for idx, nu in enumerate(alg.exps):
        cf = P.c[idx]
        if cf == 0:
            continue
        term = alg.constant(1.0) + 0j
        for i in range(d):
            if nu[i]:
                term = term * pw[i][nu[i]]
        out = out + cf * term
    return out


def _dfact(n):
    return 1 if n <= 0 else math.prod(range(n, 0, -2))


def _he0(nu):
    v = 1.0
    for n in nu:
        if n % 2:
            return 0.0
        v *= (-1) ** (n // 2) * _dfact(n - 1)
    return v


def _truncate(P, deg):
    c = P.c.copy()
    c[P.alg.deg > deg] = 0.0
    return TPoly(P.alg, c)


def studentized_density(K, q, order=3, saddle=None):
    """Saddlepoint expansions of orders ``1..order`` of the density at ``q`` of
    ``Q = X/sqrt(Y)``, ``Y > 0``, from the joint cumulant generating function ``K`` of
    ``(X, Y)`` (Theorem 2 of Bao, Broda and Kan): ``f[M-1]`` keeps the terms ``m <= M-1`` of
    the expansion in powers of ``n^-1`` and has relative error ``O(n^-M)`` (the "n" is absorbed
    in ``K``, see :func:`~saddlepoint_expansions.sample_mean`). With
    ``psi(s, t) = K(s, t) + q^2 s^2/(4t)`` and the saddlepoint of
    :func:`studentized_saddlepoint`, ``f(q) ~ vhat^2 exp(psi) / (sqrt(pi) {that |psi''|}^(1/2))
    {1 + sum_m b_m(h_0) + b_{m-1}(h_1)}``, the amplitude expansion of a bivariate integral
    with amplitude ``(s/shat)^2 (t/that)^(-5/2) + (t/that)^(-3/2)/(2 that vhat^2)``; the Hessian
    is indefinite where ``that < 0`` and the Hermite moments are evaluated in the coordinates
    of a complex factorization ``psi'' = L L'``. The terms of high order cancel by many orders of
    magnitude when ``|that|`` is large: ``info["kappa"]`` is the cancellation indicator, and in
    double precision the terms beyond the fourth are often not reliable (the Julia package
    repeats the computation in higher precision; ``status`` reports ``"coefficients
    inaccurate"`` when ``kappa`` exceeds ``1e10``). ``saddle`` may supply the saddlepoint (a dict
    with ``s``, ``t``, ``v``). Returns a dict with ``f``, ``saddle``, ``status`` and ``info``."""
    _check(K, "studentized_density")
    if order < 1:
        raise ValueError("studentized_density: order must be a positive integer")
    q = float(q)
    M = order
    sd = studentized_saddlepoint(K, q) if saddle is None else saddle
    s, t, v = float(sd["s"]), float(sd["t"]), float(sd["v"])
    if t == 0:
        raise RuntimeError("studentized_density: the saddlepoint has t = 0 (q is the point q0 = E(X)/E(Y)^(1/2)); the expansion is not evaluated there")
    D = max(6 * (M - 1), 2)
    P = _jet(K, s, t, D)
    alg = get_algebra(2, D)
    v1, v2 = alg.variables()
    psi = P + (s + v1) ** 2 * (q * q / 4) / (t + v2)
    psi0 = psi.constant_term()
    H = np.array([[psi.derivative((2, 0)), psi.derivative((1, 1))], [psi.derivative((1, 1)), psi.derivative((0, 2))]])
    detH = H[0, 0] * H[1, 1] - H[0, 1] ** 2
    if not t * detH > 0:
        raise RuntimeError("studentized_density: that*det(psi'') is not positive at the saddlepoint")
    pref = v ** 2 * math.exp(psi0) / math.sqrt(math.pi * t * detH)
    Lit = _complex_factor(H)
    w = [Lit[0, 0] * v1 + Lit[0, 1] * v2, Lit[1, 0] * v1 + Lit[1, 1] * v2]     # v = L^-T u
    psiu = _subs(psi, w)
    theta = {nu: psiu.c[alg.index[nu]] for nu in mi_list(2, min(2 * M, D)) if sum(nu) >= 3}
    Theta = from_coefficients(2, D, theta) + 0j
    a0 = _truncate(((s + w[0]) / s) ** 2 * ((t + w[1]) / t) ** (-2.5), 2 * (M - 1))
    a1 = _truncate(((t + w[1]) / t) ** (-1.5) / (2 * t * v * v), 2 * (M - 1))
    he = np.array([_he0(nu) for nu in alg.exps])
    R = 2 * (M - 1)
    Pr = [from_coefficients(2, D, {(0, 0): 1.0}) + 0j]
    for r in range(1, R + 1):
        Pr.append(Pr[r - 1] * Theta / r)

    def bcoef(m, a):
        tot = 0j
        abssum = 0.0
        for r in range(0, 2 * m + 1):
            Qp = Pr[r] * a
            sel = alg.deg == 2 * m + 2 * r
            terms = Qp.c[sel] * he[sel]
            tot += np.sum(terms)
            abssum += float(np.sum(np.abs(terms)))
        return tot, abssum

    f = np.zeros(M)
    kappa = np.zeros(M)
    br, ab = 1.0, 1.0
    f[0] = pref
    kappa[0] = 1.0
    for m in range(1, M):
        b0, s0 = bcoef(m, a0)
        b1, s1 = bcoef(m - 1, a1)
        br += (b0 + b1).real
        ab += s0 + s1
        f[m] = pref * br
        kappa[m] = ab / abs(br)
    status = "coefficients inaccurate" if kappa[-1] > 1e10 else "ok"
    return dict(f=f, saddle=dict(s=s, t=t, v=v), status=status, info=dict(pref=pref, psi0=psi0, H=H, kappa=kappa))


# ----- the distribution function: Theorem 3 after the reduction to one dimension

def _uni(P, j, alg1):
    """The univariate polynomial in ds of the coefficient of dt^j of the bivariate P."""
    c = np.zeros(alg1.M, dtype=P.c.dtype)
    N = P.alg.N
    for i in range(N + 1 - j):
        c[i] = P.c[P.alg.index[(i, j)]]
    return TPoly(alg1, c)


def _dt(P, k):
    """The bivariate polynomial of the k-th t-derivative."""
    alg = P.alg
    N = alg.N
    c = np.zeros(alg.M, dtype=P.c.dtype)
    for i in range(N + 1):
        for j in range(N + 1 - i - k):
            c[alg.index[(i, j)]] = P.c[alg.index[(i, j + k)]] * (math.factorial(j + k) // math.factorial(j))
    return TPoly(alg, c)


def _compose(P, tau, alg1):
    """P(shat + ds, that + tau(ds)) as a univariate polynomial in ds (tau nilpotent)."""
    N = P.alg.N
    r = _uni(P, N, alg1)
    for j in range(N - 1, -1, -1):
        r = r * tau + _uni(P, j, alg1)
    return r


def _gbinomial(a, k):
    out = 1.0
    for i in range(k):
        out *= (a - i) / (i + 1)
    return out


def _reduction(K, q, s0, t0, p):
    """Taylor coefficients Psi_c[k], k = 0..p+2, of Psi(s) = psi(s, t(s)) and Hk_c[k][l],
    l = 0..p-2k, of H_k(s) = H_0(s) beta_k(s) at shat."""
    jmax = p // 2
    N = max(p + 2, 2 * jmax + 2 + p + 2)
    J = _jet(K, s0, t0, N)
    alg = get_algebra(2, N)
    alg1 = get_algebra(1, N)
    v1, v2 = alg.variables()
    psi = J + (s0 + v1) ** 2 * (q * q / 4) / (t0 + v2)
    psi_t = _dt(psi, 1)
    psi_tt = _dt(psi, 2)
    tau = alg1.constant(0.0)                               # t(s) - that, nilpotent in ds
    for _ in range(N + 1):
        g = _compose(psi_t, tau, alg1)
        gt = _compose(psi_tt, tau, alg1)
        tau = tau - g / gt
        tau.c[0] = 0.0
    Psi = _compose(psi, tau, alg1)
    ds1 = alg1.variables()[0]
    su = s0 + ds1
    tu = t0 + tau
    v = -su * q / (2 * tu)
    Ktt = _compose(_dt(J, 2), tau, alg1)
    arg = 2 * tu * Ktt + 4 * v * v
    if not arg.constant_term() > 0:
        raise RuntimeError("studentized_tailprob: 2 t K_tt + 4 v^2 is not positive at the saddlepoint")
    H0 = 2 * v / arg.sqrt()
    p2 = _compose(psi_tt, tau, alg1)
    sq = (p2 + 0j).sqrt()                                  # complex where psi_tt < 0
    theta = {k: (_compose(_dt(psi, k), tau, alg1) + 0j) / (sq ** k * math.factorial(k)) for k in range(3, 2 * jmax + 3)}

    def gam(l):
        return ((1.0 / tu) ** l + 0j) / sq ** l * _gbinomial(-1.5, l)

    P2 = 2 * jmax
    zero1 = alg1.constant(0.0) + 0j
    d = [[zero1 for _ in range(P2 + 1)] for _ in range(P2 + 1)]
    d[0][0] = alg1.constant(1.0) + 0j
    for r in range(1, P2 + 1):
        for j in range(r, P2 + 1):
            acc = zero1
            for k in range(1, j - r + 2):
                acc = acc + theta[k + 2] * d[r - 1][j - k]
            d[r][j] = acc / r
    hh = hcoef(2 * P2 + 2)

    def a_ij(i, j):
        acc = zero1
        for k in range(0, j + 1):
            acc = acc + d[k][j] * hh[i + k]
        return acc

    beta = []
    for j in range(1, jmax + 1):
        acc = zero1
        for k in range(0, 2 * j + 1):
            acc = acc + gam(2 * j - k) * a_ij(j, k)
        beta.append(acc)
    Psi_c = np.array([Psi.c[k] for k in range(p + 3)])
    Hk_c = [np.array([H0.c[l] for l in range(p + 1)])]
    for k in range(1, jmax + 1):
        Hk = (H0 + 0j) * beta[k - 1]
        for l in range(p - 2 * k + 1):
            if abs(Hk.c[l].imag) > 1e-8 * (1 + abs(Hk.c[l])):
                raise RuntimeError(f"studentized_tailprob: a coefficient of H_{k} is not real ({Hk.c[l]}); the expansion does not apply at this q")
        Hk_c.append(np.array([Hk.c[l].real for l in range(p - 2 * k + 1)]))
    return Psi_c, Hk_c


def _Itable(u, mmax):
    """I_m(u), m = 0..mmax, u >= 0: I_0 = Phi(-u), I_{m+1} = He_m(0) phi(u) - u I_m (forward
    recursion), replaced for u > 2.5 by the Gauss-Hermite representation of spa_uni."""
    phi = float(normpdf(u))
    if u > 2.5:
        return phi * _Iherm(u, mmax)
    h = hcoef(mmax // 2 + 1)
    I = np.zeros(mmax + 1)
    I[0] = float(normcdf(-u))
    for m in range(mmax):
        I[m + 1] = (h[m // 2] if m % 2 == 0 else 0.0) * phi - u * I[m]
    return I


def _hrsum(Psi_c, Hk_c, s0, p):
    """The Hauschildt-Robinson type sum (Theorem 3): approximations to P(Q > q) with the terms up
    to n^(-j/2), j = 0..p, with the reflection for shat < 0."""
    sg = 1 if s0 >= 0 else -1
    Psi2 = 2 * Psi_c[2]
    if not Psi2 > 0:
        raise RuntimeError("studentized_tailprob: Psi''(shat) is not positive")
    theta = {k: sg ** k * Psi_c[k] / Psi2 ** (k / 2) for k in range(3, p + 3)}
    gam = [[sg ** l * Hk_c[k][l] / Psi2 ** (l / 2) for l in range(len(Hk_c[k]))] for k in range(len(Hk_c))]
    d = {(0, 0): 1.0}
    for r in range(1, p + 1):
        for j in range(r, p + 1):
            d[(r, j)] = sum(theta[k + 2] * d.get((r - 1, j - k), 0.0) for k in range(1, j - r + 2)) / r
    w = math.sqrt(-2 * Psi_c[0])
    u = abs(s0) * math.sqrt(Psi2)
    En = math.exp((u ** 2 - w ** 2) / 2)
    I = _Itable(u, 3 * p + 1)
    out = np.zeros(p + 1)
    kappa = np.zeros(p + 1)
    tot = 0.0
    abssum = 0.0
    for j in range(p + 1):
        for k in range(j // 2 + 1):
            for m in range(j - 2 * k + 1):
                l = j - 2 * k - m
                for r in range(m + 1):
                    drm = 1.0 if (r == 0 and m == 0) else (0.0 if r == 0 else d.get((r, m), 0.0))
                    if drm == 0:
                        continue
                    term = drm * gam[k][l] * I[m + l + 2 * r]
                    tot += term
                    abssum += abs(term)
        val = En * tot
        out[j] = val if sg == 1 else 1 - val
        kappa[j] = abssum / abs(tot)
    return out, kappa


def studentized_tailprob(K, q, order=3, lower=False, saddle=None):
    """Saddlepoint expansions of orders ``1..order`` of ``P(Q >= q)`` for ``Q = X/sqrt(Y)``,
    ``Y > 0``, from the joint cumulant generating function ``K`` of ``(X, Y)`` (Theorem 3 of Bao,
    Broda and Kan): ``F[M-1]`` keeps the terms of orders ``n^(-m/2)``, ``m <= 2M-1``, and has
    relative error ``O(n^-M)``; ``info["Fp"][j]``, ``j = 0..2M-1``, are the expansions with the
    terms up to ``n^(-j/2)``. ``lower=True`` gives ``P(Q <= q)``. The expansion is of
    Hauschildt-Robinson type in ``s`` after the ``t``-integral is expanded by Laplace's method:
    with ``Psi(s) = psi(s, t(s))``, ``psi_t(s, t(s)) = 0``, the leading term is
    ``E_n {H_0(shat) Phibar(uhat) + ...}``, and all Taylor coefficients of ``Psi`` and of the
    amplitudes ``H_k`` at ``shat`` are obtained from the Taylor polynomial of ``K`` by Newton's
    method and composition on power series, exact to rounding. ``saddle`` may supply the
    saddlepoint; ``info["kappa"]`` is the cancellation indicator. Returns a dict with ``F``,
    ``saddle``, ``status`` and ``info``."""
    _check(K, "studentized_tailprob")
    if order < 1:
        raise ValueError("studentized_tailprob: order must be a positive integer")
    q = float(q)
    M = order
    sd = studentized_saddlepoint(K, q) if saddle is None else saddle
    s, t, v = float(sd["s"]), float(sd["t"]), float(sd["v"])
    if t == 0:
        raise RuntimeError("studentized_tailprob: the saddlepoint has t = 0 (q is the point q0 = E(X)/E(Y)^(1/2)); the expansion is not evaluated there")
    p = 2 * M - 1
    Psi_c, Hk_c = _reduction(K, q, s, t, p)
    Fp, kappa = _hrsum(Psi_c, Hk_c, s, p)
    if lower:
        Fp = 1 - Fp
    F = np.array([Fp[2 * m - 1] for m in range(1, M + 1)])
    status = "coefficients inaccurate" if kappa[-1] > 1e8 else "ok"
    return dict(F=F, saddle=dict(s=s, t=t, v=v), status=status, info=dict(Fp=Fp, kappa=kappa, Psi_c=Psi_c, Hk_c=Hk_c))


def studentized_quantile(K, prob, order=3, tol=1e-10):
    """The quantile of the saddlepoint distribution function of order ``order``
    (:func:`studentized_tailprob` with ``lower=True``): ``P(Q <= q) = prob``. The search runs
    over ``shat`` along the saddlepoint locus (:func:`studentized_locus`), where each trial point
    gives its ``q`` and its distribution function directly, so that no saddlepoint equation is
    solved: a bracket in ``shat`` from the origin, then the Illinois iteration. Returns a dict
    with ``q`` and ``saddle``."""
    _check(K, "studentized_quantile")
    if order < 1:
        raise ValueError("studentized_quantile: order must be a positive integer")
    prob = float(prob)
    if not 0 < prob < 1:
        raise ValueError("studentized_quantile: prob must be in (0, 1)")
    P0 = _jet(K, 0.0, 0.0, 2)
    EU, EV = P0.derivative((1, 0)), P0.derivative((0, 1))
    slope = -EU / (2 * EV)                                   # t'(0) along the locus

    def at(s, tg):
        r = studentized_locus(K, s, tg)
        F = studentized_tailprob(K, r["q"], order=order, lower=True, saddle=dict(s=s, t=r["t"], v=r["v"]))["F"][-1]
        return F, r["q"], r["t"]

    step = 0.5
    s = 0.1
    t = slope * s
    F, q, t = at(s, t)
    direction = 1 if F < prob else -1
    if direction == -1:
        s = -s
        t = slope * s
        F, q, t = at(s, t)
    sa, Fa, ta = s, F, t
    while (direction == 1 and F < prob) or (direction == -1 and F > prob):
        sa, Fa, ta = s, F, t
        s += direction * step
        F, q, t = at(s, t)
        if abs(s) > 1e4:
            raise RuntimeError("studentized_quantile: no bracket found")
    sb, Fb, tb = s, F, t
    fa, fb = Fa - prob, Fb - prob
    side = 0
    for _ in range(200):
        sm = (sa * fb - sb * fa) / (fb - fa)
        tg = ta + (tb - ta) * (sm - sa) / (sb - sa)
        Fm, qm, tm = at(sm, tg)
        fm = Fm - prob
        if abs(fm) < tol:
            r = studentized_locus(K, sm, tm)
            return dict(q=qm, saddle=dict(s=sm, t=r["t"], v=r["v"]))
        if np.sign(fm) == np.sign(fb):
            sb, fb, tb = sm, fm, tm
            if side == 1:
                fa /= 2
            side = 1
        else:
            sa, fa, ta = sm, fm, tm
            if side == -1:
                fb /= 2
            side = -1
    raise RuntimeError("studentized_quantile: the iteration did not converge")
