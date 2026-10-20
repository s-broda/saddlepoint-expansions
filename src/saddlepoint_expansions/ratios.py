"""The joint density of several ratios (Theorem 4 and Corollary 1 of the paper) from a CGF: all
the Taylor coefficients, of K_W and of the amplitude functions, come from one Taylor polynomial
of K(s, -r*s + tau) in (u, tau), in the whitened coordinates s = shat + L^-T u (port of the
MATLAB ratio_mv of R. Kan)."""
from __future__ import annotations

import math

import numpy as np

from .cgf import _taylor, kfun, linear_combination
from .multivariate import mi_list
from .taylor import from_coefficients, get_algebra
from .util import spa_saddle


def ratio_cgf2(K, d, name):
    """The cgf of (X_1..X_d, Y_1..Y_d) from one of dimension 2d or, for a common denominator,
    d + 1."""
    if K.d == 2 * d:
        return K
    if K.d == d + 1:
        B = np.zeros((d + 1, 2 * d))
        for j in range(d):
            B[j, j] = 1.0
            B[d, d + j] = 1.0
        return linear_combination(K, B)
    raise ValueError(f"{name}: K must have dimension 2d (numerators, denominators) or d + 1 (common denominator), d = {d}")


def _setparts(d):
    """All partitions of 0..d-1, each a list of block bit masks."""
    parts = [[1]]
    for i in range(1, d):
        new = []
        b = 1 << i
        for bl in parts:
            for j in range(len(bl)):
                q = list(bl)
                q[j] += b
                new.append(q)
            new.append(bl + [b])
        parts = new
    return parts


def _dfact(n):
    return 1 if n <= 0 else math.prod(range(n, 0, -2))


def _he0(nu):
    v = 1.0
    for n in nu:
        if n % 2:
            return 0.0
        v *= (-1) ** (n // 2) * _dfact(n - 1)
    return v


def ratio_mv(K2, r, order):
    """Saddlepoint expansion of orders ``1..order`` of the joint density at ``r`` of
    ``R_j = X_j/Y_j`` from the cgf ``K2`` of ``(X_1..X_d, Y_1..Y_d)``, ``Y_j > 0``: with
    ``W = X - r*Y``, ``K_W(s) = K(s, -r*s)``, ``K_W'(shat) = 0``, ``K_W'' = L L'``,
    ``f(r) = (2 pi i)^-d int exp(K_W(s)) g(s) ds`` with ``g`` the sum over the partitions of the
    index set of the products of the mixed derivatives ``K_B(s, -r*s)``, expanded in whitened
    coordinates (Corollary 1 of the paper). Returns ``(f, info)``."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    d = len(r)
    M = order
    B = np.vstack([np.eye(d), -np.diag(r)])
    KW = linear_combination(K2, B)
    s, sinfo = spa_saddle(kfun(KW, 2), np.zeros(d), None)
    P2 = _taylor(KW, s, 2)
    if P2 is None:
        raise RuntimeError("ratio_density: the saddlepoint is outside the domain of K")
    H = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            e = [0] * d
            e[i] += 1
            e[j] += 1
            H[i, j] = P2.derivative(tuple(e))
    try:
        L = np.linalg.cholesky((H + H.T) / 2)
    except np.linalg.LinAlgError:
        raise RuntimeError("ratio_density: K_W'' is not positive definite at the saddlepoint")
    Lit = np.linalg.inv(L).T
    N = max(2 * M, 2 * M - 2 + d)
    Mmap = np.block([[Lit, np.zeros((d, d))], [-np.diag(r) @ Lit, np.eye(d)]])
    P = _taylor(K2, np.concatenate([s, -r * s]), N, Mmap)
    if P is None:
        raise RuntimeError("ratio_density: the saddlepoint is outside the domain of K")
    D = max(6 * (M - 1), 2 * M)
    zd = (0,) * d
    # Theta: the terms of degree 3..2M of K_W(shat + L^-T u)
    theta = {u: P.coeff(u + zd) for u in mi_list(d, min(2 * M, N)) if sum(u) >= 3}
    Theta = from_coefficients(d, D, theta)
    K0 = P.coeff((0,) * (2 * d))
    # K_B(shat + L^-T u) for every nonempty subset B (bit mask), degree <= 2(M-1) in u
    G = [None] * (2 ** d)
    for mask in range(1, 2 ** d):
        tv = tuple((mask >> j) & 1 for j in range(d))
        cf = {u: P.coeff(u + tv) for u in mi_list(d, min(2 * (M - 1), N - sum(tv)))}
        G[mask] = from_coefficients(d, D, cf)
    # g_k: the partitions into d - k blocks, truncated at degree 2(M-1)
    alg = get_algebra(d, D)
    g = [from_coefficients(d, D, {}) for _ in range(d)]
    for bl in _setparts(d):
        q = G[bl[0]]
        for j in range(1, len(bl)):
            q = q * G[bl[j]]
        k = d - len(bl)
        g[k] = g[k] + q
    low = alg.deg <= 2 * (M - 1)
    for k in range(d):
        c = g[k].c.copy()
        c[~low] = 0.0
        g[k] = from_coefficients(d, D, {alg.exps[i]: c[i] for i in np.nonzero(c)[0]})
    he = np.array([_he0(nu) for nu in alg.exps])
    R = 2 * (M - 1)
    Pr = [from_coefficients(d, D, {zd: 1.0})]
    for q in range(1, R + 1):
        Pr.append(Pr[q - 1] * Theta / q)
    a = np.zeros(M)
    for k in range(0, min(M - 1, d - 1) + 1):
        h = g[k]
        for q in range(0, R + 1):
            Q = Pr[q] * h
            for m in range(-(-q // 2), M - 1 - k + 1):
                sel = alg.deg == 2 * m + 2 * q
                a[m + k] += float(np.sum(Q.c[sel] * he[sel]))
    logC = K0 - d / 2 * math.log(2 * math.pi) - float(np.sum(np.log(np.diag(L))))
    f = math.exp(logC) * np.cumsum(a)
    return f, dict(shat=s, a=a, logC=logC, saddle=sinfo)
