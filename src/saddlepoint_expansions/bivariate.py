"""Bivariate expansions: density (Theorem 3 of the paper) and orthant probability (Theorem 1),
in whitened coordinates, with the orthant integrals by Gauss quadrature (Appendix B)."""
from __future__ import annotations

import math

import numpy as np

from .util import SQRT2PI, addstatus, fact, gauher, gauleg, spa_saddle


def spa_bi(Kfun, x, order, tau0=None, nG=200, Kwhite=None, direct=False, logscale=0.0,
           qtol=1e-8, nGmax=None):
    """Bivariate saddlepoint approximations of orders ``1..order`` at ``x = [x1, x2]`` from
    ``Kfun(nu, t)``, the derivative of order ``nu`` (a tuple) of the cumulant generating
    function at ``t`` (NaN outside the domain). See :func:`spa_bi_at` for the result and the
    keywords; ``Kwhite(nu, u, T)`` optionally returns the derivatives of ``K_Y(u) = K(T u)``."""
    x = np.asarray(x, dtype=float)
    if x.shape != (2,) or not np.all(np.isfinite(x)):
        raise ValueError("spa_bi: x must be a finite vector of length 2")
    if order < 1:
        raise ValueError("spa_bi: order must be a positive integer")
    tau, sinfo = spa_saddle(Kfun, x, tau0)
    p = 2 * order - 1
    Kd = np.full((p + 3, p + 3), np.nan)
    for k in range(p + 3):
        for n1 in range(k + 1):
            Kd[n1, k - n1] = Kfun((n1, k - n1), tau)
    r = spa_bi_at(Kd, tau, x, order, nG=nG, Kwhite=Kwhite, direct=direct, logscale=logscale,
                  qtol=qtol, nGmax=nGmax)
    r["tau"] = tau
    r["saddle"] = sinfo
    return r


def spa_bi_at(Kd, tau, x, order, nG=200, Kwhite=None, direct=False, logscale=0.0, qtol=1e-8,
              nGmax=None):
    """The bivariate expansions from the derivatives at the saddlepoint ``tau``:
    ``Kd[n1, n2] = K^(n1,n2)(tau)``, ``n1 + n2 <= 2*order + 1``.

    Returns a dict with ``pdf[M]`` (density expansion with the terms ``m <= M-1``), ``Fup[M]``
    (``P(X1 >= x1, X2 >= x2)`` with the terms ``m <= 2M-1``; NaN when the saddlepoint has a
    negative component unless ``direct``), ``canc`` and ``kappa`` (cancellation indicators),
    ``qerr`` (relative change of ``Fup`` with half the quadrature nodes; the number of nodes is
    doubled while ``qerr[-1] > qtol``, up to ``nGmax``, default ``4 nG``), ``status`` and ``nG``.
    ``logscale = c`` scales ``Fup`` by ``exp(-c)``. The terms are evaluated for the whitened
    vector ``Y = L^-1 X``, ``K'' = L L'``; the Taylor coefficients of ``K_Y`` come from
    ``Kwhite(nu, u, T)`` (``T = L^-T``) when given and by polynomial substitution otherwise."""
    Kd = np.asarray(Kd, dtype=float)
    tau = np.asarray(tau, dtype=float)
    x = np.asarray(x, dtype=float)
    if nGmax is None:
        nGmax = 4 * nG
    K0 = Kd[0, 0]
    K2 = np.array([[Kd[2, 0], Kd[1, 1]], [Kd[1, 1], Kd[0, 2]]])
    p = 2 * order - 1
    Nn = max(3 * p, 6 * (order - 1))
    Lc = np.linalg.cholesky(K2)
    Tm = np.linalg.inv(Lc.T)
    uh = Lc.T @ tau
    theta = np.zeros((Nn + 1, Nn + 1))
    if Kwhite is not None:
        for k in range(3, p + 3):
            for n1 in range(k + 1):
                theta[n1, k - n1] = Kwhite((n1, k - n1), uh, Tm) / (fact(n1) * fact(k - n1))
    else:
        for k in range(3, p + 3):
            for n1 in range(k + 1):
                theta[n1, k - n1] = Kd[n1, k - n1] / (fact(n1) * fact(k - n1))
        theta = subs2(theta, Tm)
    R = max(p, 2 * (order - 1))
    a = [None] * (max(R, 1) + 1)
    a[1] = theta
    for r in range(2, R + 1):
        a[r] = polyprod(theta, a[r - 1], 3 * r, Nn) / r
    # density: Hermite polynomials at 0 in the whitened coordinates, He_mu(0, I) = He_mu1(0) He_mu2(0)
    h0 = np.zeros(Nn + 1)
    h0[0] = 1.0
    for m in range(2, Nn + 1, 2):
        h0[m] = -(m - 1) * h0[m - 2]
    He = np.outer(h0, h0)
    logdet2 = np.sum(np.log(np.diag(Lc)))
    f0 = math.exp(K0 - tau @ x - math.log(2 * math.pi) - logdet2)
    pdf = np.zeros(order)
    s = 1.0
    pdf[0] = f0
    for M in range(2, order + 1):
        m = M - 1
        for r in range(1, 2 * m + 1):
            k = 2 * m + 2 * r
            for n1 in range(k + 1):
                s += a[r][n1, k - n1] * He[n1, k - n1]
        pdf[M - 1] = f0 * s
    # orthant probability
    Fup = np.full(order, np.nan)
    canc = np.full(order, np.nan)
    kappa = np.full(order, np.nan)
    qerr = np.full(order, np.nan)
    nGu = 0
    status = "ok"

    def res():
        return dict(pdf=pdf, Fup=Fup, canc=canc, kappa=kappa, qerr=qerr, status=status, nG=nGu)

    if np.any(tau < 0) and not direct:
        status = addstatus(status, "negative saddlepoint component (use spa_orthant2)")
        return res()
    if np.all(tau >= 0) and K0 - tau @ x - logscale < math.log(np.finfo(float).tiny):
        Fup = np.zeros(order)
        qerr = np.zeros(order)
        canc = np.zeros(order)
        kappa = np.ones(order)
        status = addstatus(status, "probability below the double-precision range (use logscale)")
        return res()
    L = 3 * p

    def osums(In, E):
        F = np.zeros(order)
        cn = np.zeros(order)
        kp = np.zeros(order)
        os_ = In[0, 0]
        osa = abs(os_)
        om = 0
        obig = 0.0
        for oM in range(1, order + 1):
            while om < 2 * oM - 1:
                om += 1
                for orr in range(1, om + 1):
                    ok_ = om + 2 * orr
                    for on1 in range(ok_ + 1):
                        oterm = a[orr][on1, ok_ - on1] * In[on1, ok_ - on1]
                        os_ += oterm
                        osa += abs(oterm)
                        obig = max(obig, abs(oterm))
            F[oM - 1] = E * os_
            cn[oM - 1] = obig / In[0, 0]
            kp[oM - 1] = osa / abs(os_)
        return F, cn, kp

    def resolved(In):
        return np.isfinite(In[0, 0]) and In[0, 0] > 0

    Dk = np.diag(1.0 / np.sqrt(np.diag(K2)))
    Si = Dk @ np.linalg.inv(Dk @ K2 @ Dk) @ Dk
    uu = tau / np.sqrt(np.array([Si[0, 0], Si[1, 1]]))
    if np.all(uu > 2.5):
        Inu, Ih = Ifreq(Tm, tau, L, 80)
        E = math.exp(K0 - tau @ x - logscale - math.log(2 * math.pi) - logdet2)
        nGu = 80
        if resolved(Inu):
            Fup, canc, kappa = osums(Inu, E)
            qerr = np.abs(osums(Ih, E)[0] - Fup) / np.abs(Fup)
    else:
        t = K2 @ tau
        logE = K0 - tau @ x + tau @ K2 @ tau / 2 - logscale
        Gn = nG
        Ih = None
        while True:
            Inu = Ileg(K2, Lc, t, Gn, logE, L)
            if not resolved(Inu):
                break
            if Ih is None:
                Ih = Ileg(K2, Lc, t, Gn // 2, logE, L)
            Fup, canc, kappa = osums(Inu, 1.0)
            qerr = np.abs(osums(Ih, 1.0)[0] - Fup) / np.abs(Fup)
            if not (qerr[order - 1] > qtol and 2 * Gn <= nGmax):
                break
            Gn *= 2
            Ih = Inu
        nGu = Gn
    if not resolved(Inu):
        Fup = np.full(order, np.nan)
        canc = np.full(order, np.nan)
        kappa = np.full(order, np.nan)
        qerr = np.full(order, np.nan)
        status = addstatus(status, "leading integral not resolved")
        return res()
    if not qerr[order - 1] <= qtol:
        status = addstatus(status, "integration not resolved")
    return res()


def polyprod(A, B, kmin, N):
    """Coefficients of degree ``kmin..N`` of the product of the bivariate polynomials A and B."""
    C = np.zeros((N + 1, N + 1))
    for k in range(kmin, N + 1):
        for n1 in range(k + 1):
            n2 = k - n1
            s = 0.0
            for j in range(k + 1):
                for l1 in range(max(0, j - n2), min(j, n1) + 1):
                    l2 = j - l1
                    av = A[l1, l2]
                    if av == 0:
                        continue
                    s += av * B[n1 - l1, n2 - l2]
            C[n1, n2] = s
    return C


def subs2(P, Tm):
    """Coefficients of P(T u) for the bivariate polynomial P, T upper triangular."""
    n = P.shape[0]
    Q = np.zeros((n, n))
    T11, T12, T22 = Tm[0, 0], Tm[0, 1], Tm[1, 1]
    for n1 in range(n):
        for n2 in range(n - n1):
            c = P[n1, n2]
            if c == 0:
                continue
            for j in range(n1 + 1):
                Q[j, n1 - j + n2] += c * math.comb(n1, j) * T11 ** j * T12 ** (n1 - j) * T22 ** n2
    return Q


def hermite1(V, L):
    """H[m] = He_m(V), m = 0..L, by the three-term recursion (V an array)."""
    H = [np.ones_like(V)]
    if L == 0:
        return H
    H.append(V.copy())
    for m in range(2, L + 1):
        H.append(V * H[m - 1] - (m - 1) * H[m - 2])
    return H


def Ileg(K2, Lc, t, G, logE, L):
    """Whitened orthant integrals by Gauss-Legendre quadrature in conditioned coordinates."""
    W1, W2, phiW = condgrid(K2, t, G, logE)
    V1 = W1 / Lc[0, 0]
    V2 = (W2 - Lc[1, 0] * V1) / Lc[1, 1]
    H1 = hermite1(V1, L)
    H2 = hermite1(V2, L)
    Inu = np.zeros((L + 1, L + 1))
    for k in range(L + 1):
        for n1 in range(k + 1):
            Inu[n1, k - n1] = np.sum(H1[n1] * H2[k - n1] * phiW)
    return Inu


def Ifreq(Tm, tau, L, G):
    """Whitened form of ``I(nu)/phi(t, Sigma)`` by G x G Gauss-Hermite quadrature of the
    frequency-domain representation, and with G/2 nodes."""
    def run(n):
        x, w = gauher(n)
        Z1 = np.repeat(x[:, None], n, axis=1)
        Z2 = np.repeat(x[None, :], n, axis=0)
        W1 = Tm[0, 0] * Z1 + Tm[0, 1] * Z2
        W2 = Tm[1, 1] * Z2
        q = np.outer(w, w) / ((tau[0] + 1j * W1) * (tau[1] + 1j * W2))
        P1 = [np.ones((n, n), dtype=complex)]
        P2 = [np.ones((n, n), dtype=complex)]
        for k in range(1, L + 1):
            P1.append(P1[k - 1] * (1j * Z1))
            P2.append(P2[k - 1] * (1j * Z2))
        I = np.zeros((L + 1, L + 1))
        for k in range(L + 1):
            for n1 in range(k + 1):
                I[n1, k - n1] = np.real(np.sum(q * P1[n1] * P2[k - n1]))
        return I
    return run(G), run(G // 2)


def condgrid(S, t, G, logE=0.0):
    """Quadrature grid for ``E int_{z >= t} h(z - t) phi(z, S) dz`` in conditioned coordinates:
    the nodes ``W1, W2 = z - t`` and the weights ``phiW`` (with ``E = exp(logE)`` combined with
    the normal density in the log domain)."""
    xg, wg = gauleg(G)
    sd1 = math.sqrt(S[0, 0])
    b = S[0, 1] / S[0, 0]
    s2 = math.sqrt(S[1, 1] - S[0, 1] ** 2 / S[0, 0])
    m1 = modez1(S, t)
    a1 = max(t[0], m1 - 16 * sd1)
    b1 = m1 + 16 * sd1
    bp = layerbp(a1, b1, b, 0.0, s2, t[1], 16) + tailbp(a1, b1, 0.0, sd1, a1, 16)
    z1, w1 = glpieces(a1, b1, bp, xg, wg)
    n1 = len(z1)
    mu = b * z1
    lo = np.maximum(t[1], mu - 16 * s2)
    hi = np.maximum(mu, t[1]) + 16 * s2
    Z2 = (lo[:, None] + hi[:, None]) / 2 + (hi[:, None] - lo[:, None]) / 2 * xg[None, :]
    W2w = (hi[:, None] - lo[:, None]) / 2 * wg[None, :]
    phiW = np.exp(logE - z1[:, None] ** 2 / (2 * S[0, 0]) - math.log(SQRT2PI * sd1)
                  - (Z2 - mu[:, None]) ** 2 / (2 * s2 ** 2) - math.log(SQRT2PI * s2)) * w1[:, None] * W2w
    W1 = np.repeat((z1 - t[0])[:, None], G, axis=1)
    W2 = Z2 - t[1]
    return W1, W2, phiW


def glpieces(lo, hi, bp, xg, wg):
    """Gauss-Legendre nodes and weights on [lo, hi], split at the breakpoints inside (lo, hi),
    with the full number of nodes on each piece."""
    e = sorted(v for v in bp if lo < v < hi)
    e = [lo] + e + [hi]
    z = []
    w = []
    for k in range(len(e) - 1):
        if not e[k + 1] > e[k]:
            continue
        z.append((e[k] + e[k + 1]) / 2 + (e[k + 1] - e[k]) / 2 * xg)
        w.append((e[k + 1] - e[k]) / 2 * wg)
    return np.concatenate(z), np.concatenate(w)


def layerbp(lo, hi, slope, icpt, sd, tj, R):
    if slope == 0 or not np.isfinite(slope):
        return []
    zA = (tj - R * sd - icpt) / slope
    zB = (tj + R * sd - icpt) / slope
    return [zA, zB] if abs(zB - zA) < (hi - lo) / 4 else []


def tailbp(lo, hi, m, s, t, R):
    if t <= m:
        return []
    h = m + math.sqrt((t - m) ** 2 + R ** 2 * s ** 2)
    return [h] if h - lo < (hi - lo) / 4 else []


def modez1(S, t):
    """First coordinate of the maximizer of ``-z'S^-1 z/2`` on ``{z >= t}`` (2 x 2)."""
    Dk = np.diag(1.0 / np.sqrt(np.diag(S)))
    P = Dk @ np.linalg.inv(Dk @ S @ Dk) @ Dk
    if np.all(P @ t >= 0):
        return t[0]
    cand = []
    if np.all(t <= 0):
        cand.append(np.array([0.0, 0.0]))
    cand.append(np.array([t[0], max(t[1], S[0, 1] / S[0, 0] * t[0])]))
    cand.append(np.array([max(t[0], S[0, 1] / S[1, 1] * t[1]), t[1]]))
    q = [c @ P @ c for c in cand]
    return cand[int(np.argmin(q))][0]
