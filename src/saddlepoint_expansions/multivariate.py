"""Expansions in any dimension d: density (Theorem 3 of the paper) and orthant probability
(Theorem 1), the orthant integrals by Gauss quadrature (d = 3) or randomized quasi-Monte Carlo
(d >= 4), and the inclusion-exclusion of Section 3.3 for saddlepoints with negative components."""
from __future__ import annotations

import math

import numpy as np
from scipy.special import erfc, erfcinv, erfcx

from .bivariate import glpieces, layerbp, spa_bi, tailbp
from .taylor import compositions
from .univariate import spa_uni
from .util import addstatus, fact, gauher, gauleg, spa_saddle


# ----- multi-indices with |nu| <= N in graded order, located by rank

def mi_list(d, N):
    out = []
    for k in range(N + 1):
        out.extend(compositions(k, d))
    return out


class MIndex:
    def __init__(self, d, N):
        self.d = d
        self.N = N
        self.list = mi_list(d, N)
        self.pos = {nu: i for i, nu in enumerate(self.list)}
        self.deg = np.array([sum(nu) for nu in self.list])

    def rank(self, nu):
        return self.pos[tuple(nu)]


# ----- the expansions from the derivatives at the saddlepoint

def spa_mv(Kfun, x, order, tau0=None, method="auto", Kwhite=None, **kwargs):
    """Saddlepoint approximations of orders ``1..order`` in dimension ``d = len(x)`` of the
    density and of the orthant probability ``P(X >= x)`` from ``Kfun(nu, t)``, the derivative of
    order ``nu`` (a tuple) of the cumulant generating function at ``t`` (NaN outside the
    domain); see :func:`spa_mv_at`."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("spa_mv: x must be a nonempty finite vector")
    if order < 1:
        raise ValueError("spa_mv: order must be a positive integer")
    d = len(x)
    tau, sinfo = spa_saddle(Kfun, x, tau0)
    p = 2 * order - 1
    Kd = {nu: Kfun(nu, tau) for nu in mi_list(d, p + 2)}
    r = spa_mv_at(Kd, tau, x, order, method=method, Kwhite=Kwhite, **kwargs)
    r["tau"] = tau
    r["saddle"] = sinfo
    return r


def spa_mv_at(Kd, tau, x, order, method="auto", nG=64, nGH=48, nqmc=2 ** 14, nshift=8, seed=1,
              Kwhite=None, qtol=None, nGmax=None):
    """The expansions from the derivatives at a given saddlepoint: ``Kd`` is a dict from
    multi-indices (tuples) to ``K^(nu)(tau)``, ``|nu| <= 2*order + 1``.

    Returns a dict with ``pdf[M]`` (density expansion with the terms ``m <= M-1``), ``Fup[M]``
    (``P(X >= x)`` with the terms ``m <= 2M-1``; NaN when the saddlepoint has a negative
    component, use :func:`spa_orthant`), ``kappa`` and ``kappa_pdf`` (cancellation indicators),
    ``err`` (quadrature check: the change with half the nodes for ``method='quad'``, the
    standard error over the random shifts for ``'qmc'``), ``method_cdf``, ``nodes`` and
    ``status``. ``method`` is ``'quad'`` (Gauss quadrature, ``d = 3``), ``'qmc'`` (randomized
    quasi-Monte Carlo, any ``d``), ``'none'`` (density only) or ``'auto'`` (``'quad'`` for
    ``d = 3``, ``'qmc'`` otherwise). The terms are evaluated in whitened coordinates; the
    Taylor coefficients of ``K_Y(u) = K(T u)`` come from ``Kwhite(nu, u, T)`` when given and by
    polynomial substitution otherwise."""
    tau = np.asarray(tau, dtype=float)
    x = np.asarray(x, dtype=float)
    d = len(x)
    if method not in ("auto", "quad", "qmc", "none"):
        raise ValueError(f"spa_mv: unknown method {method!r}")
    meth = ("quad" if d == 3 else "qmc") if method == "auto" else method
    p = 2 * order - 1
    Nn = max(3 * p, 6 * (order - 1))
    I = MIndex(d, Nn)
    nN = len(I.list)
    deg = I.deg

    def e(j):
        return tuple(1 if i == j else 0 for i in range(d))

    K0 = float(Kd[(0,) * d])
    K2 = np.array([[float(Kd[tuple(a + b for a, b in zip(e(i), e(j)))]) for j in range(d)] for i in range(d)])
    Lc = np.linalg.cholesky(K2)
    Tm = np.linalg.inv(Lc.T)
    uh = Lc.T @ tau
    logdet2 = float(np.sum(np.log(np.diag(Lc))))
    sel = [k for k in range(nN) if 3 <= deg[k] <= p + 2]
    fsel = [np.prod([fact(n) for n in I.list[k]]) for k in sel]
    if Kwhite is not None:
        theta = np.zeros(nN)
        for i, k in enumerate(sel):
            theta[k] = Kwhite(I.list[k], uh, Tm) / fsel[i]
    else:
        theta0 = np.zeros(nN)
        for i, k in enumerate(sel):
            theta0[k] = float(Kd[I.list[k]]) / fsel[i]
        theta = polysub_mv(theta0, Tm, I)
    supp = [(I.list[k], theta[k]) for k in sel if theta[k] != 0]
    R = max(p, 2 * (order - 1))

    def mulpoly(prev, terms, r):
        A = np.zeros(nN)
        for kidx, ka in enumerate(I.list):
            v = prev[kidx]
            if v == 0 or deg[kidx] < 3 * (r - 1):
                continue
            for mu, th in terms:
                if deg[kidx] + sum(mu) > Nn:
                    continue
                A[I.rank(tuple(a + b for a, b in zip(ka, mu)))] += th * v
        return A

    a = [None] * (max(R, 1) + 1)
    a[1] = theta
    for r in range(2, R + 1):
        a[r] = mulpoly(a[r - 1], supp, r) / r
    # density: Hermite polynomials at 0 in whitened coordinates
    He0 = hermite_mv(I, np.eye(d), np.zeros((d, 1)))[0, :]
    f0 = math.exp(K0 - tau @ x - (d / 2) * math.log(2 * math.pi) - logdet2)
    pdf = np.zeros(order)
    kappa_pdf = np.ones(order)
    s = 1.0
    sa = 1.0
    pdf[0] = f0
    bydeg = {}
    for k in range(nN):
        bydeg.setdefault(int(deg[k]), []).append(k)
    for M in range(2, order + 1):
        m = M - 1
        for r in range(1, 2 * m + 1):
            for k in bydeg.get(2 * m + 2 * r, []):
                term = a[r][k] * He0[k]
                s += term
                sa += abs(term)
        pdf[M - 1] = f0 * s
        kappa_pdf[M - 1] = sa / abs(s)
    # orthant probability
    Fup = np.full(order, np.nan)
    kappa = np.full(order, np.nan)
    err = np.full(order, np.nan)
    nodes = dict(method=meth)

    def res(status):
        return dict(pdf=pdf, Fup=Fup, kappa=kappa, err=err, kappa_pdf=kappa_pdf, method_cdf=meth,
                    nodes=nodes, status=status)

    if meth == "none":
        return res("no orthant probability requested")
    if np.any(tau < 0):
        return res("negative saddlepoint component (use spa_orthant)")
    if K0 - tau @ x < math.log(np.finfo(float).tiny):
        Fup = np.zeros(order)
        err = np.zeros(order)
        kappa = np.ones(order)
        return res("probability below the double-precision range")
    L = 3 * p
    IL = MIndex(d, L)
    t = K2 @ tau
    Dk = np.diag(1.0 / np.sqrt(np.diag(K2)))
    Si = Dk @ np.linalg.inv(Dk @ K2 @ Dk) @ Dk
    uu = tau / np.sqrt(np.diag(Si))
    qt = (1e-3 if meth == "qmc" else 1e-8) if qtol is None else qtol
    degL = IL.deg
    rk = [I.rank(nu) for nu in IL.list]
    idxk = {k: [kk for kk in range(len(IL.list)) if degL[kk] == k] for k in range(L + 1)}
    E = 1.0

    def sums(Inu):
        F = np.zeros(order)
        ka = np.zeros(order)
        ss = float(Inu[0])
        sb = abs(ss)
        mm = 0
        for MM in range(1, order + 1):
            while mm < 2 * MM - 1:
                mm += 1
                for rr in range(1, mm + 1):
                    for kk in idxk[mm + 2 * rr]:
                        tt = a[rr][rk[kk]] * Inu[kk]
                        ss += tt
                        sb += abs(tt)
            F[MM - 1] = E * ss
            ka[MM - 1] = sb / abs(ss)
        return F, ka

    def resolved(v):
        return np.isfinite(v[0]) and v[0] > 0

    status = "ok"
    if meth == "quad":
        if d != 3:
            raise ValueError("spa_mv: method 'quad' needs d = 3")
        if np.all(uu > 2.5):
            Inu, Ih = Ifreq_mv(IL, Tm, tau, nGH)
            E = math.exp(K0 - tau @ x - (d / 2) * math.log(2 * math.pi) - logdet2)
            nodes["nGH"] = nGH
            if resolved(Inu):
                Fup, kappa = sums(Inu)
                err = np.abs(sums(Ih)[0] - Fup)
        else:
            logE = K0 - tau @ x + tau @ K2 @ tau / 2
            Gmax = 2 * nG if nGmax is None else nGmax
            Gn = nG
            Ih = None
            while True:
                Inu = Iquad3(IL, K2, Lc, t, Gn, logE)
                if not resolved(Inu):
                    break
                if Ih is None:
                    Ih = Iquad3(IL, K2, Lc, t, Gn // 2, logE)
                Fup, kappa = sums(Inu)
                err = np.abs(sums(Ih)[0] - Fup)
                if not (err[order - 1] > qt * abs(Fup[order - 1]) and 2 * Gn <= Gmax):
                    break
                Gn *= 2
                Ih = Inu
            nodes["nG"] = Gn
        Isets = [Inu]
    else:
        Isets = Iqmc(IL, K2, Lc, t, nqmc, nshift, seed, K0 - tau @ x + tau @ K2 @ tau / 2)
        nodes["nqmc"] = nqmc
        nodes["nshift"] = nshift
        if all(resolved(Is) for Is in Isets):
            Fs = [sums(Is) for Is in Isets]
            Fall = np.column_stack([F for F, _ in Fs])
            Fup = Fall.mean(axis=1)
            nsh = Fall.shape[1]
            err = np.sqrt(np.sum((Fall - Fup[:, None]) ** 2, axis=1) / (nsh * (nsh - 1)))
            _, kappa = sums(sum(Isets) / len(Isets))
    if not all(resolved(Is) for Is in Isets):
        Fup = np.full(order, np.nan)
        kappa = np.full(order, np.nan)
        err = np.full(order, np.nan)
        return res("leading integral not resolved")
    if not err[order - 1] <= qt * abs(Fup[order - 1]):
        status = addstatus(status, "integration not resolved")
    return res(status)


def polysub_mv(th0, Tm, I):
    """Coefficients of P(T u) for the polynomial with coefficients th0 on the list of I, T upper
    triangular (slot by slot substitution, last variable first)."""
    d = I.d
    th = th0.copy()
    for j in range(d - 1, -1, -1):
        new = np.zeros(len(th))
        for q in np.nonzero(th)[0]:
            nu = I.list[q]
            m = nu[j]
            if m == 0:
                new[q] += th[q]
                continue
            for c in compositions(m, d - j):
                mu = list(nu)
                mu[j] -= m
                coef = fact(m)
                for i, ci in enumerate(c):
                    mu[j + i] += ci
                    coef *= Tm[j, j + i] ** ci / fact(ci)
                new[I.rank(tuple(mu))] += th[q] * coef
        th = new
    return th


def hermite_mv(I, Si, W):
    """He[j, k] = He_{nu_k}(W[:, j], Sigma), nu_k = I.list[k], by the recursion of Takemura and
    Takeuchi (1988); Si = Sigma^-1, W has one column per point."""
    nN = len(I.list)
    n = W.shape[1]
    Y = (Si @ W).T
    He = np.zeros((n, nN))
    He[:, 0] = 1.0
    d = I.d
    for k, nu in enumerate(I.list):
        if k == 0:
            continue
        i = next(j for j in range(d) if nu[j] > 0)
        pr = list(nu)
        pr[i] -= 1
        kp = I.rank(tuple(pr))
        He[:, k] = Y[:, i] * He[:, kp]
        for l in range(d):
            if pr[l] <= 0:
                continue
            prl = list(pr)
            prl[l] -= 1
            kpp = I.rank(tuple(prl))
            He[:, k] -= Si[i, l] * pr[l] * He[:, kpp]
    return He


def Isum(IL, Si, W, wts):
    """I(nu) = sum_n wts[n] He_nu(W[:, n]) over chunks of nodes."""
    nN = len(IL.list)
    n = W.shape[1]
    chunk = max(256, min(n, int(1e7 // nN)))
    Inu = np.zeros(nN)
    for c0 in range(0, n, chunk):
        c1 = min(n, c0 + chunk)
        He = hermite_mv(IL, Si, W[:, c0:c1])
        Inu += He.T @ wts[c0:c1]
    return Inu


def Iquad3(IL, S, Lc, t, G, logE):
    """d = 3: Gauss-Legendre quadrature of the orthant-integral form in conditioned coordinates,
    whitened Hermite polynomials; the weights include exp(logE) combined with the normal density
    in the log domain."""
    xg, wg = gauleg(G)
    sd1 = math.sqrt(S[0, 0])
    b21 = S[1, 0] / S[0, 0]
    s2 = math.sqrt(S[1, 1] - S[1, 0] ** 2 / S[0, 0])
    c31 = S[2, 0] / S[0, 0]
    s31 = math.sqrt(S[2, 2] - S[2, 0] ** 2 / S[0, 0])
    S12 = S[0:2, 0:2]
    beta = np.linalg.solve(S12, S[2, 0:2])
    s3 = math.sqrt(S[2, 2] - beta @ S[0:2, 2])
    lo1 = t[0]
    hi1 = t[0] + 16 * sd1
    bp = layerbp(lo1, hi1, b21, 0.0, s2, t[1], 16) + layerbp(lo1, hi1, c31, 0.0, s31, t[2], 16) \
        + tailbp(lo1, hi1, 0.0, sd1, t[0], 16)
    z1s, w1s = glpieces(lo1, hi1, bp, xg, wg)
    P1 = []
    P2 = []
    LW = []
    for z1, w1 in zip(z1s, w1s):
        lw1 = math.log(w1) - z1 ** 2 / (2 * S[0, 0]) - math.log(math.sqrt(2 * math.pi) * sd1)
        mu2 = b21 * z1
        lo2 = max(t[1], mu2 - 16 * s2)
        hi2 = max(mu2, t[1]) + 16 * s2
        bp2 = layerbp(lo2, hi2, beta[1], beta[0] * z1, s3, t[2], 16) + tailbp(lo2, hi2, mu2, s2, t[1], 16)
        z2s, w2s = glpieces(lo2, hi2, bp2, xg, wg)
        for z2, w2 in zip(z2s, w2s):
            P1.append(z1)
            P2.append(z2)
            LW.append(lw1 + math.log(w2) - (z2 - mu2) ** 2 / (2 * s2 ** 2) - math.log(math.sqrt(2 * math.pi) * s2))
    n12 = len(P1)
    if n12 == 0:
        return np.zeros(len(IL.list))
    P1 = np.array(P1)
    P2 = np.array(P2)
    LW = np.array(LW)
    mu3 = beta[0] * P1 + beta[1] * P2
    lo3 = np.maximum(t[2], mu3 - 16 * s3)
    hi3 = np.maximum(mu3, t[2]) + 16 * s3
    Z3 = (lo3[:, None] + hi3[:, None]) / 2 + (hi3[:, None] - lo3[:, None]) / 2 * xg[None, :]
    lw3 = np.log((hi3[:, None] - lo3[:, None]) / 2 * wg[None, :]) - (Z3 - mu3[:, None]) ** 2 / (2 * s3 ** 2) \
        - math.log(math.sqrt(2 * math.pi) * s3)
    W = np.empty((3, n12 * G))
    W[0, :] = np.repeat(P1 - t[0], G)
    W[1, :] = np.repeat(P2 - t[1], G)
    W[2, :] = (Z3 - t[2]).ravel()
    wc = np.exp(logE + LW[:, None] + lw3).ravel()
    V = np.linalg.solve(Lc, W)
    return Isum(IL, np.eye(3), V, wc)


def Ifreq_mv(IL, Tm, tau, G):
    """Whitened form of ``I(nu)/phi(t, Sigma)`` by G^d Gauss-Hermite quadrature of the
    frequency-domain representation, and with G/2 nodes."""
    d = IL.d
    C = np.asarray(Tm, dtype=float)

    def run(G):
        xg, wg = gauher(G)
        nN = len(IL.list)
        Inu = np.zeros(nN)
        grids = np.meshgrid(*([np.arange(G)] * d), indexing="ij")
        idx = np.stack([g.ravel() for g in grids], axis=1)          # all node tuples
        n_all = idx.shape[0]
        chunk = max(256, int(5e6 // nN))
        for c0 in range(0, n_all, chunk):
            c1 = min(n_all, c0 + chunk)
            ci = idx[c0:c1]
            n = c1 - c0
            Z = xg[ci].T                                            # d x n
            q = np.prod(wg[ci], axis=1).astype(complex)
            W = C @ Z
            for j in range(d):
                q /= (tau[j] + 1j * W[j, :])
            V = np.zeros((n, nN), dtype=complex)
            V[:, 0] = 1.0
            iZ = (1j * Z).T
            for k, nu in enumerate(IL.list):
                if k == 0:
                    continue
                i = next(j for j in range(d) if nu[j] > 0)
                pr = list(nu)
                pr[i] -= 1
                kp = IL.rank(tuple(pr))
                V[:, k] = V[:, kp] * iZ[:, i]
            Inu += np.real(V.T @ q)
        return Inu

    return run(G), run(G // 2)


class _Lcg:
    """Small deterministic generator for the random shifts (the same as in the Julia package)."""

    def __init__(self, seed):
        self.s = np.uint64(seed)

    def rand(self):
        with np.errstate(over="ignore"):
            self.s = self.s * np.uint64(0x5851f42d4c957f2d) + np.uint64(0x14057b7ef767814f)
        return float(self.s >> np.uint64(11)) * 2.0 ** -53


def logPhibar(a):
    """log P(Z >= a) without underflow."""
    a = np.asarray(a, dtype=float)
    return np.where(a > 0, np.log(erfcx(a / math.sqrt(2)) / 2) - a * a / 2, np.log(erfc(a / math.sqrt(2)) / 2))


def tailinv(a, lq):
    """y >= a with log P(Z >= y) = lq."""
    y = np.nan if lq < -600 else math.sqrt(2) * erfcinv(2 * math.exp(lq))
    if not np.isfinite(y):
        y = a + (float(logPhibar(a)) - lq) / max(a, 1.0)
        for _ in range(20):
            f = float(logPhibar(y)) - lq
            h = -math.sqrt(2 / math.pi) / erfcx(y / math.sqrt(2))
            y = max(a, y - f / h)
    return max(y, a)


def genz_order(S, t):
    """Variable prioritization of Genz and Bretz."""
    d = len(t)
    perm = list(range(d))
    Sp = S.copy()
    tp = t.copy()
    L = np.zeros((d, d))
    y = np.zeros(d)
    for j in range(d):
        best = j
        bestp = np.inf
        for i in range(j, d):
            v = Sp[i, i] - np.sum(L[i, :j] ** 2)
            a = (tp[i] - L[i, :j] @ y[:j]) / math.sqrt(max(v, np.finfo(float).eps))
            pa = erfc(a / math.sqrt(2)) / 2
            if pa < bestp:
                best, bestp = i, pa
        if best != j:
            perm[j], perm[best] = perm[best], perm[j]
            Sp[[j, best], :] = Sp[[best, j], :]
            Sp[:, [j, best]] = Sp[:, [best, j]]
            tp[[j, best]] = tp[[best, j]]
            L[[j, best], :] = L[[best, j], :]
        L[j, j] = math.sqrt(Sp[j, j] - np.sum(L[j, :j] ** 2))
        for i in range(j + 1, d):
            L[i, j] = (Sp[i, j] - L[i, :j] @ L[j, :j]) / L[j, j]
        a = (tp[j] - L[j, :j] @ y[:j]) / L[j, j]
        pa = erfc(a / math.sqrt(2)) / 2
        y[j] = math.exp(-a * a / 2) / math.sqrt(2 * math.pi) / pa if pa > 0 else a
    return perm


def Iqmc(IL, S, Lc, t, nqmc, nshift, seed, logE):
    """Randomized QMC: Richtmyer lattice with antithetic points and nshift random shifts;
    Genz's sequential conditioning of the normal vector truncated to z >= t. One vector of
    whitened integrals per shift."""
    d = len(t)
    perm = genz_order(S, t)
    Sp = S[np.ix_(perm, perm)]
    tp = t[perm]
    Lp = np.linalg.cholesky(Sp)
    primes = []
    c = 2
    while len(primes) < d:
        if all(c % q != 0 for q in primes):
            primes.append(c)
        c += 1
    alpha = np.sqrt(np.array(primes, dtype=float))
    rng = _Lcg(seed)
    out = []
    for _ in range(nshift):
        shift = np.array([rng.rand() for _ in range(d)])
        n = 2 * nqmc
        kk = np.arange(1, nqmc + 1)
        U = np.empty((d, n))
        for j in range(d):
            u = np.mod(kk * alpha[j] + shift[j], 1.0)
            U[j, :nqmc] = u
            U[j, nqmc:] = 1 - u
        W = np.zeros((d, n))
        lw = np.zeros(n)
        Y = np.zeros((d, n))
        for j in range(d):
            s = Lp[j, :j] @ Y[:j, :] if j > 0 else np.zeros(n)
            aj = (tp[j] - s) / Lp[j, j]
            lpa = logPhibar(aj)
            lw += lpa
            Y[j, :] = np.array([tailinv(aj[k], lpa[k] + math.log(U[j, k])) for k in range(n)])
        z = Lp @ Y
        for j in range(d):
            W[perm[j], :] = z[j, :] - tp[j]
        V = np.linalg.solve(Lc, W)
        out.append(Isum(IL, np.eye(d), V, np.exp(logE + lw) / n))
    return out


# ----- orthant probabilities for saddlepoints of any sign

def spa_orthant(Kfun, x, order, Kwhite=None, **kwargs):
    """``P(X >= x)`` of orders ``1..order`` in any dimension, with the inclusion-exclusion of
    the paper when the saddlepoint has negative components: every term is the orthant
    probability of a marginal with some components negated, computed by :func:`spa_uni` (one
    dimension), :func:`spa_bi` (two) or :func:`spa_mv` (three or more), recursively. Returns a
    dict with ``Fup``, ``tau`` and ``err`` (the sum of the absolute error estimates of the terms;
    NaN when a term has none). ``Kwhite`` is transformed for every term (it must accept a
    rectangular ``d x k`` matrix)."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("spa_orthant: x must be a nonempty finite vector")
    if order < 1:
        raise ValueError("spa_orthant: order must be a positive integer")
    d = len(x)
    if d == 1:
        K1 = lambda k, s: Kfun((k,), np.array([s]))
        return dict(Fup=uni_upper(K1, x[0], order), tau=np.array([np.nan]), err=np.zeros(order))
    tau, _ = spa_saddle(Kfun, x, np.zeros(d))
    if np.all(tau >= 0):
        r = _orthant_core(Kfun, x, order, tau, Kwhite=Kwhite, **kwargs)
        return dict(Fup=r["Fup"], tau=tau, err=r["err"])
    Nn = [j for j in range(d) if tau[j] < 0]
    Fup = np.zeros(order)
    err = np.zeros(order)
    for mask in range(2 ** len(Nn)):
        Tset = [Nn[i] for i in range(len(Nn)) if (mask >> i) & 1]
        sgn = (-1) ** len(Tset)
        keep = [j for j in range(d) if j not in set(Nn) - set(Tset)]
        flip = np.ones(d)
        flip[Tset] = -1
        if not keep:
            Fup += sgn
            continue
        k = len(keep)

        def Km(nu, t, keep=keep, flip=flip):
            nf = [0] * d
            tf = np.zeros(d)
            for c, j in enumerate(keep):
                nf[j] = nu[c]
                tf[j] = t[c]
            return float(np.prod(flip[keep] ** np.array(nu))) * Kfun(tuple(nf), flip * tf)

        xk = (flip * x)[keep]
        Kw = None
        if Kwhite is not None:
            B = np.zeros((d, k))
            for c, j in enumerate(keep):
                B[j, c] = flip[j]
            Kw = (lambda nu, u, M, B=B: Kwhite(nu, u, B @ M))
        if k == d:
            r = _orthant_core(Km, xk, order, np.abs(tau), Kwhite=Kw, **kwargs)
        else:
            r = spa_orthant(Km, xk, order, Kwhite=Kw, **kwargs)
        Fup += sgn * r["Fup"]
        err += r["err"]
    return dict(Fup=Fup, tau=tau, err=err)


def _orthant_core(Kfun, x, order, tau0, Kwhite=None, **kwargs):
    d = len(x)
    if d == 2:
        bikw = {k: v for k, v in kwargs.items() if k in ("nG", "logscale", "qtol", "nGmax")}
        r = spa_bi(Kfun, x, order, tau0=tau0, Kwhite=Kwhite, **bikw)
        return dict(Fup=r["Fup"], tau=r["tau"], err=np.abs(r["Fup"]) * r["qerr"])
    r = spa_mv(Kfun, x, order, tau0=tau0, Kwhite=Kwhite, **kwargs)
    return dict(Fup=r["Fup"], tau=r["tau"], err=r["err"])


def uni_upper(K1, xj, order):
    """P[X >= xj] for the univariate cgf K1(k, s) (NaN outside the domain), by spa_uni."""
    N = 2 * order + 1
    s, _ = spa_saddle(lambda nu, t: K1(nu[0], t[0]), np.array([xj]), np.array([0.0]))
    s = s[0]
    Kd = [K1(k, s) for k in range(N + 1)]
    return spa_uni(Kd, s, xj, order)["upp"]


def spa_orthant2(Kfun, x, order, nG=200, Kwhite=None, logscale=0.0, qtol=1e-8):
    """``P(X1 >= x1, X2 >= x2)`` of orders ``1..order`` by the bivariate expansion, with the
    inclusion-exclusion when the saddlepoint has negative components; the one-dimensional terms
    use :func:`spa_uni`. With ``logscale = c`` every term is scaled by ``exp(-c)``. Returns a
    dict with ``Fup``, ``tau``, ``canc``, ``kappa``, ``status`` (those of the bivariate term of
    the highest order)."""
    x = np.asarray(x, dtype=float)
    r = spa_bi(Kfun, x, order, nG=nG, Kwhite=Kwhite, logscale=logscale, qtol=qtol)
    tau = r["tau"]
    canc = r["canc"][-1]
    kappa = r["kappa"][-1]
    status = r["status"]
    if np.all(tau >= 0):
        return dict(Fup=r["Fup"], tau=tau, canc=canc, kappa=kappa, status=status, qerr=r["qerr"])
    Nn = [j for j in range(2) if tau[j] < 0]
    Fup = np.zeros(order)
    qerr = np.zeros(order)
    for mask in range(2 ** len(Nn)):
        Tset = [Nn[i] for i in range(len(Nn)) if (mask >> i) & 1]
        sgn = (-1) ** len(Tset)
        keep = [j for j in range(2) if j not in set(Nn) - set(Tset)]
        flip = np.ones(2)
        flip[Tset] = -1
        if len(keep) == 2:
            Kf = (lambda nu, t, flip=flip: flip[0] ** nu[0] * flip[1] ** nu[1] * Kfun(nu, flip * t))
            Kw = None if Kwhite is None else (lambda nu, u, M, flip=flip: Kwhite(nu, u, np.diag(flip) @ M))
            r2 = spa_bi(Kf, flip * x, order, tau0=np.abs(tau), nG=nG, Kwhite=Kw, logscale=logscale, qtol=qtol)
            canc, kappa, status = r2["canc"][-1], r2["kappa"][-1], r2["status"]
            Fup += sgn * r2["Fup"]
            qerr += np.abs(r2["Fup"]) * r2["qerr"]
        elif len(keep) == 1:
            j = keep[0]
            f = flip[j]

            def K1(k, s, j=j, f=f):
                nu = (k, 0) if j == 0 else (0, k)
                t = np.array([f * s, 0.0]) if j == 0 else np.array([0.0, f * s])
                return f ** k * Kfun(nu, t)

            F1 = uni_upper(K1, f * x[j], order)
            Fup += sgn * np.sign(F1) * np.exp(np.log(np.abs(F1)) - logscale)
        else:
            Fup += sgn * math.exp(-logscale)
    return dict(Fup=Fup, tau=tau, canc=canc, kappa=kappa, status=status, qerr=qerr)
