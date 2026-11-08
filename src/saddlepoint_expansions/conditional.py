"""Conditional saddlepoint expansions: the conditional distribution function
``P(X_A >= x_A | X_B = x_B)`` and the conditional density of ``X_A`` given ``X_B = x_B``, for
continuous and for lattice variables (Broda and Kan, "Conditional saddlepoint expansions of
arbitrary order", Theorems 1 to 3; port of conditional.jl of the Julia package).

The numerator of the conditional probability is the orthant-probability integral with the poles in
the components of A only, the denominator the density expansion of X_B at its own saddlepoint. The
numerator is evaluated in the canonical coordinates of Section 5.2 of that paper,
``Y_A = X_A - C X_B``, ``Y_B = L^{-1} X_B``, ``C = K''_AB (K''_BB)^{-1}``, ``LL' = K''_BB``, in which
the tilted covariance matrix is block diagonal, ``diag(K''_{AA.B}, I)``; the tail components are
then whitened as well, with the Hermite polynomials taken as products of univariate ones integrated
over the same orthant (Appendix B of the first paper). The density expansions are evaluated in
fully whitened coordinates.
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from .cgf import _indomain, _taylor, kfun, kwhite, reflect
from .multivariate import hermite_mv, mi_list, MIndex
from .taylor import DomainError, compositions
from .util import SaddlepointError, addstatus, gauleg, hcoef, normcdf, normpdf, spa_saddle


# ----- multi-indices as integer arrays, ranked by the binomial formula (the graded order of
# MIndex and of the Taylor algebra)

class _MI:
    def __init__(self, d, N):
        self.d = d
        self.N = N
        self.E = np.array(mi_list(d, N), dtype=np.int64).reshape(-1, d)
        self.deg = self.E.sum(axis=1)
        nb = N + d + 2
        B = np.zeros((nb + 1, nb + 1), dtype=np.int64)
        for n in range(nb + 1):
            for k in range(n + 1):
                B[n, k] = math.comb(n, k)
        self.B = B
        self.start = np.searchsorted(self.deg, np.arange(N + 2))
        self.n = self.E.shape[0]

    def _b(self, n, k):
        n = np.asarray(n, dtype=np.int64)
        out = np.zeros(n.shape, dtype=np.int64)
        ok = (n >= k) & (n >= 0)
        out[ok] = self.B[n[ok], k]
        return out

    def rank(self, E):
        """0-based positions of the rows of the integer array E (n x d)."""
        E = np.asarray(E, dtype=np.int64).reshape(-1, self.d)
        d = self.d
        k = E.sum(axis=1)
        r = np.where(k > 0, self._b(k - 1 + d, d), 0)
        ki = k.copy()
        for i in range(d - 1):
            r = r + self._b(ki - E[:, i] - 1 + d - (i + 1), d - (i + 1))
            ki = ki - E[:, i]
        return r

    def degrange(self, k):
        return slice(int(self.start[k]), int(self.start[k + 1]))

    def count(self, k):
        """Number of multi-indices with |nu| <= k."""
        return int(self.start[min(k, self.N) + 1]) if k >= 0 else 0


@lru_cache(maxsize=None)
def _mi(d, N):
    return _MI(d, N)


@lru_cache(maxsize=None)
def _comps_arr(m, p):
    return np.array(compositions(m, p), dtype=np.int64).reshape(-1, p)


def _cond_set(A, d, name):
    Av = [int(A)] if np.ndim(A) == 0 else [int(a) for a in A]
    if len(Av) == 0 or len(set(Av)) != len(Av) or not all(0 <= a < d for a in Av):
        raise ValueError(f"{name}: A must be a nonempty set of distinct indices in 0..{d - 1}")
    return sorted(Av)


def _he0_list(I):
    """He_nu(0, I) = prod_j He_{nu_j}(0) on the list of I."""
    h = hcoef(I.N // 2 + 1)
    E = I.E
    out = np.ones(I.n)
    for j in range(I.d):
        col = E[:, j]
        v = np.where(col % 2 == 0, h[col // 2], 0.0)
        out = out * v
    return out


def _polysub_slot(th, I, j, ell):
    """Substitution of the variable in slot j by the linear form ell'u in the polynomial with the
    coefficients th on the list of I."""
    supp = np.nonzero(np.asarray(ell) != 0)[0]
    ns = len(supp)
    new = np.zeros(len(th))
    fac = [math.factorial(k) for k in range(I.N + 1)]
    for q in np.nonzero(th)[0]:
        v = th[q]
        nu = I.E[q]
        m = int(nu[j])
        if m == 0:
            new[q] += v
            continue
        if ns == 0:
            continue
        C = _comps_arr(m, ns)
        base = nu.copy()
        base[j] = 0
        tg = np.repeat(base[None, :], C.shape[0], axis=0)
        tg[:, supp] += C
        coef = np.full(C.shape[0], float(fac[m]))
        for i, si in enumerate(supp):
            coef *= np.asarray(ell[si], dtype=float) ** C[:, i] / np.array([fac[c] for c in C[:, i]], dtype=float)
        np.add.at(new, I.rank(tg), v * coef)
    return new


def _cond_coeffs(theta, I, R, dmax):
    """c[r][k] = [u^nu_k] G^r/r!, r = 1..R (c[0] unused), only for 3r <= |nu| <= dmax(r)."""
    c = [None] * (R + 1)
    if R < 1:
        return c
    c[1] = theta
    supp = np.nonzero(theta)[0]
    supp = supp[np.argsort(I.deg[supp], kind="stable")]
    for r in range(2, R + 1):
        dm = dmax(r)
        prev = c[r - 1]
        A = np.zeros(I.n)
        src = np.nonzero(prev[:I.count(dm - 3)])[0]
        if len(src):
            dsrc = I.deg[src]
            Esrc = I.E[src]
            vsrc = prev[src]
            for q in supp:
                dq = I.deg[q]
                ok = dsrc + dq <= dm
                if not ok.any():
                    if dq + dsrc.min() > dm:
                        break
                    continue
                tgt = I.rank(Esrc[ok] + I.E[q])
                A[tgt] += theta[q] * vsrc[ok]          # distinct targets for a fixed q
        c[r] = A / r
    return c


def _cond_theta(Kfun, Kwhite, tau, uh, Tm, I, nmax, subs):
    """Taylor coefficients [v^nu] K(tau + Tm v), 3 <= |nu| <= nmax, on the list of I: from Kwhite
    at uh (Tm uh = tau), or from the derivatives of K at tau by the substitutions subs."""
    theta = np.zeros(I.n)
    for k in range(int(I.start[3]) if I.N >= 3 else I.n, I.count(nmax)):
        nu = tuple(int(v) for v in I.E[k])
        f = 1.0
        for n in nu:
            f *= math.factorial(n)
        theta[k] = (Kfun(nu, tau) if Kwhite is None else Kwhite(nu, uh, Tm)) / f
    if Kwhite is not None:
        return theta
    for j, ell in subs:
        theta = _polysub_slot(theta, I, j, ell)
    return theta


def _cond_K02(Kfun, tau):
    d = len(tau)

    def e(j):
        return tuple(1 if i == j else 0 for i in range(d))

    K0 = float(Kfun((0,) * d, tau))
    K2 = np.array([[Kfun(tuple(a + b for a, b in zip(e(i), e(j))), tau) for j in range(d)] for i in range(d)],
                  dtype=float)
    if not (np.isfinite(K0) and np.all(np.isfinite(K2))):
        raise RuntimeError("conditional: the saddlepoint is outside the domain of K")
    return K0, (K2 + K2.T) / 2


# ----- the density expansion in whitened coordinates

def _cond_dens(Kfun, Kwhite, tau, x, order):
    """log of the leading term and the bracket of orders 1..order at the saddlepoint tau of x,
    with the cancellation indicators of the bracket."""
    d = len(x)
    K0, K2 = _cond_K02(Kfun, tau)
    Lc = np.linalg.cholesky(K2)
    logf0 = K0 - float(tau @ x) - d / 2 * math.log(2 * math.pi) - float(np.sum(np.log(np.diag(Lc))))
    s = np.ones(order)
    kappa = np.ones(order)
    if order == 1:
        return logf0, s, kappa
    Tm = np.linalg.inv(Lc.T)
    uh = Lc.T @ tau
    Lx = 6 * (order - 1)
    I = _mi(d, Lx)
    theta = _cond_theta(Kfun, Kwhite, tau, uh, Tm, I, 2 * order, [(j, Tm[j, :]) for j in range(d - 1, -1, -1)])
    R = 2 * (order - 1)
    c = _cond_coeffs(theta, I, R, lambda r: min(Lx, 2 * (order - 1) + 2 * r))
    He0 = _he0_list(I)
    acc = 1.0
    ab = 1.0
    for M in range(2, order + 1):
        m = M - 1
        for r in range(1, 2 * m + 1):
            sl = I.degrange(2 * m + 2 * r)
            t = c[r][sl] * He0[sl]
            acc += float(np.sum(t))
            ab += float(np.sum(np.abs(t)))
        s[M - 1] = acc
        kappa[M - 1] = ab / abs(acc)
    return logf0, s, kappa


# ----- the lattice amplitude g(s) = s/(2 sinh(s/2))

def _ginv_series(n):
    """Coefficients of 1/Q(u), Q(u) = 2 sinh(u/2)/u = sum_k u^(2k)/(4^k (2k+1)!), degrees 0..n."""
    Q = np.zeros(n + 1)
    for k in range(n // 2 + 1):
        Q[2 * k] = 1.0 / (4.0 ** k * math.factorial(2 * k + 1))
    co = np.zeros(n + 1)
    co[0] = 1.0
    for m in range(1, n + 1):
        co[m] = -sum(Q[j] * co[m - j] for j in range(1, m + 1))
    return co


def _shiftpoly(co, s0, L):
    g = np.zeros(L + 1)
    for k in range(len(co)):
        for j in range(min(k, L) + 1):
            g[j] += co[k] * math.comb(k, j) * s0 ** (k - j)
    return g


def _sinhcoef(s0, L, mode, deg=0):
    """Taylor coefficients at s0, degrees 0..L, of g(s) = s/(2 sinh(s/2)) (mode 'saddle'), or of
    its expansion about the origin truncated after s^deg (mode 'origin', Kolassa 2003)."""
    if mode == "origin":
        return _shiftpoly(_ginv_series(deg), s0, L)
    if abs(s0) >= 1:
        q = np.array([(2 * math.sinh(s0 / 2) if k % 2 == 0 else 2 * math.cosh(s0 / 2)) / (2.0 ** k * math.factorial(k))
                      for k in range(L + 1)])
        pp = np.zeros(L + 1)
        pp[0] = s0
        if L >= 1:
            pp[1] = 1.0
        g = np.zeros(L + 1)
        g[0] = pp[0] / q[0]
        for m in range(1, L + 1):
            g[m] = (pp[m] - sum(q[j] * g[m - j] for j in range(1, m + 1))) / q[0]
        return g
    n = L + 40
    return _shiftpoly(_ginv_series(n), s0, L)


# ----- the integrals over the orthant of the tail components

def _cond_panels(u, G, c, L):
    """Nodes v >= 0 and weights of three Gauss-Legendre panels for He_k(v) exp(-(u + v)^2/2),
    k <= L, truncated where L log(max(v, 1)) - (u + v)^2/2 falls c below its maximum."""
    xg, wg = gauleg(G)

    def h(v):
        return L * math.log(max(v, 1.0)) - (u + v) ** 2 / 2

    vq = max(0.0, -u)
    vs = (-u + math.sqrt(u * u + 4 * L)) / 2 if L > 0 else 0.0
    vst = max(vq, vs) if vs > 1 else vq
    hm = max(h(vq), h(vs) if vs > 1 else h(vq))
    lo, hi = vst, vst + 1.0
    while h(hi) > hm - c:
        lo, hi = hi, 2 * hi
    for _ in range(200):
        mid = (lo + hi) / 2
        if h(mid) > hm - c:
            lo = mid
        else:
            hi = mid
        if hi - lo <= 8 * np.finfo(float).eps * hi:
            break
    vmax = hi
    if u >= 0:
        br = (0.0, min(1 / (1 + u), vmax / 3), min(4 / (1 + u), 2 * vmax / 3), vmax)
    else:
        v0 = max(0.0, -u - math.sqrt(2 * c))
        b1 = min(max(-u - 2, v0), vmax)
        b2 = min(max(-u + 2, b1), vmax)
        br = (v0, b1, b2, vmax)
    v = []
    w = []
    for k in range(3):
        hw = (br[k + 1] - br[k]) / 2
        if hw <= 0:
            continue
        m = (br[k] + br[k + 1]) / 2
        v.append(m + hw * xg)
        w.append(hw * wg)
    return np.concatenate(v), np.concatenate(w)


def _cond_grid(S, tA, G, c, L):
    """Nodes w = z_A - t_A >= 0 of the a-dimensional orthant in sequentially conditioned
    coordinates of N(0, S), t_A = S tA, with the logs of the products of the panel weights."""
    a = len(tA)
    t = S @ tA
    s1 = math.sqrt(S[0, 0])
    v, w = _cond_panels(t[0] / s1, G, c, L)
    Z = (t[0] + s1 * v)[None, :]
    lq = np.log(s1 * w)
    for i in range(1, a):
        beta = np.linalg.solve(S[:i, :i], S[:i, i])
        si = math.sqrt(S[i, i] - beta @ S[:i, i])
        Gi = G if a == 2 else max(8, G // 2)
        Zn = []
        lqn = []
        for q in range(Z.shape[1]):
            mu = float(beta @ Z[:, q])
            vi, wi = _cond_panels((t[i] - mu) / si, Gi, c, L)
            Zn.append(np.vstack([np.repeat(Z[:, q:q + 1], len(vi), axis=1), (t[i] + si * vi)[None, :]]))
            lqn.append(lq[q] + np.log(si * wi))
        Z = np.hstack(Zn)
        lq = np.concatenate(lqn)
    return Z - t[:, None], lq


def _cond_IA(IA, S, LA, tA, logE0, G):
    """The integrals over the orthant of the tail components in canonical coordinates, with the
    Hermite polynomials products of univariate ones in v = LA^{-1} w (Appendix B of the first
    paper); G and G/2 nodes per panel."""
    a = len(tA)
    L = IA.N
    c = 40.0
    if a == 1:
        sd = LA[0, 0]
        u = sd * tA[0]

        def run1(G):
            vv, ww = _cond_panels(u, G, c, L)
            W = ww * np.exp(logE0 - u * vv - vv ** 2 / 2) / math.sqrt(2 * math.pi)
            out = np.zeros(L + 1)
            h0 = np.ones(len(vv))
            h1 = vv.copy()
            out[0] = W.sum()
            if L >= 1:
                out[1] = W @ h1
            for k in range(1, L):
                h0, h1 = h1, vv * h1 - k * h0
                out[k + 1] = W @ h1
            return out

        return run1(G), run1(max(8, G // 2))
    Si = np.linalg.inv(S)
    Si = (Si + Si.T) / 2
    ldS = 2 * float(np.sum(np.log(np.diag(LA))))
    mi = MIndex(a, L)

    def runa(G):
        W, lq = _cond_grid(S, tA, G, c, L)
        lw = lq + logE0 - tA @ W - np.sum(W * (Si @ W), axis=0) / 2 - a / 2 * math.log(2 * math.pi) - ldS / 2
        wt = np.exp(lw)
        V = np.linalg.solve(LA, W)
        out = np.zeros(len(mi.list))
        n = W.shape[1]
        chunk = max(64, int(4e6 // len(mi.list)))
        for c0 in range(0, n, chunk):
            c1 = min(n, c0 + chunk)
            He = hermite_mv(mi, np.eye(a), V[:, c0:c1])
            out += He.T @ wt[c0:c1]
        return out

    return runa(G), runa(max(8, G // 2))


# ----- the numerator of the conditional tail probability (Theorems 1 and 3)

def _cond_lattice_ct(c, Gc, IL, A, UA, p, L):
    """ct_r = sum_mu c_{r, . - mu} gamma_mu, r = 0..p, gamma the Taylor coefficients of
    prod_{j in A} g(s_j) at tau in the whitened variables of the tail components."""
    a = len(A)
    IA = _mi(a, L)
    h = np.ones(IA.n)
    for i in range(a):
        h = h * Gc[i][IA.E[:, i]]
    for i in range(a - 1, -1, -1):
        h = _polysub_slot(h, IA, i, UA[i, :])
    nz = np.nonzero(h)[0]
    gE = np.zeros((len(nz), IL.d), dtype=np.int64)
    gE[:, A] = IA.E[nz]
    gv = h[nz]
    gd = IA.deg[nz]
    ct = [None] * (p + 1)
    for r in range(p + 1):
        dm = p if r == 0 else min(L, p + 2 * r)
        v = np.zeros(IL.n)
        if r == 0:
            ok = gd <= dm
            v[IL.rank(gE[ok])] += gv[ok]
        else:
            prev = c[r]
            src = np.nonzero(prev[:IL.count(dm)])[0]
            if len(src):
                Es = IL.E[src]
                ds = IL.deg[src]
                for q in range(len(gv)):
                    ok = ds + gd[q] <= dm
                    if ok.any():
                        v[IL.rank(Es[ok] + gE[q])] += gv[q] * prev[src[ok]]
        ct[r] = v
    return ct


def _cond_numer(Kfun, Kwhite, tau, x, A, order, logscale, lattice, mode, nG):
    d = len(x)
    B = [j for j in range(d) if j not in A]
    a = len(A)
    b = len(B)
    p = 2 * order - 1
    L = 3 * p
    K0, K2 = _cond_K02(Kfun, tau)
    T1 = np.eye(d)
    uh = tau.copy()
    if b > 0:
        KBB = K2[np.ix_(B, B)]
        Lb = np.linalg.cholesky(KBB)
        C = np.linalg.solve(KBB, K2[np.ix_(B, A)]).T              # K_AB K_BB^{-1}
        T1[np.ix_(B, A)] = -C.T
        T1[np.ix_(B, B)] = np.linalg.inv(Lb.T)
        SA = K2[np.ix_(A, A)] - C @ K2[np.ix_(B, A)]
        uh[B] = Lb.T @ (tau[B] + C.T @ tau[A])
        logdetL = float(np.sum(np.log(np.diag(Lb))))
    else:
        SA = K2.copy()
        logdetL = 0.0
    SA = (SA + SA.T) / 2
    try:
        LA = np.linalg.cholesky(SA)
    except np.linalg.LinAlgError:
        raise RuntimeError("conditional: K''_{AA.B} is not positive definite at the saddlepoint")
    UA = np.linalg.inv(LA.T)
    T2 = np.eye(d)
    T2[np.ix_(A, A)] = UA
    Tm = T1 @ T2
    uh[A] = LA.T @ tau[A]
    IL = _mi(d, L)
    subs = [(j, T1[j, :]) for j in reversed(B)] + [(j, T2[j, :]) for j in reversed(A)]
    theta = _cond_theta(Kfun, Kwhite, tau, uh, Tm, IL, p + 2, subs)
    c = _cond_coeffs(theta, IL, p, lambda r: min(L, p + 2 * r))
    delta = np.zeros(IL.n)
    delta[0] = 1.0
    if not lattice:
        ct = [delta] + c[1:]
        ctM = [ct] * order
    else:
        ctM = [None] * order
        for M in range(1, order + 1):
            if mode == "saddle" and M > 1:
                ctM[M - 1] = ctM[0]
                continue
            dg = 2 * M - 2 if mode == "origin" else 0
            Gc = [_sinhcoef(float(tau[A[i]]), L, mode, dg) for i in range(a)]
            ctM[M - 1] = _cond_lattice_ct(c, Gc, IL, A, UA, p, L)
    logE0 = K0 - float(tau @ x) - logdetL - b / 2 * math.log(2 * math.pi) - logscale
    IA = _mi(a, L)
    Iv, Ih = _cond_IA(IA, SA, LA, tau[A], logE0, nG)
    hB = hcoef(L // 2 + 1)
    E = IL.E
    hb = np.ones(IL.n)
    for j in B:
        col = E[:, j]
        hb = hb * np.where(col % 2 == 0, hB[col // 2], 0.0)
    rA = IA.rank(E[:, A])

    def full(Iv):
        return hb * Iv[rA]

    def sums(Inu):
        F = np.zeros(p + 1)
        ab = np.zeros(p + 1)
        for j in range(p + 1):
            ct = ctM[j // 2]
            s = 0.0
            sa = 0.0
            for m in range(j + 1):
                for r in range(m + 1):
                    sl = IL.degrange(m + 2 * r)
                    tt = ct[r][sl] * Inu[sl]
                    s += float(np.sum(tt))
                    sa += float(np.sum(np.abs(tt)))
            F[j] = s
            ab[j] = sa
        return F, ab

    D, Dabs = sums(full(Iv))
    err = np.abs(sums(full(Ih))[0] - D)
    return D, Dabs, err


# ----- the tail probability, with the inclusion-exclusion over the negative components of A

def _cond_tail(Kfun, Kwhite, x, A, order, den, tau0, lattice, mode, nG):
    d = len(x)
    xc = x.copy()
    if lattice:
        xc[A] -= 0.5
    try:
        tau, _ = spa_saddle(Kfun, xc, np.zeros(d) if tau0 is None else np.asarray(tau0, dtype=float))
    except SaddlepointError:
        if not lattice:
            raise
        raise DomainError("conditional: no saddlepoint at the continuity-corrected point; it is outside "
                          "the interior of the convex hull of the support (the probability is 0 or 1 there)")
    Nn = [j for j in A if tau[j] < 0]
    logf0B, sB, _ = den
    if not Nn:
        D, Dabs, err = _cond_numer(Kfun, Kwhite, tau, xc, A, order, logf0B, lattice, mode, nG)
        sj = np.array([sB[j // 2] for j in range(2 * order)])
        return D / sj, Dabs / np.abs(sj), err / np.abs(sj), tau
    B = [j for j in range(d) if j not in A]
    Apos = [j for j in A if j not in Nn]
    P = np.zeros(2 * order)
    abssum = np.zeros(2 * order)
    err = np.zeros(2 * order)
    for mask in range(2 ** len(Nn)):
        Tset = [Nn[i] for i in range(len(Nn)) if (mask >> i) & 1]
        sgn = -1 if len(Tset) % 2 else 1
        tail = sorted(Apos + Tset)
        if not tail:
            P += sgn
            abssum += 1
            continue
        keep = sorted(tail + B)
        k = len(keep)
        flip = np.ones(d)
        flip[Tset] = -1
        fk = flip[keep]

        def Km(nu, t, keep=keep, flip=flip, fk=fk):
            nf = [0] * d
            tf = np.zeros(d)
            for cc, j in enumerate(keep):
                nf[j] = nu[cc]
                tf[j] = t[cc]
            return float(np.prod(fk ** np.array(nu))) * Kfun(tuple(nf), flip * tf)

        Kw = None
        if Kwhite is not None:
            Bm = np.zeros((d, k))
            for cc, j in enumerate(keep):
                Bm[j, cc] = flip[j]
            Kw = (lambda nu, u, M, Bm=Bm: Kwhite(nu, u, Bm @ M))
        x2 = flip * x
        if lattice:
            x2[Tset] += 1
        A2 = [keep.index(j) for j in tail]
        P2, a2, e2, _ = _cond_tail(Km, Kw, x2[keep], A2, order, den, (flip * tau)[keep], lattice, mode, nG)
        P += sgn * P2
        abssum += a2
        err += e2
    return P, abssum, err, tau


def _spa_cond(Kfun, Kwhite, x, A, order, tail=True, density=True, lattice=False, sinh="saddle", nG=64,
              tau0=None):
    if order < 1:
        raise ValueError("spa_cond: order must be a positive integer")
    x = np.atleast_1d(np.asarray(x, dtype=float)).copy()
    if x.ndim != 1 or len(x) == 0 or not np.all(np.isfinite(x)):
        raise ValueError("spa_cond: x must be a nonempty finite vector")
    if sinh not in ("saddle", "origin"):
        raise ValueError("spa_cond: sinh must be 'saddle' or 'origin'")
    if nG < 8:
        raise ValueError("spa_cond: nG must be at least 8")
    d = len(x)
    A = _cond_set(A, d, "spa_cond")
    B = [j for j in range(d) if j not in A]
    b = len(B)
    t0 = None if tau0 is None else np.asarray(tau0, dtype=float)
    if b > 0:
        def KB(nu, s):
            nf = [0] * d
            tf = np.zeros(d)
            for cc, j in enumerate(B):
                nf[j] = nu[cc]
                tf[j] = s[cc]
            return Kfun(tuple(nf), tf)

        KwB = None
        if Kwhite is not None:
            EB = np.zeros((d, b))
            for cc, j in enumerate(B):
                EB[j, cc] = 1.0
            KwB = (lambda nu, u, M: Kwhite(nu, u, EB @ M))
        taut, _ = spa_saddle(KB, x[B], np.zeros(b) if t0 is None else t0[B])
        den = _cond_dens(KB, KwB, taut, x[B], order)
        tautf = np.zeros(d)
        tautf[B] = taut
    else:
        den = (0.0, np.ones(order), np.ones(order))
        tautf = np.zeros(d)
    status = "ok"

    def canc(v):
        return not np.all(np.isfinite(v)) or np.max(v) * np.finfo(float).eps > 1e-8

    if canc(den[2]):
        status = addstatus(status, "large cancellation in the denominator")
    f = np.full(order, np.nan)
    kappaf = np.full(order, np.nan)
    tauf = np.full(d, np.nan)
    if density:
        tauf, _ = spa_saddle(Kfun, x, np.zeros(d) if t0 is None else t0)
        logf0, s, kappaf = _cond_dens(Kfun, Kwhite, tauf, x, order)
        f = math.exp(logf0 - den[0]) * s / den[1]
        if canc(kappaf):
            status = addstatus(status, "large cancellation in the density")
    P = np.full(order, np.nan)
    kappaP = np.full(order, np.nan)
    err = np.full(order, np.nan)
    Pp = np.full(2 * order, np.nan)
    tau = np.full(d, np.nan)
    if tail:
        st = tauf if density else t0
        Pp, abssum, errp, tau = _cond_tail(Kfun, Kwhite, x, A, order, den, st, lattice, sinh, nG)
        P = Pp[1::2].copy()
        err = errp[1::2].copy()
        kappaP = abssum[1::2] / np.abs(P)
        if canc(kappaP):
            status = addstatus(status, "large cancellation in the numerator")
        if np.all(np.isfinite(err)) and err[-1] > 1e-8 * abs(P[-1]):
            status = addstatus(status, "integration not resolved")
    return dict(P=P, f=f, tau=tau, tauf=tauf, taut=tautf, kappaP=kappaP, kappaf=kappaf, kappaB=den[2],
                err=err, Pp=Pp, status=status)


def spa_cond(Kfun, x, A, order, lattice=False, sinh="saddle", nG=64, tau0=None, Kwhite=None):
    """The conditional expansions of Broda and Kan, "Conditional saddlepoint expansions of
    arbitrary order", of orders ``1..order``, from ``Kfun(nu, t)``, the derivative of order ``nu``
    (a ``d``-tuple) of the cumulant generating function at ``t`` (NaN outside the domain), at the
    point ``x``; ``A`` holds the (0-based) indices of the components whose tail is wanted, and the
    others, ``B``, are conditioned on. The counterpart of ``spa_cond.m`` of the paper and of
    ``spa_cond`` of the Julia package.

    Returns a dict with ``P`` (``P(X_A >= x_A | X_B = x_B)`` of orders ``1..order``: the numerator
    with the terms ``m <= 2M-1`` over the density expansion of ``X_B`` with ``M`` terms, relative
    error ``O(n^-M)``; saddlepoints with negative components in ``A`` by the inclusion-exclusion of
    Remark 1 of the paper, the components not kept in a term marginalized), ``f`` (the conditional
    density, the ratio of the density expansions of ``X`` and ``X_B`` with the terms ``m <= M-1``),
    ``tau`` (the saddlepoint at ``x``, continuity-corrected in the lattice case), ``tauf`` (that of
    the density), ``taut`` (the saddlepoint of the marginal of ``X_B``, embedded as
    ``(0_A, s_B)``), ``kappaP``, ``kappaf``, ``kappaB`` (cancellation indicators of the numerator,
    the inclusion-exclusion included, of the density of ``X`` and of that of ``X_B``), ``err`` (the
    change of ``P`` with half the quadrature nodes), ``Pp`` (``Pp[j]``, ``j = 0..2 order - 1``, the
    expansions with the terms up to ``n^(-j/2)``; ``P[M-1] = Pp[2M-1]``; ``Pp[0]`` is the leading
    term, for one tail component the conditional analogue ``(u/z) exp((u^2 - w^2)/2) Phibar(u)`` of
    the Hauschildt-Robinson approximation in the notation of Skovgaard 1987) and ``status``.

    ``lattice=True``: the components in ``A`` are integer-valued and so is ``x[A]`` (Theorem 3);
    ``P`` is ``P(S_A >= k_A | S_B = k_B)`` with the continuity correction (the saddlepoint at
    ``k_A - 1/2``) and the amplitude ``prod_j s_j/(2 sinh(s_j/2))``, expanded about the saddlepoint
    (``sinh='saddle'``) or about the origin, keeping ``s^(2M-2)`` at order ``M`` (``'origin'``,
    Kolassa 2003); ``f`` is ``P(S_A = k_A | S_B = k_B)``.

    The coefficients are taken in the canonical coordinates ``Y_A = X_A - C X_B``,
    ``Y_B = L^{-1} X_B`` of Section 5.2 of the paper, the tail components whitened in addition;
    the integrals factor into ``He_{nu_B}(0)`` times an integral over the orthant of the tail
    components, one-dimensional for one tail component (three Gauss-Legendre panels of ``nG``
    nodes, truncated where ``v^L exp(-u v - v^2/2)`` has fallen by ``e^-40``), in sequentially
    conditioned coordinates otherwise. ``Kwhite(nu, u, M)`` (derivative at ``u`` of ``K(M u)``,
    ``M`` possibly rectangular) supplies the coefficients in these coordinates; without it they
    are obtained by polynomial substitution. See the Julia package for the details."""
    return _spa_cond(Kfun, Kwhite, x, A, order, lattice=lattice, sinh=sinh, nG=nG, tau0=tau0)


def _cond_signs(name, d, A, lower, signs):
    sg = np.ones(d)
    if signs is None:
        if lower:
            sg[A] = -1
        return sg
    s = np.asarray(signs, dtype=float)
    if len(s) != len(A):
        raise ValueError(f"{name}: signs must have one entry per tail component ({len(A)})")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError(f"{name}: signs must be +-1")
    sg[A] = s
    return sg


def conditional_tailprob(K, x, A, order=3, lower=False, signs=None, lattice=False, sinh="saddle", nG=64):
    """Saddlepoint expansions of orders ``1..order`` of the conditional distribution function
    ``P(X_A >= x_A | X_B = x_B)`` (componentwise in ``A``) for the random vector with cumulant
    generating function ``K`` at ``x``; ``A`` holds the 0-based indices of the components whose
    tail is wanted (an integer or a sequence), the others, ``B``, are conditioned on (Broda and Kan,
    "Conditional saddlepoint expansions of arbitrary order", Theorem 1).

    Returns a dict with ``F`` (array over the orders: the numerator, the orthant-probability
    integral with the poles in the components of ``A`` only, with the terms of orders ``n^(-m/2)``,
    ``m <= 2M-1``, over the density expansion of ``X_B`` with ``M`` terms; relative error
    ``O(n^-M)``), ``tau`` (the saddlepoint at ``x``, corrected for lattice variables), ``taut``
    (that of the marginal of ``X_B``, embedded as ``(0_A, s_B)``), ``status`` and ``info``
    (``Fp``, the expansions with the terms up to ``n^(-j/2)``, ``Fp[0]`` the leading term;
    ``kappa`` and ``kappa_den``, the cancellation indicators of the numerator and the denominator;
    ``err``, the change with half the quadrature nodes; ``nG``). With ``B`` empty the expansion is
    that of :func:`~saddlepoint_expansions.tailprob`.

    ``lower=True`` gives ``P(X_A <= x_A | X_B = x_B)`` and ``signs`` (one +-1 per component of
    ``A``) a mixed corner. Saddlepoints with negative components in ``A`` are handled by the
    inclusion-exclusion of Remark 1 of the paper. ``lattice=True``: integer-valued components in
    ``A`` and integer ``x[A]``; ``F`` is ``P(S_A >= k_A | S_B = k_B)`` (Theorem 3) with the
    continuity correction and the amplitude ``prod_j s_j/(2 sinh(s_j/2))`` expanded about the
    saddlepoint (``sinh='saddle'``) or about the origin (``'origin'``, Kolassa 2003). All Taylor
    coefficients are generated by Taylor-mode differentiation directly in the canonical
    coordinates of Section 5.2 of the paper, in which the terms do not cancel. See
    :func:`spa_cond`."""
    if order < 1:
        raise ValueError("conditional_tailprob: order must be a positive integer")
    d = K.d
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    if len(xx) != d:
        raise ValueError(f"conditional_tailprob: x must have length {d}")
    Av = _cond_set(A, d, "conditional_tailprob")
    sg = _cond_signs("conditional_tailprob", d, Av, lower, signs)
    Kr = K if np.all(sg == 1) else reflect(K, sg)
    maxdeg = 2 * order + 1
    r = _spa_cond(kfun(Kr, maxdeg), kwhite(Kr, maxdeg), sg * xx, Av, order, density=False, lattice=lattice,
                  sinh=sinh, nG=nG)
    return dict(F=r["P"], tau=sg * r["tau"], taut=r["taut"], status=r["status"],
                info=dict(Fp=r["Pp"], kappa=r["kappaP"], kappa_den=r["kappaB"], err=r["err"], nG=nG))


def conditional_density(K, x, A, order=3, lattice=False):
    """Saddlepoint expansions of orders ``1..order`` of the conditional density ``f(x_A | x_B)`` of
    the components ``A`` (0-based) given the others, ``B`` (Theorem 2 of Broda and Kan,
    "Conditional saddlepoint expansions of arbitrary order"): the ratio of the density expansions of
    ``X`` at its saddlepoint and of ``X_B`` at its own, with the terms ``m <= M-1`` at order ``M``.
    The leading term is the double saddlepoint density approximation of Barndorff-Nielsen and Cox
    (1979). With ``lattice=True`` the same formula is ``P(S_A = k_A | S_B = k_B)`` (Theorem 3) at the
    uncorrected point; the keyword only records this interpretation. Returns a dict with ``f``,
    ``tau``, ``taut``, ``status`` and ``info`` (``kappa``, ``kappa_den``)."""
    if order < 1:
        raise ValueError("conditional_density: order must be a positive integer")
    d = K.d
    xx = np.atleast_1d(np.asarray(x, dtype=float))
    if len(xx) != d:
        raise ValueError(f"conditional_density: x must have length {d}")
    Av = _cond_set(A, d, "conditional_density")
    maxdeg = 2 * order + 1
    r = _spa_cond(kfun(K, maxdeg), kwhite(K, maxdeg), xx, Av, order, tail=False)
    return dict(f=r["f"], tau=r["tauf"], taut=r["taut"], status=r["status"],
                info=dict(kappa=r["kappaf"], kappa_den=r["kappaB"]))
