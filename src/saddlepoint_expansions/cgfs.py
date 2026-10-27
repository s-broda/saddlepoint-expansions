"""A library of cumulant generating functions, written generically for Taylor-mode
differentiation."""
from __future__ import annotations

import math

import numpy as np
from scipy.linalg import sqrtm

from .cgf import CGF
from .multivariate import mi_list
from .taylor import DomainError, TPoly, from_coefficients, sinhc


def normal_cgf(mu, Sigma):
    """``N(mu, Sigma)``: ``K(t) = mu't + t'Sigma t/2``. The expansions are exact at order one."""
    m = np.atleast_1d(np.asarray(mu, dtype=float))
    S = np.atleast_2d(np.asarray(Sigma, dtype=float))
    d = len(m)
    if S.shape != (d, d):
        raise ValueError(f"normal_cgf: Sigma must be {d} x {d}")
    return CGF(lambda t: np.sum(m * t) + np.sum(t * (S @ t)) / 2, d)


def gamma_cgf(a, b=1.0):
    """Gamma distribution with shape ``a`` and scale ``b`` (mean ``ab``): ``K(t) = -a log(1 - bt)``,
    ``t < 1/b``. ``gamma_cgf(1, b)`` is the exponential distribution with mean ``b``."""
    return CGF(lambda t: -a * np.log(1 - b * t[0]), 1, domain=lambda t: b * t[0] < 1)


def ncx2_cgf(m, lam=0.0):
    """Noncentral chi-squared ``chi^2_m(lam)``: ``K(t) = -(m/2) log(1 - 2t) + lam t/(1 - 2t)``,
    ``t < 1/2``."""
    return CGF(lambda t: -m / 2 * np.log(1 - 2 * t[0]) + lam * t[0] / (1 - 2 * t[0]), 1,
               domain=lambda t: t[0] < 0.5)


def exponential_sums_cgf(A, n=1.0, shift=None):
    """The mean of ``n`` independent copies of ``A'U + l``, ``U`` a vector of independent unit
    exponential variables, ``A`` an ``m x d`` matrix and ``shift = l`` (default 0):
    ``K(t) = -n sum_i log(1 - a_i't/n) + l't``, domain ``a_i't < n``. With ``n = 1`` this is
    the cumulant generating function of ``A'U + l`` itself (the examples of Wang 1990 and
    Kolassa and Li 2010 in the paper)."""
    Am = np.asarray(A, dtype=float)
    d = Am.shape[1]
    if not n > 0:
        raise ValueError("exponential_sums_cgf: n must be positive")
    if shift is None:
        K = lambda t: -n * np.sum(np.log(1 - Am @ t / n))
    else:
        l = np.asarray(shift, dtype=float)
        if len(l) != d:
            raise ValueError(f"exponential_sums_cgf: shift must have length {d}")
        K = lambda t: -n * np.sum(np.log(1 - Am @ t / n)) + np.sum(l * t)
    return CGF(K, d, domain=lambda t: bool(np.all(Am @ t < n)))


def _chol_lower(A):
    """Lower Cholesky factor in generic arithmetic (object arrays of polynomials); np.sqrt of a
    polynomial with a nonpositive constant term raises DomainError."""
    n = A.shape[0]
    L = np.empty((n, n), dtype=object)
    z = A[0, 0] * 0.0
    for i in range(n):
        for j in range(n):
            L[i, j] = z
    for j in range(n):
        v = A[j, j]
        for k in range(j):
            v = v - L[j, k] * L[j, k]
        if not isinstance(v, (int, float, np.floating)):
            L[j, j] = np.sqrt(v)
        else:
            if not v > 0:
                raise DomainError("quadratic_forms_cgf: I - 2C is not positive definite")
            L[j, j] = np.sqrt(v)
        for i in range(j + 1, n):
            w = A[i, j]
            for k in range(j):
                w = w - L[i, k] * L[j, k]
            L[i, j] = w / L[j, j]
    return L


def _chol_solve(L, b):
    n = len(b)
    y = [None] * n
    for i in range(n):
        v = b[i]
        for k in range(i):
            v = v - L[i, k] * y[k]
        y[i] = v / L[i, i]
    w = [None] * n
    for i in range(n - 1, -1, -1):
        v = y[i]
        for k in range(i + 1, n):
            v = v - L[k, i] * w[k]
        w[i] = v / L[i, i]
    return w


def _qf_taylor(B, m, scale, t, N, M):
    """Taylor polynomial of degree N in v of K(t + M v) for the quadratic forms w'B_j w,
    w ~ N(m, I), K = -1/2 log det(I - 2C) + 1/2 m'{(I - 2C)^-1 - I}m, times scale, in closed
    form by the trace recursion in the directions sum_j M_ji B_j (the MATLAB qf_taylor of R. Kan):
    with D = (I - 2C)^-1 = L L', G_i = L' Dir_i L and y = L'm, the coefficient of v^a is
    scale 2^(|a|-1)/|a| {tr F_a + |a| y'F_a y}, F_a the symmetrized sum of the products of the G_i
    in every order; F is formed up to degree ceil(N/2) and the higher traces by splitting.
    None outside the domain."""
    d = len(B)
    n = B[0].shape[0]
    t = np.asarray(t, dtype=float)
    Mm = np.eye(d) if M is None else np.asarray(M, dtype=float)
    k = Mm.shape[1]
    N = max(N, 1)
    C = sum(t[j] * B[j] for j in range(d))
    try:
        R = np.linalg.cholesky(np.eye(n) - 2 * C).T          # I - 2C = R'R
    except np.linalg.LinAlgError:
        return None
    L = np.linalg.inv(R)                                     # D = L L'
    y = L.T @ m
    coeffs = {(0,) * k: scale * (-np.sum(np.log(np.diag(R))) + (y @ y - m @ m) / 2)}
    G = []
    for i in range(k):
        Dir = np.zeros((n, n))
        for j in range(d):
            if Mm[j, i] != 0:
                Dir = Dir + Mm[j, i] * B[j]
        Gi = L.T @ Dir @ L
        G.append((Gi + Gi.T) / 2)
    h = -(-N // 2)
    hasmean = bool(np.any(m != 0))
    F = {}
    Fy = {}
    for a in mi_list(k, N):
        l = sum(a)
        if l == 0:
            continue
        if l <= h:
            if l == 1:
                Fa = G[a.index(1)]
            else:
                Fa = np.zeros((n, n))
                for i in range(k):
                    if a[i] > 0:
                        b = list(a)
                        b[i] -= 1
                        Fa = Fa + G[i] @ F[tuple(b)]
                Fa = (Fa + Fa.T) / 2
            F[a] = Fa
            val = np.trace(Fa)
            if hasmean:
                Fy[a] = Fa @ y
                val += l * (y @ Fy[a])
            coeffs[a] = scale * 2.0 ** (l - 1) / l * val
        else:
            kk = l - h
            tval = 0.0
            mq = 0.0
            for g in mi_list(k, kk):
                if sum(g) != kk or any(gi > ai for gi, ai in zip(g, a)):
                    continue
                r = tuple(ai - gi for ai, gi in zip(a, g))
                tval += np.sum(F[g] * F[r])
                if hasmean:
                    mq += Fy[g] @ Fy[r]
            coeffs[a] = scale * 2.0 ** (l - 1) / l * (tval + l * mq)
    return from_coefficients(k, N, coeffs)


def quadratic_forms_cgf(Mats, Sigma=None, mu=None, scale=1.0, closedform=True):
    """Joint cumulant generating function of the quadratic forms ``Q_i = z'A_i z``,
    ``Mats = [A_1, ..., A_d]`` (symmetric ``p x p``), in ``z ~ N(mu, Sigma)`` (default standard
    normal): ``K(t) = -1/2 log det(I - 2C) + nu'C(I - 2C)^-1 nu``, ``C = sum_i t_i S^1/2 A_i S^1/2``,
    ``nu = S^-1/2 mu``, multiplied by ``scale`` (the sum of ``scale`` independent copies when
    ``scale`` is an integer). The domain is where ``I - 2C`` is positive definite. The
    determinant and the solve use a Cholesky factorization in generic arithmetic; the cost of
    one evaluation with derivatives is about ``p^3/6`` polynomial products, so this is for
    small and moderate ``p``. With ``closedform=True`` (the default) the Taylor polynomials,
    including the whitened ones, are instead computed in closed form by the trace recursion of
    the paper in the whitened directions (the ``taylor`` option of
    :class:`~saddlepoint_expansions.CGF`), so that large ``p`` costs matrix products rather than
    polynomial products; ``closedform=False`` differentiates the generic formula."""
    d = len(Mats)
    if d < 1:
        raise ValueError("quadratic_forms_cgf: Mats must be nonempty")
    p = np.asarray(Mats[0]).shape[0]
    if Sigma is None:
        B = [np.asarray((M + np.asarray(M).T) / 2, dtype=float) for M in Mats]
        nu = None if mu is None else np.asarray(mu, dtype=float)
    else:
        Sh = np.real(sqrtm(np.asarray(Sigma, dtype=float)))
        B = [Sh @ ((np.asarray(M) + np.asarray(M).T) / 2) @ Sh for M in Mats]
        B = [(Bi + Bi.T) / 2 for Bi in B]
        nu = None if mu is None else np.linalg.solve(Sh, np.asarray(mu, dtype=float))
    Ip = np.eye(p)

    def K(t):
        C = sum(t[i] * B[i] for i in range(d))
        Q = Ip - 2 * C
        L = _chol_lower(Q)
        v = -sum(np.log(L[j, j]) for j in range(p))
        if nu is not None:
            Cnu = C @ nu
            w = _chol_solve(L, nu)
            v = v + sum(Cnu[i] * w[i] for i in range(p))
        return v if scale == 1 else scale * v

    def dom(t):
        Q = Ip - 2 * sum(t[i] * B[i] for i in range(d))
        try:
            np.linalg.cholesky((Q + Q.T) / 2)
            return True
        except np.linalg.LinAlgError:
            return False

    m0 = np.zeros(p) if nu is None else np.asarray(nu, dtype=float)
    tf = (lambda t, N, M: _qf_taylor(B, m0, scale, t, N, M)) if closedform else None
    return CGF(K, d, domain=dom, taylor=tf)


def wishart_diag_cgf(V, n):
    """The diagonal of a Wishart matrix ``W ~ W_d(n, V)``: ``d`` quadratic forms ``z'e_i e_i'z``
    in ``z ~ N(0, V)``, summed over ``n`` independent copies (:func:`quadratic_forms_cgf` with
    ``scale = n``)."""
    V = np.asarray(V, dtype=float)
    d = V.shape[0]
    E = [np.outer(np.eye(d)[:, i], np.eye(d)[:, i]) for i in range(d)]
    return quadratic_forms_cgf(E, Sigma=V, scale=n)


def common_component_cgf(m, lam=None):
    """``Q_j = X_j + X_0``, ``j = 1..d``, with independent ``X_i ~ chi^2_{m_i}(lam_i)``,
    ``m = [m_1, ..., m_d, m_0]`` and ``lam`` likewise: ``K(t) = sum_j K_j(t_j) + K_0(sum_j t_j)``
    (the pairs of quadratic forms of the paper, and the trivariate example)."""
    mm = np.asarray(m, dtype=float)
    d = len(mm) - 1
    if d < 1:
        raise ValueError("common_component_cgf: m must have at least two entries")
    ll = np.zeros(d + 1) if lam is None else np.asarray(lam, dtype=float)
    if len(ll) != d + 1:
        raise ValueError("common_component_cgf: lam must have the length of m")

    def k1(u, mi, li):
        return -mi / 2 * np.log(1 - 2 * u) + li * u / (1 - 2 * u)

    def K(t):
        v = k1(t[0], mm[0], ll[0])
        for j in range(1, d):
            v = v + k1(t[j], mm[j], ll[j])
        return v + k1(np.sum(t), mm[d], ll[d])

    return CGF(K, d, domain=lambda t: bool(np.all(t < 0.5) and np.sum(t) < 0.5))


def uniform_cgf(a=-1.0, b=1.0):
    """Uniform distribution on ``[a, b]``: ``K(t) = log sinhc((b - a)t/2) + (a + b)t/2``, with
    ``sinhc(x) = sinh(x)/x`` (:func:`~saddlepoint_expansions.taylor.sinhc`, which handles the
    removable singularity at ``t = 0``, where the saddlepoint iteration starts). Finite for all
    ``t``."""
    if not b > a:
        raise ValueError("uniform_cgf: b must exceed a")
    h = (b - a) / 2
    m = (a + b) / 2
    return CGF(lambda t: np.log(sinhc(h * t[0])) + m * t[0], 1)


# ----- the Dickey-Fuller t-ratio (Bao, Broda and Kan, "Inversion formulas and saddlepoint
# expansions for Studentized means", Section 7)

def _entire_coeffs(z0, N, kind):
    """Taylor coefficient k = 0..N at z0 of C = sum z^j/(2j)!, S = sum z^j/(2j+1)! or
    Q = -sum z^j/(2j+2)!: sum_j binom(j+k, k) z0^j/(2j+2k+off)!, summed by the ratio of
    consecutive terms (no overflow, no cancellation for z0 > 0)."""
    off, sign = {"C": (0, 1.0), "S": (1, 1.0), "Q": (2, -1.0)}[kind]
    out = np.empty(N + 1)
    for k in range(N + 1):
        term = sign / math.factorial(2 * k + off)
        tot = term
        j = 0
        while True:
            j += 1
            term *= z0 * (j + k) / j / ((2 * j + 2 * k + off) * (2 * j + 2 * k + off - 1))
            tot += term
            if j > 2 and abs(term) < 1e-18 * abs(tot) and abs(z0) < (2 * j + 2 * k + off) * (2 * j + 2 * k + off - 1) * j / (j + k):
                break
        out[k] = tot
    return out


def _entire_csq(z, nser=40):
    """cosh(mu), (1 - cosh mu)/z and sinh(mu)/mu with mu = sqrt(z), as entire functions of z, for
    floats, complex numbers and polynomials. For a polynomial the three functions are expanded
    about its constant term by their Taylor coefficients in z (:func:`_entire_coeffs`), which is
    exact to rounding; for numbers they are evaluated by their power series near z = 0 (where
    (1 - cosh mu)/z cancels), by cosh and sinh of the real square root for z > 0, and by cos and
    sin for z < 0."""
    if isinstance(z, TPoly):
        z0 = z.constant_term()
        N = z.alg.N
        return (z._compose(_entire_coeffs(z0, N, "C")), z._compose(_entire_coeffs(z0, N, "Q")),
                z._compose(_entire_coeffs(z0, N, "S")))
    z0 = z
    if abs(z0) < 1e-2:
        C = 0 * z + 1 / math.factorial(2 * nser)
        S = 0 * z + 1 / math.factorial(2 * nser + 1)
        Q = 0 * z - 1 / math.factorial(2 * nser + 2)
        for j in range(nser - 1, -1, -1):
            C = C * z + 1 / math.factorial(2 * j)
            S = S * z + 1 / math.factorial(2 * j + 1)
            Q = Q * z - 1 / math.factorial(2 * j + 2)
        return C, Q, S
    if np.iscomplexobj(z0):
        mu = np.sqrt(z)
        C = np.cosh(mu)
        S = np.sinh(mu) / mu
    elif z0 > 0:
        mu = np.sqrt(z)
        C = np.cosh(mu)
        S = np.sinh(mu) / mu
    else:
        mu = np.sqrt(-z)
        C = np.cos(mu)
        S = np.sin(mu) / mu
    return C, (1 - C) / z, S


def dickey_fuller_cgf(model, c=0.0, window=8.0):
    """The joint cumulant generating function of ``(U, V)``, ``U = int ytilde dy`` and
    ``V = int ytilde^2``, for the near-unit-root diffusion ``dy = c y dt + dW`` on ``[0, 1]``
    with ``y(0) = 0``, where ``ytilde`` is ``y`` (``model=1``, no deterministic terms), ``y``
    demeaned (``model=2``, a constant) or ``y`` detrended (``model=3``, a constant and a linear
    trend). The limiting Dickey-Fuller t-ratio is ``U/sqrt(V)``, so that
    :func:`~saddlepoint_expansions.studentized_tailprob` and
    :func:`~saddlepoint_expansions.studentized_density` give its limiting distribution and
    density under the unit root (``c = 0``) and under the local alternatives ``c < 0``. The
    functions are those of Nabeya and Tanaka (1990) and Nabeya and Sorensen (1994):
    ``K(s, t) = -s/2 + e - log D(s, t)/2`` with ``D`` built from ``cosh mu``, ``sinh mu/mu`` and
    ``(1 - cosh mu)/mu^2``, ``mu^2 = c^2 - 2t``, as entire functions of ``mu^2``. Near
    ``mu^2 = 0``, for ``|mu^2| < window``, ``D`` is evaluated by its Taylor series in ``mu^2``
    (its closed form divides by ``mu^2`` or ``mu^6`` and cancels there)."""
    if model not in (1, 2, 3):
        raise ValueError("dickey_fuller_cgf: model must be 1, 2 or 3")
    from ._df_series import D1_SERIES, D2_SERIES
    cc = float(c)

    def K(t):
        s, tt = t[0], t[1]
        z = cc * cc - 2 * tt
        C, Q, S = _entire_csq(z)
        e = math.exp(cc)
        if model == 1:
            D = C - (cc + s) * S
            return -s / 2 - cc / 2 - np.log(D) / 2
        lam = -s
        lx = tt
        z0 = z.constant_term() if isinstance(z, TPoly) else z
        if model == 2:
            if abs(z0) < window:
                D = sum(D1_SERIES[j](lam, cc) * z ** j for j in range(len(D1_SERIES))) * e
            else:
                A = 2 * cc * cc * lam - 4 * cc * lx - 2 * lam * lam
                B = -cc ** 3 + cc * cc * lam - 2 * lx - lam * lam
                D = (cc * cc * (1 - z * Q) + A * Q + B * S) / z * e
        else:
            if abs(z0) < window:
                D = sum(D2_SERIES[j](lam, cc) * z ** j for j in range(len(D2_SERIES))) * e
            else:
                c2, c3, c4, c5, c6, c7 = cc ** 2, cc ** 3, cc ** 4, cc ** 5, cc ** 6, cc ** 7
                P1 = c6 - 12 * c4 * lam + 2 * lx * (-c4 + 12 * c3 - 12 * c2) + 48 * lam * lam * (c2 - 3 * cc + 3)
                P2 = (4 * lam * (-c6 + 6 * c4) + 16 * lam * lam * (c4 - 3 * c3 - 3 * c2 + 18 * cc - 18)
                      + 8 * lx * (c5 - 3 * c4 + c4 * lam - 4 * lam * lam * (c2 - 3 * cc + 3)) + 16 * lx * lx * (-c3 + 3 * c2 - 6 * cc + 6))
                P3 = (-c7 + lam * (c6 + 24 * c4) + 4 * lam * lam * (-c4 + 3 * c3 - 27 * c2 + 72 * cc - 72)
                      + 2 * lx * (c5 - 4 * c4 - 12 * c3 + 12 * c2 - c4 * lam + 4 * lam * lam * (c2 - 3 * cc + 3)) + 16 * lx * lx * (c2 - 3 * cc + 3))
                D = ((1 - z * Q) * P1 + Q * P2 + S * P3) / z ** 3 * e
        return -s / 2 - np.log(D) / 2

    return CGF(K, 2)
