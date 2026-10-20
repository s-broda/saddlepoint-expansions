"""Cumulant generating functions as Python callables, their derivatives by Taylor-mode
differentiation, and transformations."""
from __future__ import annotations

import math

import numpy as np

from .multivariate import mi_list
from .taylor import DomainError, TPoly, get_algebra


class CGF:
    """A cumulant generating function ``K`` of a random vector of dimension ``d``.

    ``K`` is a callable ``K(t)`` for a numpy vector ``t`` of length ``d`` (also when ``d = 1``:
    write ``t[0]``). It must be written generically, with arithmetic and numpy functions
    (``np.log``, ``np.exp``, ``np.sqrt``, powers, ``np.sum``, products with constant matrices),
    because it is evaluated on vectors of truncated Taylor polynomials to generate its
    derivatives. Avoid ``math.log`` and the like, ``float(...)`` conversions, and library calls
    that require floats. Outside the domain it may raise a :class:`DomainError` (``np.log`` and
    ``np.sqrt`` of a negative constant term do this automatically) or return a non-finite value.

    ``domain`` is an optional predicate ``t -> bool`` for the domain of ``K``. It is needed when
    the formula is finite outside the true domain (``1/(1 - t)`` for ``t > 1``), and it is
    cheaper than an exception.

    ``taylor`` supplies the Taylor polynomials in closed form instead of by automatic
    differentiation: a function ``taylor(t, N, M)`` returning the polynomial of degree ``N`` in
    ``v`` of ``K(t + M v)`` as a :class:`~saddlepoint_expansions.taylor.TPoly` (``M = None`` for
    the identity; ``M`` may be rectangular), or ``None`` outside the domain; build it from a
    coefficient table with :func:`~saddlepoint_expansions.taylor.from_coefficients`. The library
    uses it where closed forms are faster (:func:`~saddlepoint_expansions.quadratic_forms_cgf`),
    and the transformations carry it along.

    A ``CGF`` is callable. See :func:`derivatives`, :func:`~saddlepoint_expansions.api.saddlepoint`,
    :func:`~saddlepoint_expansions.api.tailprob`, :func:`~saddlepoint_expansions.api.density`, and
    the transformations :func:`reflect`, :func:`marginal`, :func:`linear_combination`,
    :func:`sample_mean`, :func:`shift` and ``K1 + K2`` (sum of independent vectors).

    Example::

        K = CGF(lambda t: -3*np.log(1 - t[0]) - 2*np.log(1 - t[1]) - np.log(1 - t[0] - t[1]), 2,
                domain=lambda t: t[0] < 1 and t[1] < 1 and t[0] + t[1] < 1)
    """

    def __init__(self, K, d, domain=None, taylor=None):
        if int(d) < 1:
            raise ValueError("CGF: the dimension must be a positive integer")
        self.K = K
        self.d = int(d)
        self.domain = domain
        self.taylor = taylor

    def __call__(self, t):
        return self.K(t)

    def __repr__(self):
        return (f"CGF of dimension {self.d}" + (" with domain predicate" if self.domain is not None else "")
                + (" and closed-form Taylor polynomials" if self.taylor is not None else ""))

    def __add__(self, other):
        if not isinstance(other, CGF):
            return NotImplemented
        if self.d != other.d:
            raise ValueError("+: the cumulant generating functions must have the same dimension")
        K1, K2 = self.K, other.K
        d1, d2 = self.domain, other.domain
        dom = d2 if d1 is None else d1 if d2 is None else (lambda t: d1(t) and d2(t))
        tf = None
        if self.taylor is not None or other.taylor is not None:
            a, b = self, other

            def tf(t, N, M):
                p1 = _taylor(a, t, N, M)
                p2 = _taylor(b, t, N, M)
                return None if (p1 is None or p2 is None) else p1 + p2
        return CGF(lambda t: K1(t) + K2(t), self.d, domain=dom, taylor=tf)


def _indomain(K, t):
    return K.domain is None or bool(K.domain(t))


def _taylor(K, t, N, M=None):
    """Taylor polynomial of degree N in v of K(t + M v) (M = I when None), or None when t is
    outside the domain. The coefficient of v^nu is K_Y^(nu)(0)/nu! for K_Y(v) = K(t + M v)."""
    t = np.asarray(t, dtype=float)
    if not _indomain(K, t):
        return None
    if K.taylor is not None:
        p = K.taylor(t, N, M)
        if p is None or not np.isfinite(p.constant_term()):
            return None
        return p
    k = K.d if M is None else M.shape[1]
    alg = get_algebra(k, max(N, 1))
    v = np.empty(k, dtype=object)
    for i, p in enumerate(alg.variables()):
        v[i] = p
    arg = t + v if M is None else t + M @ v
    try:
        p = K.K(arg)
    except DomainError:
        return None
    if not isinstance(p, TPoly):
        p = np.asarray(p).item()
        if not np.isfinite(p):
            return None
        return alg.constant(float(p))
    if not np.isfinite(p.constant_term()):
        return None
    return p


def _fact(nu):
    f = 1.0
    for n in nu:
        f *= math.factorial(n)
    return f


def derivatives(K, t, N):
    """All derivatives ``K^(nu)(t)`` of orders ``|nu| <= N``, as a dict keyed by the
    multi-index tuple (``D[(0, 0)]`` is ``K(t)``, ``D[(1, 0)]`` is ``dK/dt_1``, ...), or
    ``None`` when ``t`` is outside the domain."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    if len(t) != K.d:
        raise ValueError(f"derivatives: t must have length {K.d}")
    if N < 0:
        raise ValueError("derivatives: N must be nonnegative")
    p = _taylor(K, t, N)
    if p is None:
        return None
    return {nu: p.coeff(nu) * _fact(nu) for nu in mi_list(K.d, N)}


def kfun(K, maxdeg):
    """``Kfun(nu, t)`` for the derivative interface, from a :class:`CGF`: the Taylor polynomial
    at the last ``t`` is memoized (degree 2 while only derivatives of order <= 2 are requested,
    as in the saddlepoint iteration, otherwise ``maxdeg``). Not thread-safe: one per task."""
    state = {"t": None, "N": -1, "p": None}

    def Kfun(nu, t):
        k = sum(nu)
        N = 2 if k <= 2 else max(maxdeg, k)
        tt = np.asarray(t, dtype=float)
        if state["t"] is None or not np.array_equal(state["t"], tt) or N > state["N"]:
            state["p"] = _taylor(K, tt, N)
            state["t"] = tt.copy()
            state["N"] = N
        p = state["p"]
        if p is None:
            return np.nan
        return p.coeff(nu) * _fact(nu)

    return Kfun


def kwhite(K, maxdeg):
    """``Kwhite(nu, u, M)`` for the derivative interface: derivatives at ``u`` of
    ``K_Y(u) = K(M u)`` from the Taylor polynomial of ``K(M u + M v)``, which the expansions
    evaluate in whitened coordinates; ``M`` may be rectangular. Memoized at the last (u, M)."""
    state = {"u": None, "M": None, "p": None}

    def Kwhite(nu, u, M):
        uu = np.asarray(u, dtype=float)
        MM = np.asarray(M, dtype=float)
        if state["u"] is None or not np.array_equal(state["u"], uu) or not np.array_equal(state["M"], MM):
            state["p"] = _taylor(K, MM @ uu, max(maxdeg, sum(nu)), MM)
            state["u"] = uu.copy()
            state["M"] = MM.copy()
        p = state["p"]
        if p is None:
            return np.nan
        return p.coeff(nu) * _fact(nu)

    return Kwhite


# ----- transformations

def reflect(K, signs=None):
    """Cumulant generating function of ``(s_1 X_1, ..., s_d X_d)`` for signs ``+-1``:
    ``K(s * t)``. ``reflect(K)`` negates every component."""
    s = np.full(K.d, -1.0) if signs is None else np.asarray(signs, dtype=float)
    if len(s) != K.d:
        raise ValueError(f"reflect: signs must have length {K.d}")
    if not np.all((s == 1) | (s == -1)):
        raise ValueError("reflect: signs must be +-1")
    return linear_combination(K, np.diag(s))


def _embed(s, idx, d):
    t = np.empty(d, dtype=object)
    z = s[0] * 0.0
    for j in range(d):
        t[j] = z
    for i, j in enumerate(idx):
        t[j] = s[i]
    return t


def marginal(K, idx):
    """Cumulant generating function of the components ``idx`` of ``X``."""
    ii = [int(i) for i in idx]
    if len(ii) == 0 or len(set(ii)) != len(ii) or not all(0 <= i < K.d for i in ii):
        raise ValueError(f"marginal: idx must be distinct indices in 0..{K.d - 1}")
    E = np.zeros((K.d, len(ii)))
    for c, j in enumerate(ii):
        E[j, c] = 1.0
    return linear_combination(K, E)


def linear_combination(K, B):
    """Cumulant generating function of ``Y = B'X`` for a ``d x k`` matrix ``B``: ``K(B u)``."""
    Bm = np.asarray(B, dtype=float)
    if Bm.ndim == 1:
        Bm = Bm.reshape(-1, 1)
    if Bm.shape[0] != K.d:
        raise ValueError(f"linear_combination: B must have {K.d} rows")
    Kf = K.K
    dom = None if K.domain is None else (lambda u, D=K.domain: D(Bm @ np.asarray(u, dtype=float)))
    tf = None
    if K.taylor is not None:
        tf = lambda u, N, M: _taylor(K, Bm @ u, N, Bm if M is None else Bm @ M)
    return CGF(lambda u: Kf(Bm @ u), Bm.shape[1], domain=dom, taylor=tf)


def sample_mean(K, n):
    """Cumulant generating function of the mean of ``n`` independent copies: ``n K(t/n)``."""
    if not n > 0:
        raise ValueError("sample_mean: n must be positive")
    Kf = K.K
    dom = None if K.domain is None else (lambda t, D=K.domain: D(t / n))
    tf = None
    if K.taylor is not None:
        def tf(t, N, M):
            p = _taylor(K, np.asarray(t, dtype=float) / n, N, (np.eye(K.d) if M is None else M) / n)
            return None if p is None else n * p
    return CGF(lambda t: n * Kf(t / n), K.d, domain=dom, taylor=tf)


def shift(K, c):
    """Cumulant generating function of ``X + c``: ``K(t) + c't``."""
    cc = np.atleast_1d(np.asarray(c, dtype=float))
    if len(cc) != K.d:
        raise ValueError(f"shift: c must have length {K.d}")
    Kf = K.K
    tf = None
    if K.taylor is not None:
        def tf(t, N, M):
            p = _taylor(K, t, N, M)
            if p is None:
                return None
            Mm = np.eye(K.d) if M is None else np.asarray(M, dtype=float)
            alg = get_algebra(Mm.shape[1], max(N, 1))
            lin = np.zeros(alg.M)
            lin[0] = float(cc @ np.asarray(t, dtype=float))
            cM = cc @ Mm
            for i in range(Mm.shape[1]):
                e = tuple(1 if j == i else 0 for j in range(Mm.shape[1]))
                lin[alg.index[e]] = cM[i]
            return p + TPoly(alg, lin)
    return CGF(lambda t: Kf(t) + np.sum(cc * t), K.d, domain=K.domain, taylor=tf)
