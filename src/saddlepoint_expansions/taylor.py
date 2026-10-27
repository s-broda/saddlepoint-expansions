"""Truncated multivariate Taylor polynomials (Taylor-mode automatic differentiation).

A :class:`TPoly` is a polynomial in ``d`` variables truncated at total degree ``N``, with
``float64`` (or ``complex128``) coefficients stored in the graded monomial order of its
:class:`Algebra`. The
arithmetic operations and the elementary functions (``exp``, ``log``, ``sqrt``, powers, division,
the trigonometric and hyperbolic functions, ``arctan``, and ``sincu``/``sinhc`` for ``sin(x)/x``
and ``sinh(x)/x`` with their removable singularity)
propagate the coefficients exactly (up to rounding), so that evaluating a function written in
ordinary arithmetic on the variables of an algebra returns its Taylor polynomial, whose
coefficient of ``u^nu`` is the derivative of order ``nu`` divided by ``nu!``.

Numpy ufuncs work on :class:`TPoly` objects (``np.log(p)``, ``np.exp(p)``, ``np.sqrt(p)``, and
the arithmetic ufuncs), also elementwise on object arrays of them, so that a cumulant generating
function written with numpy for float arguments usually works unchanged.
"""
from __future__ import annotations

import cmath
import math
from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np


class DomainError(ValueError):
    """Raised by ``log``, ``sqrt`` and non-integer powers of a polynomial whose constant term
    is outside the domain of the function (negative, or zero for ``log``)."""


def compositions(m, p):
    """All nonnegative integer ``p``-tuples with sum ``m``, first component descending (the
    order of the Julia package)."""
    if p == 1:
        return [(m,)]
    out = []
    for k in range(m, -1, -1):
        for r in compositions(m - k, p - 1):
            out.append((k,) + r)
    return out


@lru_cache(maxsize=None)
def get_algebra(d, N):
    """The algebra of ``d`` variables truncated at degree ``N`` (cached)."""
    return Algebra(d, N)


class Algebra:
    """Index tables of the truncated algebra of ``d`` variables and total degree ``N``."""

    def __init__(self, d, N):
        if d < 1 or N < 0:
            raise ValueError("Algebra: d must be positive and N nonnegative")
        self.d = int(d)
        self.N = int(N)
        exps = []
        for k in range(N + 1):
            exps.extend(compositions(k, d))
        self.exps = exps
        self.index = {nu: i for i, nu in enumerate(exps)}
        self.M = len(exps)
        self.deg = np.array([sum(nu) for nu in exps])
        self.degstart = np.searchsorted(self.deg, np.arange(N + 2))   # first index of each degree
        # product table: pairs (i, j) with deg_i + deg_j <= N and the index of the product monomial
        I = []
        J = []
        Kk = []
        E = np.array(exps)
        for i in range(self.M):
            di = self.deg[i]
            jmax = self.degstart[N - di + 1]
            if jmax == 0:
                continue
            Ej = E[:jmax]
            tgt = Ej + E[i]
            for j in range(jmax):
                I.append(i)
                J.append(j)
                Kk.append(self.index[tuple(tgt[j])])
        self.pI = np.array(I, dtype=np.intp)
        self.pJ = np.array(J, dtype=np.intp)
        self.pK = np.array(Kk, dtype=np.intp)

    def variables(self):
        """The ``d`` variables as polynomials."""
        out = []
        for i in range(self.d):
            c = np.zeros(self.M)
            e = tuple(1 if j == i else 0 for j in range(self.d))
            c[self.index[e]] = 1.0
            out.append(TPoly(self, c))
        return out

    def constant(self, value):
        c = np.zeros(self.M)
        c[0] = value
        return TPoly(self, c)

    def mul(self, a, b):
        w = a[self.pI] * b[self.pJ]
        if np.iscomplexobj(w):
            return (np.bincount(self.pK, weights=w.real, minlength=self.M)
                    + 1j * np.bincount(self.pK, weights=w.imag, minlength=self.M))
        return np.bincount(self.pK, weights=w, minlength=self.M)


class TPoly:
    """A truncated multivariate Taylor polynomial; see the module docstring."""

    __slots__ = ("alg", "c")
    __array_priority__ = 1000

    def __init__(self, alg, c):
        self.alg = alg
        self.c = c

    # ----- access
    @property
    def order(self):
        return self.alg.N

    def constant_term(self):
        return self.c[0].item()

    def coeff(self, nu):
        """The coefficient of ``u^nu`` (``nu`` a tuple), which is ``f^(nu)(0)/nu!``."""
        return self.c[self.alg.index[tuple(nu)]].item()

    def derivative(self, nu):
        """The derivative of order ``nu`` at the expansion point: ``nu! * coeff(nu)``."""
        f = 1
        for n in nu:
            f *= math.factorial(n)
        return f * self.coeff(nu)

    def __repr__(self):
        return f"TPoly(d={self.alg.d}, N={self.alg.N}, constant={self.c[0]!r})"

    def __float__(self):
        raise TypeError("a TPoly cannot be converted to a float; use constant_term()")

    def __bool__(self):
        return bool(np.any(self.c != 0))

    # comparisons compare the constant terms, so that code with tests on the argument can be
    # differentiated (as in the MATLAB counterpart)
    def _c0cmp(self, other, op):
        if isinstance(other, TPoly):
            other = other.c[0]
        return op(self.c[0], other)

    def __lt__(self, other):
        return self._c0cmp(other, lambda a, b: a < b)

    def __le__(self, other):
        return self._c0cmp(other, lambda a, b: a <= b)

    def __gt__(self, other):
        return self._c0cmp(other, lambda a, b: a > b)

    def __ge__(self, other):
        return self._c0cmp(other, lambda a, b: a >= b)

    def __eq__(self, other):
        return self._c0cmp(other, lambda a, b: a == b)

    def __ne__(self, other):
        return self._c0cmp(other, lambda a, b: a != b)

    __hash__ = None

    # ----- helpers
    def _coerce(self, other):
        if isinstance(other, TPoly):
            if other.alg is not self.alg:
                raise ValueError("TPoly: operands belong to different algebras")
            return other.c, True
        if isinstance(other, (int, float, np.integer, np.floating)):
            return float(other), False
        if isinstance(other, (complex, np.complexfloating)):
            return complex(other), False
        return NotImplemented, None

    def _ccopy(self, o):
        """A copy of the coefficients, promoted to complex when the scalar o is complex."""
        return self.c.astype(np.result_type(self.c, o))

    # ----- arithmetic
    def __neg__(self):
        return TPoly(self.alg, -self.c)

    def __pos__(self):
        return self

    def __add__(self, other):
        o, isp = self._coerce(other)
        if isp is None:
            return NotImplemented
        if isp:
            return TPoly(self.alg, self.c + o)
        c = self._ccopy(o)
        c[0] += o
        return TPoly(self.alg, c)

    __radd__ = __add__

    def __sub__(self, other):
        o, isp = self._coerce(other)
        if isp is None:
            return NotImplemented
        if isp:
            return TPoly(self.alg, self.c - o)
        c = self._ccopy(o)
        c[0] -= o
        return TPoly(self.alg, c)

    def __rsub__(self, other):
        o, isp = self._coerce(other)
        if isp is None:
            return NotImplemented
        c = -self._ccopy(o)
        c[0] += o
        return TPoly(self.alg, c)

    def __mul__(self, other):
        o, isp = self._coerce(other)
        if isp is None:
            return NotImplemented
        if isp:
            return TPoly(self.alg, self.alg.mul(self.c, o))
        return TPoly(self.alg, self.c * o)

    __rmul__ = __mul__

    def __truediv__(self, other):
        o, isp = self._coerce(other)
        if isp is None:
            return NotImplemented
        if isp:
            return self * TPoly(self.alg, o).reciprocal()
        return TPoly(self.alg, self.c / o)

    def __rtruediv__(self, other):
        o, isp = self._coerce(other)
        if isp is None:
            return NotImplemented
        return self.reciprocal() * o

    def __pow__(self, a):
        if isinstance(a, TPoly):
            return (a * self.log()).exp()
        if isinstance(a, (int, np.integer)) and a >= 0:
            a = int(a)
            result = self.alg.constant(1.0)
            base = self
            while a:
                if a & 1:
                    result = result * base
                a >>= 1
                if a:
                    base = base * base
            return result
        a = float(a)
        c0 = self.c[0]
        iscomplex = np.iscomplexobj(c0)
        if not iscomplex and c0 <= 0:
            if c0 == 0 or a != int(a):
                raise DomainError("TPoly: power of a polynomial with nonpositive constant term")
        if iscomplex and c0 == 0:
            raise DomainError("TPoly: power of a polynomial with zero constant term")
        # f(x) = x^a: f^(k)(c0)/k! = binom(a, k) c0^(a-k)  (principal branch for complex c0)
        N = self.alg.N
        coefs = np.empty(N + 1, dtype=complex if iscomplex else float)
        b = 1.0
        for k in range(N + 1):
            coefs[k] = b * c0 ** (a - k)
            b *= (a - k) / (k + 1)
        return self._compose(coefs)

    def __rpow__(self, base):
        return (self * math.log(float(base))).exp()

    def reciprocal(self):
        c0 = self.c[0]
        if c0 == 0:
            raise DomainError("TPoly: division by a polynomial with zero constant term")
        N = self.alg.N
        coefs = np.array([(-1) ** k / c0 ** (k + 1) for k in range(N + 1)])
        return self._compose(coefs)

    def _compose(self, coefs):
        """sum_k coefs[k] q^k with q = self - constant term (the Taylor series of a univariate
        function through its derivatives at the constant term)."""
        alg = self.alg
        q = self.c.copy()
        q[0] = 0.0
        out = np.zeros(alg.M, dtype=np.result_type(q, coefs))
        out[0] = coefs[0]
        qk = q
        for k in range(1, alg.N + 1):
            if k > 1:
                qk = alg.mul(qk, q)
            out += coefs[k] * qk
        return TPoly(alg, out)

    # ----- elementary functions (also reachable as numpy ufuncs)
    def exp(self):
        c0 = self.c[0]
        e = cmath.exp(c0) if np.iscomplexobj(c0) else math.exp(c0)
        N = self.alg.N
        coefs = np.array([e / math.factorial(k) for k in range(N + 1)])
        return self._compose(coefs)

    def log(self):
        c0 = self.c[0]
        iscomplex = np.iscomplexobj(c0)
        if iscomplex:
            if c0 == 0:
                raise DomainError("TPoly: log of a polynomial with zero constant term")
        elif not c0 > 0:
            raise DomainError("TPoly: log of a polynomial with nonpositive constant term")
        N = self.alg.N
        coefs = np.empty(N + 1, dtype=complex if iscomplex else float)
        coefs[0] = cmath.log(c0) if iscomplex else math.log(c0)
        for k in range(1, N + 1):
            coefs[k] = (-1) ** (k + 1) / (k * c0 ** k)
        return self._compose(coefs)

    def log1p(self):
        return (self + 1.0).log()

    def expm1(self):
        return self.exp() - 1.0

    def sqrt(self):
        c0 = self.c[0]
        if not np.iscomplexobj(c0) and not c0 > 0:
            raise DomainError("TPoly: sqrt of a polynomial with nonpositive constant term")
        return self ** 0.5

    def square(self):
        return self * self

    def __abs__(self):
        """|p| as the polynomial with the sign of the constant term (as in TaylorSeries.jl);
        undefined, and an error, when the constant term is zero."""
        c0 = self.c[0]
        if c0 == 0.0:
            raise DomainError("TPoly: abs of a polynomial with zero constant term")
        return self if c0 > 0 else -self

    def absolute(self):
        return abs(self)

    # trigonometric and hyperbolic functions: Taylor coefficients f^(k)(c0)/k! at the constant term
    def _cycle(self, vals):
        N = self.alg.N
        coefs = np.array([vals[k % 4] / math.factorial(k) for k in range(N + 1)])
        return self._compose(coefs)

    def sin(self):
        c0 = self.c[0]
        s, c = math.sin(c0), math.cos(c0)
        return self._cycle((s, c, -s, -c))

    def cos(self):
        c0 = self.c[0]
        s, c = math.sin(c0), math.cos(c0)
        return self._cycle((c, -s, -c, s))

    def tan(self):
        return self.sin() / self.cos()

    def sinh(self):
        c0 = self.c[0]
        s, c = math.sinh(c0), math.cosh(c0)
        return self._cycle((s, c, s, c))

    def cosh(self):
        c0 = self.c[0]
        s, c = math.sinh(c0), math.cosh(c0)
        return self._cycle((c, s, c, s))

    def tanh(self):
        return self.sinh() / self.cosh()

    def arctan(self):
        # f' (1 + x^2) = 1 differentiated k times: f^(k+1) = -(2 c k f^(k) + k (k-1) f^(k-1))/(1 + c^2)
        c0 = self.c[0]
        N = self.alg.N
        f = np.empty(N + 1)
        f[0] = math.atan(c0)
        if N >= 1:
            f[1] = 1.0 / (1 + c0 * c0)
        for k in range(1, N):
            f[k + 1] = -(2 * c0 * k * f[k] + k * (k - 1) * f[k - 1]) / (1 + c0 * c0)
        coefs = np.array([f[k] / math.factorial(k) for k in range(N + 1)])
        return self._compose(coefs)

    def sincu(self):
        """sin(x)/x (unnormalized), with the removable singularity at a zero constant term."""
        if self.c[0] == 0.0:
            N = self.alg.N
            coefs = np.array([((-1) ** (k // 2)) / math.factorial(k + 1) if k % 2 == 0 else 0.0
                              for k in range(N + 1)])
            return self._compose(coefs)
        return self.sin() / self

    def sinhc(self):
        """sinh(x)/x, with the removable singularity at a zero constant term."""
        if self.c[0] == 0.0:
            N = self.alg.N
            coefs = np.array([1.0 / math.factorial(k + 1) if k % 2 == 0 else 0.0 for k in range(N + 1)])
            return self._compose(coefs)
        return self.sinh() / self

    def negative(self):
        return -self

    def reciprocal_ufunc(self):
        return self.reciprocal()

    def conjugate(self):
        return self

    # ----- numpy interoperability
    _UNARY = {"exp": "exp", "log": "log", "sqrt": "sqrt", "log1p": "log1p", "expm1": "expm1",
              "square": "square", "negative": "negative", "positive": "__pos__",
              "reciprocal": "reciprocal", "conjugate": "conjugate", "sin": "sin", "cos": "cos",
              "tan": "tan", "sinh": "sinh", "cosh": "cosh", "tanh": "tanh", "arctan": "arctan",
              "absolute": "absolute"}
    _BINARY = {"add": "__add__", "subtract": "__sub__", "multiply": "__mul__",
               "true_divide": "__truediv__", "divide": "__truediv__", "power": "__pow__"}

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method != "__call__" or kwargs.get("out") is not None:
            return NotImplemented
        name = ufunc.__name__
        if name in self._UNARY and len(inputs) == 1:
            return getattr(self, self._UNARY[name])()
        if name in self._BINARY and len(inputs) == 2:
            a, b = inputs
            if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
                # broadcast elementwise, producing an object array of polynomials
                op = {"add": lambda x, y: x + y, "subtract": lambda x, y: x - y,
                      "multiply": lambda x, y: x * y, "true_divide": lambda x, y: x / y,
                      "divide": lambda x, y: x / y, "power": lambda x, y: x ** y}[name]
                def wrap(v):                # 0-d object array: no ufunc dispatch on the scalar
                    if isinstance(v, np.ndarray):
                        return v.astype(object)
                    w = np.empty((), dtype=object)
                    w[()] = v
                    return w
                return np.frompyfunc(op, 2, 1)(wrap(a), wrap(b))
            if a is self:
                r = getattr(self, self._BINARY[name])(b)
            else:
                r = getattr(self, {"add": "__radd__", "subtract": "__rsub__", "multiply": "__rmul__",
                                   "true_divide": "__rtruediv__", "divide": "__rtruediv__",
                                   "power": "__rpow__"}[name])(a)
            return r
        return NotImplemented


def exp(x):
    return x.exp() if isinstance(x, TPoly) else np.exp(x)


def log(x):
    return x.log() if isinstance(x, TPoly) else np.log(x)


def sqrt(x):
    return x.sqrt() if isinstance(x, TPoly) else np.sqrt(x)


def sincu(x):
    """``sin(x)/x`` (unnormalized, unlike ``np.sinc``), equal to 1 at 0; works on floats, arrays
    and polynomials, where the removable singularity at a zero constant term is handled."""
    if isinstance(x, TPoly):
        return x.sincu()
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x == 0, 1.0, np.sin(x) / np.where(x == 0, 1.0, x))


def sinhc(x):
    """``sinh(x)/x``, equal to 1 at 0; works on floats, arrays and polynomials, where the
    removable singularity at a zero constant term is handled. The cumulant generating function
    of the uniform distribution on ``[-1, 1]`` is ``log(sinhc(t))``."""
    if isinstance(x, TPoly):
        return x.sinhc()
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(x == 0, 1.0, np.sinh(x) / np.where(x == 0, 1.0, x))


def from_coefficients(d, N, coeffs):
    """A polynomial of degree ``N`` in ``d`` variables from a table of its coefficients: a dict
    from exponent tuples to the coefficient of ``u^nu`` (``f^(nu)/nu!``), missing entries zero.
    For closed-form ``taylor`` functions of a :class:`~saddlepoint_expansions.CGF`."""
    alg = get_algebra(d, max(N, 1))
    c = np.zeros(alg.M, dtype=complex if any(np.iscomplexobj(v) for v in coeffs.values()) else float)
    for nu, v in coeffs.items():
        i = alg.index.get(tuple(nu))
        if i is not None:
            c[i] = v
    return TPoly(alg, c)


def variables(d, N):
    """The ``d`` variables of the algebra of degree ``N``, as a numpy object array."""
    alg = get_algebra(d, N)
    v = np.empty(d, dtype=object)
    for i, p in enumerate(alg.variables()):
        v[i] = p
    return v


def constant_term(x):
    return x.constant_term() if isinstance(x, TPoly) else float(x)


def is_tpoly(x):
    return isinstance(x, TPoly)
