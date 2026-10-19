"""Normal orthant probabilities in two and three dimensions by adaptive quadrature."""
from __future__ import annotations

import math

import numpy as np
from scipy.integrate import quad
from scipy.special import erfc


def orthant2(t, S, rtol=1e-13):
    """``P(Z_1 >= t_1, Z_2 >= t_2)``, ``Z ~ N(0, S)``, by one-dimensional adaptive quadrature."""
    rtol = max(rtol, 2e-14)                 # scipy's quad needs epsrel > 50 eps with epsabs = 0
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    sd1 = math.sqrt(S[0, 0])
    sd2 = math.sqrt(S[1, 1])
    r = S[0, 1] / (sd1 * sd2)
    h = t[0] / sd1
    k = t[1] / sd2
    c = math.sqrt(1 - r * r)

    def f(z):
        return math.exp(-z * z / 2) / math.sqrt(2 * math.pi) * erfc((k - r * z) / (c * math.sqrt(2))) / 2

    v, _ = quad(f, h, np.inf, epsabs=0.0, epsrel=rtol, limit=500)
    return v


def orthant3(t, S, rtol=1e-13):
    """``P(Z >= t)``, ``Z ~ N(0, S)`` trivariate, by adaptive quadrature over ``z_1`` of the
    bivariate conditional orthant probability."""
    t = np.asarray(t, dtype=float)
    S = np.asarray(S, dtype=float)
    s1 = math.sqrt(S[0, 0])
    b = S[1:3, 0] / S[0, 0]
    Sc = S[1:3, 1:3] - np.outer(S[1:3, 0], S[0, 1:3]) / S[0, 0]

    def f(z):
        return math.exp(-z * z / (2 * S[0, 0])) / (math.sqrt(2 * math.pi) * s1) * orthant2(t[1:3] - b * z, Sc, rtol=rtol / 10)

    v, _ = quad(f, t[0], np.inf, epsabs=0.0, epsrel=rtol, limit=500)
    return v
