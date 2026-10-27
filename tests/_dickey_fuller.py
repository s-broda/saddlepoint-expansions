"""The joint cumulant generating function of the limiting Dickey-Fuller t-ratio (Bao, Broda and Kan,
"Inversion formulas and saddlepoint expansions for Studentized means", Section 7): a test fixture
for the Studentized-mean functions, not part of the package."""
import math

import numpy as np

from saddlepoint_expansions import CGF
from saddlepoint_expansions.taylor import TPoly

D1_SERIES = [lambda lam, c: -1/12*c**3 + (1/12)*c**2*lam + (1/3)*c**2 - c - 1/12*lam**2 + 1, lambda lam, c: -1/180*c**3 + (1/180)*c**2*lam + (1/30)*c**2 - 1/12*c - 1/180*lam**2 + 1/6, lambda lam, c: -1/6720*c**3 + (1/6720)*c**2*lam + (1/840)*c**2 - 1/360*c - 1/6720*lam**2 + 1/120, lambda lam, c: -1/453600*c**3 + (1/453600)*c**2*lam + (1/45360)*c**2 - 1/20160*c - 1/453600*lam**2 + 1/5040, lambda lam, c: -1/47900160*c**3 + (1/47900160)*c**2*lam + (1/3991680)*c**2 - 1/1814400*c - 1/47900160*lam**2 + 1/362880, lambda lam, c: -1/7264857600*c**3 + (1/7264857600)*c**2*lam + (1/518918400)*c**2 - 1/239500800*c - 1/7264857600*lam**2 + 1/39916800, lambda lam, c: -1/1494484992000*c**3 + (1/1494484992000)*c**2*lam + (1/93405312000)*c**2 - 1/43589145600*c - 1/1494484992000*lam**2 + 1/6227020800, lambda lam, c: -1/400148356608000*c**3 + (1/400148356608000)*c**2*lam + (1/22230464256000)*c**2 - 1/10461394944000*c - 1/400148356608000*lam**2 + 1/1307674368000, lambda lam, c: -1/135161222676480000*c**3 + (1/135161222676480000)*c**2*lam + (1/6758061133824000)*c**2 - 1/3201186852864000*c - 1/135161222676480000*lam**2 + 1/355687428096000, lambda lam, c: -1/56200036388880384000*c**3 + (1/56200036388880384000)*c**2*lam + (1/2554547108585472000)*c**2 - 1/1216451004088320000*c - 1/56200036388880384000*lam**2 + 1/121645100408832000, lambda lam, c: -1/28202200078783610880000*c**3 + (1/28202200078783610880000)*c**2*lam + (1/1175091669949317120000)*c**2 - 1/562000363888803840000*c - 1/28202200078783610880000*lam**2 + 1/51090942171709440000, lambda lam, c: -1/16803810880275234816000000*c**3 + (1/16803810880275234816000000)*c**2*lam + (1/646300418472124416000000)*c**2 - 1/310224200866619719680000*c - 1/16803810880275234816000000*lam**2 + 1/25852016738884976640000, lambda lam, c: -1/11726474792758225403904000000*c**3 + (1/11726474792758225403904000000)*c**2*lam + (1/418802671169936621568000000)*c**2 - 1/201645730563302817792000000*c - 1/11726474792758225403904000000*lam**2 + 1/15511210043330985984000000, lambda lam, c: -1/9473316421863966379868160000000*c**3 + (1/9473316421863966379868160000000)*c**2*lam + (1/315777214062132212662272000000)*c**2 - 1/152444172305856930250752000000*c - 1/9473316421863966379868160000000*lam**2 + 1/10888869450418352160768000000, lambda lam, c: -1/8771027897789784338907267072000000*c**3 + (1/8771027897789784338907267072000000)*c**2*lam + (1/274094621805930760590852096000000)*c**2 - 1/132626429906095529318154240000000*c - 1/8771027897789784338907267072000000*lam**2 + 1/8841761993739701954543616000000, lambda lam, c: -1/9226024969987629401488081551360000000*c**3 + (1/9226024969987629401488081551360000000)*c**2*lam + (1/271353675587871452984943575040000000)*c**2 - 1/131565418466846765083609006080000000*c - 1/9226024969987629401488081551360000000*lam**2 + 1/8222838654177922817725562880000000]
D2_SERIES = [lambda lam, c: -1/720*c**5 + (1/720)*c**4*lam + (1/80)*c**4 - 1/10*c**3 - 1/180*c**2*lam**2 + (13/30)*c**2 + (1/60)*c*lam**2 - c - 1/60*lam**2 + 1, lambda lam, c: -1/16800*c**5 + (1/16800)*c**4*lam + (11/16800)*c**4 - 13/2520*c**3 - 1/4200*c**2*lam**2 + (23/840)*c**2 + (1/1400)*c*lam**2 - 1/15*c - 1/1400*lam**2 + 1/15, lambda lam, c: -1/907200*c**5 + (1/907200)*c**4*lam + (13/907200)*c**4 - 17/151200*c**3 - 1/226800*c**2*lam**2 + (107/151200)*c**2 + (1/75600)*c*lam**2 - 1/560*c - 1/75600*lam**2 + 1/560, lambda lam, c: -1/83825280*c**5 + (1/83825280)*c**4*lam + (1/5588352)*c**4 - 1/712800*c**3 - 1/20956320*c**2*lam**2 + (17/1663200)*c**2 + (1/6985440)*c*lam**2 - 1/37800*c - 1/6985440*lam**2 + 1/37800, lambda lam, c: -1/11623772160*c**5 + (1/11623772160)*c**4*lam + (17/11623772160)*c**4 - 5/435891456*c**3 - 1/2905943040*c**2*lam**2 + (23/242161920)*c**2 + (1/968647680)*c*lam**2 - 1/3991680*c - 1/968647680*lam**2 + 1/3991680, lambda lam, c: -1/2241727488000*c**5 + (1/2241727488000)*c**4*lam + (19/2241727488000)*c**4 - 29/435891456000*c**3 - 1/560431872000*c**2*lam**2 + (269/435891456000)*c**2 + (1/186810624000)*c*lam**2 - 1/605404800*c - 1/186810624000*lam**2 + 1/605404800, lambda lam, c: -1/571640509440000*c**5 + (1/571640509440000)*c**4*lam + (1/27220976640000)*c**4 - 1/3464487936000*c**3 - 1/142910127360000*c**2*lam**2 + (113/38109367296000)*c**2 + (1/47636709120000)*c*lam**2 - 1/124540416000*c - 1/47636709120000*lam**2 + 1/124540416000, lambda lam, c: -1/185846681180160000*c**5 + (1/185846681180160000)*c**4*lam + (23/185846681180160000)*c**4 - 37/38014093877760000*c**3 - 1/46461670295040000*c**2*lam**2 + (139/12671364625920000)*c**2 + (1/15487223431680000)*c*lam**2 - 1/33345696384000*c - 1/15487223431680000*lam**2 + 1/33345696384000, lambda lam, c: -1/74933381851840512000*c**5 + (1/74933381851840512000)*c**4*lam + (1/2997335274073620480)*c**4 - 41/15611121219133440000*c**3 - 1/18733345462960128000*c**2*lam**2 + (503/15611121219133440000)*c**2 + (1/6244448487653376000)*c*lam**2 - 1/11263435223040000*c - 1/6244448487653376000*lam**2 + 1/11263435223040000, lambda lam, c: -1/36662860102418694144000*c**5 + (1/36662860102418694144000)*c**4*lam + (1/1357883707496988672000)*c**4 - 1/172346778259233177600*c**3 - 1/9165715025604673536000*c**2*lam**2 + (199/2585201673888497664000)*c**2 + (1/3055238341868224512000)*c*lam**2 - 1/4683336365740032000*c - 1/3055238341868224512000*lam**2 + 1/4683336365740032000, lambda lam, c: -1/21386668393077571584000000*c**5 + (1/21386668393077571584000000)*c**4*lam + (29/21386668393077571584000000)*c**4 - 1/93527704342904832000000*c**3 - 1/5346667098269392896000000*c**2*lam**2 + (233/1527619170934112256000000)*c**2 + (1/1782222366089797632000000)*c*lam**2 - 1/2350183339898634240000*c - 1/1782222366089797632000000*lam**2 + 1/2350183339898634240000, lambda lam, c: -1/14658093490947781754880000000*c**5 + (1/14658093490947781754880000000)*c**4*lam + (31/14658093490947781754880000000)*c**4 - 53/3175920256372019380224000000*c**3 - 1/3664523372736945438720000000*c**2*lam**2 + (809/3175920256372019380224000000)*c**2 + (1/1221507790912315146240000000)*c*lam**2 - 1/1400317573356269568000000*c - 1/1221507790912315146240000000*lam**2 + 1/1400317573356269568000000, lambda lam, c: -1/11659466365371035544453120000000*c**5 + (1/11659466365371035544453120000000)*c**4*lam + (1/353317162587001077104640000000)*c**4 - 1/44745759077630070620160000000*c**3 - 1/2914866591342758886113280000000*c**2*lam**2 + (103/283389807491657113927680000000)*c**2 + (1/971622197114252962037760000000)*c*lam**2 - 1/977206232729852116992000000*c - 1/971622197114252962037760000000*lam**2 + 1/977206232729852116992000000, lambda lam, c: -1/10650533875887595268673110016000000*c**5 + (1/10650533875887595268673110016000000)*c**4*lam + (1/304300967882502721962088857600000)*c**4 - 61/2349382472622263662207303680000000*c**3 - 1/2662633468971898817168277504000000*c**2*lam**2 + (1/2231132452632729023938560000000)*c**2 + (1/887544489657299605722759168000000)*c*lam**2 - 1/789443035155330531655680000000*c - 1/887544489657299605722759168000000*lam**2 + 1/789443035155330531655680000000, lambda lam, c: -1/11071229963985155281785697861632000000*c**5 + (1/11071229963985155281785697861632000000)*c**4*lam + (37/11071229963985155281785697861632000000)*c**4 - 1/37850358851231300108669052518400000*c**3 - 1/2767807490996288820446424465408000000*c**2*lam**2 + (1187/2460273325330034507063488413696000000)*c**2 + (1/922602496998762940148808155136000000)*c*lam**2 - 1/730918991482482028242272256000000*c - 1/922602496998762940148808155136000000*lam**2 + 1/730918991482482028242272256000000, lambda lam, c: -1/12992413987147285168919098372915200000000*c**5 + (1/12992413987147285168919098372915200000000)*c**4*lam + (1/333138820183263722279976881356800000000)*c**4 - 1/42118809645595699441576024473600000000*c**3 - 1/3248103496786821292229774593228800000000*c**2*lam**2 + (443/968732621848701087156248562892800000000)*c**2 + (1/1082701165595607097409924864409600000000)*c*lam**2 - 1/768835414165635783457340129280000000*c - 1/1082701165595607097409924864409600000000*lam**2 + 1/768835414165635783457340129280000000]


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
