# Guide

## Cumulant generating functions

A {class}`~saddlepoint_expansions.CGF` wraps a Python function `K(t)` of a numpy vector `t` of
length $d$ (also for $d = 1$, where you write `t[0]`), together with the dimension and,
optionally, a predicate describing the domain:

```python
K = CGF(lambda t: -3*np.log(1 - t[0]) - 2*np.log(1 - 2*t[1]), 2,
        domain=lambda t: t[0] < 1 and t[1] < 1/2)
```

The function is evaluated on vectors of truncated multivariate Taylor polynomials
({class}`~saddlepoint_expansions.taylor.TPoly`) to generate its derivatives, so it must be
written generically: arithmetic, `np.exp`, `np.log`, `np.sqrt`, powers, `np.sum`, products
with constant matrices and vectors (`A @ t`), `np.sin`, `np.cos`, `np.tan`, `np.sinh`,
`np.cosh`, `np.tanh`, `np.arctan` all work, and {func}`~saddlepoint_expansions.sinhc` and
{func}`~saddlepoint_expansions.sincu` give $\sinh(x)/x$ and $\sin(x)/x$ with their removable
singularity at zero (write `np.log(sinhc(t[0]))` for the uniform distribution, not
`np.log(np.sinh(t[0])/t[0])`, which is $0/0$ at the origin where the saddlepoint iteration starts); `math.log`, `float(...)` and calls into
numerical libraries that require floats do not. For determinants and inverses of matrices that
depend on `t`, write the factorization in generic arithmetic as
{func}`~saddlepoint_expansions.quadratic_forms_cgf` does, or use that constructor.

Outside its domain, `K` may raise a {class}`~saddlepoint_expansions.DomainError` (`np.log` and
`np.sqrt` of a polynomial with a negative constant term do this automatically) or return a
non-finite value. When the formula is finite outside the true domain, as with $1/(1-t)$, the
`domain` predicate is needed; it is also cheaper than an exception.

Transformations: {func}`~saddlepoint_expansions.reflect` (sign changes),
{func}`~saddlepoint_expansions.marginal`, {func}`~saddlepoint_expansions.linear_combination`
($Y = B'X$), {func}`~saddlepoint_expansions.sample_mean` ($nK(t/n)$),
{func}`~saddlepoint_expansions.shift` ($X + c$), and `K1 + K2` for the sum of independent
vectors. The library: {func}`~saddlepoint_expansions.normal_cgf`,
{func}`~saddlepoint_expansions.gamma_cgf`, {func}`~saddlepoint_expansions.ncx2_cgf`,
{func}`~saddlepoint_expansions.exponential_sums_cgf`,
{func}`~saddlepoint_expansions.quadratic_forms_cgf`,
{func}`~saddlepoint_expansions.wishart_diag_cgf`,
{func}`~saddlepoint_expansions.common_component_cgf`, {func}`~saddlepoint_expansions.uniform_cgf`.

## Tail probabilities

{func}`~saddlepoint_expansions.tailprob`` (K, x, order=M)` returns the expansions of orders
$1, \dots, M$ of $P(X \ge x)$ in `F`, with the saddlepoint `tau`, a `status` string and the
full result of the underlying routine in `info`. The expansion of order $M$ keeps the terms of
orders $n^{-m/2}$, $m \le 2M - 1$, and has relative error $O(n^{-M})$; in practice each order
gains one to two digits at moderate sample sizes, and the first order is accurate to a few
percent.

`lower=True` gives $P(X \le x)$ and `signs` a mixed corner, e.g. `signs=[1, -1]` for
$P(X_1 \ge x_1, X_2 \le x_2)$. When the saddlepoint has negative components, the expansion is
applied through the inclusion-exclusion identity of the paper, which preserves the relative
error bound. In three dimensions the orthant integrals are computed by Gauss quadrature and in
four or more by randomized quasi-Monte Carlo (`info["err"]` is then a standard error); keyword
arguments are passed on to {func}`~saddlepoint_expansions.spa_mv` (`nG`, `nqmc`, ...) and, for
$d = 2$, to {func}`~saddlepoint_expansions.spa_orthant2` (`nG`, `logscale` for probabilities
below the double-precision range, ...). For $d = 1$, `x` may be a scalar.

## Densities

{func}`~saddlepoint_expansions.density`` (K, x, order=M)` returns the expansions of orders
$1, \dots, M$ of the density (the terms $m \le M-1$ of the expansion in powers of $n^{-1}$,
relative error $O(n^{-M})$); a finite sum of Hermite polynomials, no integration, any
dimension. In three or more dimensions the terms can cancel strongly at high orders; check
`info["kappa_pdf"]`, the ratio of the sum of the absolute terms to the sum (about $\log_{10}$
of it digits are lost). The Julia package repeats such computations in higher precision; this
package works in double precision.

## Ratios

For a bivariate `K` with a positive denominator $X_2 > 0$:
{func}`~saddlepoint_expansions.ratio_density`` (K, r)` is the Rice expansion of the density of
$X_1/X_2$ at $r$ through the saddlepoint of $K_W(s) = K(s, -rs)$, and
{func}`~saddlepoint_expansions.ratio_tailprob`` (K, r, lower=...)` is the univariate expansion
for $W = X_1 - rX_2$ at $0$. Example, the noncentral $F$ with $Q_1 \sim \chi^2_m(\lambda)$ and
$Q_2 \sim \chi^2_n$:

```python
K = CGF(lambda t: ncx2_cgf(m, lam)(t[:1]) + ncx2_cgf(n)(t[1:]), 2,
        domain=lambda t: t[0] < 0.5 and t[1] < 0.5)
c = f*m/n
ratio_tailprob(K, c, lower=True)["F"]     # P(F <= f), orders 1..3
ratio_density(K, c)["f"] * (m/n)          # density of F at f
```

*Several ratios.* With a vector `r` of length $d$, `K` is the cgf of $(X_1, \dots, X_d, Y_1, \dots, Y_d)$,
or of $(X_1, \dots, X_d, Y)$ for a common denominator, and `ratio_density` returns the joint density
of $(X_1/Y_1, \dots, X_d/Y_d)$ at `r` (Theorem 4 with the amplitude expansion of Corollary 1 of
the paper, every Taylor coefficient from one polynomial of $K(s, -r \circ s + \tau)$), while
`ratio_tailprob` returns the orthant probability $P(R \ge r)$ with `lower` and `signs` as in
`tailprob`. The sample autocorrelations at lags one and two, for example, are two ratios of
quadratic forms with a common denominator:

```python
N = 50; Mc = np.eye(N) - np.ones((N, N))/N
A = lambda j: (np.diag(np.ones(N - j), j) + np.diag(np.ones(N - j), -j))/2
K = quadratic_forms_cgf([Mc @ A(1) @ Mc, Mc @ A(2) @ Mc, Mc])
ratio_density(K, [-1, -1]/np.sqrt(N), order=5)["f"]
```

## Studentized means

For $Q = X/\sqrt{Y}$ with $Y > 0$, with `K` the joint cumulant generating function of $(X, Y)$,
the expansions of Bao, Broda and Kan, *Inversion formulas and saddlepoint expansions for
Studentized means*:

```python
studentized_density(K, q, order=3)           # density of X/sqrt(Y) at q, orders 1..3
studentized_tailprob(K, q, order=3)          # P(X/sqrt(Y) >= q); lower=True for P(X/sqrt(Y) <= q)
studentized_quantile(K, 0.05, order=3)       # the q with P(X/sqrt(Y) <= q) = 0.05 under the expansion
```

Unlike a ratio, $X - q\sqrt{Y}$ is not linear in $(X, Y)$, and the inversion is a double
integral with a kernel in $s$ and $t$. The saddlepoint $(\hat s, \hat t, \hat v)$ solves
$K_s = qv$, $K_t = v^2$, $qs + 2tv = 0$ ({func}`~saddlepoint_expansions.studentized_saddlepoint`);
the density is the bivariate amplitude expansion about it, and the distribution function is a
Hauschildt-Robinson type expansion in $s$ after the $t$-integral is expanded by Laplace's method,
all Taylor coefficients being generated from the Taylor polynomial of `K` at the saddlepoint by
Newton's method and composition on power series. `order=M` keeps the terms $m \le M - 1$ of the
density and $m \le 2M - 1$ of the distribution function (relative error $O(n^{-M})$);
`info["Fp"]` holds every half-order. The saddlepoints for all $q$ lie on the curve
$sK_s + 2tK_t = 0$ ({func}`~saddlepoint_expansions.studentized_locus`), which is how
`studentized_quantile` searches: along the curve each $\hat s$ gives its $q$ and its distribution
function directly, so that no saddlepoint equation is solved for a trial $q$.

The noncentral $t$ distribution, for example, with $X \sim N(\mu, 1)$ and
$Y \sim \chi^2_\nu/\nu$ independent, $n$ pairs: $\bar X/\sqrt{\bar Y} = T/\sqrt{n}$, $T$
noncentral $t$ with $n\nu$ degrees of freedom and noncentrality $\sqrt{n}\mu$:

```python
K = sample_mean(CGF(lambda t: mu*t[0] + t[0]**2/2 - nu/2*np.log(1 - 2*t[1]/nu), 2,
                    domain=lambda t: t[1] < nu/2), n)
studentized_tailprob(K, x/np.sqrt(n), lower=True)["F"]     # P(T <= x)
```

The terms of high order of the density cancel by many orders of magnitude where $|\hat t|$ is
large; `info["kappa"]` is the cancellation indicator, and in double precision the terms beyond
the fourth are often unreliable there (the Julia package repeats the computation in higher
precision; here `status` reports `"coefficients inaccurate"`). The Taylor algebra carries complex
coefficients for these expansions (the Hessian of the exponent is indefinite where $\hat t < 0$).

## Conditional distributions

{func}`~saddlepoint_expansions.conditional_tailprob`` (K, x, A, order=M)` returns the expansions
of orders $1, \dots, M$ of the conditional distribution function $P(X_A \ge x_A \mid X_B = x_B)$,
where `A` holds the (0-based) indices of the components whose tail is wanted and $B$ is the rest,
and {func}`~saddlepoint_expansions.conditional_density`` (K, x, A, order=M)` those of the
conditional density $f(x_A \mid x_B)$. These are the expansions of Broda and Kan, *Conditional
saddlepoint expansions of arbitrary order* (2026, draft). The double saddlepoint approximation of
Skovgaard (1987) to the distribution function of one component of a mean given the others is the
conditional counterpart of the Lugannani-Rice formula; Kolassa (1996) showed that it is the first
term of an asymptotic expansion and gave the second. The paper gives every term of every order,
for any numbers of tail and conditioning components: the numerator of the conditional probability
is the orthant-probability integral with the poles in the components of $A$ only, expanded to the
terms of orders $n^{-m/2}$, $m \le 2M-1$, the denominator is the density expansion of $X_B$ at its
own saddlepoint with $M$ terms, and the relative error is $O(n^{-M})$. The expansion is of
Hauschildt-Robinson type and has no singular terms, its leading term being
$(\hat u/\hat z)e^{(\hat u^2-\hat w^2)/2}\bar\Phi(\hat u)$ in Skovgaard's notation for one tail
component (`info["Fp"][0]`); saddlepoints with negative tail components are handled by the
inclusion-exclusion identity of the paper, and `lower` and `signs` select other corners.

An example with an exact answer: $U \sim \mathrm{Gamma}(16)$ and $V \sim \mathrm{Gamma}(16)$
independent, $X = (U, U + V)$; given $X_2 = 32$, $X_1/32$ has the $\mathrm{Beta}(16, 16)$
distribution.

```python
from scipy.special import betaincinv
K = CGF(lambda t: -16*np.log(1 - t[0] - t[1]) - 16*np.log(1 - t[1]), 2,
        domain=lambda t: t[0] + t[1] < 1 and t[1] < 1)
x1 = 32*betaincinv(16, 16, 0.9)                       # the 90% point
r = conditional_tailprob(K, [x1, 32.0], 0, order=5)
r["F"]                  # P(X1 >= x1 | X2 = 32), orders 1 to 5; exact 0.1
conditional_density(K, [x1, 32.0], 0, order=5)["f"]
```

The five approximations of the tail probability have relative errors $2.5\times10^{-4}$,
$-7.0\times10^{-5}$, $-5.6\times10^{-7}$, $5.5\times10^{-8}$ and $1.3\times10^{-9}$.

The Taylor coefficients are taken in the canonical coordinates of the paper,
$Y_A = X_A - CX_B$, $Y_B = L^{-1}X_B$, $C = K''_{AB}(K''_{BB})^{-1}$, $LL' = K''_{BB}$, in which the
tail components are uncorrelated with the conditioning components under the tilted distribution
and the latter are whitened; the conditional probability is unchanged, and the terms, which in the
original coordinates cancel by up to sixteen orders of magnitude when the components are strongly
correlated, no longer do, so that double precision suffices. With one tail component every
integral is one-dimensional, whatever the number of conditioning variables.

*Lattice variables.* With `lattice=True` the components in $A$ are integer-valued, and so is
`x[A]`; `F` is then $P(S_A \ge k_A \mid S_B = k_B)$ with the continuity correction (the saddlepoint
at $k_A - 1/2$) and the amplitude $\prod_{j\in A}s_j/\{2\sinh(s_j/2)\}$, expanded about the
saddlepoint (`sinh="saddle"`, the default) or about the origin (`sinh="origin"`, keeping
$s^{2M-2}$ at order $M$, as in Kolassa 2003), and `conditional_density` gives
$P(S_A = k_A \mid S_B = k_B)$. For the sufficient statistic of a logistic regression with integer
covariates ({func}`~saddlepoint_expansions.logistic_regression_cgf`):

```python
X = np.column_stack([np.ones(30, dtype=int), np.arange(30) % 5, np.arange(30) % 3])
K = logistic_regression_cgf(X)                        # S = X'y under beta = 0
conditional_tailprob(K, [15, 37, 15], 1, order=4, lattice=True)["F"]   # P(S1 >= 37 | S0 = 15, S2 = 15)
conditional_density(K, [15, 37, 15], 1, order=4, lattice=True)["f"]    # P(S1 = 37 | S0 = 15, S2 = 15)
```

{func}`~saddlepoint_expansions.spa_cond` is the same computation from user-supplied derivatives.

## Lattice variables

For an integer-valued vector without conditioning, `tailprob(K, k, lattice=True)` returns the
continuity-corrected expansions of $P(S \ge k)$ and `density(K, k, lattice=True)` those of
$P(S = k)$, the case without conditioning components of the lattice theorem of the conditional
paper (the computation of `conditional_tailprob` with every component in `A`, with the same
numbers); in one dimension the approximation of order one has the accuracy of the
continuity-corrected approximations of Daniels (1987). {func}`~saddlepoint_expansions.poisson_cgf`,
{func}`~saddlepoint_expansions.binomial_cgf` and {func}`~saddlepoint_expansions.negative_binomial_cgf`
are in the library:

```python
tailprob(poisson_cgf(5.0), 11, order=4, lattice=True)["F"]    # P(S >= 11), exact 0.0136953
density(poisson_cgf(5.0), 11, order=4, lattice=True)["f"]     # P(S = 11)
```

Where the continuity-corrected point lies outside the convex hull of the support, as for
$P(S \ge 0)$ of a count, there is no saddlepoint and a `DomainError` is raised.

## Quantiles

{func}`~saddlepoint_expansions.quantile`` (K, p, order=M)` returns the point at which the
saddlepoint distribution function of order $M$ of a univariate variable equals `p`
(`lower=False` for the upper tail), and {func}`~saddlepoint_expansions.ratio_quantile` the same for
a ratio with a positive denominator. The search runs in saddlepoint space: the point is
parametrized by its saddlepoint, $x = K'(s)$, so that each iterate comes with its saddlepoint and
no saddlepoint equation is solved, and Newton's method in $s$ uses the density expansion times
$K''(s)$ as the derivative, safeguarded by bracketing and bisection. For a ratio the parameter is
the saddlepoint of $X_1 - rX_2$, which moves along the curve $sK_1 + tK_2 = 0$ on which
$r = K_1/K_2$.

## Normalized densities

`density(K, x, normalize=True)` divides the expansion of every order by its total mass, computed
in saddlepoint space, $\int f(x)\,dx = \int f(K'(s))\det K''(s)\,ds$, so that the saddlepoint is
known at every quadrature node, by `scipy.integrate.quad_vec` over the domain of $K$ (nested over
the coordinates in more dimensions, which is slow); the masses are returned in `info["mass"]` and
can be passed back as `normalize`. For the gamma distribution every normalized order is exact.
`ratio_density` accepts the same option for one ratio. The Studentized density does not: its terms
of high order cancel catastrophically in the tails, where the Julia package computes them in
higher precision.

## Closed-form Taylor polynomials

Automatic differentiation is the default, but a `CGF` can carry its Taylor polynomials in closed
form through the argument `taylor`, a function `taylor(t, N, M)` returning the polynomial of
degree `N` of $K(t + Mv)$ as a {class}`~saddlepoint_expansions.taylor.TPoly` (built from a
coefficient table with {func}`~saddlepoint_expansions.taylor.from_coefficients`), or `None`
outside the domain. {func}`~saddlepoint_expansions.quadratic_forms_cgf` uses this with the trace
recursion of the paper evaluated directly in the whitened directions, which makes large matrices
cheap, and the transformations carry the option along; pass `closedform=False` to differentiate
the generic formula instead. Comparisons between polynomials, or with numbers, compare the
constant terms, so a cgf written with an `if` on its argument can be differentiated.

## Derivatives and the saddlepoint

{func}`~saddlepoint_expansions.derivatives`` (K, t, N)` returns all derivatives
$K^{(\nu)}(t)$, $|\nu| \le N$, as a dict keyed by the multi-index, and
{func}`~saddlepoint_expansions.saddlepoint`` (K, x)` solves $K'(\tau) = x$ by Newton's method
with backtracking. The derivatives at the saddlepoint are generated once, to the degree $2M + 1$
needed by order $M$, and the whitened coefficients (of $K_Y(u) = K(L^{-T}u)$,
$K''(\tau) = LL'$) are generated by differentiating the composed function directly, which keeps
them as accurate as the original ones.

## The derivative interface

The engine can be used with derivatives supplied by the user, as `Kfun(nu, t)` returning the
derivative of multi-index order `nu` (a tuple) at `t`, NaN outside the domain:
{func}`~saddlepoint_expansions.spa_saddle`, {func}`~saddlepoint_expansions.spa_uni`,
{func}`~saddlepoint_expansions.spa_ratio_pdf`, {func}`~saddlepoint_expansions.spa_bi`,
{func}`~saddlepoint_expansions.spa_bi_at`, {func}`~saddlepoint_expansions.spa_orthant2`,
{func}`~saddlepoint_expansions.spa_mv`, {func}`~saddlepoint_expansions.spa_mv_at`,
{func}`~saddlepoint_expansions.spa_orthant`. These follow the Julia package and the MATLAB
programs of the paper. {func}`~saddlepoint_expansions.kfun` and
{func}`~saddlepoint_expansions.kwhite` turn a `CGF` into the functions they take.

## The Taylor algebra

{mod}`saddlepoint_expansions.taylor` is a self-contained implementation of truncated
multivariate Taylor polynomials: `variables(d, N)` returns the $d$ variables of degree $N$ as
an object array, and any generic function evaluated on them returns a
{class}`~saddlepoint_expansions.taylor.TPoly` whose `coeff(nu)` is $f^{(\nu)}(0)/\nu!$ and
`derivative(nu)` is $f^{(\nu)}(0)$. Products use precomputed index tables and `np.bincount`; a
product of two polynomials of degree 12 in three variables takes about 0.1 ms, an `exp` or `log`
about 1 ms.
