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

and the limiting Dickey-Fuller $t$-ratio, the Studentized mean $U/\sqrt{V}$ of two functionals
of a near-unit-root diffusion ({func}`~saddlepoint_expansions.dickey_fuller_cgf`):

```python
K = dickey_fuller_cgf(2)                                    # a constant in the regression, unit root
studentized_tailprob(K, -2.862, lower=True)["F"]            # the limiting size of the 5% test, ~ 0.05
studentized_tailprob(dickey_fuller_cgf(2, c=-5.0), -2.862, lower=True)["F"]   # local power at c = -5
```

The terms of high order of the density cancel by many orders of magnitude where $|\hat t|$ is
large; `info["kappa"]` is the cancellation indicator, and in double precision the terms beyond
the fourth are often unreliable there (the Julia package repeats the computation in higher
precision; here `status` reports `"coefficients inaccurate"`). The Taylor algebra carries complex
coefficients for these expansions (the Hessian of the exponent is indefinite where $\hat t < 0$).

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
