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
with constant matrices and vectors (`A @ t`), all work; `math.log`, `float(...)` and calls into
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
{func}`~saddlepoint_expansions.common_component_cgf`.

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
