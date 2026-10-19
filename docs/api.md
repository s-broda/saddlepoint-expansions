# API reference

## Cumulant generating functions and the high-level interface

```{eval-rst}
.. autoclass:: saddlepoint_expansions.CGF
   :members: __add__

.. autofunction:: saddlepoint_expansions.derivatives
.. autofunction:: saddlepoint_expansions.saddlepoint
.. autofunction:: saddlepoint_expansions.tailprob
.. autofunction:: saddlepoint_expansions.density
.. autofunction:: saddlepoint_expansions.ratio_density
.. autofunction:: saddlepoint_expansions.ratio_tailprob
.. autofunction:: saddlepoint_expansions.reflect
.. autofunction:: saddlepoint_expansions.marginal
.. autofunction:: saddlepoint_expansions.linear_combination
.. autofunction:: saddlepoint_expansions.sample_mean
.. autofunction:: saddlepoint_expansions.shift
```

## Library of cumulant generating functions

```{eval-rst}
.. automodule:: saddlepoint_expansions.cgfs
   :members:
```

## The derivative interface

```{eval-rst}
.. autofunction:: saddlepoint_expansions.spa_saddle
.. autofunction:: saddlepoint_expansions.spa_uni
.. autofunction:: saddlepoint_expansions.spa_ratio_pdf
.. autofunction:: saddlepoint_expansions.spa_bi
.. autofunction:: saddlepoint_expansions.spa_bi_at
.. autofunction:: saddlepoint_expansions.spa_orthant2
.. autofunction:: saddlepoint_expansions.spa_mv
.. autofunction:: saddlepoint_expansions.spa_mv_at
.. autofunction:: saddlepoint_expansions.spa_orthant
.. autofunction:: saddlepoint_expansions.orthant2
.. autofunction:: saddlepoint_expansions.orthant3
.. autofunction:: saddlepoint_expansions.kfun
.. autofunction:: saddlepoint_expansions.kwhite
```

## The Taylor algebra

```{eval-rst}
.. automodule:: saddlepoint_expansions.taylor
   :members: TPoly, Algebra, get_algebra, variables, DomainError
```
