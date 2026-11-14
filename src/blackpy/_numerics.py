"""Shared numerical toolbox: autodiff derivatives, root finding, quadrature.

Everything here is written with ``jax.numpy`` and ``jax.lax`` control flow,
so every function can be wrapped in ``jax.jit`` and ``jax.vmap``.

Derivatives are always obtained by automatic differentiation
(:func:`jax.grad`); there are no finite differences in this module.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np

_EPS = float(np.finfo(np.float64).eps)
#: Relative size of a Newton step below which the next iterate is converged to round-off.
_NEWTON_STOP = 1e-10


def value_and_derivative(fn: Callable, x):
    """Return ``(fn(x), fn'(x))`` in a single forward-mode pass."""
    return jax.jvp(fn, (x,), (jnp.ones_like(x),))


def derivative(fn: Callable, order: int = 1) -> Callable:
    """Return the ``order``-th derivative of a scalar function of a scalar.

    Forward-mode automatic differentiation (``jax.jvp``) is used: for
    scalar-to-scalar functions nested forward mode produces much smaller
    computational graphs than nested reverse mode (measured: about 6x faster
    for the third derivative of a metric function).

    Parameters
    ----------
    fn : callable
        JAX-traceable function ``fn(x) -> scalar`` of a scalar.
    order : int
        Derivative order (0 returns ``fn`` itself).

    Returns
    -------
    callable
        ``x -> d^order fn / dx^order``.
    """
    if order < 0:
        raise ValueError(f"Derivative order must be >= 0, got {order}.")
    for _ in range(order):
        fn = (lambda inner: lambda x: value_and_derivative(inner, x)[1])(fn)
    return fn


def refine_root(fn: Callable, a, b, *, max_iter: int = 100, xtol: float = 4.0 * _EPS):
    """Refine a bracketed root with a safeguarded Newton iteration.

    This is the classic ``rtsafe`` scheme (Press et al., *Numerical Recipes*,
    3rd ed., section 9.4): a Newton step is taken when it stays inside the
    current bracket and converges fast enough, otherwise the bracket is
    bisected. The derivative comes from automatic differentiation.

    Parameters
    ----------
    fn : callable
        Scalar function of a scalar with ``fn(a)`` and ``fn(b)`` of opposite
        sign (or one of them exactly zero).
    a, b : float
        Bracket end points.
    max_iter : int
        Maximum number of iterations.
    xtol : float
        Relative tolerance on the root position.

    Returns
    -------
    float
        Root location, or NaN if ``[a, b]`` does not bracket a sign change.
    """
    fa = fn(a)
    fb = fn(b)
    # Orient the bracket so that fn(x_lo) < 0 <= fn(x_hi).
    x_lo = jnp.where(fa < 0.0, a, b)
    x_hi = jnp.where(fa < 0.0, b, a)
    # Start from the secant (linear-interpolation) point: scan cells are small, so the
    # function is nearly linear across them and Newton then converges in 2-3 steps.
    secant = a - fa * (b - a) / jnp.where(fb == fa, 1.0, fb - fa)
    inside = jnp.isfinite(secant) & (secant > jnp.minimum(a, b)) & (secant < jnp.maximum(a, b))
    x0 = jnp.where(inside, secant, 0.5 * (a + b))
    f0, d0 = value_and_derivative(fn, x0)
    x_lo = jnp.where(f0 < 0.0, x0, x_lo)
    x_hi = jnp.where(f0 < 0.0, x_hi, x0)
    width = jnp.abs(b - a)

    def cond(state):
        i, _x, _f, _d, _lo, _hi, _dx_old, _dx, done = state
        return (i < max_iter) & (~done)

    def body(state):
        i, x, f, d, lo, hi, dx_old, dx, _done = state
        newton_leaves_bracket = ((x - hi) * d - f) * ((x - lo) * d - f) > 0.0
        too_slow = jnp.abs(2.0 * f) > jnp.abs(dx_old * d)
        bad_slope = (~jnp.isfinite(d)) | (d == 0.0)
        bisect = newton_leaves_bracket | too_slow | bad_slope
        safe_d = jnp.where(bad_slope, 1.0, d)
        dx_new = jnp.where(bisect, 0.5 * (hi - lo), f / safe_d)
        x_new = jnp.where(bisect, lo + dx_new, x - dx_new)
        f_new, d_new = value_and_derivative(fn, x_new)
        lo = jnp.where(f_new < 0.0, x_new, lo)
        hi = jnp.where(f_new < 0.0, hi, x_new)
        # A Newton step of relative size <= 1e-10 leaves an error of order 1e-20
        # (quadratic convergence), so the iteration can stop one step earlier.
        step_tol = jnp.where(bisect, xtol, _NEWTON_STOP)
        converged = (jnp.abs(dx_new) <= step_tol * jnp.abs(x_new)) | (f_new == 0.0)
        return i + 1, x_new, f_new, d_new, lo, hi, dx, dx_new, converged

    # An invalid bracket (no sign change) is skipped at once: under ``vmap`` the loop
    # runs until every batch element is done, so padded slots must not iterate.
    bracketed = ((fa < 0.0) != (fb < 0.0)) & jnp.isfinite(fa) & jnp.isfinite(fb)
    init = (0, x0, f0, d0, x_lo, x_hi, width, width, (f0 == 0.0) | (~bracketed))
    _, x, *_ = jax.lax.while_loop(cond, body, init)
    x = jnp.where(fa == 0.0, a, x)
    x = jnp.where(fb == 0.0, b, x)
    return jnp.where(bracketed | (fa == 0.0) | (fb == 0.0), x, jnp.nan)


def _outermost_sign_changes(values, max_roots: int):
    """Indices ``i`` of the outermost cells ``[i, i+1]`` across which ``values`` changes sign.

    Returns ``(idx, valid)``, both of length ``max_roots``; ``idx`` is 0 where
    ``valid`` is false.
    """
    left, right = values[:-1], values[1:]
    mask = ((left < 0.0) != (right < 0.0)) & jnp.isfinite(left) & jnp.isfinite(right)
    score = jnp.where(mask, jnp.arange(mask.shape[0]), -1)
    top, _ = jax.lax.top_k(score, max_roots)
    valid = top >= 0
    return jnp.where(valid, top, 0), valid


def scan_grid(lo, hi, n_scan: int, log_spacing: bool = True):
    """Grid of ``n_scan`` points on ``[lo, hi]`` (logarithmic by default)."""
    t = jnp.linspace(0.0, 1.0, n_scan)
    if log_spacing:
        return lo * (hi / lo) ** t
    return lo + (hi - lo) * t


def find_roots(
    fn: Callable,
    lo,
    hi,
    *,
    n_scan: int = 256,
    max_roots: int = 6,
    log_spacing: bool = True,
    detect_extrema: bool = True,
    degenerate_atol: float = 1e-10,
):
    """Find all roots of a scalar function on ``[lo, hi]``.

    Algorithm
    ---------
    1. Evaluate ``fn`` on a scan grid of ``n_scan`` points and refine every
       sign change with a safeguarded Newton iteration (:func:`refine_root`).
    2. If ``detect_extrema`` is true, also locate the extrema of ``fn`` (sign
       changes of its autodiff derivative) and search for roots on both sides
       of each extremum inside its grid cell. A pair of roots that falls
       inside a single grid cell always encloses an extremum, so this
       recovers closely spaced roots that plain sign-change scanning misses.
       An extremum at which ``|fn| <= degenerate_atol`` is reported as a
       single *degenerate* (double) root, e.g. the horizon of an extremal
       black hole, where there is no sign change at all. Consequently two
       roots so close that ``|fn|`` stays below the tolerance between them
       are merged into one degenerate root.

    The output has a fixed length so that the function is compatible with
    ``jax.jit`` and ``jax.vmap``.

    Parameters
    ----------
    fn : callable
        JAX-traceable scalar function of a scalar.
    lo, hi : float
        Search interval, ``0 < lo < hi`` when ``log_spacing`` is true.
    n_scan : int
        Number of scan points.
    max_roots : int
        Maximum number of roots returned. If there are more, the outermost
        ones are kept.
    log_spacing : bool
        Use a logarithmic scan grid (recommended for radial problems).
    detect_extrema : bool
        Enable step 2 above. Costs two extra derivative orders.
    degenerate_atol : float
        Absolute tolerance on ``|fn|`` at an extremum for a degenerate root.

    Returns
    -------
    roots : array, shape (max_roots,)
        Roots in ascending order, padded with NaN.
    degenerate : array of bool, shape (max_roots,)
        True where the root is a degenerate (double) root.
    """
    grid = scan_grid(lo, hi, n_scan, log_spacing)
    refine = jax.vmap(lambda a, b: refine_root(fn, a, b))
    values = jax.vmap(fn)(grid)
    idx, valid = _outermost_sign_changes(values, max_roots)
    simple_lo = grid[idx]
    simple_hi = jnp.where(valid, grid[idx + 1], simple_lo)
    if not detect_extrema:
        return jnp.sort(refine(simple_lo, simple_hi)), jnp.zeros(max_roots, dtype=bool)

    dfn = derivative(fn)
    slopes = jax.vmap(dfn)(grid)
    eidx, has_extremum = _outermost_sign_changes(slopes, max_roots)
    cell_lo = grid[eidx]
    cell_hi = jnp.where(has_extremum, grid[eidx + 1], cell_lo)
    extrema = jax.vmap(lambda a, b: refine_root(dfn, a, b))(cell_lo, cell_hi)
    has_extremum = has_extremum & jnp.isfinite(extrema)
    safe_ext = jnp.where(has_extremum, extrema, cell_lo)
    # One refinement pass over the sign-change cells and over both sides of every
    # extremum (roots hidden inside a single grid cell).
    simple = refine(
        jnp.concatenate([simple_lo, cell_lo, safe_ext]),
        jnp.concatenate([simple_hi, safe_ext, cell_hi]),
    )
    values_at_extrema = jax.vmap(fn)(safe_ext)
    is_degenerate = has_extremum & (jnp.abs(values_at_extrema) <= degenerate_atol)
    degenerate = jnp.where(is_degenerate, extrema, jnp.nan)

    # Remove duplicates. (i) Where the extremum itself is a root within tolerance, it
    # is reported once, as a degenerate root: drop the pair of simple roots on either
    # side of it, and any simple root that coincides with it.
    beside_degenerate = jnp.concatenate(
        [jnp.zeros(max_roots, dtype=bool), is_degenerate, is_degenerate]
    )
    simple = jnp.where(beside_degenerate, jnp.nan, simple)
    near_degenerate = jnp.abs(simple[:, None] - degenerate[None, :]) <= 1e-7 * jnp.abs(
        degenerate[None, :]
    )
    simple = jnp.sort(jnp.where(jnp.any(near_degenerate, axis=1), jnp.nan, simple))
    # (ii) The same simple root found from a grid cell and from a sub-cell.
    repeated = jnp.abs(simple[1:] - simple[:-1]) <= 1e-12 * jnp.abs(simple[1:])
    simple = simple.at[1:].set(jnp.where(repeated, jnp.nan, simple[1:]))

    all_roots = jnp.concatenate([simple, degenerate])
    all_flags = jnp.concatenate([jnp.zeros(simple.shape[0], dtype=bool), is_degenerate])
    order = jnp.argsort(all_roots)  # NaN sorts last
    n_all = all_roots.shape[0]
    n_found = jnp.sum(jnp.isfinite(all_roots))
    # Keep the outermost ``max_roots`` roots if more were found.
    shift = jnp.clip(n_found - max_roots, 0, n_all - max_roots)
    take = jnp.arange(max_roots) + shift
    return all_roots[order][take], all_flags[order][take]


def nanmax_or_nan(values, mask):
    """Largest element of ``values`` where ``mask`` is true, NaN if none."""
    picked = jnp.where(mask & jnp.isfinite(values), values, -jnp.inf)
    best = jnp.max(picked)
    return jnp.where(jnp.isfinite(best), best, jnp.nan)


def nanmin_or_nan(values, mask):
    """Smallest element of ``values`` where ``mask`` is true, NaN if none."""
    picked = jnp.where(mask & jnp.isfinite(values), values, jnp.inf)
    best = jnp.min(picked)
    return jnp.where(jnp.isfinite(best), best, jnp.nan)


def gauss_legendre(fn: Callable, a, b, n: int = 64):
    """Integrate ``fn`` over ``[a, b]`` with ``n``-point Gauss-Legendre quadrature.

    Parameters
    ----------
    fn : callable
        JAX-traceable scalar function of a scalar.
    a, b : float
        Integration limits.
    n : int
        Number of quadrature nodes. The rule is exact for polynomials of
        degree ``2 n - 1``.

    Returns
    -------
    float
        Approximation of the integral.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    half = 0.5 * (b - a)
    mid = 0.5 * (b + a)
    x = mid + half * jnp.asarray(nodes)
    return half * jnp.sum(jnp.asarray(weights) * jax.vmap(fn)(x))
