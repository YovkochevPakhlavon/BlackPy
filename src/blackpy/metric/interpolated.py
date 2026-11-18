"""Smooth JAX interpolants for metrics without a closed form.

A numerically obtained metric function (e.g. the solution of a field
equation on a radial grid) can be turned into a JAX-traceable callable with
:func:`spline`. The interpolant is a B-spline of degree ``k`` (default 5), so
it has ``k - 1`` continuous derivatives: enough for the ISCO, which needs
the second derivative of ``f`` and, for Newton refinement, the third.
"""

from __future__ import annotations

from collections.abc import Callable

import jax.numpy as jnp
import numpy as np
from scipy.interpolate import PPoly, make_interp_spline


def spline(r_grid, values, k: int = 5) -> Callable:
    """Return a JAX-traceable spline interpolant ``fn(r)``.

    Parameters
    ----------
    r_grid : array_like
        Strictly increasing radial grid.
    values : array_like
        Function values on the grid.
    k : int
        Spline degree; the interpolant is ``C^(k-1)``. Use ``k >= 4`` if the
        ISCO or epicyclic frequencies are needed.

    Returns
    -------
    callable
        ``fn(r)``, differentiable with JAX autodiff. Outside the grid the
        function returns NaN (no extrapolation), so scans simply find
        nothing there.

    Notes
    -----
    The spline coefficients are computed once with SciPy; evaluation is a
    Horner scheme on the piecewise-polynomial form, written with
    ``jax.numpy``. Wrap it for use in a metric as
    ``StaticSpherical(lambda r: fn(r), r_range=(r_grid[0], r_grid[-1]))``.
    """
    r_grid = np.asarray(r_grid, dtype=float)
    values = np.asarray(values, dtype=float)
    if r_grid.ndim != 1 or r_grid.shape != values.shape:
        raise ValueError("r_grid and values must be 1-D arrays of the same length.")
    if np.any(np.diff(r_grid) <= 0):
        raise ValueError("r_grid must be strictly increasing.")
    pp = PPoly.from_spline(make_interp_spline(r_grid, values, k=k))
    # Drop the repeated boundary knots, which produce zero-length pieces.
    keep = np.diff(pp.x) > 0
    breaks = jnp.asarray(pp.x[:-1][keep])
    coeffs = jnp.asarray(pp.c[:, keep])  # shape (k + 1, n_pieces), highest power first
    lo, hi = float(r_grid[0]), float(r_grid[-1])
    n_pieces = int(breaks.shape[0])

    def fn(r):
        r = jnp.asarray(r, dtype=jnp.float64)
        idx = jnp.clip(jnp.searchsorted(breaks, r, side="right") - 1, 0, n_pieces - 1)
        dx = r - breaks[idx]
        out = jnp.zeros_like(r)
        for j in range(coeffs.shape[0]):
            out = out * dx + coeffs[j, idx]
        return jnp.where((r >= lo) & (r <= hi), out, jnp.nan)

    return fn


__all__ = ["spline"]
