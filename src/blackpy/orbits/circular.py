r"""Circular, equatorial, timelike geodesics and their epicyclic frequencies.

Metric: ``ds^2 = -f dt^2 + dr^2/g + h dOmega^2`` (G = c = 1, lengths in M).
Primes denote d/dr. The full derivation is in ``docs/theory.md``; summary:

* Angular velocity seen from infinity (``Omega = dphi/dt``), from the radial
  geodesic equation:  ``Omega^2 = f' / h'``.
* Four-velocity normalisation: ``u^t = 1 / sqrt(D)`` with
  ``D = f - h f'/h'``, hence the specific energy and angular momentum
  ``E = f / sqrt(D)``, ``L = h Omega / sqrt(D)``.
* Circular orbits exist where ``f > 0``, ``g > 0``, ``h' > 0``, ``f' >= 0``
  and ``D > 0`` (the last condition fails at and inside the photon sphere).
* Radial epicyclic frequency (seen from infinity):

  .. math::

      \Omega_r^2 = \frac{g\,N}{2 f h h'}, \qquad
      N = f f'' h h' - 2 f'^2 h h' + f f' (2 h'^2 - h h'')

  For ``h = r^2`` the marginal-stability condition ``N = 0`` reduces to the
  familiar ``r f f'' - 2 r f'^2 + 3 f f' = 0``.
* Vertical epicyclic frequency: ``Omega_theta = Omega_phi`` exactly, for any
  spherically symmetric metric (orbital planes do not precess).

Schwarzschild limit: ``Omega^2 = 1/r^3`` and
``Omega_r^2 = Omega^2 (1 - 6/r)``.

All derivatives are automatic (forward-mode ``jax.jvp``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from blackpy._numerics import derivative, find_roots, nanmax_or_nan
from blackpy.errors import NotAsymptoticallyFlatError
from blackpy.geometry.horizons import static_domain_core

#: Relative offset used to keep scans strictly inside the static domain.
_EDGE = 1e-9


class _Derivs(NamedTuple):
    f: jax.Array
    f1: jax.Array
    f2: jax.Array
    g: jax.Array
    h: jax.Array
    h1: jax.Array
    h2: jax.Array


def _derivs(metric, p, r) -> _Derivs:
    f_fn = lambda x: metric.f_array(x, p)
    h_fn = lambda x: metric.h_array(x, p)
    return _Derivs(
        f=f_fn(r),
        f1=derivative(f_fn)(r),
        f2=derivative(f_fn, 2)(r),
        g=metric.g_array(r, p),
        h=h_fn(r),
        h1=derivative(h_fn)(r),
        h2=derivative(h_fn, 2)(r),
    )


def _exists(d: _Derivs):
    """Existence condition of a circular timelike geodesic."""
    return (
        (d.f > 0.0) & (d.g > 0.0) & (d.h1 > 0.0) & (d.f1 >= 0.0) & (d.f * d.h1 - d.h * d.f1 > 0.0)
    )


def _stability_numerator(d: _Derivs):
    """``N(r)``; circular orbits are radially stable where ``N > 0``."""
    return (
        d.f * d.f2 * d.h * d.h1
        - 2.0 * d.f1**2 * d.h * d.h1
        + d.f * d.f1 * (2.0 * d.h1**2 - d.h * d.h2)
    )


# ----------------------------------------------------------------------------- core functions


def exists_core(metric, p, r):
    """True where a circular timelike geodesic exists at radius ``r``."""
    return _exists(_derivs(metric, p, r))


def omega_phi_core(metric, p, r):
    """Orbital angular velocity ``Omega_phi = dphi/dt`` (NaN where no orbit exists)."""
    d = _derivs(metric, p, r)
    return jnp.where(_exists(d), jnp.sqrt(jnp.where(_exists(d), d.f1 / d.h1, 1.0)), jnp.nan)


def energy_core(metric, p, r):
    """Specific energy ``E = -u_t`` of the circular orbit at ``r``."""
    d = _derivs(metric, p, r)
    ok = _exists(d)
    denom = jnp.where(ok, d.f - d.h * d.f1 / d.h1, 1.0)
    return jnp.where(ok, d.f / jnp.sqrt(denom), jnp.nan)


def angular_momentum_core(metric, p, r):
    """Specific angular momentum ``L = u_phi`` of the circular orbit at ``r``."""
    d = _derivs(metric, p, r)
    ok = _exists(d)
    denom = jnp.where(ok, d.f - d.h * d.f1 / d.h1, 1.0)
    omega = jnp.sqrt(jnp.where(ok, d.f1 / d.h1, 1.0))
    return jnp.where(ok, d.h * omega / jnp.sqrt(denom), jnp.nan)


def omega_r_squared_core(metric, p, r):
    """Squared radial epicyclic frequency ``Omega_r^2`` (negative: unstable orbit)."""
    d = _derivs(metric, p, r)
    ok = _exists(d)
    denom = jnp.where(ok, 2.0 * d.f * d.h * d.h1, 1.0)
    return jnp.where(ok, d.g * _stability_numerator(d) / denom, jnp.nan)


def omega_r_core(metric, p, r):
    """Radial epicyclic frequency ``Omega_r`` (NaN where the orbit is unstable or absent)."""
    sq = omega_r_squared_core(metric, p, r)
    ok = sq >= 0.0
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, sq, 1.0)), jnp.nan)


def omega_theta_core(metric, p, r):
    """Vertical epicyclic frequency; equals ``Omega_phi`` in spherical symmetry."""
    return omega_phi_core(metric, p, r)


def frequencies_core(metric, p, r):
    """``(Omega_phi, Omega_r, Omega_theta)`` in units of 1/M at radius ``r``."""
    omega_phi = omega_phi_core(metric, p, r)
    return omega_phi, omega_r_core(metric, p, r), omega_phi


def marginally_stable_core(metric, p, n_scan=256, max_roots=6, detect_extrema=True):
    """All marginally stable circular orbits in the static domain.

    Returns
    -------
    roots : array (max_roots,)
        Radii with ``N = 0`` at which a circular orbit exists; NaN padded.
    inner_edge : array of bool
        True where the orbit is an inner edge of a stable region (``N``
        increasing: unstable inside, stable outside), i.e. an ISCO. False
        marks an outer edge (OSCO).
    """
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)
    numerator = lambda r: _stability_numerator(_derivs(metric, p, r))
    roots, degenerate = find_roots(
        numerator,
        r_in * (1.0 + _EDGE),
        r_out * (1.0 - _EDGE),
        n_scan=n_scan,
        max_roots=max_roots,
        detect_extrema=detect_extrema,
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    slope = jax.vmap(derivative(numerator))(safe)
    exists = jax.vmap(lambda r: exists_core(metric, p, r))(safe)
    valid = jnp.isfinite(roots) & exists & (~degenerate)
    roots = jnp.where(valid, roots, jnp.nan)
    order = jnp.argsort(roots)
    return roots[order], (slope > 0.0)[order] & valid[order]


def isco_core(metric, p, n_scan=256, detect_extrema=True):
    """ISCO radius: the outermost inner edge of a stable region (NaN if none).

    Only the three outermost marginally stable orbits are examined, which
    keeps the vectorised fast path cheap.
    """
    roots, inner_edge = marginally_stable_core(
        metric, p, n_scan=n_scan, max_roots=3, detect_extrema=detect_extrema
    )
    return nanmax_or_nan(roots, inner_edge)


def _bound_function(d: _Derivs):
    """``B(r) = (f^2 - f) h' + h f'``; ``B = 0`` is equivalent to ``E = 1``."""
    return (d.f**2 - d.f) * d.h1 + d.h * d.f1


def marginally_bound_core(metric, p, n_scan=256):
    """Marginally bound circular orbit (``E = 1``): outermost solution inside the ISCO."""
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)
    bound = lambda r: _bound_function(_derivs(metric, p, r))
    roots, _ = find_roots(bound, r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE), n_scan=n_scan)
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    exists = jax.vmap(lambda r: exists_core(metric, p, r))(safe)
    return nanmax_or_nan(roots, exists)


# ----------------------------------------------------------------------------- public API


def circular_orbit_exists(metric, r, **params):
    """Whether a circular, equatorial, timelike geodesic exists at radius ``r``."""
    return metric.evaluate(exists_core, params, r)


def angular_velocity(metric, r, **params):
    """Orbital angular velocity ``Omega_phi = dphi/dt = sqrt(f'/h')``.

    Parameters
    ----------
    metric : StaticSpherical
    r : float or array
        Orbit radius in units of M.
    **params
        Metric parameters.

    Returns
    -------
    jax.Array
        ``Omega_phi`` in units of 1/M, as measured by a static observer at
        infinity; NaN where no circular orbit exists.
    """
    return metric.evaluate(omega_phi_core, params, r)


def specific_energy(metric, r, **params):
    """Specific energy ``E`` of the circular orbit at ``r`` (dimensionless)."""
    return metric.evaluate(energy_core, params, r)


def specific_angular_momentum(metric, r, **params):
    """Specific angular momentum ``L`` of the circular orbit at ``r`` (units of M)."""
    return metric.evaluate(angular_momentum_core, params, r)


def radial_epicyclic_frequency(metric, r, **params):
    """Radial epicyclic angular frequency ``Omega_r`` in units of 1/M.

    NaN where the circular orbit is radially unstable or does not exist; use
    :func:`radial_epicyclic_frequency_squared` to see the sign.
    """
    return metric.evaluate(omega_r_core, params, r)


def radial_epicyclic_frequency_squared(metric, r, **params):
    """``Omega_r^2`` in units of 1/M^2 (negative for unstable circular orbits)."""
    return metric.evaluate(omega_r_squared_core, params, r)


def vertical_epicyclic_frequency(metric, r, **params):
    """Vertical epicyclic angular frequency ``Omega_theta`` in units of 1/M.

    Identical to :func:`angular_velocity` in spherical symmetry.
    """
    return metric.evaluate(omega_theta_core, params, r)


def isco(metric, *, n_scan: int = 256, detect_extrema: bool = True, **params):
    """Innermost stable circular orbit (vectorised over parameters).

    Definition used: the outermost radius at which circular orbits change
    from unstable (inside) to stable (outside). For ordinary black holes
    this is the unique ISCO; if several stable regions exist it is the inner
    edge of the outermost one. Use :func:`marginally_stable_orbits` to see
    all solutions.

    Parameters
    ----------
    metric : StaticSpherical
    n_scan : int
        Number of scan points between the horizon and the outer boundary.
    detect_extrema : bool
        Also resolve pairs of marginally stable orbits closer than the scan
        resolution (needed close to a merger of ISCO and OSCO). Setting it to
        False is roughly twice as fast.
    **params
        Metric parameters (scalars or arrays).

    Returns
    -------
    jax.Array
        ISCO radius in units of M; NaN if no stable circular orbit exists.
    """
    return metric.evaluate(isco_core, params, n_scan=n_scan, detect_extrema=detect_extrema)


def marginally_bound_orbit(metric, *, n_scan: int = 256, **params):
    """Marginally bound circular orbit, ``E(r_mb) = 1`` (vectorised).

    Raises
    ------
    NotAsymptoticallyFlatError
        If the metric is not asymptotically flat: ``E = 1`` then has no
        meaning as the energy of a particle at rest at infinity.
    """
    if not metric.is_asymptotically_flat(**params):
        raise NotAsymptoticallyFlatError(
            f"The marginally bound orbit is defined by E = 1, the energy of a particle at rest "
            f"at infinity, but metric '{metric.name}' is not asymptotically flat for these "
            "parameters. If it is flat but approaches Minkowski slowly, declare "
            "asymptotically_flat=True when constructing the metric."
        )
    return metric.evaluate(marginally_bound_core, params, n_scan=n_scan)


@dataclass(frozen=True)
class MarginallyStableOrbit:
    """A marginally stable circular orbit.

    Attributes
    ----------
    radius : float
        Radius in units of M.
    kind : str
        ``"ISCO"`` (inner edge of a stable region) or ``"OSCO"`` (outer edge).
    """

    radius: float
    kind: str


def marginally_stable_orbits(metric, *, n_scan: int = 256, **params) -> list[MarginallyStableOrbit]:
    """All marginally stable circular orbits for one set of scalar parameters.

    Returns
    -------
    list of MarginallyStableOrbit
        In ascending radius; empty if there are none.
    """
    roots, inner_edge = metric.evaluate(marginally_stable_core, params, n_scan=n_scan)
    roots, inner_edge = np.asarray(roots), np.asarray(inner_edge)
    if roots.ndim != 1:
        raise ValueError(
            "marginally_stable_orbits() needs scalar parameters; use isco() for arrays."
        )
    return [
        MarginallyStableOrbit(float(r), "ISCO" if edge else "OSCO")
        for r, edge in zip(roots, inner_edge, strict=True)
        if np.isfinite(r)
    ]


def _existence_boundaries_core(metric, p, n_scan=256, max_roots=6):
    """Radii where the existence condition can change: roots of f' and of f h' - h f'."""
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)
    lo, hi = r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE)
    f_prime = lambda r: _derivs(metric, p, r).f1
    photon = lambda r: (lambda d: d.f * d.h1 - d.h * d.f1)(_derivs(metric, p, r))
    a, _ = find_roots(f_prime, lo, hi, n_scan=n_scan, max_roots=max_roots)
    b, _ = find_roots(photon, lo, hi, n_scan=n_scan, max_roots=max_roots)
    return jnp.sort(jnp.concatenate([a, b])), lo, hi


def circular_orbit_region(metric, *, n_scan: int = 256, **params) -> list[tuple[float, float]]:
    """Radial intervals in which circular timelike geodesics exist.

    The boundaries are photon spheres (``f h' - h f' = 0``, where the orbit
    becomes null) and static radii (``f' = 0``, where gravity is balanced,
    e.g. by a cosmological constant), together with the edges of the static
    domain.

    Parameters
    ----------
    metric : StaticSpherical
    **params
        Scalar metric parameters.

    Returns
    -------
    list of (float, float)
        Open intervals ``(r_low, r_high)`` in units of M. ``r_high`` equal
        to ``metric.r_range[1]`` means "up to the edge of the scan range".
    """
    bounds, lo, hi = metric.evaluate(_existence_boundaries_core, params, n_scan=n_scan)
    bounds = np.asarray(bounds)
    if bounds.ndim != 1:
        raise ValueError("circular_orbit_region() needs scalar parameters.")
    edges = [float(lo), *[float(b) for b in bounds if np.isfinite(b)], float(hi)]
    intervals = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        if b <= a:
            continue
        mid = float(np.sqrt(a * b))
        if bool(circular_orbit_exists(metric, mid, **params)):
            if intervals and np.isclose(intervals[-1][1], a, rtol=1e-12):
                intervals[-1] = (intervals[-1][0], b)
            else:
                intervals.append((a, b))
    return intervals
