r"""Equatorial circular orbits and epicyclic frequencies of rotating metrics.

Metric: stationary, axisymmetric and symmetric under
:math:`\theta \to \pi - \theta`,

.. math::

    ds^2 = g_{tt}dt^2 + 2g_{t\phi}dt\,d\phi + g_{\phi\phi}d\phi^2
           + g_{rr}dr^2 + g_{\theta\theta}d\theta^2 .

Notation (all on the equatorial plane): :math:`A = -g_{tt}`,
:math:`B = g_{t\phi}`, :math:`C = g_{\phi\phi}`,
:math:`\varrho^2 = B^2 + AC`; primes are radial derivatives and a subscript
:math:`\theta\theta` is a second polar derivative. The derivation is in
``docs/theory.md`` (section 12); summary:

* Angular velocity (radial geodesic equation,
  :math:`A' - 2B'\Omega - C'\Omega^2 = 0`):

  .. math:: \Omega_\pm = \frac{-B' \pm \sqrt{B'^2 + A'C'}}{C'} .

  The upper sign is the *prograde* direction (positive :math:`\phi`
  direction; for :math:`a > 0` in Kerr this is co-rotation), the lower sign
  the *retrograde* one.
* Normalisation: :math:`(u^t)^{-2} = D \equiv A - 2B\Omega - C\Omega^2`,
  :math:`E = e/\sqrt D`, :math:`L = \ell/\sqrt D` with
  :math:`e = A - B\Omega`, :math:`\ell = B + C\Omega`.
* Epicyclic frequencies seen from infinity:

  .. math::

      \Omega_r^2 = \frac{N_r}{2 g_{rr}\varrho^2}, \quad
      N_r = D(\varrho^2)'' - C''e^2 - 2B''e\ell + A''\ell^2 ,

      \Omega_\theta^2 = \frac{N_\theta}{2 g_{\theta\theta}\varrho^2}, \quad
      N_\theta = D(\varrho^2)_{\theta\theta} - C_{\theta\theta}e^2
                 - 2B_{\theta\theta}e\ell + A_{\theta\theta}\ell^2 .

* ISCO: :math:`N_r = 0` with :math:`N_r` increasing outward; marginally
  bound orbit: :math:`E = 1`, i.e. :math:`e^2 - D = 0`.

For :math:`B = 0`, :math:`A = f`, :math:`C = h\sin^2\theta` these reduce to
the static formulas of :mod:`blackpy.orbits.circular`
(:math:`N_r = N/h'`, :math:`\Omega_\theta = \Omega_\phi`).

All derivatives are automatic (forward-mode ``jax.jvp``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp

from blackpy._numerics import derivative, find_roots, nanmax_or_nan
from blackpy.geometry.horizons import static_domain_core

HALF_PI = 0.5 * math.pi

#: Relative offset used to keep scans strictly outside the horizon.
_EDGE = 1e-9


class Equatorial(NamedTuple):
    """Equatorial metric functions with their first and second radial derivatives."""

    a: jax.Array
    a1: jax.Array
    a2: jax.Array
    b: jax.Array
    b1: jax.Array
    b2: jax.Array
    c: jax.Array
    c1: jax.Array
    c2: jax.Array


def _abc(metric, p, r, theta):
    return jnp.stack(
        [
            -metric.gtt_array(r, theta, p),
            metric.gtphi_array(r, theta, p),
            metric.gphiphi_array(r, theta, p),
        ]
    )


def equatorial(metric, p, r) -> Equatorial:
    """``A, B, C`` and their first two radial derivatives at radius ``r``."""
    fn = lambda x: _abc(metric, p, x, HALF_PI)
    v, d1, d2 = fn(r), derivative(fn)(r), derivative(fn, 2)(r)
    return Equatorial(v[0], d1[0], d2[0], v[1], d1[1], d2[1], v[2], d1[2], d2[2])


class Orbit(NamedTuple):
    """Quantities of the circular orbit at one radius (NaN-free; see ``exists``)."""

    exists: jax.Array
    omega: jax.Array
    d: jax.Array
    e: jax.Array
    ell: jax.Array
    rho2: jax.Array


def _orbit(metric, p, r, q: Equatorial, direction) -> Orbit:
    disc = q.b1**2 + q.a1 * q.c1
    real = (disc >= 0.0) & (q.c1 != 0.0)
    omega = (-q.b1 + direction * jnp.sqrt(jnp.where(real, disc, 0.0))) / jnp.where(
        q.c1 != 0.0, q.c1, 1.0
    )
    d = q.a - 2.0 * q.b * omega - q.c * omega**2
    rho2 = q.b**2 + q.a * q.c
    g_rr = metric.grr_array(r, HALF_PI, p)
    exists = real & (q.c1 > 0.0) & (q.c > 0.0) & (rho2 > 0.0) & (g_rr > 0.0) & (d > 0.0)
    return Orbit(exists, omega, d, q.a - q.b * omega, q.b + q.c * omega, rho2)


def _radial_numerator(q: Equatorial, o: Orbit):
    """``N_r``; the orbit is radially stable where ``N_r > 0``."""
    rho2_dd = 2.0 * q.b1**2 + 2.0 * q.b * q.b2 + q.a2 * q.c + 2.0 * q.a1 * q.c1 + q.a * q.c2
    return o.d * rho2_dd - q.c2 * o.e**2 - 2.0 * q.b2 * o.e * o.ell + q.a2 * o.ell**2


def _radial_numerator_at(metric, p, r, direction):
    q = equatorial(metric, p, r)
    return _radial_numerator(q, _orbit(metric, p, r, q, direction))


def _sqrt_or_nan(value, ok):
    ok = ok & (value >= 0.0)
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, value, 1.0)), jnp.nan)


# ----------------------------------------------------------------------------- core functions


def exists_core(metric, p, r, direction=1):
    """True where a circular, equatorial, timelike geodesic exists at radius ``r``."""
    return _orbit(metric, p, r, equatorial(metric, p, r), direction).exists


def omega_phi_core(metric, p, r, direction=1):
    """``Omega_phi = dphi/dt`` (signed; NaN where no orbit exists)."""
    o = _orbit(metric, p, r, equatorial(metric, p, r), direction)
    return jnp.where(o.exists, o.omega, jnp.nan)


def energy_core(metric, p, r, direction=1):
    """Specific energy ``E = -u_t`` of the circular orbit at ``r``."""
    o = _orbit(metric, p, r, equatorial(metric, p, r), direction)
    return jnp.where(o.exists, o.e / jnp.sqrt(jnp.where(o.exists, o.d, 1.0)), jnp.nan)


def angular_momentum_core(metric, p, r, direction=1):
    """Specific angular momentum ``L = u_phi`` of the circular orbit at ``r`` (signed)."""
    o = _orbit(metric, p, r, equatorial(metric, p, r), direction)
    return jnp.where(o.exists, o.ell / jnp.sqrt(jnp.where(o.exists, o.d, 1.0)), jnp.nan)


def omega_r_squared_core(metric, p, r, direction=1):
    """Squared radial epicyclic frequency (negative: radially unstable orbit)."""
    q = equatorial(metric, p, r)
    o = _orbit(metric, p, r, q, direction)
    denom = 2.0 * metric.grr_array(r, HALF_PI, p) * o.rho2
    return jnp.where(o.exists, _radial_numerator(q, o) / jnp.where(o.exists, denom, 1.0), jnp.nan)


def omega_theta_squared_core(metric, p, r, direction=1):
    """Squared vertical epicyclic frequency (negative: vertically unstable orbit)."""
    q = equatorial(metric, p, r)
    o = _orbit(metric, p, r, q, direction)
    polar = derivative(lambda th: _abc(metric, p, r, th), 2)(jnp.asarray(HALF_PI))
    a_tt, b_tt, c_tt = polar[0], polar[1], polar[2]
    rho2_tt = 2.0 * q.b * b_tt + a_tt * q.c + q.a * c_tt
    numerator = o.d * rho2_tt - c_tt * o.e**2 - 2.0 * b_tt * o.e * o.ell + a_tt * o.ell**2
    denom = 2.0 * metric.gthth_array(r, HALF_PI, p) * o.rho2
    return jnp.where(o.exists, numerator / jnp.where(o.exists, denom, 1.0), jnp.nan)


def omega_r_core(metric, p, r, direction=1):
    """Radial epicyclic frequency (NaN where the orbit is unstable or absent)."""
    sq = omega_r_squared_core(metric, p, r, direction)
    return _sqrt_or_nan(sq, jnp.isfinite(sq))


def omega_theta_core(metric, p, r, direction=1):
    """Vertical epicyclic frequency (NaN where the orbit is unstable or absent)."""
    sq = omega_theta_squared_core(metric, p, r, direction)
    return _sqrt_or_nan(sq, jnp.isfinite(sq))


def frequencies_core(metric, p, r, direction=1):
    """``(|Omega_phi|, Omega_r, Omega_theta)`` in units of 1/M at radius ``r``."""
    return (
        jnp.abs(omega_phi_core(metric, p, r, direction)),
        omega_r_core(metric, p, r, direction),
        omega_theta_core(metric, p, r, direction),
    )


def marginally_stable_core(metric, p, n_scan=256, max_roots=6, detect_extrema=True, direction=1):
    """All marginally stable circular orbits outside the horizon.

    Returns
    -------
    roots : array (max_roots,)
        Radii with ``N_r = 0`` at which a circular orbit exists; NaN padded.
    inner_edge : array of bool
        True for an ISCO (``N_r`` increasing outward), False for an OSCO.
    """
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)
    numerator = lambda r: _radial_numerator_at(metric, p, r, direction)
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
    exists = jax.vmap(lambda r: exists_core(metric, p, r, direction))(safe)
    valid = jnp.isfinite(roots) & exists & (~degenerate)
    roots = jnp.where(valid, roots, jnp.nan)
    order = jnp.argsort(roots)
    return roots[order], (slope > 0.0)[order] & valid[order]


def isco_core(metric, p, n_scan=256, detect_extrema=True, direction=1):
    """ISCO radius: the outermost inner edge of a stable region (NaN if none)."""
    roots, inner_edge = marginally_stable_core(
        metric, p, n_scan=n_scan, max_roots=3, detect_extrema=detect_extrema, direction=direction
    )
    return nanmax_or_nan(roots, inner_edge)


def marginally_bound_core(metric, p, n_scan=256, direction=1):
    """Marginally bound circular orbit (``E = 1``): outermost solution."""
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)

    def bound(r):
        o = _orbit(metric, p, r, equatorial(metric, p, r), direction)
        return o.e**2 - o.d

    roots, _ = find_roots(bound, r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE), n_scan=n_scan)
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)

    def valid(r):
        o = _orbit(metric, p, r, equatorial(metric, p, r), direction)
        return o.exists & (o.e > 0.0)

    return nanmax_or_nan(roots, jax.vmap(valid)(safe))


def photon_orbit_core(metric, p, n_scan=256, direction=1):
    """Equatorial circular photon orbit: outermost root of ``D`` outside the horizon."""
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)

    def null(r):
        return _orbit(metric, p, r, equatorial(metric, p, r), direction).d

    roots, _ = find_roots(null, r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE), n_scan=n_scan)
    return nanmax_or_nan(roots, jnp.isfinite(roots))


def existence_boundaries_core(metric, p, n_scan=256, max_roots=6, direction=1):
    """Radii where the existence condition can change: roots of ``D``, of the
    discriminant ``B'^2 + A'C'`` and of ``C'``."""
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)
    lo, hi = r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE)

    def functions(r):
        q = equatorial(metric, p, r)
        return _orbit(metric, p, r, q, direction).d, q.b1**2 + q.a1 * q.c1, q.c1

    found = [
        find_roots(lambda r, i=i: functions(r)[i], lo, hi, n_scan=n_scan, max_roots=max_roots)[0]
        for i in range(3)
    ]
    return jnp.sort(jnp.concatenate(found)), lo, hi
