r"""Photon spheres and the critical impact parameter.

Null geodesics of ``ds^2 = -f dt^2 + dr^2/g + h dOmega^2`` obey
:math:`\dot r^2 = g\,(E^2/f - L^2/h)`, i.e. they move in the effective
potential :math:`V = f/h` (per unit :math:`L^2`). Circular photon orbits sit
at the extrema of ``f/h``:

.. math::

    P(r) \equiv f' h - f h' = 0 .

A maximum of ``f/h`` (``P' < 0``) is an *unstable* photon sphere, which
forms the edge of the shadow; a minimum (``P' > 0``) is a *stable* photon
sphere (an "anti-photon sphere"). The critical impact parameter is
:math:`b_c = \sqrt{h/f}` evaluated at the unstable photon sphere.

Schwarzschild: ``r_ph = 3``, ``b_c = 3 sqrt(3)``.

Reference: V. Perlick & O. Yu. Tsupko, "Calculating black hole shadows:
Review of analytical studies", Phys. Rep. 947 (2022) 1, arXiv:2105.07101.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from blackpy._numerics import derivative, find_roots, nanmax_or_nan
from blackpy.geometry.horizons import static_domain_core

_EDGE = 1e-9


def _photon_function(metric, p):
    f_fn = lambda x: metric.f_array(x, p)
    h_fn = lambda x: metric.h_array(x, p)
    return lambda r: derivative(f_fn)(r) * h_fn(r) - f_fn(r) * derivative(h_fn)(r)


def photon_spheres_core(metric, p, n_scan=256, max_roots=6):
    """All photon spheres in the static domain.

    Returns
    -------
    roots : array (max_roots,)
        Ascending radii, NaN padded.
    unstable : array of bool
        True for an unstable photon sphere (maximum of ``f/h``).
    """
    _, _, r_in, r_out = static_domain_core(metric, p, n_scan=n_scan)
    photon = _photon_function(metric, p)
    roots, degenerate = find_roots(
        photon, r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE), n_scan=n_scan, max_roots=max_roots
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    slope = jax.vmap(derivative(photon))(safe)
    static = jax.vmap(lambda r: (metric.f_array(r, p) > 0.0) & (metric.g_array(r, p) > 0.0))(safe)
    valid = jnp.isfinite(roots) & static & (~degenerate)
    return jnp.where(valid, roots, jnp.nan), (slope < 0.0) & valid


def photon_sphere_core(metric, p, n_scan=256):
    """Outermost unstable photon sphere (NaN if none)."""
    roots, unstable = photon_spheres_core(metric, p, n_scan=n_scan, max_roots=3)
    return nanmax_or_nan(roots, unstable)


def critical_impact_parameter_core(metric, p, n_scan=256):
    """``b_c = sqrt(h/f)`` at the outermost unstable photon sphere."""
    r_ph = photon_sphere_core(metric, p, n_scan=n_scan)
    safe = jnp.where(jnp.isfinite(r_ph), r_ph, 1.0)
    ratio = metric.h_array(safe, p) / metric.f_array(safe, p)
    ok = jnp.isfinite(r_ph) & (ratio > 0.0)
    return jnp.where(ok, jnp.sqrt(jnp.where(ok, ratio, 1.0)), jnp.nan)


def _static_only(metric, what: str) -> None:
    if hasattr(metric, "plasma"):
        raise TypeError(
            f"{what} computes the vacuum quantity, but '{metric.name}' carries a plasma. Use "
            "blackpy.light.photon_sphere_in_plasma / photon_spheres_in_plasma / "
            "shadow_in_plasma(metric, plasma, ...), or pass the plain metric for the vacuum value."
        )
    if metric.is_rotating_class:
        raise TypeError(
            f"{what} is defined for static, spherically symmetric metrics. A rotating metric "
            f"such as '{metric.name}' has a photon region instead of a photon sphere: use "
            "blackpy.light.photon_region (spherical photon orbits), "
            "blackpy.light.critical_impact_parameters, or blackpy.orbits.photon_orbit "
            "(equatorial circular photon orbits)."
        )


def photon_sphere(metric, *, n_scan: int = 256, **params):
    """Radius of the (outermost) unstable photon sphere, vectorised.

    Parameters
    ----------
    metric : StaticSpherical
    n_scan : int
        Number of scan points between the horizon and the outer boundary.
    **params
        Metric parameters (scalars or arrays).

    Returns
    -------
    jax.Array
        ``r_ph`` in units of M; NaN if there is no unstable photon sphere.
    """
    _static_only(metric, "photon_sphere")
    return metric.evaluate(photon_sphere_core, params, n_scan=n_scan)


def critical_impact_parameter(metric, *, n_scan: int = 256, **params):
    """Critical impact parameter ``b_c = sqrt(h(r_ph) / f(r_ph))``, vectorised.

    Returns
    -------
    jax.Array
        ``b_c`` in units of M; NaN if there is no unstable photon sphere.
    """
    _static_only(metric, "critical_impact_parameter")
    return metric.evaluate(critical_impact_parameter_core, params, n_scan=n_scan)


@dataclass(frozen=True)
class PhotonSphere:
    """A circular photon orbit.

    Attributes
    ----------
    radius : float
        Radius in units of M.
    stable : bool
        False for the usual unstable photon sphere, True for a stable one.
    impact_parameter : float
        ``sqrt(h/f)`` at this radius.
    """

    radius: float
    stable: bool
    impact_parameter: float


def photon_spheres(metric, *, n_scan: int = 256, **params) -> list[PhotonSphere]:
    """All photon spheres for one set of scalar parameters, with stability.

    Returns
    -------
    list of PhotonSphere
        In ascending radius; empty if there are none.
    """
    _static_only(metric, "photon_spheres")
    roots, unstable = metric.evaluate(photon_spheres_core, params, n_scan=n_scan)
    roots, unstable = np.asarray(roots), np.asarray(unstable)
    if roots.ndim != 1:
        raise ValueError(
            "photon_spheres() needs scalar parameters; use photon_sphere() for arrays."
        )
    out = []
    for radius, is_unstable in zip(roots, unstable, strict=True):
        if np.isfinite(radius):
            b = float(np.sqrt(metric.h(radius, **params) / metric.f(radius, **params)))
            out.append(PhotonSphere(float(radius), not bool(is_unstable), b))
    return sorted(out, key=lambda ps: ps.radius)
