r"""Strong deflection limit and its observables.

Close to the critical impact parameter :math:`b_c` the deflection angle
diverges logarithmically (Bozza 2002, Phys. Rev. D 66, 103001, Eq. (9)),

.. math::

    \alpha(b) = -\bar a\,\ln\Bigl(\frac{b}{b_c} - 1\Bigr) + \bar b
                + O\bigl((b - b_c)\ln(b - b_c)\bigr).

The coefficients are computed in the improved form of Tsukamoto (2017),
Phys. Rev. D 95, 064035, Eqs. (28)-(55), which uses the variable
:math:`z = 1 - r_0/r` and works for any asymptotically flat, static,
spherically symmetric metric. With :math:`A = f`, :math:`B = 1/g`,
:math:`C = h` and everything evaluated at the photon sphere :math:`r_m`:

.. math::

    \bar a = \sqrt{\frac{2B_mA_m}{C_m''A_m - C_mA_m''}}, \qquad
    \bar b = \bar a\,\ln\Bigl[r_m^2\Bigl(\frac{C_m''}{C_m}
             - \frac{A_m''}{A_m}\Bigr)\Bigr] + I_R(r_m) - \pi,

    I_R(r_m) = \int_0^1\Bigl[\frac{2r_m}{\sqrt{G(z)}}
               - \frac{2r_m}{\sqrt{c_2}\,z}\Bigr]dz ,

with :math:`G` and :math:`c_2` as in :mod:`blackpy.lensing.deflection`. In
BlackPy both coefficients are written with :math:`H = h/f` (or
:math:`h_{\rm pl}^2` in a plasma), which is the same thing in vacuum:
:math:`\bar a = r_m/\sqrt{c_2}`,
:math:`\bar b = \bar a\ln(r_m^2H_m''/H_m) + I_R - \pi`. The derivatives come
from automatic differentiation.

Observables (Bozza 2002, section IV), for a lens at distance
:math:`D_{OL}`: the angular radius of the innermost image
:math:`\theta_\infty = b_c/D_{OL}` (his Eq. (85)), the separation between
the outermost relativistic image and the others
:math:`s = \theta_\infty e^{(\bar b - 2\pi)/\bar a}` (Eq. (89)), and their
flux ratio :math:`r = e^{2\pi/\bar a}` (Eq. (90)), quoted in magnitudes as
:math:`r_m = 2.5\log_{10}r`.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from blackpy import constants
from blackpy._numerics import value_and_derivative
from blackpy.errors import BlackPyError
from blackpy.lensing.deflection import (
    _g_function,
    _require_flat,
    _static_only,
    _view,
    effective_h2,
    refractive_index_squared_at_infinity,
    regular_integral,
    taylor_coefficients,
)


class StrongDeflection(NamedTuple):
    r"""Coefficients of the strong deflection limit.

    Attributes
    ----------
    a_bar, b_bar : the coefficients :math:`\bar a`, :math:`\bar b`
    b_c : critical impact parameter (units of M); Bozza's :math:`u_m`
    r_m : radius of the photon sphere (units of M)
    """

    a_bar: jax.Array
    b_bar: jax.Array
    b_c: jax.Array
    r_m: jax.Array


def strong_deflection_core(view, p, n=128, n_scan=256) -> StrongDeflection:
    if hasattr(view, "plasma"):
        from blackpy.light.plasma import static_shadow_core

        r_m = static_shadow_core(view, p, jnp.asarray(jnp.inf), n_scan=n_scan, finite=False).r_ph
    else:
        from blackpy.light.photon_sphere import photon_sphere_core

        r_m = photon_sphere_core(view, p, n_scan=n_scan)
    ok = jnp.isfinite(r_m)
    safe = jnp.where(ok, r_m, 3.0)
    _, c2, c3, c4 = taylor_coefficients(view, p, safe)
    ok = ok & (c2 > 0.0)
    c2 = jnp.where(ok, c2, 1.0)
    big_g = _g_function(view, p, safe)
    h2 = lambda r: effective_h2(view, p, r)
    h2_m = h2(safe)
    curvature = value_and_derivative(lambda r: value_and_derivative(h2, r)[1], safe)[1]
    # At the photon sphere c_1 = 0: the subtracted term is 1 / (sqrt(c_2) z).
    regular = 2.0 * safe * regular_integral(big_g, 0.0, c2, c3, c4, n)
    a_bar = safe / jnp.sqrt(c2)
    b_bar = a_bar * jnp.log(safe**2 * curvature / h2_m) + regular - math.pi
    n2 = refractive_index_squared_at_infinity(view, p)
    b_c = jnp.sqrt(h2_m / n2)
    nan = jnp.nan
    return StrongDeflection(
        jnp.where(ok, a_bar, nan), jnp.where(ok, b_bar, nan), jnp.where(ok, b_c, nan), r_m
    )


def strong_deflection_limit(
    metric, *, plasma=None, n: int = 128, n_scan: int = 256, **params
) -> StrongDeflection:
    r"""Coefficients :math:`\bar a`, :math:`\bar b`, :math:`b_c` of the strong deflection limit.

    .. math:: \alpha(b) \simeq -\bar a\,\ln(b/b_c - 1) + \bar b .

    Parameters
    ----------
    metric : StaticSpherical
        Asymptotically flat, with an unstable photon sphere.
    plasma : blackpy.light.PlasmaModel, optional
    **params
        Metric (and plasma) parameters, scalars or arrays.

    Returns
    -------
    StrongDeflection
        NaN where there is no photon sphere. Known values (tests):
        Schwarzschild :math:`\bar a = 1`,
        :math:`\bar b = \ln[216(7 - 4\sqrt3)] - \pi = -0.4002`,
        :math:`b_c = 3\sqrt3` (Bozza 2002, Eqs. (47)-(50)); extremal
        Reissner-Nordstrom :math:`\bar a = \sqrt2`,
        :math:`\bar b = 2\sqrt2\ln[4(2 - \sqrt2)] - \pi`, :math:`b_c = 4`
        (Tsukamoto 2017, Eqs. (78)-(80)).

    Notes
    -----
    The formulas are those of Tsukamoto (2017); see the module docstring.
    The error of the limit is of order :math:`(b - b_c)\ln(b - b_c)`: at the
    outermost relativistic image of a Schwarzschild lens it is 0.06% of the
    deflection angle (Bozza 2002), which the tests reproduce.
    """
    _static_only(metric, "strong_deflection_limit")
    _require_flat(metric, params, plasma)
    return _view(metric, plasma).evaluate(strong_deflection_core, params, n=n, n_scan=n_scan)


def strong_deflection_angle(metric, b, *, plasma=None, **params):
    r"""Deflection angle in the strong deflection limit, :math:`-\bar a\ln(b/b_c - 1) + \bar b`.

    An approximation valid for ``b`` close to the critical impact parameter;
    compare with the exact :func:`blackpy.lensing.deflection_angle`.
    """
    limit = strong_deflection_limit(metric, plasma=plasma, **params)
    ratio = jnp.asarray(b) / limit.b_c - 1.0
    return jnp.where(ratio > 0.0, -limit.a_bar * jnp.log(ratio) + limit.b_bar, jnp.nan)


class StrongFieldObservables(NamedTuple):
    r"""Observables of the relativistic images (Bozza 2002, section IV).

    Attributes
    ----------
    theta_infinity : angular radius of the innermost images, :math:`b_c/D_{OL}` (microarcsec)
    separation : :math:`s`, separation of the outermost image from the others (microarcsec)
    flux_ratio : :math:`r`, flux of the outermost image over the sum of the others
    magnitude_difference : :math:`r_m = 2.5\log_{10}r` (magnitudes)
    theta_g : angular gravitational radius :math:`GM/(c^2D_{OL})` (microarcsec)
    a_bar, b_bar, b_c : the strong-deflection coefficients used
    """

    theta_infinity: jax.Array
    separation: jax.Array
    flux_ratio: jax.Array
    magnitude_difference: jax.Array
    theta_g: jax.Array
    a_bar: jax.Array
    b_bar: jax.Array
    b_c: jax.Array


def _lens(mass_msun, distance_kpc, dataset, mass_prior):
    """Mass (solar masses) and distance (kpc), given directly or by a shadow dataset.

    Warns (``UnverifiedDataWarning``) when the dataset has not been verified by a human;
    the warning points at the caller of :func:`strong_field_observables`.
    """
    if dataset is None:
        if mass_msun is None or distance_kpc is None:
            raise TypeError(
                "Give the lens either as mass_msun=... and distance_kpc=..., or as "
                "dataset='eht/sgra' (a shadow dataset with mass priors and a distance)."
            )
        return float(mass_msun), float(distance_kpc)
    from blackpy.data import ShadowDataset, load
    from blackpy.data.registry import warn_if_unverified

    data = load(dataset) if isinstance(dataset, str) else dataset
    if not isinstance(data, ShadowDataset) or data.distance is None or not data.mass_priors:
        raise BlackPyError(
            "The dataset must be a shadow dataset that stores a distance and mass priors "
            "(for example 'eht/sgra' or 'eht/m87')."
        )
    key = mass_prior or (data.assumed_mass_priors[0] if data.assumed_mass_priors else None)
    if key not in data.mass_priors:
        raise BlackPyError(
            f"Choose the mass of '{data.id}' with mass_prior=<one of {sorted(data.mass_priors)}>."
        )
    units = {"Msun": 1.0, "1e6 Msun": 1e6, "1e9 Msun": 1e9}
    lengths = {"kpc": 1.0, "Mpc": 1e3, "pc": 1e-3}
    prior, distance = data.mass_priors[key], data.distance
    if prior.unit not in units or distance.unit not in lengths:
        raise BlackPyError(
            f"Unknown unit in '{data.id}': mass in '{prior.unit}', distance in '{distance.unit}'. "
            "Pass the lens as mass_msun=... and distance_kpc=... instead."
        )
    warn_if_unverified(data, "strong_field_observables", stacklevel=4)
    return prior.value * units[prior.unit], distance.value * lengths[distance.unit]


def strong_field_observables(
    metric,
    *,
    mass_msun: float | None = None,
    distance_kpc: float | None = None,
    dataset=None,
    mass_prior: str | None = None,
    plasma=None,
    **params,
) -> StrongFieldObservables:
    r"""Positions and brightness of the relativistic images of a distant source.

    Predictions for a lens of given mass and distance: the angular radius
    :math:`\theta_\infty` at which the relativistic images accumulate, the
    separation :math:`s` of the outermost one from the rest, and their flux
    ratio (Bozza 2002, Eqs. (85), (89), (90)).

    **These are predictions, not constraints.** Relativistic images have not
    been resolved for any black hole, so no data exist to compare with.

    Parameters
    ----------
    metric : StaticSpherical
    mass_msun, distance_kpc : float, optional
        Mass of the lens in solar masses and its distance in kiloparsec.
    dataset : ShadowDataset or registry id, optional
        Take mass and distance from a shadow dataset instead (``"eht/sgra"``,
        ``"eht/m87"``); ``mass_prior`` selects which of its mass
        measurements (default: the first one its shadow bound assumes). Mass
        and distance of such sources are correlated: only their ratio enters
        here. A dataset not yet verified by a human emits
        :class:`~blackpy.errors.UnverifiedDataWarning`, as in fits.
    plasma, **params
        As in :func:`strong_deflection_limit`.

    Returns
    -------
    StrongFieldObservables
        Angles in microarcseconds.
    """
    mass, distance = _lens(mass_msun, distance_kpc, dataset, mass_prior)
    limit = strong_deflection_limit(metric, plasma=plasma, **params)
    theta_g = (
        constants.gravitational_radius_m(mass)
        / (distance * 1e3 * constants.PARSEC)
        / constants.MICROARCSECOND
    )
    theta_infinity = limit.b_c * theta_g
    flux_ratio = jnp.exp(2.0 * math.pi / limit.a_bar)
    return StrongFieldObservables(
        theta_infinity=theta_infinity,
        separation=theta_infinity * jnp.exp((limit.b_bar - 2.0 * math.pi) / limit.a_bar),
        flux_ratio=flux_ratio,
        magnitude_difference=2.5 * jnp.log10(flux_ratio),
        theta_g=jnp.asarray(theta_g),
        a_bar=limit.a_bar,
        b_bar=limit.b_bar,
        b_c=limit.b_c,
    )


def einstein_ring_angle(
    mass_msun: float, distance_lens_kpc: float, distance_source_kpc: float
) -> float:
    r"""Angular radius of the (weak-field) Einstein ring, in microarcseconds.

    .. math:: \theta_E = \sqrt{\frac{4GM}{c^2}\,\frac{D_{LS}}{D_{OL}D_{OS}}}

    for a source exactly behind the lens, with :math:`D_{LS} = D_{OS} -
    D_{OL}` (Euclidean distances; the standard result of the lens equation
    with the first-order deflection :math:`4M/b`, which is the same for
    every asymptotically flat metric of mass :math:`M`). The relativistic
    Einstein rings lie at :math:`\theta_\infty(1 + e^{(\bar b - 2n\pi)/\bar
    a})` (Bozza 2002, Eq. (77)); see :func:`strong_field_observables`.

    Parameters
    ----------
    mass_msun : float
    distance_lens_kpc, distance_source_kpc : float
        Distances of the lens and of the source from the observer.
    """
    if not 0.0 < distance_lens_kpc < distance_source_kpc:
        raise ValueError("The source must be farther away than the lens (and both positive).")
    kpc = 1e3 * constants.PARSEC
    d_ol, d_os = distance_lens_kpc * kpc, distance_source_kpc * kpc
    radius = constants.gravitational_radius_m(mass_msun)
    return float(np.sqrt(4.0 * radius * (d_os - d_ol) / (d_ol * d_os)) / constants.MICROARCSECOND)
