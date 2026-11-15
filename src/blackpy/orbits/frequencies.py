"""Orbital and epicyclic frequencies in physical units."""

from __future__ import annotations

from typing import NamedTuple

import jax

from blackpy.constants import omega_to_hz
from blackpy.errors import NodalPrecessionUnavailableError
from blackpy.orbits.circular import _options, frequencies_core


class Frequencies(NamedTuple):
    """Fundamental frequencies of a slightly perturbed circular orbit, in Hz.

    Attributes
    ----------
    nu_phi : orbital (Keplerian) frequency, as a positive number
    nu_r : radial epicyclic frequency
    nu_theta : vertical epicyclic frequency
    """

    nu_phi: jax.Array
    nu_r: jax.Array
    nu_theta: jax.Array


def frequencies_hz_core(metric, p, r, mass_msun, direction=1) -> Frequencies:
    """``(nu_phi, nu_r, nu_theta)`` in Hz (jit/vmap-compatible)."""
    omega_phi, omega_r, omega_theta = frequencies_core(metric, p, r, direction)
    return Frequencies(
        omega_to_hz(omega_phi, mass_msun),
        omega_to_hz(omega_r, mass_msun),
        omega_to_hz(omega_theta, mass_msun),
    )


def nodal_precession_hz_core(metric, p, r, mass_msun, direction=1):
    """Nodal precession frequency ``nu_phi - nu_theta`` in Hz (jit/vmap-compatible)."""
    nu_phi, _, nu_theta = frequencies_hz_core(metric, p, r, mass_msun, direction)
    return nu_phi - nu_theta


def frequencies_hz(metric, r, mass, *, direction="prograde", **params) -> Frequencies:
    r"""Orbital and epicyclic frequencies in hertz.

    Implements :math:`\nu = \tilde\Omega\, c^3 / (2\pi G M)` with constants
    from :mod:`blackpy.constants`.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    r : float or array
        Orbit radius in units of M (i.e. of G M / c^2).
    mass : float or array
        Black hole mass in solar masses.
    direction : {"prograde", "retrograde"}
        Rotating metrics only.
    **params
        Metric parameters.

    Returns
    -------
    Frequencies
        ``nu_phi`` (positive for both directions), ``nu_r``, ``nu_theta`` in
        Hz as measured by a distant static observer. ``nu_r`` is NaN inside
        the ISCO.
    """
    return metric.evaluate(frequencies_hz_core, params, r, mass, **_options(metric, direction))


def nodal_precession_frequency(metric, r, mass, *, direction="prograde", **params):
    r"""Nodal (Lense-Thirring) precession frequency :math:`\nu_\phi - \nu_\theta`.

    Parameters
    ----------
    metric : StationaryAxisymmetric
    r : float or array
        Orbit radius in units of M.
    mass : float or array
        Black hole mass in solar masses.
    direction : {"prograde", "retrograde"}
        Orbit direction.
    **params
        Metric parameters.

    Returns
    -------
    jax.Array
        :math:`\nu_{\rm nod} = \nu_\phi - \nu_\theta` in Hz, with
        :math:`\nu_\phi > 0`. It is positive for orbits co-rotating with the
        hole and negative for counter-rotating ones; QPO models compare its
        absolute value with the observed low-frequency QPO.

    Raises
    ------
    NodalPrecessionUnavailableError
        For static, spherically symmetric metrics
        (:class:`blackpy.metric.StaticSpherical`): there
        :math:`\nu_\theta = \nu_\phi` exactly, so the nodal precession
        frequency is identically zero and cannot model a low-frequency QPO.
    """
    if not metric.is_rotating_class:
        raise NodalPrecessionUnavailableError(
            "The nodal precession frequency nu_phi - nu_theta vanishes identically in a "
            f"spherically symmetric spacetime such as '{metric.name}', so low-frequency (type-C) "
            "QPOs cannot be modelled with it. Use a rotating metric (blackpy.metric.Kerr, "
            "KerrNewman, Johannsen, or rotating_from_static(metric, a)), or fit only the twin "
            "high-frequency QPOs."
        )
    return metric.evaluate(nodal_precession_hz_core, params, r, mass, **_options(metric, direction))
