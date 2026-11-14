"""Orbital and epicyclic frequencies in physical units."""

from __future__ import annotations

from typing import NamedTuple

import jax

from blackpy.constants import omega_to_hz
from blackpy.errors import NodalPrecessionUnavailableError
from blackpy.orbits.circular import frequencies_core


class Frequencies(NamedTuple):
    """Fundamental frequencies of a slightly perturbed circular orbit, in Hz.

    Attributes
    ----------
    nu_phi : orbital (Keplerian) frequency
    nu_r : radial epicyclic frequency
    nu_theta : vertical epicyclic frequency
    """

    nu_phi: jax.Array
    nu_r: jax.Array
    nu_theta: jax.Array


def frequencies_hz_core(metric, p, r, mass_msun) -> Frequencies:
    """``(nu_phi, nu_r, nu_theta)`` in Hz (jit/vmap-compatible)."""
    omega_phi, omega_r, omega_theta = frequencies_core(metric, p, r)
    return Frequencies(
        omega_to_hz(omega_phi, mass_msun),
        omega_to_hz(omega_r, mass_msun),
        omega_to_hz(omega_theta, mass_msun),
    )


def frequencies_hz(metric, r, mass, **params) -> Frequencies:
    r"""Orbital and epicyclic frequencies in hertz.

    Implements :math:`\nu = \tilde\Omega\, c^3 / (2\pi G M)` with constants
    from :mod:`blackpy.constants`.

    Parameters
    ----------
    metric : StaticSpherical
    r : float or array
        Orbit radius in units of M (i.e. of G M / c^2).
    mass : float or array
        Black hole mass in solar masses.
    **params
        Metric parameters.

    Returns
    -------
    Frequencies
        ``nu_phi``, ``nu_r``, ``nu_theta`` in Hz as measured by a distant
        static observer. ``nu_r`` is NaN inside the ISCO.
    """
    return metric.evaluate(frequencies_hz_core, params, r, mass)


def nodal_precession_frequency(metric, r, mass, **params):
    r"""Nodal (Lense-Thirring) precession frequency :math:`\nu_\phi - \nu_\theta`.

    Raises
    ------
    NodalPrecessionUnavailableError
        Always, for spherically symmetric metrics: there
        :math:`\nu_\theta = \nu_\phi` exactly, so the nodal precession
        frequency is identically zero and cannot model a low-frequency QPO.
        This needs a rotating metric, which is outside the MVP.
    """
    if metric.is_spherically_symmetric:
        raise NodalPrecessionUnavailableError(
            "The nodal precession frequency nu_phi - nu_theta vanishes identically in a "
            f"spherically symmetric spacetime such as '{metric.name}', so low-frequency (type-C) "
            "QPOs cannot be modelled. Fit only the twin high-frequency QPOs, or wait for "
            "rotating-metric support (see DESIGN_NOTES.md, roadmap)."
        )
    raise NotImplementedError("Nodal precession for rotating metrics is not implemented yet.")
