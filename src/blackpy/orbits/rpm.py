r"""Analytic solution of the relativistic precession model in the Kerr metric.

Reference
---------
A. Ingram and S. Motta, "Solutions to the relativistic precession model",
MNRAS 444 (2014) 2065, arXiv:1408.0884. Equations (1)-(9) were transcribed
from the arXiv version.

In the relativistic precession model (Stella & Vietri 1998, 1999) the three
QPOs of a triplet are, at one radius ``r`` of the Kerr metric,

.. math::

    \nu_U = \nu_\phi, \qquad \nu_L = \nu_{\rm per} = \nu_\phi - \nu_r, \qquad
    \nu_{\rm low} = \nu_{\rm nod} = \nu_\phi - \nu_\theta .

With :math:`\Gamma = (1 - \nu_{\rm per}/\nu_\phi)^2` and
:math:`\Delta = (1 - \nu_{\rm nod}/\nu_\phi)^2` (their Eqs. (4), (5)) the
solution for prograde orbits is (Eqs. (9), (6), (1))

.. math::

    r = \frac{2}{3}\,\frac{6 - \Delta - 5\Gamma
        + 2\sqrt{2(\Delta - \Gamma)(3 - \Delta - 2\Gamma)}}{(\Delta + \Gamma - 2)^2},
    \quad
    a = \frac{r^{3/2}}{4}\left[\Delta + \Gamma - 2 + \frac{6}{r}\right],
    \quad
    M = \frac{c^3}{2\pi G M_\odot}\,\frac{1}{\nu_\phi\,(r^{3/2} + a)} .

BlackPy evaluates :math:`c^3 / (2\pi G M_\odot)` from its own constants
(32 312.5 Hz); Ingram & Motta quote 3.237e4 Hz, which is 0.18% larger and
shifts the mass by the same fraction.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp

from blackpy.constants import T_SUN


class RPMSolution(NamedTuple):
    """Solution of the relativistic precession model for one QPO triplet.

    Attributes
    ----------
    radius : orbital radius in units of G M / c^2
    spin : dimensionless spin a (prograde solution, ``a >= 0``)
    mass : black hole mass in solar masses
    """

    radius: jnp.ndarray
    spin: jnp.ndarray
    mass: jnp.ndarray


def rpm_kerr_solution(nu_low, nu_lower, nu_upper) -> RPMSolution:
    """Radius, spin and mass from a QPO triplet (Ingram & Motta 2014, Eqs. (4)-(9)).

    Parameters
    ----------
    nu_low : float or array
        Low-frequency (type-C) QPO in Hz, identified with the nodal
        precession frequency.
    nu_lower : float or array
        Lower high-frequency QPO in Hz (periastron precession frequency).
    nu_upper : float or array
        Upper high-frequency QPO in Hz (orbital frequency).

    Returns
    -------
    RPMSolution
        ``(radius, spin, mass)`` for prograde orbits. As noted in the paper,
        the retrograde solution has the same radius and mass and the
        opposite spin. NaN where the frequencies admit no solution.

    Notes
    -----
    The function is JAX-traceable and broadcasts, so uncertainties can be
    propagated by evaluating it on Monte Carlo draws of the frequencies.
    """
    nu_low, nu_lower, nu_upper = (
        jnp.asarray(v, dtype=jnp.float64) for v in (nu_low, nu_lower, nu_upper)
    )
    gamma = (1.0 - nu_lower / nu_upper) ** 2
    delta = (1.0 - nu_low / nu_upper) ** 2
    disc = 2.0 * (delta - gamma) * (3.0 - delta - 2.0 * gamma)
    ok = disc >= 0.0
    root = jnp.sqrt(jnp.where(ok, disc, 0.0))
    radius = (2.0 / 3.0) * (6.0 - delta - 5.0 * gamma + 2.0 * root) / (delta + gamma - 2.0) ** 2
    spin = 0.25 * radius**1.5 * (delta + gamma - 2.0 + 6.0 / radius)
    mass = 1.0 / (2.0 * math.pi * T_SUN * nu_upper * (radius**1.5 + spin))
    nan = jnp.nan
    return RPMSolution(
        jnp.where(ok, radius, nan), jnp.where(ok, spin, nan), jnp.where(ok, mass, nan)
    )
