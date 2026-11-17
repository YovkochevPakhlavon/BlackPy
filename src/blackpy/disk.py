r"""Observables of a geometrically thin, optically thick accretion disk.

Radiative efficiency
--------------------
Matter that spirals in through a sequence of circular orbits and falls into
the hole from the inner edge :math:`r_{\rm in}` (by default the ISCO) has
radiated the fraction

.. math:: \eta = 1 - E(r_{\rm in})

of its rest mass, with :math:`E = -u_t` the specific energy of the circular
orbit (Page & Thorne 1974, section I; for Kerr their Eq. (15g) gives
:math:`E = (1 - 2/r + a r^{-3/2})/\sqrt{1 - 3/r + 2a r^{-3/2}}` in units of
M).

Page-Thorne flux
----------------
D. N. Page & K. S. Thorne, "Disk-accretion onto a black hole. I.
Time-averaged structure of accretion disk", Astrophys. J. 191 (1974) 499,
Eqs. (11b) and (12) (read from the scanned journal version provided by the
NASA ADS). For a stationary, axisymmetric, reflection-symmetric metric
written near the equatorial plane as their Eq. (1),
:math:`ds^2 = -e^{2\nu}dt^2 + e^{2\psi}(d\phi - \omega dt)^2 + e^{2\mu}dr^2 + dz^2`,
the time-averaged flux emitted from each face of the disk per unit proper
time and proper area, measured by an observer comoving with the disk, is

.. math::

    F(r) = \frac{\dot M_0}{4\pi}\,e^{-(\nu+\psi+\mu)}\,f(r), \qquad
    f(r) = -\frac{\Omega_{,r}}{(E - \Omega L)^2}
           \int_{r_{\rm in}}^r (E - \Omega L)\,L_{,r}\,dr ,

with :math:`E`, :math:`L`, :math:`\Omega` the specific energy, specific
angular momentum and angular velocity of circular geodesics. In BlackPy's
notation (**derived**, see ``docs/theory.md``, section 24)

.. math::

    e^{\nu+\psi+\mu} = \sqrt{(g_{t\phi}^2 - g_{tt}g_{\phi\phi})\,g_{rr}}
    \Big|_{\theta = \pi/2},

which is :math:`\sqrt{fh/g}` for a static metric and :math:`r` for Kerr
(their Eq. (15d)). The flux vanishes at the inner edge (zero torque there,
their section IIIc) and falls as :math:`r^{-3}` at large radius. The
derivative :math:`L_{,r}` and :math:`\Omega_{,r}` come from automatic
differentiation, the integral from Gauss-Legendre quadrature.

Results are in units :math:`G = c = M = 1`: ``flux`` is
:math:`F\,M^2/\dot M_0`. The effective temperature follows from
:math:`\sigma T_{\rm eff}^4 = F` and is returned up to its normalisation.

Not in this module
------------------
Continuum-fitting and iron-line likelihoods. Comparing a disk with an
observed spectrum needs the image of the disk on the observer's sky (ray
tracing with light bending, Doppler and gravitational redshift), which is
outside the scope of BlackPy; no likelihood is offered that would ignore
it.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np

from blackpy._numerics import find_roots, gauss_legendre, value_and_derivative
from blackpy.errors import BlackPyError
from blackpy.orbits.circular import (
    _options,
    angular_momentum_core,
    direction_sign,
    energy_core,
    exists_core,
    marginally_stable_orbits,
    omega_phi_core,
    require_equatorial_orbits,
)

HALF_PI = 0.5 * math.pi


# ----------------------------------------------------------------------------- cores


def _sqrt_minus_g(metric, p, r):
    r""":math:`e^{\nu+\psi+\mu}` of Page & Thorne: the (t, r, phi) volume element on the equator."""
    if metric.is_rotating_class:
        g_tt, g_tphi, g_phiphi, g_rr, _ = metric.components_array(r, HALF_PI, p)
        return jnp.sqrt((g_tphi**2 - g_tt * g_phiphi) * g_rr)
    return jnp.sqrt(metric.f_array(r, p) * metric.h_array(r, p) / metric.g_array(r, p))


def _orbit(metric, p, r, direction):
    """``(E, L, Omega)`` of the circular orbit at ``r``."""
    return (
        energy_core(metric, p, r, direction),
        angular_momentum_core(metric, p, r, direction),
        omega_phi_core(metric, p, r, direction),
    )


def efficiency_core(metric, p, r_in, direction=1):
    """Radiative efficiency ``1 - E(r_in)``."""
    return 1.0 - energy_core(metric, p, r_in, direction)


def flux_core(metric, p, r, r_in, direction=1, n=96):
    r"""Page-Thorne flux :math:`F M^2/\dot M_0` at radius ``r`` for inner edge ``r_in``."""

    def integrand(x):
        energy, _, omega = _orbit(metric, p, x, direction)
        l_slope = value_and_derivative(lambda y: angular_momentum_core(metric, p, y, direction), x)[
            1
        ]
        return (energy - omega * angular_momentum_core(metric, p, x, direction)) * l_slope

    outside = r > r_in
    upper = jnp.where(outside, r, r_in * 1.5)
    # The integrand is smooth on [r_in, r]; a logarithmic substitution keeps the nodes
    # dense near the inner edge, where the flux rises steeply.
    log_lo, log_hi = jnp.log(r_in), jnp.log(upper)
    integral = gauss_legendre(lambda s: integrand(jnp.exp(s)) * jnp.exp(s), log_lo, log_hi, n)
    energy, angular, omega = _orbit(metric, p, upper, direction)
    omega_slope = value_and_derivative(lambda y: omega_phi_core(metric, p, y, direction), upper)[1]
    f = -omega_slope / (energy - omega * angular) ** 2 * integral
    flux = f / (4.0 * math.pi * _sqrt_minus_g(metric, p, upper))
    flux = jnp.where(outside, flux, 0.0)
    ok = exists_core(metric, p, upper, direction) & jnp.isfinite(r_in)
    return jnp.where(ok | (~outside), flux, jnp.nan)


def flux_maximum_core(metric, p, r_in, direction=1, n=96, n_scan=96):
    """Radius of maximum flux and the maximum flux (NaN if not found)."""
    slope = lambda r: value_and_derivative(
        lambda x: flux_core(metric, p, x, r_in, direction, n), r
    )[1]
    roots, _ = find_roots(
        slope, r_in * (1.0 + 1e-6), 200.0 * r_in, n_scan=n_scan, max_roots=2,
        detect_extrema=False,
    )  # fmt: skip
    radius = roots[0]
    safe = jnp.where(jnp.isfinite(radius), radius, 2.0 * r_in)
    return radius, jnp.where(
        jnp.isfinite(radius), flux_core(metric, p, safe, r_in, direction, n), jnp.nan
    )


# ----------------------------------------------------------------------------- inner edge


def inner_edge(metric, *, direction="prograde", r_in=None, n_scan: int = 256, **params) -> float:
    """Inner edge of a thin disk: the ISCO, or an explicit radius.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    direction : {"prograde", "retrograde"}
    r_in : float, optional
        Explicit inner radius in units of M. A circular orbit must exist
        there.
    **params
        Scalar metric parameters.

    Returns
    -------
    float

    Raises
    ------
    BlackPyError
        If there is no ISCO, or if there are several marginally stable
        orbits that are inner edges of stable regions (then the disk's inner
        edge is not well defined and must be chosen with ``r_in``).
    """
    if metric.is_rotating_class:
        require_equatorial_orbits(metric)
    if r_in is not None:
        r_in = float(r_in)
        exists = bool(
            np.asarray(metric.evaluate(exists_core, params, r_in, **_options(metric, direction)))
        )
        if not exists:
            raise BlackPyError(
                f"No circular orbit exists at r_in = {r_in} in '{metric.name}' for these "
                "parameters, so it cannot be the inner edge of a disk."
            )
        return r_in
    orbits = marginally_stable_orbits(metric, n_scan=n_scan, direction=direction, **params)
    edges = [o.radius for o in orbits if o.kind == "ISCO"]
    if not edges:
        raise BlackPyError(
            f"'{metric.name}' has no innermost stable circular orbit for these parameters, so "
            "the inner edge of a thin disk is not defined. Pass an explicit inner radius, "
            "r_in=<radius in M>, if you have a physical reason for one."
        )
    if len(edges) > 1:
        raise BlackPyError(
            f"'{metric.name}' has several marginally stable orbits that are inner edges of "
            f"stable regions, at r = {[round(x, 6) for x in edges]} (all marginally stable "
            f"orbits: {[(round(o.radius, 6), o.kind) for o in orbits]}). The inner edge of a "
            "thin disk is then not well defined: choose it explicitly with r_in=<radius in M>."
        )
    return edges[0]


# ----------------------------------------------------------------------------- public API


def radiative_efficiency(metric, *, direction="prograde", r_in=None, **params):
    r"""Radiative efficiency :math:`\eta = 1 - E(r_{\rm in})` of a thin disk.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
        Rotating metrics must be reflection symmetric.
    direction : {"prograde", "retrograde"}
        Orbit direction of the disk.
    r_in : float, optional
        Inner edge in units of M; default: the ISCO (see :func:`inner_edge`).
    **params
        Metric parameters. With the default inner edge they must be scalars
        (the ISCO is determined first); with an explicit ``r_in`` they may be
        arrays.

    Returns
    -------
    jax.Array
        Schwarzschild: :math:`1 - \sqrt{8/9} = 0.0572`. Kerr prograde tends to
        :math:`1 - 1/\sqrt3 = 0.423` as :math:`a \to 1`.
    """
    sign = direction_sign(direction)
    if r_in is None:
        r_in = inner_edge(metric, direction=direction, **params)
    elif metric.is_rotating_class:
        require_equatorial_orbits(metric)
    return metric.evaluate(efficiency_core, params, r_in, direction=sign)


class DiskProfile(NamedTuple):
    """Radial structure of a Page-Thorne disk (see :func:`page_thorne_flux`).

    Attributes
    ----------
    r : numpy.ndarray
        Radii in units of M.
    flux : numpy.ndarray
        :math:`F M^2/\\dot M_0` at each radius (zero at the inner edge).
    temperature : numpy.ndarray
        :math:`F^{1/4}` divided by its maximum: the effective temperature
        profile up to its normalisation (:math:`\\sigma T_{\\rm eff}^4 = F`).
    r_in : float
        Inner edge used.
    r_max : float
        Radius of maximum flux.
    flux_max : float
        Maximum of :math:`F M^2/\\dot M_0`.
    efficiency : float
        :math:`1 - E(r_{\\rm in})`.
    """

    r: np.ndarray
    flux: np.ndarray
    temperature: np.ndarray
    r_in: float
    r_max: float
    flux_max: float
    efficiency: float


def page_thorne_flux(
    metric, r=None, *, direction="prograde", r_in=None, n: int = 96, **params
) -> DiskProfile:
    r"""Page-Thorne flux of a geometrically thin, optically thick disk.

    .. math::

        F(r) = -\frac{\dot M_0}{4\pi e^{\nu+\psi+\mu}}
               \frac{\Omega_{,r}}{(E - \Omega L)^2}
               \int_{r_{\rm in}}^r (E - \Omega L)\,L_{,r}\,dr

    (Page & Thorne 1974, Eqs. (11b), (12); see the module docstring).

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
        Rotating metrics must be reflection symmetric (equatorial orbits).
    r : array, optional
        Radii in units of M. Default: 200 points from the inner edge to
        :math:`10^3\,r_{\rm in}`, logarithmically spaced.
    direction : {"prograde", "retrograde"}
    r_in : float, optional
        Inner edge; default the ISCO. A metric with no ISCO or with several
        possible inner edges is refused with a message (see
        :func:`inner_edge`).
    n : int
        Gauss-Legendre nodes for the radial integral.
    **params
        Scalar metric parameters.

    Returns
    -------
    DiskProfile
        Flux per unit accretion rate in units :math:`G = c = M = 1`, the
        temperature profile up to normalisation, the radius and value of
        the maximum flux, and the efficiency.

    Notes
    -----
    Checks (tests): the Kerr closed form of Page & Thorne, their Eq. (15n);
    the general (rotating) code against the static one through
    ``as_axisymmetric()``; zero flux at the inner edge and
    :math:`F \to 3/(8\pi r^3)` at large radius.

    Comparing with observed spectra needs ray tracing of the disk image and
    is outside the scope of BlackPy.
    """
    sign = direction_sign(direction)
    edge = inner_edge(metric, direction=direction, r_in=r_in, **params)
    radii = np.geomspace(edge, 1.0e3 * edge, 200) if r is None else np.asarray(r, dtype=float)
    flux = np.asarray(metric.evaluate(flux_core, params, radii, edge, direction=sign, n=n))
    r_max, flux_max = (
        float(v) for v in metric.evaluate(flux_maximum_core, params, edge, direction=sign, n=n)
    )
    with np.errstate(invalid="ignore"):
        temperature = np.clip(flux, 0.0, None) ** 0.25 / flux_max**0.25
    efficiency = float(metric.evaluate(efficiency_core, params, edge, direction=sign))
    return DiskProfile(radii, flux, temperature, edge, r_max, flux_max, efficiency)


__all__ = [
    "DiskProfile",
    "inner_edge",
    "page_thorne_flux",
    "radiative_efficiency",
]
