"""Physical constants and unit conversions.

All constants are in SI units.

Sources
-------
* ``C``: exact by definition of the metre (SI, 2019 redefinition). Equal to
  ``scipy.constants.c``.
* ``G``: CODATA recommended value, 6.67430(15)e-11 m^3 kg^-1 s^-2. The
  CODATA 2018 adjustment (Tiesinga et al. 2021, Rev. Mod. Phys. 93, 025010)
  and the CODATA 2022 adjustment give the same value and uncertainty. Equal
  to ``scipy.constants.G``; SciPy 1.18 ships CODATA 2022 as its current set
  (SciPy 1.8 to 1.14 shipped CODATA 2018). Both facts are checked by the
  test-suite against the installed SciPy.
* ``GM_SUN``: IAU 2015 Resolution B3 *nominal solar mass parameter*
  (GM)^N_sun = 1.3271244e20 m^3 s^-2, adopted as an exact number
  (Resolution B3, recommendation 3, arXiv:1510.07674; Prsa et al. 2016,
  AJ 152, 41, Table 1, arXiv:1605.09788). Read from both arXiv texts on
  2026-11-15. SciPy does not provide this constant.
* ``M_SUN``: derived as ``GM_SUN / G``.

Unit conversions use ``GM_SUN`` directly, because the product G*M_sun is
known about five orders of magnitude more precisely than G or M_sun alone.

Geometric units
---------------
BlackPy works in units G = c = 1 with lengths in units of the black hole
mass M. A dimensionless angular frequency ``omega`` (in units of 1/M) maps
to a frequency in hertz through::

    nu = omega * c^3 / (2 pi G M)
"""

from __future__ import annotations

import math

#: Speed of light in vacuum [m s^-1] (exact).
C: float = 299_792_458.0

#: Newtonian constant of gravitation [m^3 kg^-1 s^-2] (CODATA 2018 and 2022: same value).
G: float = 6.67430e-11

#: Nominal solar mass parameter G*M_sun [m^3 s^-2] (IAU 2015 Resolution B3).
GM_SUN: float = 1.3271244e20

#: Solar mass [kg], derived as GM_SUN / G.
M_SUN: float = GM_SUN / G

#: Gravitational time G*M_sun/c^3 [s] (light-crossing time of one solar gravitational radius).
T_SUN: float = GM_SUN / C**3

#: Gravitational radius G*M_sun/c^2 [m].
R_SUN_GRAV: float = GM_SUN / C**2


def gravitational_radius_m(mass_msun):
    """Gravitational radius G M / c^2 in metres.

    Parameters
    ----------
    mass_msun : float or array
        Mass in solar masses.

    Returns
    -------
    float or array
        G M / c^2 in metres.
    """
    return R_SUN_GRAV * mass_msun


def omega_to_hz(omega, mass_msun):
    """Convert a dimensionless angular frequency to a frequency in hertz.

    Implements ``nu = omega * c^3 / (2 pi G M)``.

    Parameters
    ----------
    omega : float or array
        Angular frequency in geometric units (units of 1/M), as measured by
        a static observer at infinity.
    mass_msun : float or array
        Black hole mass in solar masses.

    Returns
    -------
    float or array
        Frequency in Hz.
    """
    return omega / (2.0 * math.pi * T_SUN * mass_msun)


def hz_to_omega(nu_hz, mass_msun):
    """Convert a frequency in hertz to a dimensionless angular frequency.

    Inverse of :func:`omega_to_hz`.

    Parameters
    ----------
    nu_hz : float or array
        Frequency in Hz.
    mass_msun : float or array
        Black hole mass in solar masses.

    Returns
    -------
    float or array
        Angular frequency in units of 1/M.
    """
    return nu_hz * (2.0 * math.pi * T_SUN * mass_msun)
