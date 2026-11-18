"""Constants and unit conversion (validation item 6)."""

import math

import numpy as np

import blackpy as bp
from blackpy import constants as c


def test_x64_enabled():
    import jax.numpy as jnp

    assert jnp.asarray(1.0).dtype == jnp.float64


def test_constant_values():
    assert c.C == 299_792_458.0
    assert c.G == 6.67430e-11
    assert c.GM_SUN == 1.3271244e20
    assert math.isclose(c.M_SUN, 1.98841e30, rel_tol=1e-5)


def test_constants_match_scipy_codata():
    """c and G against scipy.constants; records which CODATA edition SciPy ships."""
    import scipy.constants as sc
    from scipy.constants import _codata

    assert c.C == sc.c
    assert c.G == sc.G
    # G is identical in the CODATA 2018 and CODATA 2022 adjustments.
    name = "Newtonian constant of gravitation"
    assert _codata._physical_constants_2018[name][0] == c.G
    if hasattr(_codata, "_physical_constants_2022"):  # SciPy >= 1.15
        assert _codata._physical_constants_2022[name][0] == c.G
    assert sc.physical_constants[name][2] == 1.5e-15
    # The edition SciPy treats as current (CODATA 2022 for SciPy >= 1.15).
    current = _codata._current_codata
    assert current in ("CODATA 2018", "CODATA 2022"), current
    print(f"scipy.constants uses {current}")


def test_gm_sun_is_the_iau_2015_b3_nominal_value():
    """(GM)^N_sun = 1.3271244e20 m^3 s^-2 exactly (IAU 2015 Resolution B3)."""
    assert c.GM_SUN == 1.3271244e20
    assert math.isclose(c.R_SUN_GRAV, 1476.6250, rel_tol=1e-7)  # metres
    assert math.isclose(c.T_SUN, 4.925490947e-6, rel_tol=1e-9)  # seconds


def test_omega_hz_round_trip():
    omega = np.array([0.01, 0.068, 0.2])
    np.testing.assert_allclose(c.hz_to_omega(c.omega_to_hz(omega, 7.3), 7.3), omega, rtol=1e-15)


def test_schwarzschild_isco_frequency_10_msun(schwarzschild):
    """nu = Omega c^3 / (2 pi G M) with Omega = 6^(-3/2) at the Schwarzschild ISCO.

    The expected value is computed from the constants (not hard-coded); the
    well-known rounded value is about 220 Hz.
    """
    mass = 10.0
    expected = 6.0**-1.5 * c.C**3 / (2.0 * math.pi * c.GM_SUN * mass)
    r_isco = float(bp.orbits.isco(schwarzschild))
    nu_phi = float(bp.orbits.frequencies_hz(schwarzschild, r_isco, mass).nu_phi)
    assert math.isclose(nu_phi, expected, rel_tol=1e-12)
    assert 219.0 < nu_phi < 221.0


def test_gravitational_radius():
    # G M_sun / c^2 is about 1.4767 km.
    assert math.isclose(c.gravitational_radius_m(1.0), c.GM_SUN / c.C**2, rel_tol=1e-15)
    assert 1476.0 < c.gravitational_radius_m(1.0) < 1477.5
