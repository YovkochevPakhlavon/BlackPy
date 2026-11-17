"""Gravitational lensing by static metrics (B6).

Sources: Bozza (2002, gr-qc/0208075); Tsukamoto (2017, arXiv:1612.08251); Keeton &
Petters (2005, gr-qc/0511019); Gibbons & Werner (2008, arXiv:0807.0854);
Bisnovatyi-Kogan & Tsupko (2010, arXiv:1006.2321). All were read.
"""

import math

import numpy as np
import pytest
import scipy.constants

import blackpy as bp
from blackpy import constants
from blackpy.errors import BlackPyError, NotAsymptoticallyFlatError

ln = bp.lensing
B_CRIT = 3.0 * math.sqrt(3.0)


def test_parsec_and_astronomical_unit_match_scipy():
    assert constants.AU == scipy.constants.au
    assert constants.PARSEC == pytest.approx(scipy.constants.parsec, rel=1e-15)
    assert constants.MICROARCSECOND == pytest.approx(scipy.constants.arcsecond * 1e-6, rel=1e-15)


def test_weak_deflection_series(schwarzschild, rn):
    """4M/b and the higher-order terms of Keeton & Petters (2005), Eqs. (24), (25), (52)."""
    for b in (30.0, 1.0e2, 1.0e3, 1.0e4, 1.0e6):
        exact = float(ln.deflection_angle(schwarzschild, b))
        first = 4.0 / b
        second = first + 15.0 * math.pi / (4.0 * b**2)
        third = second + 128.0 / (3.0 * b**3)
        fourth = third + 3465.0 * math.pi / (64.0 * b**4)
        # Each truncation is wrong by about the next term (mandatory: first and second order).
        assert exact - first == pytest.approx(15.0 * math.pi / (4.0 * b**2), rel=5.0 / b + 1e-5)
        assert exact - second == pytest.approx(128.0 / (3.0 * b**3), rel=6.0 / b + 2e-4, abs=3e-15)
        # The next term is (3584/5) (M/b)^5, i.e. 179 / b^4 relative to the leading one.
        assert exact == pytest.approx(fourth, rel=400.0 / b**4 + 2e-9)
    for charge in (0.4, 0.9):  # Reissner-Nordstrom, q = Q / M
        b = 500.0
        series = (
            4.0 / b
            + (5.0 - charge**2) * 3.0 * math.pi / (4.0 * b**2)
            + (128.0 / 3.0 - 16.0 * charge**2) / b**3
        )
        # Truncated after the third order: the next term is of relative size ~ 40 / b^3.
        assert float(ln.deflection_angle(rn, b, Q=charge)) == pytest.approx(
            series, rel=100.0 / b**3
        )
    # Vectorised; the angle grows as the ray passes closer and diverges at b_c.
    angles = np.asarray(ln.deflection_angle(schwarzschild, np.array([5.3, 6.0, 10.0, 50.0])))
    assert np.all(np.diff(angles) < 0.0) and angles[0] > math.pi


def test_closest_approach_and_deflection_at(schwarzschild):
    b = 20.0
    r0 = float(ln.closest_approach(schwarzschild, b))
    assert r0**3 / (r0 - 2.0) == pytest.approx(b**2, rel=1e-12)  # b^2 = h/f at the turning point
    assert float(ln.deflection_angle_at(schwarzschild, r0)) == pytest.approx(
        float(ln.deflection_angle(schwarzschild, b)), rel=1e-12
    )
    assert np.isnan(float(ln.deflection_angle_at(schwarzschild, 2.9)))  # inside the photon sphere
    assert np.isnan(float(ln.closest_approach(schwarzschild, 5.0)))  # captured: b < 3 sqrt(3)
    with pytest.raises(BlackPyError, match="captured"):
        ln.deflection_angle(schwarzschild, 5.0)
    # More quadrature nodes change nothing: the singular part is integrated in closed form.
    assert float(ln.deflection_angle(schwarzschild, 5.4, n=400)) == pytest.approx(
        float(ln.deflection_angle(schwarzschild, 5.4, n=64)), rel=1e-9
    )


def test_gauss_bonnet_weak_deflection(schwarzschild, rn, bardeen):
    """Gibbons & Werner (2008): first-order agreement with the exact angle."""
    for b in (1.0e2, 1.0e3, 1.0e5):
        gauss_bonnet = float(ln.weak_deflection_angle(schwarzschild, b))
        assert gauss_bonnet == pytest.approx(4.0 / b, rel=1.0 / b)  # their Eq. (14)
        # With the straight-line boundary the second-order coefficient is 3 pi / 4, not the
        # exact 15 pi / 4: the method is first order.
        assert (gauss_bonnet - 4.0 / b) * b**2 == pytest.approx(0.75 * math.pi, rel=5.0 / b)
        for metric, params in ((rn, {"Q": 0.8}), (bardeen, {"q": 0.6})):
            approximate = float(ln.weak_deflection_angle(metric, b, **params))
            exact = float(ln.deflection_angle(metric, b, **params))
            assert approximate == pytest.approx(exact, rel=4.0 / b)
    assert ln.weak_deflection_angle(schwarzschild, np.array([50.0, 500.0])).shape == (2,)


def test_strong_deflection_limit_of_schwarzschild(schwarzschild):
    """Bozza (2002), Eqs. (47)-(50): a = 1, b = ln[216 (7 - 4 sqrt 3)] - pi, u_m = 3 sqrt 3 M."""
    limit = ln.strong_deflection_limit(schwarzschild)
    b_bar = math.log(216.0 * (7.0 - 4.0 * math.sqrt(3.0))) - math.pi
    assert b_bar == pytest.approx(-0.4002, abs=5e-5)  # the value printed in the paper
    assert float(limit.a_bar) == pytest.approx(1.0, rel=1e-12)
    assert float(limit.b_bar) == pytest.approx(b_bar, abs=1e-9)
    assert float(limit.b_c) == pytest.approx(B_CRIT, rel=1e-12)
    assert float(limit.r_m) == pytest.approx(3.0, rel=1e-12)


def test_exact_deflection_approaches_the_strong_deflection_limit(schwarzschild, rn):
    """The exact integral agrees with the logarithmic formula as b -> b_c (mandatory)."""
    for metric, params in ((schwarzschild, {}), (rn, {"Q": 0.7})):
        limit = ln.strong_deflection_limit(metric, **params)
        b_c = float(limit.b_c)
        errors = []
        for delta in (1e-2, 1e-3, 1e-4, 1e-6, 1e-8):
            b = b_c * (1.0 + delta)
            exact = float(ln.deflection_angle(metric, b, **params))
            approximate = float(ln.strong_deflection_angle(metric, b, **params))
            assert approximate == pytest.approx(
                -float(limit.a_bar) * math.log(delta) + float(limit.b_bar), rel=1e-9
            )
            errors.append(abs(exact - approximate))
            # Error of order delta * ln(delta), as Tsukamoto (2017) shows.
            assert errors[-1] < 1.5 * delta * abs(math.log(delta)) + 2e-8
        assert all(later < earlier for earlier, later in zip(errors, errors[1:], strict=False))
        assert float(ln.deflection_angle(metric, b_c * (1.0 + 1e-8), **params)) > 5.0 * math.pi
    # Bozza (2002), section III.1: the outermost image (alpha = 2 pi) is at u - u_m = 0.003264
    # in units of 2M, where the strong field limit is off by about 0.06%.
    b = B_CRIT + 2.0 * 0.003264
    exact = float(ln.deflection_angle(schwarzschild, b))
    assert exact == pytest.approx(2.0 * math.pi, rel=1e-5)
    approximate = float(ln.strong_deflection_angle(schwarzschild, b))
    assert abs(approximate / exact - 1.0) == pytest.approx(6.0e-4, rel=0.1)
    assert np.isnan(float(ln.strong_deflection_angle(schwarzschild, 5.0)))


def test_reissner_nordstrom_strong_deflection_coefficients(rn):
    """Tsukamoto (2017), section III.2 (exact), and Bozza (2002), Table 1 (mandatory)."""
    charge = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 0.95, 1.0])
    limit = ln.strong_deflection_limit(rn, Q=charge)
    r_m = (3.0 + np.sqrt(9.0 - 8.0 * charge**2)) / 2.0
    a_bar = r_m / np.sqrt(3.0 * r_m - 4.0 * charge**2)
    b_bar = (
        a_bar
        * np.log(
            8.0
            * (3.0 * r_m - 4.0 * charge**2) ** 3
            / (r_m**2 * (r_m - charge**2) ** 2)
            * (2.0 * np.sqrt(r_m - charge**2) - np.sqrt(3.0 * r_m - 4.0 * charge**2)) ** 2
        )
        - math.pi
    )
    np.testing.assert_allclose(limit.r_m, r_m, rtol=1e-11)
    np.testing.assert_allclose(limit.b_c, r_m**2 / np.sqrt(r_m - charge**2), rtol=1e-11)
    np.testing.assert_allclose(limit.a_bar, a_bar, rtol=1e-10)
    np.testing.assert_allclose(limit.b_bar, b_bar, atol=2e-8)
    # Extremal case, Tsukamoto Eqs. (78)-(80).
    assert float(limit.b_c[-1]) == pytest.approx(4.0, rel=1e-11)
    assert float(limit.a_bar[-1]) == pytest.approx(math.sqrt(2.0), rel=1e-10)
    assert float(limit.b_bar[-1]) == pytest.approx(
        2.0 * math.sqrt(2.0) * math.log(4.0 * (2.0 - math.sqrt(2.0))) - math.pi, abs=2e-8
    )
    # Bozza's Table 1: q = Q / (2M) = 0.1 ... 0.4; lengths in units of R_S = 2M.
    table = {
        0.1: (1.005, -0.3993, 2.58),
        0.2: (1.02, -0.3972, 2.53),
        0.3: (1.052, -0.3965, 2.43),
        0.4: (1.123, -0.4136, 2.27),
    }
    for q, (a_printed, b_printed, u_printed) in table.items():
        one = ln.strong_deflection_limit(rn, Q=2.0 * q)
        assert float(one.a_bar) == pytest.approx(a_printed, abs=6e-4)
        assert float(one.b_bar) == pytest.approx(b_printed, abs=6e-5)
        assert float(one.b_c) / 2.0 == pytest.approx(u_printed, abs=6e-3)


def test_strong_field_observables_of_the_galactic_centre(schwarzschild, rn):
    """Bozza (2002), Table 1: M = 2.8e6 Msun at 8.5 kpc."""
    lens = {"mass_msun": 2.8e6, "distance_kpc": 8.5}
    observables = ln.strong_field_observables(schwarzschild, **lens)
    # theta_g = G M / (c^2 D) by hand.
    theta_g = (constants.GM_SUN * 2.8e6 / constants.C**2) / (8.5e3 * constants.PARSEC)
    assert float(observables.theta_g) == pytest.approx(
        theta_g / constants.MICROARCSECOND, rel=1e-12
    )
    assert float(observables.theta_infinity) == pytest.approx(B_CRIT * float(observables.theta_g))
    # Printed: theta_inf = 16.87 muas, s = 0.0211 muas, r_m = 6.82 mag. The 0.15% offset in
    # theta_inf is the precision of the constants used in 2002.
    assert float(observables.theta_infinity) == pytest.approx(16.87, rel=3e-3)
    assert float(observables.separation) == pytest.approx(0.0211, rel=5e-3)
    assert float(observables.magnitude_difference) == pytest.approx(6.82, abs=5e-3)
    assert float(observables.flux_ratio) == pytest.approx(math.exp(2.0 * math.pi), rel=1e-10)
    charged = ln.strong_field_observables(rn, Q=0.8, **lens)  # the column q = 0.4
    assert float(charged.theta_infinity) == pytest.approx(14.76, rel=3e-3)
    assert float(charged.separation) == pytest.approx(0.038, rel=1e-2)
    assert float(charged.magnitude_difference) == pytest.approx(6.07, abs=1e-2)
    # Lens taken from the registry (schema only: the result is not compared with a number).
    for dataset, key in (("eht/sgra", "vlti"), ("eht/m87", "stellar_dynamics")):
        data = bp.data.load(dataset)
        from_registry = ln.strong_field_observables(schwarzschild, dataset=dataset, mass_prior=key)
        assert float(from_registry.theta_g) == pytest.approx(
            data.mass_priors[key].theta_g.value, rel=0.02
        )  # consistent with the angular gravitational radius stored with the mass
        assert float(from_registry.theta_infinity) > 0.0
    default = ln.strong_field_observables(schwarzschild, dataset="eht/m87")
    assert float(default.theta_g) == pytest.approx(
        float(
            ln.strong_field_observables(
                schwarzschild, dataset="eht/m87", mass_prior="stellar_dynamics"
            ).theta_g
        )
    )
    with pytest.raises(TypeError, match="mass_msun"):
        ln.strong_field_observables(schwarzschild)
    with pytest.raises(BlackPyError, match="mass_prior"):
        ln.strong_field_observables(schwarzschild, dataset="eht/m87", mass_prior="unknown")
    with pytest.raises(BlackPyError, match="shadow dataset"):
        ln.strong_field_observables(schwarzschild, dataset="qpo/gro_j1655_40")
    # Einstein ring of a source 1 kpc behind the lens, by hand.
    ring = ln.einstein_ring_angle(2.8e6, 8.5, 9.5)
    r_g = constants.GM_SUN * 2.8e6 / constants.C**2
    kpc = 1e3 * constants.PARSEC
    expected = math.sqrt(4.0 * r_g * (1.0 * kpc) / (8.5 * kpc * 9.5 * kpc))
    assert ring == pytest.approx(expected / constants.MICROARCSECOND, rel=1e-12)
    with pytest.raises(ValueError, match="farther"):
        ln.einstein_ring_angle(2.8e6, 8.5, 8.0)


def test_deflection_in_a_homogeneous_plasma(schwarzschild):
    """Bisnovatyi-Kogan & Tsupko (2010), Eq. (38), and the exact angle with a plasma."""
    plasma = bp.light.HomogeneousPlasma()
    for b in (1.0e3, 1.0e5):
        vacuum = float(ln.deflection_angle(schwarzschild, b))
        assert float(ln.deflection_angle(schwarzschild, b, plasma=plasma)) == pytest.approx(
            vacuum, rel=1e-9
        )  # omega_p = 0
        for omega_p in (0.3, 0.6, 0.9):
            x = omega_p**2
            weak = (2.0 / b) * (1.0 + 1.0 / (1.0 - x))  # R_S / b (1 + 1 / (1 - x)), R_S = 2M
            exact = float(ln.deflection_angle(schwarzschild, b, plasma=plasma, omega_p=omega_p))
            assert exact == pytest.approx(weak, rel=30.0 / (b * (1.0 - x)))
            assert exact > vacuum  # a homogeneous plasma increases the deflection
    # Strong deflection with a plasma: the exact integral still approaches the limit.
    limit = ln.strong_deflection_limit(schwarzschild, plasma=plasma, omega_p=0.5)
    assert float(limit.b_c) == pytest.approx(
        float(bp.light.shadow_in_plasma(schwarzschild, plasma, omega_p=0.5)), rel=1e-12
    )
    b = float(limit.b_c) * (1.0 + 1e-6)
    exact = float(ln.deflection_angle(schwarzschild, b, plasma=plasma, omega_p=0.5))
    approximate = float(ln.strong_deflection_angle(schwarzschild, b, plasma=plasma, omega_p=0.5))
    assert exact == pytest.approx(approximate, abs=1e-4)
    # A plasma that thins out: the impact parameter is L/E and refraction reduces the angle.
    power_law = bp.light.PowerLawPlasma(k=2.0)
    thin = float(ln.deflection_angle(schwarzschild, 50.0, plasma=power_law, omega_p=3.0))
    assert thin < float(ln.deflection_angle(schwarzschild, 50.0))


def test_lensing_refusals(sds, kerr, schwarzschild):
    with pytest.raises(TypeError, match="static"):
        ln.deflection_angle(kerr, 10.0, a=0.5)
    with pytest.raises(TypeError, match="static"):
        ln.strong_deflection_limit(kerr, a=0.5)
    with pytest.raises(NotAsymptoticallyFlatError):
        ln.deflection_angle(sds, 10.0, Lambda=0.01)
    # The name reserved in earlier versions is the implementation.
    assert bp.light.deflection_angle is ln.deflection_angle
    assert bp.light.future.deflection_angle is ln.deflection_angle
    assert "predictions, not constraints" in ln.__doc__
