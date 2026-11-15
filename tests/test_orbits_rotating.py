"""Equatorial orbits, epicyclic frequencies and QPO models for rotating metrics (B3)."""

import math
import time

import jax
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy import orbits as ob
from blackpy.errors import NodalPrecessionUnavailableError, ReflectionSymmetryError
from blackpy.metric import StationaryAxisymmetric
from blackpy.orbits import axisymmetric as axi

SPINS = (0.0, 0.3, 0.7, 0.95, 0.998)


def kerr_frequencies(r, a):
    """Kerr closed forms (M = 1), prograde for a > 0.

    Source: Ingram & Motta (2014), Eqs. (1)-(3), and Motta et al. (2014),
    Eqs. (1)-(3), both read from the arXiv versions; they are the forms given
    in the v0.2 specification.
    """
    omega = 1.0 / (r**1.5 + a)
    radial = omega**2 * (1 - 6 / r + 8 * a / r**1.5 - 3 * a**2 / r**2)
    vertical = omega**2 * (1 - 4 * a / r**1.5 + 3 * a**2 / r**2)
    return omega, radial, vertical


def bpt_isco(a):
    """Bardeen, Press & Teukolsky (1972) ISCO, as printed in Motta et al. (2014), Eq. (6)
    (with their a/r_g read as the dimensionless spin). Prograde for a > 0."""
    z1 = 1 + np.cbrt(1 - a**2) * (np.cbrt(1 + a) + np.cbrt(1 - a))
    z2 = np.sqrt(3 * a**2 + z1**2)
    return 3 + z2 - np.sign(a) * np.sqrt((3 - z1) * (3 + z1 + 2 * z2))


# ----------------------------------------------------------------------------- Kerr


def test_kerr_frequencies_against_closed_forms(kerr):
    r = np.geomspace(1.3, 60.0, 80)
    for a in (*SPINS, -0.5, -0.9):
        omega, radial, vertical = kerr_frequencies(r, a)
        exists = np.asarray(ob.circular_orbit_exists(kerr, r, a=a))
        assert exists[-1] and exists.sum() > 30
        np.testing.assert_allclose(
            ob.angular_velocity(kerr, r, a=a)[exists], omega[exists], rtol=1e-13
        )
        np.testing.assert_allclose(
            ob.radial_epicyclic_frequency_squared(kerr, r, a=a)[exists],
            radial[exists],
            rtol=1e-10,
            atol=1e-13,
        )
        np.testing.assert_allclose(
            ob.vertical_epicyclic_frequency_squared(kerr, r, a=a)[exists],
            vertical[exists],
            rtol=1e-10,
            atol=1e-13,
        )
    # Radial frequency: NaN inside the ISCO, real outside.
    assert np.isnan(float(ob.radial_epicyclic_frequency(kerr, 3.0, a=0.3)))
    value = float(ob.radial_epicyclic_frequency(kerr, 8.0, a=0.3))
    assert math.isclose(value, math.sqrt(kerr_frequencies(8.0, 0.3)[1]), rel_tol=1e-12)


def test_retrograde_orbits_equal_prograde_orbits_of_opposite_spin(kerr):
    r, a = np.geomspace(9.5, 40.0, 12), 0.6
    retro = dict(direction="retrograde", a=a)
    np.testing.assert_allclose(
        ob.angular_velocity(kerr, r, **retro),
        -np.asarray(ob.angular_velocity(kerr, r, a=-a)),
        rtol=1e-13,
    )
    for fn in (ob.radial_epicyclic_frequency, ob.vertical_epicyclic_frequency, ob.specific_energy):
        np.testing.assert_allclose(fn(kerr, r, **retro), fn(kerr, r, a=-a), rtol=1e-12)
    np.testing.assert_allclose(
        ob.specific_angular_momentum(kerr, r, **retro),
        -np.asarray(ob.specific_angular_momentum(kerr, r, a=-a)),
        rtol=1e-12,
    )
    assert float(ob.isco(kerr, **retro)) == pytest.approx(float(ob.isco(kerr, a=-a)), rel=1e-13)
    freqs = ob.frequencies_hz(kerr, 10.0, 7.0, **retro)
    assert float(freqs.nu_phi) > 0  # frequencies are reported as positive numbers
    with pytest.raises(ValueError, match="prograde"):
        ob.isco(kerr, direction="sideways", a=0.5)


def test_kerr_isco_equals_bardeen_press_teukolsky(kerr):
    """Mandatory validation: Kerr ISCO = BPT formula for a in [-0.999, 0.999] to 1e-10."""
    a = np.linspace(-0.999, 0.999, 1999)
    expected = bpt_isco(a)
    # The transcribed formula is itself checked: it solves the Kerr marginal-stability
    # condition 1 - 6/r + 8a/r^(3/2) - 3a^2/r^2 = 0 (the radial frequency above).
    residual = 1 - 6 / expected + 8 * a / expected**1.5 - 3 * a**2 / expected**2
    assert np.max(np.abs(residual)) < 1e-13
    np.testing.assert_allclose(ob.isco(kerr, a=a), expected, rtol=0, atol=1e-10)
    np.testing.assert_allclose(
        ob.isco(kerr, direction="retrograde", a=a), bpt_isco(-a), rtol=0, atol=1e-10
    )
    assert float(ob.isco(kerr, a=0.0)) == pytest.approx(6.0, abs=1e-12)
    orbits = ob.marginally_stable_orbits(kerr, a=0.9)
    assert [o.kind for o in orbits] == ["ISCO"]


def test_kerr_energy_and_angular_momentum(kerr):
    """E and L satisfy the effective-potential conditions U = dU/dr = 0, computed
    independently from the inverse metric; dE/dr = dL/dr = 0 at the ISCO."""
    for a, direction in ((0.7, "prograde"), (0.7, "retrograde"), (0.0, "prograde")):
        sign = ob.circular.direction_sign(direction)
        r0 = 9.5
        energy = float(ob.specific_energy(kerr, r0, direction=direction, a=a))
        momentum = float(ob.specific_angular_momentum(kerr, r0, direction=direction, a=a))

        def potential(r, a=a, energy=energy, momentum=momentum):
            g_tt = kerr.gtt_array(r, math.pi / 2, np.array([a]))
            g_tp = kerr.gtphi_array(r, math.pi / 2, np.array([a]))
            g_pp = kerr.gphiphi_array(r, math.pi / 2, np.array([a]))
            det = g_tt * g_pp - g_tp**2
            # g^tt E^2 - 2 g^tphi E L + g^phiphi L^2 + 1
            return (g_pp * energy**2 + 2 * g_tp * energy * momentum + g_tt * momentum**2) / det + 1

        assert abs(float(potential(r0))) < 1e-12
        assert abs(float(jax.grad(potential)(r0))) < 1e-12
        r_isco = float(ob.isco(kerr, direction=direction, a=a))
        p = np.array([a])
        for core in (axi.energy_core, axi.angular_momentum_core):
            slope = float(
                jax.grad(lambda r, core=core, p=p, sign=sign: core(kerr, p, r, sign))(r_isco)
            )
            assert abs(slope) < 1e-9
    # Closed forms of Bardeen, Press & Teukolsky (1972), Eqs. (2.12)-(2.13); quoted from
    # memory (the paper was not accessible), hence only a secondary check.
    r, a = np.geomspace(4.0, 50.0, 15), 0.7
    root = np.sqrt(r**1.5 - 3 * np.sqrt(r) + 2 * a)
    np.testing.assert_allclose(
        ob.specific_energy(kerr, r, a=a),
        (r**1.5 - 2 * np.sqrt(r) + a) / (r**0.75 * root),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        ob.specific_angular_momentum(kerr, r, a=a),
        (r**2 - 2 * a * np.sqrt(r) + a**2) / (r**0.75 * root),
        rtol=1e-12,
    )


def test_kerr_marginally_bound_and_photon_orbits(kerr):
    a = np.linspace(-0.99, 0.99, 199)
    r_mb = np.asarray(ob.marginally_bound_orbit(kerr, a=a))
    # Defining property: E = 1.
    np.testing.assert_allclose(ob.specific_energy(kerr, r_mb, a=a), 1.0, rtol=1e-12)
    assert float(ob.marginally_bound_orbit(kerr, a=0.0)) == pytest.approx(4.0, abs=1e-12)
    # Closed forms of Bardeen, Press & Teukolsky (1972), Eqs. (2.18)-(2.19); quoted from memory.
    np.testing.assert_allclose(r_mb, 2 - a + 2 * np.sqrt(1 - a), rtol=1e-12)
    r_ph = np.asarray(ob.photon_orbit(kerr, a=a))
    np.testing.assert_allclose(r_ph, 2 * (1 + np.cos(2 / 3 * np.arccos(-a))), rtol=1e-12)
    np.testing.assert_allclose(
        ob.photon_orbit(kerr, direction="retrograde", a=a),
        2 * (1 + np.cos(2 / 3 * np.arccos(a))),
        rtol=1e-12,
    )
    # Ordering r_h < r_ph < r_mb < r_ISCO and the region where orbits exist.
    r_isco = np.asarray(ob.isco(kerr, a=a))
    assert np.all(r_ph < r_mb) and np.all(r_mb < r_isco)
    ((low, high),) = ob.circular_orbit_region(kerr, a=0.9)
    assert low == pytest.approx(float(ob.photon_orbit(kerr, a=0.9)), rel=1e-12)
    assert high == pytest.approx(kerr.r_range[1], rel=1e-6)


def test_nodal_precession(kerr, schwarzschild):
    r, a, mass = np.geomspace(6.0, 30.0, 10), 0.4, 6.0
    omega, _, vertical = kerr_frequencies(r, a)
    expected = bp.constants.omega_to_hz(omega - np.sqrt(vertical), mass)
    np.testing.assert_allclose(
        ob.nodal_precession_frequency(kerr, r, mass, a=a), expected, rtol=1e-9
    )
    assert np.all(np.asarray(ob.nodal_precession_frequency(kerr, r, mass, a=a)) > 0)
    # Counter-rotating orbits precess the other way; a = 0 gives zero.
    assert np.all(np.asarray(ob.nodal_precession_frequency(kerr, r, mass, a=-a)) < 0)
    np.testing.assert_allclose(ob.nodal_precession_frequency(kerr, r, mass, a=0.0), 0.0, atol=1e-10)
    np.testing.assert_allclose(
        ob.QPO_MODELS["RP"].low_frequency(kerr, r, mass, a=a), expected, rtol=1e-9
    )
    # The explicit refusal stays for static metrics.
    with pytest.raises(NodalPrecessionUnavailableError, match="vanishes identically"):
        ob.nodal_precession_frequency(schwarzschild, 6.0, 5.0)


# ----------------------------------------------------------------------------- limits


def test_kerr_newman_reduces_to_rn_and_kerr(kerr, kerr_newman, rn):
    """Mandatory validation: Kerr-Newman -> RN at a = 0 and -> Kerr at Q = 0."""
    q = np.array([0.0, 0.3, 0.6, 0.9, 1.0])
    np.testing.assert_allclose(ob.isco(kerr_newman, a=0.0, Q=q), ob.isco(rn, Q=q), rtol=1e-11)
    np.testing.assert_allclose(
        ob.marginally_bound_orbit(kerr_newman, a=0.0, Q=q),
        ob.marginally_bound_orbit(rn, Q=q),
        rtol=1e-11,
    )
    r = np.geomspace(4.5, 40.0, 9)[:, None]
    for name in ("angular_velocity", "radial_epicyclic_frequency_squared", "specific_energy"):
        np.testing.assert_allclose(
            getattr(ob, name)(kerr_newman, r, a=0.0, Q=q),
            getattr(ob, name)(rn, r, Q=q),
            rtol=1e-11,
            atol=1e-14,
        )
    np.testing.assert_allclose(
        ob.vertical_epicyclic_frequency(kerr_newman, r, a=0.0, Q=q),
        ob.angular_velocity(rn, r, Q=q),
        rtol=1e-11,
    )
    a = np.array([-0.9, -0.3, 0.2, 0.8, 0.99])
    np.testing.assert_allclose(ob.isco(kerr_newman, a=a, Q=0.0), ob.isco(kerr, a=a), rtol=1e-12)
    np.testing.assert_allclose(
        ob.vertical_epicyclic_frequency(kerr_newman, r, a=a, Q=0.0),
        ob.vertical_epicyclic_frequency(kerr, r, a=a),
        rtol=1e-12,
    )


@pytest.mark.slow
def test_general_formulas_reproduce_the_static_code(rn, hayward, sds):
    """The rotating code path, applied to static metrics, equals the v0.1 formulas."""
    isotropic = bp.StaticSpherical(
        lambda rho: ((1 - 0.5 / rho) / (1 + 0.5 / rho)) ** 2,
        lambda rho: (1 + 0.5 / rho) ** -4,
        lambda rho: rho**2 * (1 + 0.5 / rho) ** 4,
        name="Schwarzschild, isotropic",
        asymptotically_flat=True,
    )
    cases = ((rn, {"Q": 0.8}), (hayward, {"l": 0.6}), (sds, {"Lambda": 2e-4}), (isotropic, {}))
    for metric, pars in cases:
        general = metric.as_axisymmetric()
        r_isco = float(ob.isco(metric, **pars))
        assert float(ob.isco(general, **pars)) == pytest.approx(r_isco, rel=1e-11)
        r = r_isco * np.array([0.8, 1.0001, 1.5, 3.0])
        for name in (
            "angular_velocity",
            "specific_energy",
            "specific_angular_momentum",
            "radial_epicyclic_frequency_squared",
        ):
            np.testing.assert_allclose(
                getattr(ob, name)(general, r, **pars),
                getattr(ob, name)(metric, r, **pars),
                rtol=1e-10,
                atol=1e-13,
            )
        # Omega_theta = Omega_phi follows from the general formula; it is not imposed.
        np.testing.assert_allclose(
            ob.vertical_epicyclic_frequency(general, r, **pars),
            ob.angular_velocity(metric, r, **pars),
            rtol=1e-11,
        )
        assert [
            (o.kind, pytest.approx(o.radius, rel=1e-10))
            for o in ob.marginally_stable_orbits(general, **pars)
        ] == [(o.kind, o.radius) for o in ob.marginally_stable_orbits(metric, **pars)]
        if metric is not sds:
            assert float(ob.marginally_bound_orbit(general, **pars)) == pytest.approx(
                float(ob.marginally_bound_orbit(metric, **pars)), rel=1e-11
            )
        np.testing.assert_allclose(
            ob.circular_orbit_region(general, **pars),
            ob.circular_orbit_region(metric, **pars),
            rtol=1e-9,
        )


def test_epicyclic_frequencies_against_the_direct_definition(johannsen):
    """Omega_r^2 and Omega_theta^2 from second derivatives of the effective potential,
    done symbolically for the Johannsen metric (Johannsen 2013, Eqs. (92)-(97)):
    Omega_x^2 = (d^2 U / dx^2) / (2 g_xx (u^t)^2), U = g^tt E^2 - 2 g^tphi E L + g^phiphi L^2 + 1.
    The code uses an algebraically reduced form; this checks the reduction."""
    s = johannsen.symbolic
    values = {"a": 0.6, "alpha13": 0.4, "alpha22": -0.3, "alpha52": 0.5, "epsilon3": 0.8}
    subs = {sym: values[sym.name] for sym in s.params}
    g_tt, g_tp, g_pp, g_rr, g_thth = (e.subs(subs) for e in s.components)
    energy, momentum = sp.symbols("E L")
    det = g_tt * g_pp - g_tp**2
    potential = (g_pp * energy**2 + 2 * g_tp * energy * momentum + g_tt * momentum**2) / det + 1
    equator = {s.theta: sp.pi / 2}
    for direction in ("prograde", "retrograde"):
        for r0 in (6.5, 11.0):
            e0 = float(ob.specific_energy(johannsen, r0, direction=direction, **values))
            l0 = float(ob.specific_angular_momentum(johannsen, r0, direction=direction, **values))
            point = {s.r: r0, energy: e0, momentum: l0}
            # u^t = -g^tt E + g^tphi L
            u_t = float(((g_pp * energy + g_tp * momentum) / -det).subs(equator).subs(point))
            radial = sp.diff(potential.subs(equator), s.r, 2) / (2 * g_rr.subs(equator))
            polar = sp.diff(potential, s.theta, 2).subs(equator) / (2 * g_thth.subs(equator))
            expected_r = float(radial.subs(point)) / u_t**2
            expected_theta = float(polar.subs(point)) / u_t**2
            got_r = float(
                ob.radial_epicyclic_frequency_squared(johannsen, r0, direction=direction, **values)
            )
            got_theta = float(
                ob.vertical_epicyclic_frequency_squared(
                    johannsen, r0, direction=direction, **values
                )
            )
            assert got_r == pytest.approx(expected_r, rel=1e-9)
            assert got_theta == pytest.approx(expected_theta, rel=1e-9)
            # The circular-orbit conditions themselves: U = dU/dr = 0.
            assert abs(float(potential.subs(equator).subs(point))) < 1e-11
            assert abs(float(sp.diff(potential.subs(equator), s.r).subs(point))) < 1e-11


def test_johannsen_orbits(johannsen, kerr):
    a = np.array([-0.8, 0.0, 0.5, 0.9])
    np.testing.assert_allclose(ob.isco(johannsen, a=a), ob.isco(kerr, a=a), rtol=1e-12)
    # Statements of Johannsen (2013), Sec. IV: the vertical frequency does not depend on
    # alpha52 (A_5 enters only g_rr); the ISCO does not depend on alpha22 for a = 0.
    base = float(ob.vertical_epicyclic_frequency(johannsen, 7.0, a=0.5))
    assert float(
        ob.vertical_epicyclic_frequency(johannsen, 7.0, a=0.5, alpha52=1.3)
    ) == pytest.approx(base, rel=1e-13)
    assert float(ob.isco(johannsen, a=0.0, alpha22=1.5)) == pytest.approx(6.0, abs=1e-10)
    assert abs(float(ob.isco(johannsen, a=0.5, alpha13=1.0)) - float(ob.isco(kerr, a=0.5))) > 0.05


# ----------------------------------------------------------------------------- QPO models


def test_all_qpo_models_work_with_rotation(kerr):
    """All 11 models of Stuchlik & Kolos (2016), Table 1, with the Kerr closed forms."""
    r, a, mass = np.geomspace(6.0, 20.0, 7), 0.5, 5.3
    omega, radial, vertical = kerr_frequencies(r, a)
    k, e, v = (
        bp.constants.omega_to_hz(x, mass) for x in (omega, np.sqrt(radial), np.sqrt(vertical))
    )
    table = {
        "RP": (k, k - e),
        "RP1": (v, k - e),
        "RP2": (k, v - e),
        "ER": (v, e),
        "ER1": (v, v - e),
        "ER2": (v - e, e),
        "ER3": (v + e, v),
        "ER4": (v + e, v - e),
        "ER5": (e, v - e),
        "TD": (k + e, k),
        "WD": (2 * k - e, 2 * (k - e)),
    }
    assert set(table) == set(ob.QPO_MODELS)
    for name, (upper, lower) in table.items():
        nu_u, nu_l = ob.QPO_MODELS[name].frequencies(kerr, r, mass, a=a)
        np.testing.assert_allclose(nu_u, upper, rtol=1e-9, err_msg=name)
        np.testing.assert_allclose(nu_l, lower, rtol=1e-9, err_msg=name)
        np.testing.assert_allclose(
            ob.QPO_MODELS[name].low_frequency(kerr, r, mass, a=a), k - v, rtol=1e-8
        )
    # With rotation the models that coincide in spherical symmetry are distinct.
    rp = ob.QPO_MODELS["RP"].frequencies(kerr, 8.0, mass, a=a)
    rp1 = ob.QPO_MODELS["RP1"].frequencies(kerr, 8.0, mass, a=a)
    assert abs(float(rp[0]) - float(rp1[0])) > 1.0
    # Resonance radii: the model frequencies are in the requested ratio there.
    for name in ("RP", "ER", "TD"):
        r_res = float(ob.resonance_radius(kerr, name, a=a))
        nu_u, nu_l = ob.QPO_MODELS[name].frequencies(kerr, r_res, mass, a=a)
        assert float(nu_u) / float(nu_l) == pytest.approx(1.5, rel=1e-10)
    assert float(ob.resonance_radius(kerr, "RP", a=0.0)) == pytest.approx(27 / 4, rel=1e-11)


def test_ingram_motta_analytic_rpm_solution(kerr):
    """Mandatory validation: the analytic RPM solution of Ingram & Motta (2014), Eqs. (4)-(9),
    inverts the frequencies that BlackPy computes for the Kerr metric."""
    for radius, spin, mass in (
        (5.68, 0.286, 5.3),
        (6.85, 0.149, 7.86),
        (4.0, 0.9, 10.0),
        (12.0, 0.02, 30.0),
    ):
        nu_u, nu_l = ob.QPO_MODELS["RP"].frequencies(kerr, radius, mass, a=spin)
        nu_low = ob.QPO_MODELS["RP"].low_frequency(kerr, radius, mass, a=spin)
        solution = ob.rpm_kerr_solution(nu_low, nu_l, nu_u)
        assert float(solution.radius) == pytest.approx(radius, rel=1e-9)
        assert float(solution.spin) == pytest.approx(spin, rel=1e-8)
        assert float(solution.mass) == pytest.approx(mass, rel=1e-9)
    # Vectorised, and NaN where no solution exists (nodal above periastron frequency).
    out = ob.rpm_kerr_solution(
        np.array([10.0, 250.0]), np.array([200.0, 200.0]), np.array([300.0, 300.0])
    )
    assert np.isfinite(out.radius[0]) and np.isnan(out.radius[1])


# ----------------------------------------------------------------------------- guards, performance


def test_reflection_symmetry_is_required(kerr):
    s = kerr.symbolic
    skewed = StationaryAxisymmetric.from_sympy(
        s.g_tt * (1 + sp.Rational(1, 10) * sp.cos(s.theta) / s.r**2),
        s.g_tphi,
        s.g_phiphi,
        s.g_rr,
        s.g_thth,
        r=s.r,
        theta=s.theta,
        parameters={"a": 0.3},
        name="skewed test metric",
    )
    for call in (
        lambda: ob.isco(skewed),
        lambda: ob.angular_velocity(skewed, 8.0),
        lambda: ob.frequencies_hz(skewed, 8.0, 5.0),
        lambda: ob.QPO_MODELS["RP"].frequencies(skewed, 8.0, 5.0),
        lambda: ob.nodal_precession_frequency(skewed, 8.0, 5.0),
    ):
        with pytest.raises(ReflectionSymmetryError, match="theta -> pi - theta"):
            call()


@pytest.mark.slow
def test_kerr_isco_performance(kerr):
    """Benchmark: 10^4 Kerr ISCO values, vectorised (budget 0.1 ms per evaluation)."""
    a = np.linspace(-0.99, 0.99, 10_000)
    np.asarray(ob.isco(kerr, a=a))  # compile
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        np.asarray(ob.isco(kerr, a=a))
        best = min(best, time.perf_counter() - start)
    per_eval_ms = 1e3 * best / a.size
    print(f"Kerr ISCO: {per_eval_ms:.4f} ms per evaluation")
    assert per_eval_ms < 0.1
