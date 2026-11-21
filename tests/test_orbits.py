"""Circular orbits, ISCO, epicyclic frequencies, QPO models (items 1, 2, 3)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import blackpy as bp
from blackpy import orbits as ob
from blackpy.errors import NodalPrecessionUnavailableError, NotAsymptoticallyFlatError
from blackpy.metric import CATALOGUE, StaticSpherical


def test_schwarzschild_special_orbits(schwarzschild):
    """Validation item 1: r_ISCO = 6, r_mb = 4."""
    assert math.isclose(float(ob.isco(schwarzschild)), 6.0, rel_tol=1e-13)
    assert math.isclose(float(ob.marginally_bound_orbit(schwarzschild)), 4.0, rel_tol=1e-13)
    orbits = ob.marginally_stable_orbits(schwarzschild)
    assert len(orbits) == 1 and orbits[0].kind == "ISCO"
    (region,) = ob.circular_orbit_region(schwarzschild)
    assert math.isclose(region[0], 3.0, rel_tol=1e-13)


def test_schwarzschild_orbit_quantities(schwarzschild):
    r = jnp.geomspace(6.0, 500.0, 40)
    omega = ob.angular_velocity(schwarzschild, r)
    np.testing.assert_allclose(omega, r**-1.5, rtol=1e-14)
    np.testing.assert_allclose(
        ob.specific_energy(schwarzschild, r), (1 - 2 / r) / jnp.sqrt(1 - 3 / r), rtol=1e-14
    )
    np.testing.assert_allclose(
        ob.specific_angular_momentum(schwarzschild, r),
        jnp.sqrt(r) / jnp.sqrt(1 - 3 / r),
        rtol=1e-14,
    )
    assert math.isclose(
        float(ob.specific_energy(schwarzschild, 6.0)), math.sqrt(8 / 9), rel_tol=1e-14
    )
    assert np.isnan(ob.angular_velocity(schwarzschild, 2.5))  # inside the photon sphere
    # No timelike circular orbit inside the photon orbit r = 3, one just outside it. At
    # r = 3 itself the existence condition is 0 = 0 and its sign is round-off (it differs
    # between JAX versions), so the boundary point is not tested.
    assert not bool(ob.circular_orbit_exists(schwarzschild, 3.0 - 1e-6))
    assert bool(ob.circular_orbit_exists(schwarzschild, 3.0 + 1e-6))


def test_schwarzschild_epicyclic_relations(schwarzschild):
    """Validation item 1: nu_theta = nu_phi and nu_r^2 = nu_phi^2 (1 - 6/r)."""
    r = jnp.geomspace(6.5, 300.0, 40)
    omega = ob.angular_velocity(schwarzschild, r)
    np.testing.assert_array_equal(ob.vertical_epicyclic_frequency(schwarzschild, r), omega)
    np.testing.assert_allclose(
        ob.radial_epicyclic_frequency(schwarzschild, r) ** 2, omega**2 * (1 - 6 / r), rtol=1e-12
    )
    # Unstable orbit: negative squared frequency, NaN frequency.
    assert float(ob.radial_epicyclic_frequency_squared(schwarzschild, 5.0)) < 0
    assert np.isnan(ob.radial_epicyclic_frequency(schwarzschild, 5.0))


def test_rn_isco(rn):
    """Validation item 2: extremal RN has r_ISCO = 4; general Q solves the known cubic."""
    assert math.isclose(float(ob.isco(rn, Q=1.0)), 4.0, rel_tol=1e-12)
    q = np.linspace(0.0, 1.0, 41)
    r = np.asarray(ob.isco(rn, Q=q))
    # RN ISCO condition: r^3 - 6 r^2 + 9 Q^2 r - 4 Q^4 = 0 (M = 1).
    assert np.max(np.abs(r**3 - 6 * r**2 + 9 * q**2 * r - 4 * q**4)) < 1e-11
    assert np.all(np.diff(r) < 0)


def test_rn_epicyclic_frequency_against_independent_formula(rn):
    """Omega_r^2 from the effective potential V = f (1 + L^2/r^2): Omega_r^2 = f^2 V''/(2 E^2)."""
    import sympy as sp

    r, q, ell = sp.symbols("r Q L", positive=True)
    f = 1 - 2 / r + q**2 / r**2
    v = f * (1 + ell**2 / r**2)
    l2 = sp.solve(sp.diff(v, r), ell**2)[0]
    energy2 = v.subs(ell**2, l2)
    omega_r2 = sp.simplify((f**2 * sp.diff(v, r, 2) / (2 * energy2)).subs(ell**2, l2))
    fn = sp.lambdify((r, q), omega_r2, "numpy")
    radii = np.linspace(5.0, 30.0, 26)
    got = ob.radial_epicyclic_frequency_squared(rn, radii, Q=0.6)
    np.testing.assert_allclose(got, fn(radii, 0.6), rtol=1e-11)


@pytest.mark.slow
def test_general_f_g_h_metric_against_isotropic_schwarzschild():
    """Schwarzschild in isotropic coordinates has f != g and h != r^2 but the same
    coordinate-invariant observables."""
    f = lambda rho: ((1 - 1 / (2 * rho)) / (1 + 1 / (2 * rho))) ** 2
    psi4 = lambda rho: (1 + 1 / (2 * rho)) ** 4
    iso = StaticSpherical(
        f,
        g=lambda rho: 1 / psi4(rho),
        h=lambda rho: rho**2 * psi4(rho),
        name="isotropic Schwarzschild",
        asymptotically_flat=True,
    )
    areal = lambda rho: rho * (1 + 1 / (2 * rho)) ** 2
    rho_isco = float(ob.isco(iso))
    assert math.isclose(areal(rho_isco), 6.0, rel_tol=1e-12)
    assert math.isclose(areal(float(bp.light.photon_sphere(iso))), 3.0, rel_tol=1e-12)
    assert math.isclose(
        float(bp.light.critical_impact_parameter(iso)), 3 * math.sqrt(3), rel_tol=1e-12
    )
    assert math.isclose(areal(float(ob.marginally_bound_orbit(iso))), 4.0, rel_tol=1e-12)
    rho = np.linspace(6.0, 40.0, 18)
    big_r = areal(rho)
    np.testing.assert_allclose(ob.angular_velocity(iso, rho), big_r**-1.5, rtol=1e-12)
    np.testing.assert_allclose(
        ob.radial_epicyclic_frequency_squared(iso, rho), big_r**-3.0 * (1 - 6 / big_r), rtol=1e-10
    )


@pytest.mark.parametrize(
    "name", ["ReissnerNordstrom", "SchwarzschildDeSitter", "Bardeen", "Hayward"]
)
def test_schwarzschild_limit_of_catalogue(name):
    """Validation item 3: ISCO, r_ph and b_c reduce to Schwarzschild to 1e-10."""
    metric = CATALOGUE[name]()
    zero = {metric.param_names[0]: 0.0}
    assert abs(float(ob.isco(metric, **zero)) - 6.0) < 1e-10
    assert abs(float(bp.light.photon_sphere(metric, **zero)) - 3.0) < 1e-10
    assert abs(float(bp.light.critical_impact_parameter(metric, **zero)) - 3 * math.sqrt(3)) < 1e-10
    assert abs(float(bp.geometry.event_horizon(metric, **zero)) - 2.0) < 1e-10
    # Continuity: a tiny parameter changes the ISCO only slightly.
    tiny = {metric.param_names[0]: 1e-6}
    assert abs(float(ob.isco(metric, **tiny)) - 6.0) < 1e-2


def test_sds_isco_osco_and_static_radius(sds):
    """Stable orbits only between ISCO and OSCO; none above Lambda = 4/5625
    (Stuchlik & Hledik 1999)."""
    lam = 1e-4
    orbits = ob.marginally_stable_orbits(sds, Lambda=lam)
    assert [o.kind for o in orbits] == ["ISCO", "OSCO"]
    assert math.isclose(float(ob.isco(sds, Lambda=lam)), orbits[0].radius)
    (region,) = ob.circular_orbit_region(sds, Lambda=lam)
    assert math.isclose(region[1], (3 / lam) ** (1 / 3), rel_tol=1e-12)  # static radius
    critical = 4 / 5625
    assert len(ob.marginally_stable_orbits(sds, Lambda=0.999 * critical)) == 2
    assert np.isnan(ob.isco(sds, Lambda=1.001 * critical))
    with pytest.raises(NotAsymptoticallyFlatError, match="E = 1"):
        ob.marginally_bound_orbit(sds, Lambda=lam)


def test_sds_critical_lambda_symbolic(sds):
    """ISCO and OSCO merge at r = 15/2, Lambda = 4/5625 (theory, section 6): the double
    root of r f f'' - 2 r f'^2 + 3 f f'; the other double root lies on the horizon."""
    import sympy as sp

    r, lam = sp.symbols("r Lambda", positive=True)
    f = 1 - 2 / r - lam * r**2 / 3
    fp = sp.diff(f, r)
    num = sp.numer(sp.together(r * f * sp.diff(fp, r) - 2 * r * fp**2 + 3 * f * fp))
    roots = sp.solve([num, sp.diff(num, r)], [r, lam], dict=True)
    merger = [s for s in roots if f.subs(s) > 0]
    assert merger == [{r: sp.Rational(15, 2), lam: sp.Rational(4, 5625)}]
    assert [f.subs(s) for s in roots if s not in merger] == [0]
    # The numerical ISCO and OSCO close in on r = 15/2 just below the critical value.
    orbits = ob.marginally_stable_orbits(sds, Lambda=(1 - 1e-6) * 4 / 5625)
    assert [o.kind for o in orbits] == ["ISCO", "OSCO"]
    assert all(abs(o.radius - 7.5) < 0.05 for o in orbits)


def test_naked_rn_singularity_orbits(rn):
    """1 < Q^2 < 9/8: two photon spheres, two disjoint regions of circular orbits."""
    regions = ob.circular_orbit_region(rn, Q=1.05)
    assert len(regions) == 2
    assert math.isclose(regions[0][0], 1.05**2, rel_tol=1e-12)  # f' = 0 at r = Q^2
    assert np.isfinite(ob.isco(rn, Q=1.05))


def test_frequencies_in_hz(schwarzschild):
    freqs = ob.frequencies_hz(schwarzschild, 8.0, 5.0)
    expected = 8.0**-1.5 * bp.constants.C**3 / (2 * math.pi * bp.constants.GM_SUN * 5.0)
    assert math.isclose(float(freqs.nu_phi), expected, rel_tol=1e-13)
    assert math.isclose(float(freqs.nu_r), expected * math.sqrt(1 - 6 / 8), rel_tol=1e-12)
    assert float(freqs.nu_theta) == float(freqs.nu_phi)
    # Frequencies scale as 1/M.
    assert math.isclose(
        float(ob.frequencies_hz(schwarzschild, 8.0, 10.0).nu_phi), expected / 2, rel_tol=1e-13
    )


def test_qpo_model_definitions(schwarzschild):
    """Definitions transcribed from Stuchlik & Kolos (2016), Table 1."""
    k, r, v = 5.0, 2.0, 4.0  # nu_phi, nu_r, nu_theta
    expected = {
        "RP": (k, k - r),
        "RP1": (v, k - r),
        "RP2": (k, v - r),
        "ER": (v, r),
        "ER1": (v, v - r),
        "ER2": (v - r, r),
        "ER3": (v + r, v),
        "ER4": (v + r, v - r),
        "ER5": (r, v - r),
        "TD": (k + r, k),
        "WD": (2 * k - r, 2 * (k - r)),
    }
    assert set(ob.QPO_MODELS) == set(expected)
    for name, (upper, lower) in expected.items():
        model = ob.QPO_MODELS[name]
        assert (model.upper(k, r, v), model.lower(k, r, v)) == (upper, lower), name
        assert "Stuchlik" in model.reference
    assert ob.get_qpo_model("rpm") is ob.QPO_MODELS["RP"]
    with pytest.raises(ValueError, match="Available models"):
        ob.get_qpo_model("nonsense")
    nu_u, nu_l = ob.QPO_MODELS["RP"].frequencies(schwarzschild, 7.0, 6.0)
    freqs = ob.frequencies_hz(schwarzschild, 7.0, 6.0)
    assert float(nu_u) == float(freqs.nu_phi)
    assert math.isclose(float(nu_l), float(freqs.nu_phi - freqs.nu_r), rel_tol=1e-14)


def test_resonance_radii_schwarzschild(schwarzschild):
    """3:2 radii: RP has 1 - sqrt(1 - 6/r) = 2/3, r = 27/4; ER has sqrt(1 - 6/r) = 2/3, r = 54/5."""
    assert math.isclose(float(ob.resonance_radius(schwarzschild, "RP")), 6.75, rel_tol=1e-12)
    assert math.isclose(float(ob.resonance_radius(schwarzschild, "ER")), 10.8, rel_tol=1e-12)


def test_nodal_precession_is_refused_explicitly(schwarzschild):
    with pytest.raises(NodalPrecessionUnavailableError, match="vanishes identically"):
        ob.nodal_precession_frequency(schwarzschild, 6.0, 5.0)
    with pytest.raises(NodalPrecessionUnavailableError):
        ob.QPO_MODELS["RP"].low_frequency(schwarzschild, 6.0, 5.0)


@pytest.mark.slow
def test_isco_performance(rn):
    """Validation item 8: ISCO over 10^4 parameter values, vectorised.

    Budget: 0.05 ms per evaluation after compilation (the design target is
    0.01 ms; the budget is looser so that slow CI machines do not fail).
    Measured timings are printed (run pytest with ``-s``) and recorded in
    verification/v0.1.md.
    """
    import time

    q = jnp.linspace(0.0, 1.0, 10_000)
    results = {}
    for label, kwargs in (("robust", {}), ("fast", {"detect_extrema": False})):
        ob.isco(rn, Q=q, **kwargs).block_until_ready()  # compile
        times = []
        for _ in range(7):
            start = time.perf_counter()
            out = ob.isco(rn, Q=q, **kwargs).block_until_ready()
            times.append(time.perf_counter() - start)
        results[label] = min(times) / q.size * 1e3
        assert not np.isnan(np.asarray(out)).any()
    print(
        f"\nISCO timing, RN, 10^4 values: robust {results['robust']:.5f} ms/eval, "
        f"fast {results['fast']:.5f} ms/eval"
    )
    assert results["robust"] < 0.05
    qn = np.asarray(q)
    r = np.asarray(out)
    assert np.max(np.abs(r**3 - 6 * r**2 + 9 * qn**2 * r - 4 * qn**4)) < 1e-11
