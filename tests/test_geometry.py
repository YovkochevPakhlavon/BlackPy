"""Horizons, curvature, stress-energy, energy conditions (items 1, 2, 5, 9)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy import geometry as geo
from blackpy.geometry.symbolic import line_element_tensors
from blackpy.metric import StaticSpherical


def test_schwarzschild_horizon(schwarzschild):
    res = geo.horizons(schwarzschild)
    assert not res.horizonless and len(res.horizons) == 1
    assert math.isclose(res.event_horizon, 2.0, rel_tol=1e-14)
    assert res.horizons[0].kind == "event" and res.cosmological_horizon is None


@pytest.mark.parametrize("q", [0.1, 0.5, 0.9, 0.999, 0.999999])
def test_rn_two_horizons(rn, q):
    """Validation items 2 and 9: r_pm = 1 +- sqrt(1 - Q^2)."""
    res = geo.horizons(rn, Q=q)
    root = math.sqrt(1 - q**2)
    assert [h.kind for h in res.horizons] == ["inner", "event"]
    # The horizon position is ill-conditioned near extremality (dr ~ eps / sqrt(1 - Q^2)).
    tol = 1e-13 / root
    assert math.isclose(res.horizons[0].radius, 1 - root, rel_tol=tol)
    assert math.isclose(res.event_horizon, 1 + root, rel_tol=tol)
    assert all(h.f_vanishes for h in res.horizons)


def test_rn_extremal_and_horizonless(rn):
    """Validation item 9: degenerate horizon and naked singularity are handled, not errors."""
    ext = geo.horizons(rn, Q=1.0)
    assert len(ext.horizons) == 1 and ext.horizons[0].kind == "extremal"
    assert math.isclose(ext.event_horizon, 1.0, rel_tol=1e-9) and ext.horizons[0].degenerate
    naked = geo.horizons(rn, Q=1.1)
    assert naked.horizonless and naked.event_horizon is None and naked.horizons == ()
    assert "horizonless" in naked.message
    assert np.isnan(geo.event_horizon(rn, Q=1.1))


def test_event_horizon_vectorised(rn):
    # The grid avoids Q = 1 exactly, where the horizon position is ill-conditioned.
    q = jnp.concatenate([jnp.linspace(0.0, 0.99, 23), jnp.array([1.01, 1.2])])
    out = np.asarray(geo.event_horizon(rn, Q=q))
    expected = np.where(q < 1.0, 1 + np.sqrt(np.clip(1 - np.asarray(q) ** 2, 0, None)), np.nan)
    np.testing.assert_allclose(out, expected, rtol=1e-9, equal_nan=True)


def test_sds_event_and_cosmological_horizon(sds):
    """The outermost root of g is the cosmological horizon, not the event horizon."""
    lam = 0.01
    res = geo.horizons(sds, Lambda=lam)
    assert [h.kind for h in res.horizons] == ["event", "cosmological"]
    for radius in res.radii:
        assert abs(1 - 2 / radius - lam * radius**2 / 3) < 1e-13
    assert res.event_horizon < res.cosmological_horizon
    assert math.isclose(float(geo.cosmological_horizon(sds, Lambda=lam)), res.cosmological_horizon)


def test_pure_de_sitter_is_horizonless_with_cosmological_horizon():
    ds = StaticSpherical(lambda r, Lambda=0.03: 1 - Lambda * r**2 / 3, name="de Sitter")
    res = geo.horizons(ds)
    assert res.horizonless and math.isclose(res.cosmological_horizon, 10.0, rel_tol=1e-13)


def test_root_of_g_without_root_of_f_is_not_a_horizon():
    """Wormhole-like throat: g vanishes, f does not."""
    wh = StaticSpherical(lambda r: 1.0 + 0.0 * r, g=lambda r: 1 - 2 / r, name="throat")
    res = geo.horizons(wh)
    assert res.horizonless and res.horizons[0].kind == "not a horizon"
    assert not res.horizons[0].f_vanishes and res.radii == ()


@pytest.mark.parametrize(
    "name,par,critical",
    [("Bardeen", "q", 4 / (3 * math.sqrt(3))), ("Hayward", "l", 4 / (3 * math.sqrt(3)))],
)
def test_regular_black_holes_lose_horizon_above_critical_value(name, par, critical):
    metric = bp.metric.CATALOGUE[name]()
    assert len(geo.horizons(metric, **{par: 0.99 * critical}).horizons) == 2
    assert geo.horizons(metric, **{par: 1.01 * critical}).horizonless


def test_closed_form_curvature_matches_full_tensor_computation():
    """The closed forms used numerically equal the textbook tensor definitions for
    arbitrary f(r), g(r), h(r)."""
    r = sp.Symbol("r", positive=True)
    f, g, h = (sp.Function(n)(r) for n in "fgh")
    out = line_element_tensors(f, g, h, r)
    f1, f2, g1, h1, h2 = (
        sp.diff(f, r),
        sp.diff(f, r, 2),
        sp.diff(g, r),
        sp.diff(h, r),
        sp.diff(h, r, 2),
    )
    e1 = g * (f2 / (2 * f) - f1**2 / (4 * f**2)) + g1 * f1 / (4 * f)
    e2 = g * f1 * h1 / (4 * f * h)
    e3 = g * (h2 / (2 * h) - h1**2 / (4 * h**2)) + g1 * h1 / (4 * h)
    e4 = 1 / h - g * h1**2 / (4 * h**2)
    ein = out["einstein_mixed"]
    assert sp.simplify(ein[0, 0] - (2 * e3 - e4)) == 0
    assert sp.simplify(ein[1, 1] - (2 * e2 - e4)) == 0
    assert sp.simplify(ein[2, 2] - (e1 + e2 + e3)) == 0
    assert sp.simplify(ein[3, 3] - ein[2, 2]) == 0
    assert all(ein[i, j] == 0 for i in range(4) for j in range(4) if i != j)
    assert sp.simplify(out["kretschmann"] - (4 * e1**2 + 8 * e2**2 + 8 * e3**2 + 4 * e4**2)) == 0
    assert sp.simplify(out["ricci_scalar"] - (-2 * e1 - 4 * e2 - 4 * e3 + 2 * e4)) == 0


def test_schwarzschild_is_vacuum(schwarzschild):
    """Validation item 5: zero effective stress-energy, symbolically and numerically."""
    assert geo.is_vacuum_symbolic(schwarzschild)
    assert all(v == 0 for v in geo.stress_energy_symbolic(schwarzschild).values())
    r = jnp.geomspace(2.1, 100.0, 30)
    for comp in geo.stress_energy(schwarzschild, r):
        assert np.max(np.abs(np.asarray(comp)) * np.asarray(r) ** 2) < 1e-14
    r_sym = schwarzschild.symbolic.r
    assert sp.simplify(geo.kretschmann_symbolic(schwarzschild) - 48 / r_sym**6) == 0
    np.testing.assert_allclose(geo.kretschmann(schwarzschild, r), 48 / r**6, rtol=1e-12)
    assert geo.ricci_scalar_symbolic(schwarzschild) == 0


def test_rn_stress_energy_is_coulomb_field(rn):
    """Validation item 5: rho = -p_r = p_t = Q^2 / (8 pi r^4), with
    rho = -T^t_t and G^mu_nu = 8 pi T^mu_nu."""
    sym = rn.symbolic
    (q,) = sym.params
    expected = q**2 / (8 * sp.pi * sym.r**4)
    se = geo.stress_energy_symbolic(rn)
    assert sp.simplify(se["rho"] - expected) == 0
    assert sp.simplify(se["p_r"] + expected) == 0
    assert sp.simplify(se["p_t"] - expected) == 0
    r = jnp.geomspace(2.0, 50.0, 20)
    num = geo.stress_energy(rn, r, Q=0.8)
    exact = 0.64 / (8 * np.pi * np.asarray(r) ** 4)
    np.testing.assert_allclose(num.rho, exact, rtol=1e-10)
    np.testing.assert_allclose(num.p_r, -exact, rtol=1e-10)
    np.testing.assert_allclose(num.p_t, exact, rtol=1e-10)
    # Kretschmann of RN: 48/r^6 - 96 Q^2/r^7 + 56 Q^4/r^8 (M = 1).
    k_exact = 48 / r**6 - 96 * 0.64 / r**7 + 56 * 0.64**2 / r**8
    np.testing.assert_allclose(geo.kretschmann(rn, r, Q=0.8), k_exact, rtol=1e-10)


def test_numeric_stress_energy_for_callable_metric_matches_symbolic(hayward):
    numeric_only = StaticSpherical(lambda r, l=0.5: 1 - 2 * r**2 / (r**3 + 2 * l**2))
    r = jnp.geomspace(2.0, 30.0, 12)
    sym = hayward.symbolic
    se = geo.stress_energy_symbolic(hayward)
    num = geo.stress_energy(numeric_only, r)
    for key, got in zip(("rho", "p_r", "p_t"), num, strict=True):
        fn = sp.lambdify(sym.r, se[key].subs(sym.params[0], 0.5), "numpy")
        np.testing.assert_allclose(got, fn(np.asarray(r)), rtol=1e-9, atol=1e-16)


def test_energy_conditions(schwarzschild, rn, sds, hayward):
    assert all(geo.energy_conditions(schwarzschild).holds(c) for c in ("NEC", "WEC", "SEC", "DEC"))
    assert all(geo.energy_conditions(rn, Q=0.9).holds(c) for c in ("NEC", "WEC", "SEC", "DEC"))
    # Positive cosmological constant: rho = -p = Lambda / 8 pi violates only the SEC.
    ec = geo.energy_conditions(sds, Lambda=0.01)
    assert ec.holds("NEC") and ec.holds("WEC") and ec.holds("DEC") and not ec.holds("SEC")
    assert ec.r[-1] < geo.horizons(sds, Lambda=0.01).cosmological_horizon
    # Hayward: p_t / rho = 2 (r^3 - l^2) / (r^3 + 2 l^2) > 1 for r^3 > 4 l^2: DEC fails there.
    ec = geo.energy_conditions(hayward, l=0.5)
    assert ec.holds("WEC") and not ec.holds("DEC")
    assert np.all(ec.violated("DEC") ** 3 > 4 * 0.25 * (1 - 1e-9))
    assert "VIOLATED" in ec.summary() and len(ec.violation_intervals("DEC")) == 1


def test_phantom_like_metric_violates_nec():
    """f = 1 - 2/r - a/r^2 with a > 0 is RN with Q^2 -> -a: negative energy density."""
    m = StaticSpherical(lambda r, a=0.5: 1 - 2 / r - a / r**2)
    ec = geo.energy_conditions(m)
    assert not ec.holds("WEC") and not ec.holds("DEC")
    assert ec.violation_intervals("WEC")[0][0] == ec.r[0]


def test_report_runs_for_symbolic_and_numeric(rn):
    text = geo.report(rn, Q=0.5)
    assert "event horizon" in text and "rho = Q**2/(8*pi*r**4)" in text
    numeric = StaticSpherical(lambda r: 1 - 2 / r)
    assert "Symbolic checks skipped" in geo.report(numeric)
