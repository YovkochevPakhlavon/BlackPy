"""Horizons, ergosurface, curvature and effective source of rotating metrics (B2)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy import geometry as geo
from blackpy.errors import SymbolicTimeoutError, SymbolicUnavailableError
from blackpy.metric import StationaryAxisymmetric, rotating_from_static

R = np.geomspace(2.3, 40.0, 25)[:, None]
THETA = np.linspace(0.2, math.pi - 0.2, 9)[None, :]


def kerr_kretschmann(r, theta, a):
    """Closed form derived with sympy in test_kerr_is_vacuum_and_kretschmann_closed_form."""
    x = np.cos(theta)
    sigma = r**2 + a**2 * x**2
    return 48 * (r**2 - a**2 * x**2) * (sigma**2 - 16 * a**2 * r**2 * x**2) / sigma**6


# ----------------------------------------------------------------------------- horizons


def test_kerr_horizons(kerr):
    a = np.linspace(-0.999999, 0.999999, 401)
    np.testing.assert_allclose(geo.event_horizon(kerr, a=a), 1 + np.sqrt(1 - a**2), rtol=1e-12)
    result = geo.horizons(kerr, a=0.6)
    assert [hz.kind for hz in result.horizons] == ["inner", "event"]
    np.testing.assert_allclose(result.radii, [0.2, 1.8], rtol=1e-13)
    assert result.theta_independent is True and "independent of theta" in str(result)
    # Extremal and over-extremal (naked singularity: a result, not an error).
    extremal = geo.horizons(kerr, a=1.0)
    assert [hz.kind for hz in extremal.horizons] == ["extremal"]
    assert math.isclose(extremal.event_horizon, 1.0, rel_tol=1e-7)
    naked = geo.horizons(kerr, a=1.05)
    assert naked.horizonless and naked.event_horizon is None
    assert np.isnan(float(geo.event_horizon(kerr, a=1.05)))
    # The same radius at every polar angle.
    theta = np.linspace(0.1, 3.0, 7)
    np.testing.assert_allclose(geo.event_horizon(kerr, theta, a=0.6), 1.8, rtol=1e-13)


def test_kerr_newman_horizons(kerr_newman):
    for a, q in ((0.3, 0.4), (0.6, 0.7), (0.0, 0.9)):
        root = math.sqrt(1 - a**2 - q**2)
        result = geo.horizons(kerr_newman, a=a, Q=q)
        np.testing.assert_allclose(result.radii, [1 - root, 1 + root], rtol=1e-12)
    assert geo.horizons(kerr_newman, a=0.6, Q=0.8).horizons[0].kind == "extremal"
    assert geo.horizons(kerr_newman, a=0.6, Q=0.81).horizonless


def test_johannsen_horizon_is_the_kerr_horizon(johannsen):
    """Johannsen (2013), Sec. III.1: the event horizon coincides with the Kerr horizon."""
    deformations = dict(alpha13=1.5, alpha22=-0.7, alpha52=0.9, epsilon3=2.0)
    r_h = float(geo.event_horizon(johannsen, a=0.8, **deformations))
    assert math.isclose(r_h, 1.6, rel_tol=1e-12)
    # A root of A_5 is a root of g^rr but not a horizon (the lapse does not vanish there).
    result = geo.horizons(johannsen, a=0.6, alpha52=-0.5)
    kinds = {round(hz.radius, 6): hz.kind for hz in result.horizons}
    assert kinds[round(math.sqrt(0.5), 6)] == "not a horizon" and kinds[1.8] == "event"


def test_ergosurface(kerr, kerr_newman, schwarzschild):
    theta = np.linspace(0.05, math.pi - 0.05, 21)
    for a in (0.3, 0.9, 0.999):
        # g_tt = 0  <=>  r^2 - 2 r + a^2 cos^2(theta) = 0.
        expected = 1 + np.sqrt(1 - a**2 * np.cos(theta) ** 2)
        np.testing.assert_allclose(geo.ergosurface(kerr, theta, a=a), expected, rtol=1e-12)
    assert math.isclose(float(geo.ergosurface(kerr, a=0.9)), 2.0, rel_tol=1e-13)  # equator
    a, q = 0.5, 0.6
    expected = 1 + np.sqrt(1 - q**2 - a**2 * np.cos(theta) ** 2)
    np.testing.assert_allclose(geo.ergosurface(kerr_newman, theta, a=a, Q=q), expected, rtol=1e-12)
    # Non-rotating: the static limit is the horizon.
    assert math.isclose(float(geo.ergosurface(schwarzschild, 1.0)), 2.0, rel_tol=1e-13)


def test_theta_dependent_root_is_reported(kerr):
    """A Kerr-like metric whose 'Delta' depends on theta: g^rr = 0 is not the horizon."""
    r, theta, a = sp.Symbol("r", positive=True), sp.Symbol("theta", real=True), sp.Symbol("a")
    s2 = sp.sin(theta) ** 2
    sigma = r**2 + a**2 * sp.cos(theta) ** 2
    delta = r**2 - 2 * r + a**2 + sp.Rational(1, 10) * sp.cos(theta) ** 2
    squashed = StationaryAxisymmetric.from_sympy(
        -(delta - a**2 * s2) / sigma,
        (delta * a * s2 - a * s2 * (r**2 + a**2)) / sigma,
        (-delta * a**2 * s2**2 + s2 * (r**2 + a**2) ** 2) / sigma,
        sigma / delta,
        sigma,
        r=r,
        theta=theta,
        parameters={"a": 0.5},
        name="squashed test metric",
    )
    profile = geo.horizon_profile(squashed)
    assert not profile.theta_independent and "depends on theta" in profile.message
    assert profile.radius.max() - profile.radius.min() > 0.01
    assert geo.horizons(squashed).theta_independent is False
    assert geo.horizon_profile(kerr, a=0.5).theta_independent
    with pytest.raises(ValueError, match="strictly between"):
        geo.horizon_profile(kerr, theta=[0.0, 1.0])


# ----------------------------------------------------------------------------- curvature


def test_kerr_kretschmann_numeric_against_closed_form(kerr):
    for a in (0.0, 0.5, 0.998):
        numeric = np.asarray(geo.kretschmann(kerr, R, THETA, a=a))
        np.testing.assert_allclose(numeric, kerr_kretschmann(R, THETA, a), rtol=1e-10)
    # Default polar angle: the equator, where K = 48 / r^6 for every spin.
    np.testing.assert_allclose(geo.kretschmann(kerr, 5.0, a=0.7), 48 / 5.0**6, rtol=1e-11)
    np.testing.assert_allclose(geo.ricci_scalar(kerr, R, THETA, a=0.9), 0.0, atol=1e-11)
    np.testing.assert_allclose(geo.einstein_tensor(kerr, 4.0, 1.0, a=0.9), 0.0, atol=1e-12)


@pytest.mark.slow
def test_kerr_is_vacuum_and_kretschmann_closed_form(kerr):
    """Full 4D symbolic computation: G^mu_nu = 0 exactly, and the Kretschmann scalar.

    This derives the closed form used in the numerical test above; it is not
    quoted from the literature.
    """
    assert geo.is_vacuum_symbolic(kerr, timeout=600)
    assert geo.ricci_scalar_symbolic(kerr) == 0
    s = kerr.symbolic
    (a,) = s.params
    x = sp.cos(s.theta)
    sigma = s.r**2 + a**2 * x**2
    closed = 48 * (s.r**2 - a**2 * x**2) * (sigma**2 - 16 * a**2 * s.r**2 * x**2) / sigma**6
    assert sp.simplify(geo.kretschmann_symbolic(kerr) - closed) == 0


@pytest.mark.slow
def test_kerr_newman_symbolic_source(kerr_newman):
    """R = 0 (traceless source) and T^r_r = -Q^2 / (8 pi Sigma^2).

    The second relation is Eq. (31) of Azreg-Ainou (2014) with 2 f = 2 r - Q^2:
    epsilon = -p_r = 2 (r f' - f) / rho^4 = Q^2 / rho^4 in units G_mu_nu = T_mu_nu.
    """
    s = kerr_newman.symbolic
    a, q = s.params
    assert geo.ricci_scalar_symbolic(kerr_newman, timeout=600) == 0
    assert not geo.is_vacuum_symbolic(kerr_newman)
    t_mixed = geo.stress_energy_symbolic(kerr_newman)
    sigma = s.r**2 + a**2 * sp.cos(s.theta) ** 2
    assert sp.simplify(t_mixed[1, 1] + q**2 / (8 * sp.pi * sigma**2)) == 0
    assert sp.simplify(t_mixed[2, 2] - q**2 / (8 * sp.pi * sigma**2)) == 0
    # Kretschmann scalar in the form quoted in docs/theory.md (derived here, not quoted).
    r, x = s.r, sp.cos(s.theta)
    closed = (
        8
        * (
            7 * q**4 * (r**4 + a**4 * x**4)
            - 34 * q**4 * a**2 * r**2 * x**2
            - 12 * q**2 * r * (r**4 - 10 * a**2 * r**2 * x**2 + 5 * a**4 * x**4)
            + 6 * (r**2 - a**2 * x**2) * (r**4 - 14 * a**2 * r**2 * x**2 + a**4 * x**4)
        )
        / sigma**6
    )
    assert sp.simplify(geo.kretschmann_symbolic(kerr_newman) - closed) == 0


def test_general_4d_curvature_agrees_with_static_closed_forms(rn, hayward, sds):
    """The 4D autodiff computation reproduces the v0.1 closed forms in f, g, h."""
    r = np.geomspace(2.5, 30.0, 12)
    for metric, pars in ((rn, {"Q": 0.7}), (hayward, {"l": 0.6}), (sds, {"Lambda": 1e-3})):
        general = metric.as_axisymmetric()
        np.testing.assert_allclose(
            geo.kretschmann(general, r, 0.9, **pars), geo.kretschmann(metric, r, **pars), rtol=1e-9
        )
        np.testing.assert_allclose(
            geo.ricci_scalar(general, r, 0.9, **pars),
            geo.ricci_scalar(metric, r, **pars),
            rtol=1e-8,
            atol=1e-12,
        )
        rho, p_r, p_t = (np.asarray(v) for v in geo.stress_energy(metric, r, **pars))
        t_mixed = np.asarray(geo.stress_energy(general, r, 0.9, **pars))
        expected = np.stack([-rho, p_r, p_t, p_t], axis=-1)
        np.testing.assert_allclose(
            np.diagonal(t_mixed, axis1=-2, axis2=-1), expected, rtol=1e-8, atol=1e-13
        )
        # einstein_tensor accepts a static metric directly.
        np.testing.assert_allclose(
            geo.einstein_tensor(metric, r, 0.9, **pars)[..., 0, 0],
            -8 * math.pi * rho,
            rtol=1e-8,
            atol=1e-12,
        )


def test_symbolic_path_time_limit_and_availability(johannsen):
    with pytest.raises(SymbolicTimeoutError, match="numerical functions"):
        geo.kretschmann_symbolic(johannsen, timeout=0.5)
    numeric = rotating_from_static(bp.StaticSpherical(lambda r, Q=0.0: 1 - 2 / r + Q**2 / r**2))
    with pytest.raises(SymbolicUnavailableError):
        geo.einstein_tensor_symbolic(numeric)


# ----------------------------------------------------------------------------- effective source


def test_classify_stress_energy_types():
    """Hand-made tensors in Minkowski space, one per Hawking-Ellis class used."""
    eta = np.diag([-1.0, 1.0, 1.0, 1.0])
    classify = lambda t: geo.classify_stress_energy(np.asarray(t, dtype=float), eta, 1.0)
    kind, rho, pressures = classify(np.diag([-2.0, 0.5, 0.5, 0.5]))
    assert kind == "perfect fluid" and rho == 2.0 and np.allclose(pressures, 0.5)
    kind, rho, pressures = classify(np.diag([-2.0, 0.1, 0.5, 0.7]))
    assert kind == "anisotropic fluid" and np.allclose(pressures, [0.1, 0.5, 0.7])
    # Electromagnetic type: rho = -p_1 (degenerate eigenvalue shared by a timelike and
    # a spacelike direction).
    kind, rho, pressures = classify(np.diag([-1.0, -1.0, 1.0, 1.0]))
    assert kind == "anisotropic fluid" and rho == 1.0 and np.allclose(pressures, [-1.0, 1.0, 1.0])
    # Boosted perfect fluid: not diagonal in these coordinates, same classification.
    v = 0.6
    gamma = 1 / math.sqrt(1 - v**2)
    u_up = np.array([gamma, gamma * v, 0.0, 0.0])
    boosted = (2.0 + 0.5) * np.outer(u_up, eta @ u_up) + 0.5 * np.eye(4)
    kind, rho, pressures = classify(boosted)
    assert kind == "perfect fluid" and math.isclose(rho, 2.0, rel_tol=1e-12)
    # Null dust T^a_b = k^a k_b: a double null eigenvector, no timelike one (type II).
    k_up = np.array([1.0, 1.0, 0.0, 0.0])
    assert classify(np.outer(k_up, eta @ k_up))[0] == "neither"
    # Complex eigenvalues (type IV).
    rotation = np.zeros((4, 4))
    rotation[0, 1], rotation[1, 0] = 1.0, -1.0
    assert classify(rotation)[0] == "neither"
    assert classify(np.zeros((4, 4))) == ("vacuum", 0.0, pytest.approx(np.zeros(3)))
    assert classify(np.full((4, 4), np.nan))[0] == "neither"


def test_kerr_and_schwarzschild_are_vacuum(kerr, schwarzschild):
    for metric, pars in ((kerr, {"a": 0.95}), (schwarzschild, {})):
        source = geo.effective_source(metric, **pars)
        assert source.kind == "vacuum" and source.fraction("vacuum") == 1.0
        assert all(source.holds(c) for c in ("NEC", "WEC", "SEC", "DEC"))
        assert "vacuum" in source.summary()


def test_kerr_newman_source_is_an_electromagnetic_anisotropic_fluid(kerr_newman):
    a, q = 0.5, 0.6
    source = geo.effective_source(kerr_newman, a=a, Q=q)
    assert source.kind == "anisotropic fluid"
    rho = q**2 / (8 * math.pi * (source.r**2 + a**2 * np.cos(source.theta) ** 2) ** 2)
    np.testing.assert_allclose(source.rho, rho, rtol=1e-8)
    np.testing.assert_allclose(source.pressures, rho[..., None] * [-1.0, 1.0, 1.0], rtol=1e-8)
    assert all(source.holds(c) for c in ("NEC", "WEC", "SEC", "DEC"))
    # energy_conditions() forwards rotating metrics to the same analysis.
    forwarded = geo.energy_conditions(kerr_newman, a=a, Q=q)
    assert isinstance(forwarded, geo.SourceResult) and forwarded.holds("DEC")


def test_static_sources_through_the_general_classifier(rn, sds, hayward):
    assert geo.effective_source(rn, Q=0.5).kind == "anisotropic fluid"
    de_sitter = geo.effective_source(sds, Lambda=1e-3)
    assert de_sitter.kind == "perfect fluid"  # rho = -p = Lambda / (8 pi)
    np.testing.assert_allclose(de_sitter.rho, 1e-3 / (8 * math.pi), rtol=1e-7)
    assert de_sitter.holds("WEC") and de_sitter.holds("DEC") and not de_sitter.holds("SEC")
    regular = geo.effective_source(hayward, l=0.5)
    assert regular.holds("WEC") and not regular.holds("DEC")  # as in the v0.1 radial analysis


def test_rotating_hayward_source_matches_azreg_ainou(hayward):
    """Azreg-Ainou (2014), Eq. (31): epsilon = -p_r = 2 (r f' - f) / rho^4,
    p_theta = p_phi = -p_r - f'' / rho^2 (units G_mu_nu = T_mu_nu), with 2 f = r^2 (1 - F).
    This checks the generator, the 4D curvature and the classifier together."""
    rotating = rotating_from_static(hayward)
    a, ell = 0.6, 0.5
    r, big_l = sp.symbols("r l", positive=True)
    f = r**2 * (1 - hayward.symbolic.f.subs(hayward.symbolic.params[0], big_l)) / 2
    f, f1, f2 = (sp.lambdify([r, big_l], e) for e in (f, sp.diff(f, r), sp.diff(f, r, 2)))
    source = geo.effective_source(rotating, a=a, l=ell)
    assert source.kind == "anisotropic fluid"
    rho2 = source.r**2 + a**2 * np.cos(source.theta) ** 2
    epsilon = 2 * (source.r * f1(source.r, ell) - f(source.r, ell)) / rho2**2 / (8 * math.pi)
    p_theta = epsilon - f2(source.r, ell) / rho2 / (8 * math.pi)
    np.testing.assert_allclose(source.rho, epsilon, rtol=1e-7)
    expected = np.sort(np.stack([-epsilon, p_theta, p_theta], axis=-1), axis=-1)
    np.testing.assert_allclose(source.pressures, expected, rtol=1e-6, atol=1e-14)


def test_johannsen_source_is_not_assumed_physical(johannsen):
    """The deformed metric is not a solution of GR: the report must say what its source is."""
    source = geo.effective_source(johannsen, a=0.5, alpha13=0.5)
    assert source.kind in ("anisotropic fluid", "neither") and not source.holds("WEC")
    assert geo.effective_source(johannsen, a=0.5).kind == "vacuum"  # Kerr limit


def test_report_for_rotating_metrics(kerr, johannsen):
    text = geo.report(kerr, a=0.9)
    for fragment in (
        "stationary, axisymmetric",
        "independent of theta",
        "Ergosurface (g_tt = 0) on the equator: r = 2",
        "Reflection symmetric (theta -> pi - theta): True",
        "Null geodesics separable: True",
        "Effective source: vacuum",
    ):
        assert fragment in text, fragment
    text = geo.report(johannsen, a=0.5, alpha22=1.0, symbolic=True, timeout=0.5)
    assert "Symbolic vacuum test not completed" in text and "VIOLATED" in text
    numeric = StationaryAxisymmetric(
        lambda r, theta: -(1 - 2 / r),
        lambda r, theta: 0.0 * r,
        lambda r, theta: r**2 * jnp.sin(theta) ** 2,
        lambda r, theta: 1 / (1 - 2 / r),
        lambda r, theta: r**2,
    )
    assert "numerical callables" in geo.report(numeric, symbolic=True)
