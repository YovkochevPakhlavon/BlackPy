"""Catalogue metrics added in v0.3 (B8): GMGHS, Simpson-Visser, Kerr-Sen, rotating
Bardeen and rotating Hayward. Every reference was read; see the module docstrings."""

import math

import numpy as np
import pytest

import blackpy as bp

mt = bp.metric


@pytest.fixture(scope="module")
def gmghs():
    return mt.GMGHS()


@pytest.fixture(scope="module")
def simpson_visser():
    return mt.SimpsonVisser()


@pytest.fixture(scope="module")
def kerr_sen():
    return mt.KerrSen()


# ----------------------------------------------------------------------------- GMGHS


def test_gmghs(gmghs, schwarzschild):
    """Schwarzschild at Q = 0; Rezzolla & Zhidenko (2014), Eqs. (37) and (43)."""
    assert not gmghs.h_is_r_squared and gmghs.reference
    r = np.geomspace(2.01, 300.0, 30)
    np.testing.assert_allclose(gmghs.f(r), schwarzschild.f(r), rtol=1e-14)
    for charge in (0.4, 0.9, 1.3):
        b, m = charge**2 / 2.0, 1.0 - charge**2 / 2.0
        # Event horizon at rho = 2m (Sen 1992, Eq. (20) at a = 0); areal radius 2 sqrt(m).
        horizon = float(bp.geometry.event_horizon(gmghs, Q=charge))
        assert horizon == pytest.approx(2.0 * m, rel=1e-12)
        assert math.sqrt(float(gmghs.h(horizon, Q=charge))) == pytest.approx(2 * math.sqrt(m))
        # RZ Eq. (37) in units of mu = m.
        beta = b / m
        impact = m * math.sqrt((27 + 36 * beta + 8 * beta**2 + (9 + 8 * beta) ** 1.5) / 2.0)
        assert float(bp.light.shadow_radius(gmghs, Q=charge)) == pytest.approx(impact, rel=1e-11)
        # RZ Eq. (43): Omega_ISCO = (1/(2 mu)) sqrt(kappa / ((1 + kappa)(1 + kappa + kappa^2)^3)).
        kappa = (1.0 + beta) ** (1.0 / 3.0)
        omega = math.sqrt(kappa / ((1 + kappa) * (1 + kappa + kappa**2) ** 3)) / (2.0 * m)
        r_isco = float(bp.orbits.isco(gmghs, Q=charge))
        assert float(bp.orbits.angular_velocity(gmghs, r_isco, Q=charge)) == pytest.approx(
            omega, rel=1e-10
        )
    report = str(bp.geometry.report(gmghs, Q=0.9))
    assert "event horizon" in report.lower()


def test_gmghs_equals_gibbons_maeda(gmghs):
    """Gibbons & Maeda (1988), Eqs. (4.17)-(4.19) with D = 4, g_2 = 1, kappa^2 = 8 pi, M = 1:
    ds^2 = F^2 ds_T^2, F = (1 + alpha/r)^(1/2); r_H = 2(1 + sqrt2 S), alpha = -2 sqrt2 S for the
    dimensionless scalar charge S, and Q_GM^2 = 4 pi alpha (Eq. 4.15, rationalised units)."""
    r = np.geomspace(2.5, 300.0, 25)
    for charge in (0.4, 0.9, 1.3):
        scalar = -(charge**2) / (2.0 * math.sqrt(2.0))  # alpha = Q_Gaussian^2 = Q_GM^2 / (4 pi)
        r_h, alpha = 2.0 * (1.0 + math.sqrt(2.0) * scalar), -2.0 * math.sqrt(2.0) * scalar
        conformal = 1.0 + alpha / r  # F^2
        lapse = 1.0 - r_h / r
        np.testing.assert_allclose(gmghs.f(r, Q=charge), lapse / conformal, rtol=1e-13)
        np.testing.assert_allclose(gmghs.g(r, Q=charge), lapse / conformal, rtol=1e-13)
        np.testing.assert_allclose(gmghs.h(r, Q=charge), conformal * r**2, rtol=1e-13)


# ----------------------------------------------------------------------------- Simpson-Visser


def test_simpson_visser(simpson_visser, schwarzschild):
    """Simpson & Visser (2019), section 7: r_H, r_photon, r_ISCO; one family, four geometries."""
    sv = simpson_visser
    r = np.geomspace(2.01, 300.0, 30)
    np.testing.assert_allclose(sv.f(r), schwarzschild.f(r), rtol=1e-14)  # ell = 0
    for ell in (0.5, 1.5, 1.99):  # regular black hole
        assert float(bp.geometry.event_horizon(sv, ell=ell)) == pytest.approx(
            math.sqrt(4.0 - ell**2), rel=1e-11
        )
    for ell in (0.5, 1.5, 2.5, 2.9):  # black hole and traversable wormhole with a photon sphere
        assert float(bp.light.photon_sphere(sv, ell=ell)) == pytest.approx(
            math.sqrt(9.0 - ell**2), rel=1e-11
        )
        # Derived: h/f at the photon sphere is 9 / (1/3) = 27 for every ell < 3.
        assert float(bp.light.shadow_radius(sv, ell=ell)) == pytest.approx(
            3.0 * math.sqrt(3.0), rel=1e-11
        )
    for ell in (0.5, 2.5, 4.0, 5.5):
        assert float(bp.orbits.isco(sv, ell=ell)) == pytest.approx(
            math.sqrt(36.0 - ell**2), rel=1e-10
        )
    # Horizonless configurations are results, not errors.
    wormhole = bp.geometry.horizons(sv, ell=2.5)
    assert wormhole.horizonless
    assert np.isnan(float(bp.light.photon_sphere(sv, ell=3.5)))  # "no photon sphere" (a > 3m)
    assert np.isnan(float(bp.orbits.isco(sv, ell=6.5)))  # "no ISCO" (a > 6m)
    # The extremal null throat (ell = 2) sits at r = 0, outside the scanned region r > 0.
    assert bp.geometry.horizons(sv, ell=2.0).horizonless
    with pytest.raises(ValueError, match="already has a parameter"):
        mt.rotating_from_static(bp.StaticSpherical(lambda r, a=0.0: 1 - 2 / np.sqrt(r**2 + a**2)))
    rotating = mt.rotating_from_static(sv)  # works because the parameter is called ell
    assert rotating.param_names == ("a", "ell")


# ----------------------------------------------------------------------------- Kerr-Sen


@pytest.mark.slow
def test_kerr_sen(kerr_sen, kerr, gmghs):
    """Sen (1992) against KRZ (2016) Eq. (6); Kerr and GMGHS limits; horizons; separability."""
    ks = kerr_sen
    assert ks.is_reflection_symmetric
    assert ks.check_separability(a=0.7, Q=0.6)  # tested, not assumed
    r, theta = np.meshgrid(np.geomspace(2.2, 60.0, 9), np.linspace(0.3, 2.8, 5), indexing="ij")
    for name in ("g_tt", "g_tphi", "g_phiphi", "g_rr", "g_thth"):
        np.testing.assert_allclose(
            getattr(ks, name)(r, theta, a=0.6), getattr(kerr, name)(r, theta, a=0.6), rtol=1e-13
        )
    static = gmghs.as_axisymmetric()
    for name in ("g_tt", "g_phiphi", "g_rr", "g_thth"):
        np.testing.assert_allclose(
            getattr(ks, name)(r, theta, Q=0.8), getattr(static, name)(r, theta, Q=0.8), rtol=1e-13
        )
    # KRZ Eq. (6): f, beta, sigma, kappa^2, omega with mu = M - b (M = 1).
    a, charge = 0.5, 0.7
    b, mu = charge**2 / 2.0, 1.0 - charge**2 / 2.0
    for rho, th in ((2.5, 0.6), (4.0, 1.2), (11.0, 2.0)):
        common = rho**2 + 2 * b * rho + a**2 * math.cos(th) ** 2
        f = (rho**2 - 2 * mu * rho + a**2) / rho**2
        sigma = common / rho**2
        kappa2 = ((rho**2 + 2 * b * rho + a**2) ** 2 - a**2 * math.sin(th) ** 2 * f * rho**2) / (
            rho**2 * common
        )
        omega = 2 * (mu + b) * a / common
        sin2 = math.sin(th) ** 2
        expected = (
            -(f - omega**2 * sin2) / kappa2,
            -omega * rho * sin2,
            kappa2 * rho**2 * sin2,
            sigma / f,
            sigma * rho**2,
        )
        got = [
            float(getattr(ks, n)(rho, th, a=a, Q=charge))
            for n in ("g_tt", "g_tphi", "g_phiphi", "g_rr", "g_thth")
        ]
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    # Horizons, Sen Eq. (20): rho = m +- sqrt(m^2 - a^2), m = 1 - Q^2/2.
    m = 1.0 - charge**2 / 2.0
    found = bp.geometry.horizons(ks, a=a, Q=charge)
    np.testing.assert_allclose(
        sorted(found.radii),
        [m - math.sqrt(m * m - a * a), m + math.sqrt(m * m - a * a)],
        rtol=1e-11,
    )
    assert bool(ks.in_domain(a=0.7, Q=0.6)) and not bool(ks.in_domain(a=0.9, Q=0.6))
    # Orbits and shadow are defined; the a = 0 shadow equals GMGHS.
    assert np.isfinite(float(bp.orbits.isco(ks, a=a, Q=charge)))
    assert float(bp.light.shadow_radius(ks, inclination=1.0, a=1e-9, Q=0.8)) == pytest.approx(
        float(bp.light.shadow_radius(gmghs, Q=0.8)), rel=1e-6
    )
    observables = bp.light.shadow_observables(ks, math.pi / 2, a=0.5, Q=0.8)
    assert float(observables.delta_c) > 0.0


# ----------------------------------------------------------------------------- rotating regular


@pytest.mark.parametrize(
    "which",
    [
        pytest.param("Bardeen", marks=pytest.mark.slow),
        pytest.param("Hayward", marks=pytest.mark.slow),
    ],
)
def test_rotating_regular_black_holes(which, bardeen, hayward, kerr):
    """Bambi & Modesto (2013), Eq. (14) with m(r) of Eqs. (15) and (26)."""
    if which == "Bardeen":
        factory, static, name, value = mt.RotatingBardeen, bardeen, "q", 0.5
        mass = lambda r: (r**2 / (r**2 + value**2)) ** 1.5
    else:
        factory, static, name, value = mt.RotatingHayward, hayward, "l", 0.5
        mass = lambda r: r**3 / (r**3 + 2.0 * value**2)  # g^3 = 2 l^2
    metric = factory()
    assert "effective_source" in factory.__doc__ and "generated model" in factory.__doc__
    assert metric.param_names == ("a", name) and metric.is_separable
    a = 0.6
    for r, th in ((2.5, 0.7), (5.0, 1.4), (20.0, 2.5)):
        sigma = r**2 + a**2 * math.cos(th) ** 2
        delta = r**2 - 2 * mass(r) * r + a**2
        sin2 = math.sin(th) ** 2
        expected = (  # Bambi & Modesto Eq. (14) with M -> m(r), signature converted to (-+++)
            -(1 - 2 * mass(r) * r / sigma),
            -2 * a * mass(r) * r * sin2 / sigma,
            (r**2 + a**2 + 2 * a**2 * mass(r) * r * sin2 / sigma) * sin2,
            sigma / delta,
            sigma,
        )
        got = [
            float(getattr(metric, n)(r, th, a=a, **{name: value}))
            for n in ("g_tt", "g_tphi", "g_phiphi", "g_rr", "g_thth")
        ]
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    # Limits: static seed at a = 0, Kerr at zero regularisation parameter.
    assert float(bp.orbits.isco(metric, a=0.0, **{name: value})) == pytest.approx(
        float(bp.orbits.isco(static, **{name: value})), rel=1e-10
    )
    assert float(bp.orbits.isco(metric, a=0.7)) == pytest.approx(
        float(bp.orbits.isco(kerr, a=0.7)), rel=1e-12
    )
    # Horizon: outermost root of Delta = r^2 - 2 m(r) r + a^2.
    from scipy.optimize import brentq

    root = brentq(lambda r: r * r - 2 * mass(r) * r + a * a, 1.2, 2.0)
    assert float(bp.geometry.event_horizon(metric, a=a, **{name: value})) == pytest.approx(
        root, rel=1e-10
    )
    assert np.isfinite(float(bp.light.shadow_radius(metric, inclination=1.0, a=a, **{name: value})))
