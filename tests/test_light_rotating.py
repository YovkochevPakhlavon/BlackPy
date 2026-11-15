"""Shadows of rotating black holes with separable null geodesics (B4)."""

import math
import time

import jax.numpy as jnp
import numpy as np
import pytest

import blackpy as bp
from blackpy import light as lt
from blackpy import orbits as ob
from blackpy.errors import NotAsymptoticallyFlatError, NotSeparableError
from blackpy.metric import StationaryAxisymmetric, rotating_from_static

B_SCHW = 3 * math.sqrt(3)


def bardeen_kerr(r_p, a, inclination):
    """Independent implementation of Bardeen's (1973) Kerr shadow formulas.

    Transcribed from Perlick & Tsupko (2022), Phys. Rep. 947, 1 (arXiv:2105.07101),
    Eqs. (55)-(56), who reproduce Bardeen's result (m = 1):
    lambda = [-r^2 (r - 3) - a^2 (r + 1)] / [a (r - 1)],
    eta = r^3 [4 a^2 - r (r - 3)^2] / [a^2 (r - 1)^2],
    alpha = -lambda / sin(i), beta^2 = eta + a^2 cos^2(i) - lambda^2 cot^2(i).
    """
    lam = (-(r_p**2) * (r_p - 3) - a**2 * (r_p + 1)) / (a * (r_p - 1))
    eta = r_p**3 * (4 * a**2 - r_p * (r_p - 3) ** 2) / (a**2 * (r_p - 1) ** 2)
    alpha = -lam / np.sin(inclination)
    beta2 = eta + a**2 * np.cos(inclination) ** 2 - lam**2 / np.tan(inclination) ** 2
    return lam, eta, alpha, beta2


def polygon_observables(alpha, beta):
    """Area radius, mean radius and Delta C straight from the contour points (numpy)."""
    x, y = np.append(alpha, alpha[0]), np.append(beta, beta[0])
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = 0.5 * abs(cross.sum())
    centre = ((x[:-1] + x[1:]) * cross).sum() / (6 * 0.5 * cross.sum())
    ell = np.hypot(x - centre, y)
    phi = np.unwrap(np.arctan2(y, x - centre))
    d_phi = np.abs(np.diff(phi))
    mean = (0.5 * (ell[:-1] + ell[1:]) * d_phi).sum() / (2 * math.pi)
    r_area = math.sqrt(area / math.pi)
    variance = (0.5 * ((ell[:-1] - r_area) ** 2 + (ell[1:] - r_area) ** 2) * d_phi).sum()
    return r_area, mean, math.sqrt(variance / (2 * math.pi)) / r_area, centre


# ----------------------------------------------------------------------------- required tests


def test_zero_spin_gives_the_circle_of_radius_3_sqrt_3(kerr, kerr_newman, johannsen, schwarzschild):
    for metric in (kerr, kerr_newman, johannsen, schwarzschild):
        for inclination in (0.0, 0.4, math.pi / 2):
            contour = lt.shadow_contour(metric, inclination)
            np.testing.assert_allclose(np.hypot(contour.alpha, contour.beta), B_SCHW, rtol=1e-13)
            obs = lt.shadow_observables(metric, inclination)
            assert float(obs.mean_radius) == pytest.approx(B_SCHW, rel=1e-13)
            assert float(obs.area_radius) == pytest.approx(B_SCHW, rel=1e-13)
            assert float(obs.delta_c) == 0.0 and float(obs.centroid) == 0.0
            assert float(obs.area) == pytest.approx(27 * math.pi, rel=1e-13)
    # Continuity through the slow-rotation regimes (centred circle, displaced circle, contour).
    spins = np.array([1e-9, 3e-7, 1e-6, 5e-6, 2e-5, 1e-4])
    obs = lt.shadow_observables(kerr, 1.0, a=spins)
    np.testing.assert_allclose(obs.mean_radius, B_SCHW, rtol=2e-8)
    np.testing.assert_allclose(obs.area_radius, B_SCHW, rtol=2e-9)
    np.testing.assert_allclose(obs.centroid, 2 * spins * math.sin(1.0), atol=1.2e-6)


@pytest.mark.parametrize("a", [0.05, 0.5, 0.9, 0.998, -0.7])
@pytest.mark.parametrize("inclination_deg", [5.0, 17.0, 60.0, 90.0])
def test_kerr_contour_agrees_with_bardeen(kerr, a, inclination_deg):
    inclination = math.radians(inclination_deg)
    contour = lt.shadow_contour(kerr, inclination, a=a, n=200)
    lam, eta, alpha, beta2 = bardeen_kerr(contour.r_p, a, inclination)
    np.testing.assert_allclose(contour.alpha, alpha, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(contour.beta**2, np.clip(beta2, 0, None), rtol=1e-9, atol=1e-9)
    critical = lt.critical_impact_parameters(kerr, contour.r_p[5:-5:20], a=a)
    np.testing.assert_allclose(critical.xi, lam[5:-5:20], rtol=1e-11)
    np.testing.assert_allclose(critical.eta, eta[5:-5:20], rtol=1e-10, atol=1e-11)
    # The contour is closed, symmetric about the alpha axis, and spans the photon region.
    assert contour.alpha.shape == (398,) and contour.inclination == inclination
    np.testing.assert_allclose(contour.beta[1:199], -contour.beta[:199:-1], rtol=1e-13)
    r_min, r_max = lt.photon_region(kerr, inclination, a=a)
    assert contour.r_p.min() == pytest.approx(float(r_min), rel=1e-12)
    assert contour.r_p.max() == pytest.approx(float(r_max), rel=1e-12)


def test_contour_flattens_with_spin_for_an_edge_on_observer(kerr):
    spins = np.array([0.0, 0.3, 0.6, 0.9, 0.99, 0.998])
    obs = lt.shadow_observables(kerr, math.pi / 2, a=spins)
    width = np.asarray(obs.alpha_max - obs.alpha_min)
    height = 2 * np.asarray(obs.beta_max)
    assert np.all(np.diff(width) < 0)  # the prograde side moves in: the shadow gets narrower
    assert width[0] == pytest.approx(2 * B_SCHW, rel=1e-12) and width[-1] < 0.88 * width[0]
    np.testing.assert_allclose(height, 2 * B_SCHW, rtol=1e-5)  # the vertical extent stays
    assert np.all(np.diff(np.asarray(obs.delta_c)) > 0) and float(obs.delta_c[-1]) > 0.04
    assert np.all(np.diff(np.asarray(obs.centroid)) > 0)  # displaced towards the retrograde side
    # Edge-on, the ends of the contour are the equatorial photon orbits.
    for a in (0.5, 0.95):
        r_min, r_max = lt.photon_region(kerr, a=a)
        assert float(r_min) == pytest.approx(float(ob.photon_orbit(kerr, a=a)), rel=1e-11)
        expected = float(ob.photon_orbit(kerr, direction="retrograde", a=a))
        assert float(r_max) == pytest.approx(expected, rel=1e-11)
    # Pole-on the shadow is a circle for any spin.
    polar = lt.shadow_observables(kerr, 0.0, a=0.95)
    assert float(polar.delta_c) < 1e-8 and abs(float(polar.centroid)) < 1e-3


def test_kerr_mean_radius_depends_weakly_on_spin_and_inclination(kerr):
    """Sgr A*: the shadow size is nearly independent of spin and inclination.

    EHT Collaboration (2022), Sgr A* Paper VI (arXiv:2311.09484), Sec. 1: "the radius
    ranges from ~4.8 GM/c^2 to ~5.2 GM/c^2"; the caption of their Fig. 12 calls it a
    "~8% range predicted for the Kerr metric". BlackPy (mean radius): 4.830 to 5.196,
    a 7.0% range, for 0 <= a <= 0.998 and all inclinations.
    """
    spins = np.linspace(0.0, 0.998, 41)[:, None]
    inclinations = np.radians(np.linspace(0.5, 90.0, 37))[None, :]
    obs = lt.shadow_observables(kerr, inclinations, a=spins)
    mean, area = np.asarray(obs.mean_radius), np.asarray(obs.area_radius)
    assert np.all(np.isfinite(mean))
    assert mean.max() == pytest.approx(B_SCHW, rel=1e-12)  # a = 0
    assert mean.min() == pytest.approx(4.830, abs=2e-3)  # a = 0.998, pole-on
    spread = (mean.max() - mean.min()) / mean.max()
    print(f"Kerr mean shadow radius: {mean.min():.4f} to {mean.max():.4f} ({100 * spread:.2f}%)")
    assert 0.06 < spread < 0.08
    assert 4.8 < mean.min() and mean.max() < 5.2
    # The two radius definitions agree to better than 0.2% everywhere.
    assert np.max(np.abs(area / mean - 1)) < 2e-3
    assert np.max(np.asarray(obs.delta_c)) < 0.05


# ----------------------------------------------------------------------------- observables


def test_observables_against_direct_polygon_computation(kerr, kerr_newman):
    for metric, pars, inclination in (
        (kerr, {"a": 0.95}, 1.2),
        (kerr_newman, {"a": 0.6, "Q": 0.5}, 0.7),
    ):
        obs = lt.shadow_observables(metric, inclination, **pars)
        contour = lt.shadow_contour(metric, inclination, n=4000, **pars)
        r_area, mean, delta_c, centre = polygon_observables(contour.alpha, contour.beta)
        assert float(obs.area_radius) == pytest.approx(r_area, rel=1e-6)
        assert float(obs.mean_radius) == pytest.approx(mean, rel=1e-6)
        assert float(obs.delta_c) == pytest.approx(delta_c, rel=1e-4)
        assert float(obs.centroid) == pytest.approx(centre, rel=1e-5)
        assert float(obs.alpha_min) == pytest.approx(contour.alpha.min(), rel=1e-9)
        assert float(obs.alpha_max) == pytest.approx(contour.alpha.max(), rel=1e-9)
        assert float(obs.beta_max) == pytest.approx(contour.beta.max(), rel=1e-4)
        # Identity: the rms radius of Bambi et al. (2019) is the area-equivalent radius,
        # hence Delta C = sqrt(2 (1 - <l> / R_A)).
        identity = math.sqrt(2 * (1 - float(obs.mean_radius) / float(obs.area_radius)))
        assert float(obs.delta_c) == pytest.approx(identity, rel=1e-7)
    # The quadrature converges exponentially: 64 points already give 1e-10.
    coarse = lt.shadow_observables(kerr, 1.2, n=64, a=0.95)
    fine = lt.shadow_observables(kerr, 1.2, n=512, a=0.95)
    assert float(coarse.mean_radius) == pytest.approx(float(fine.mean_radius), rel=1e-10)
    assert float(coarse.area_radius) == pytest.approx(float(fine.area_radius), rel=1e-10)


def test_shadow_radius_dispatches_on_the_metric_class(kerr, rn):
    obs = lt.shadow_observables(kerr, 1.0, a=0.8)
    assert float(lt.shadow_radius(kerr, inclination=1.0, a=0.8)) == float(obs.mean_radius)
    assert float(lt.shadow_radius(kerr, inclination=1.0, kind="area", a=0.8)) == float(
        obs.area_radius
    )
    with pytest.raises(TypeError, match="inclination"):
        lt.shadow_radius(kerr, a=0.8)
    with pytest.raises(ValueError, match="kind"):
        lt.shadow_radius(kerr, inclination=1.0, kind="max", a=0.8)
    for call in (
        lambda: lt.photon_sphere(kerr, a=0.5),
        lambda: lt.critical_impact_parameter(kerr, a=0.5),
        lambda: lt.photon_spheres(kerr, a=0.5),
        lambda: lt.shadow_angular_radius(kerr, 20.0, a=0.5),
    ):
        with pytest.raises(TypeError, match="static, spherically symmetric"):
            call()
    # Static metrics: unchanged v0.1 behaviour, and the same numbers from the rotating code.
    b_c = float(lt.shadow_radius(rn, Q=0.7))
    assert float(lt.shadow_observables(rn, 0.9, Q=0.7).mean_radius) == pytest.approx(b_c, rel=1e-13)


# ----------------------------------------------------------------------------- other metrics


def test_kerr_newman_and_generated_metrics(kerr, kerr_newman, rn, hayward):
    a, inclination = 0.6, 1.1
    reference = lt.shadow_observables(kerr, inclination, a=a)
    limit = lt.shadow_observables(kerr_newman, inclination, a=a, Q=0.0)
    for x, y in zip(limit, reference, strict=True):
        assert float(x) == pytest.approx(float(y), rel=1e-11, abs=1e-12)
    # a = 0: Reissner-Nordstrom circle.
    assert float(
        lt.shadow_observables(kerr_newman, inclination, Q=0.8).mean_radius
    ) == pytest.approx(float(lt.shadow_radius(rn, Q=0.8)), rel=1e-12)
    # The Azreg-Ainou rotating RN metric is Kerr-Newman.
    generated = lt.shadow_observables(rotating_from_static(rn), inclination, a=a, Q=0.5)
    direct = lt.shadow_observables(kerr_newman, inclination, a=a, Q=0.5)
    assert float(generated.mean_radius) == pytest.approx(float(direct.mean_radius), rel=1e-11)
    # Charge shrinks the shadow; a rotating regular black hole has a shadow too.
    assert float(direct.mean_radius) < float(reference.mean_radius)
    regular = lt.shadow_observables(rotating_from_static(hayward), inclination, a=a, l=0.5)
    assert 4.5 < float(regular.mean_radius) < float(reference.mean_radius)
    # A conformal factor does not change null geodesics, hence not the shadow.
    conformal = rotating_from_static(
        bp.metric.Schwarzschild(),
        psi=lambda r, theta, a: (r**2 + a**2 * jnp.cos(theta) ** 2) * (1 + 1 / r),
    )
    same = lt.shadow_observables(conformal, inclination, a=a)
    assert float(same.mean_radius) == pytest.approx(float(reference.mean_radius), rel=1e-10)


def test_johannsen_spherical_photon_orbits_solve_the_radial_equation(johannsen, kerr):
    """Independent check with the radial potential printed in Johannsen (2013), Eq. (23),
    for photons (mu = 0):  R = P^2 - Delta [(xi - a)^2 + eta],
    P = (r^2 + a^2) A_1 - a A_2 xi.  A spherical orbit has R = dR/dr = 0."""
    a, a13, a22 = 0.7, 0.8, -0.4
    pars = dict(a=a, alpha13=a13, alpha22=a22, alpha52=0.6, epsilon3=1.5)
    r_min, r_max = (float(v) for v in lt.photon_region(johannsen, **pars))
    r_p = np.linspace(r_min, r_max, 9)
    xi, eta = (np.asarray(v) for v in lt.critical_impact_parameters(johannsen, r_p, **pars))

    def radial(r):
        momentum = (r**2 + a**2) * (1 + a13 / r**3) - a * (1 + a22 / r**2) * xi
        return momentum**2 - (r**2 - 2 * r + a**2) * ((xi - a) ** 2 + eta)

    def radial_slope(r):
        a1, a2 = 1 + a13 / r**3, 1 + a22 / r**2
        momentum = (r**2 + a**2) * a1 - a * a2 * xi
        d_momentum = 2 * r * a1 - 3 * (r**2 + a**2) * a13 / r**4 + 2 * a * a22 * xi / r**3
        return 2 * momentum * d_momentum - (2 * r - 2) * ((xi - a) ** 2 + eta)

    scale = r_p**4
    np.testing.assert_allclose(radial(r_p) / scale, 0.0, atol=1e-11)
    np.testing.assert_allclose(radial_slope(r_p) / scale, 0.0, atol=1e-11)
    np.testing.assert_allclose(eta[[0, -1]], 0.0, atol=1e-9)  # equatorial at both ends
    # The shadow depends on alpha13 and alpha22 but not on alpha52 or epsilon3
    # (A_5 and the conformal factor do not enter the null geodesics' constants).
    base = float(lt.shadow_observables(johannsen, 1.0, a=a).mean_radius)
    assert base == pytest.approx(
        float(lt.shadow_observables(kerr, 1.0, a=a).mean_radius), rel=1e-11
    )
    unchanged = lt.shadow_observables(johannsen, 1.0, a=a, alpha52=0.9, epsilon3=2.0)
    assert float(unchanged.mean_radius) == pytest.approx(base, rel=1e-10)
    assert (
        abs(float(lt.shadow_observables(johannsen, 1.0, a=a, alpha13=1.0).mean_radius) - base) > 0.1
    )


# ----------------------------------------------------------------------------- finite distance


def test_observer_at_finite_distance(kerr, schwarzschild, rn, sds):
    # Static metric: the ZAMO is the static observer, so Synge's formula is recovered.
    for r_obs in (4.0, 10.0, 300.0):
        expected = float(lt.shadow_radius(schwarzschild, r_obs))
        got = lt.shadow_observables(schwarzschild, 1.0, r_obs)
        assert float(got.mean_radius) == pytest.approx(expected, rel=1e-12)
        assert expected == pytest.approx(B_SCHW * math.sqrt(1 - 2 / r_obs), rel=1e-12)
    assert float(lt.shadow_observables(rn, 0.5, 25.0, Q=0.6).mean_radius) == pytest.approx(
        float(lt.shadow_radius(rn, 25.0, Q=0.6)), rel=1e-12
    )
    # Kerr: convergence to Bardeen's result as r_obs -> infinity, like 1 / r_obs.
    at_infinity = lt.shadow_observables(kerr, 1.0, a=0.9)
    errors = []
    for r_obs in (1e3, 1e4, 1e6):
        near = lt.shadow_observables(kerr, 1.0, r_obs, a=0.9)
        errors.append(abs(float(near.mean_radius) - float(at_infinity.mean_radius)))
    assert errors[0] < 6e-3 and errors[1] < 6e-4 and errors[2] < 6e-6
    assert errors[0] / errors[1] == pytest.approx(10.0, rel=0.05)
    contour = lt.shadow_contour(kerr, 1.0, 1e9, a=0.9)
    reference = lt.shadow_contour(kerr, 1.0, a=0.9)
    np.testing.assert_allclose(contour.alpha, reference.alpha, atol=1e-7)
    assert contour.r_obs == 1e9 and reference.r_obs is None
    # Observer inside the photon region, and metrics without an observer at infinity.
    with pytest.raises(ValueError, match="outside the photon region"):
        lt.shadow_contour(kerr, 1.0, 2.5, a=0.9)
    with pytest.raises(NotAsymptoticallyFlatError, match="r_obs"):
        lt.shadow_observables(sds, 1.0, Lambda=1e-3)
    static = float(lt.shadow_radius(sds, 20.0, Lambda=1e-3))
    assert float(lt.shadow_observables(sds, 1.0, 20.0, Lambda=1e-3).mean_radius) == pytest.approx(
        static, rel=1e-11
    )


# ----------------------------------------------------------------------------- guards, performance


def test_non_separable_metric_is_refused(kerr):
    s = kerr.symbolic
    deformed = StationaryAxisymmetric.from_sympy(
        s.g_tt * (1 + s.g_thth.subs(s.r, 1) / (5 * s.r**3)),  # theta-dependent deformation of g_tt
        s.g_tphi,
        s.g_phiphi,
        s.g_rr,
        s.g_thth,
        r=s.r,
        theta=s.theta,
        parameters={"a": 0.5},
        name="non-separable test metric",
    )
    assert deformed.is_reflection_symmetric and not deformed.is_separable
    for call in (
        lambda: lt.shadow_contour(deformed, 1.0),
        lambda: lt.shadow_observables(deformed, 1.0),
        lambda: lt.critical_impact_parameters(deformed, 3.0),
        lambda: lt.photon_region(deformed),
        lambda: lt.shadow_radius(deformed, inclination=1.0),
    ):
        with pytest.raises(NotSeparableError, match=r"Gradus\.jl"):
            call()
    with pytest.raises(ValueError, match="scalar parameters"):
        lt.shadow_contour(kerr, 1.0, a=np.array([0.1, 0.2]))
    # A naked singularity has no closed shadow of this kind: reported, not silently wrong.
    with pytest.raises(ValueError, match="no shadow could be computed"):
        lt.shadow_contour(kerr, 1.0, a=1.2)


@pytest.mark.slow
def test_shadow_performance(kerr):
    """Benchmark: shadow observables for 10^4 spins, and one 256-point contour."""
    spins = np.linspace(0.0, 0.998, 10_000)
    np.asarray(lt.shadow_observables(kerr, 1.0, a=spins).mean_radius)  # compile
    best = math.inf
    for _ in range(5):
        start = time.perf_counter()
        np.asarray(lt.shadow_observables(kerr, 1.0, a=spins).mean_radius)
        best = min(best, time.perf_counter() - start)
    per_eval_ms = 1e3 * best / spins.size
    lt.shadow_contour(kerr, 1.0, a=0.9)
    start = time.perf_counter()
    for a in np.linspace(0.1, 0.9, 20):
        lt.shadow_contour(kerr, 1.0, a=float(a))
    contour_ms = 1e3 * (time.perf_counter() - start) / 20
    print(
        f"Kerr shadow observables: {per_eval_ms:.4f} ms; contour (510 points): {contour_ms:.3f} ms"
    )
    assert per_eval_ms < 0.5 and contour_ms < 50
