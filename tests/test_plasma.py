"""Shadows in a cold, non-magnetised plasma (B2).

Formulas checked against Perlick, Tsupko & Bisnovatyi-Kogan (2015, arXiv:1507.04217;
"PTB") and Perlick & Tsupko (2017, arXiv:1702.08768; "PT").
"""

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import blackpy as bp
from blackpy.errors import (
    BlackPyWarning,
    DegeneracyWarning,
    NotAsymptoticallyFlatError,
    NotSeparableError,
    PlasmaPropagationError,
    PriorDominatedWarning,
)
from blackpy.infer import Fit, Uniform, build_problem, simulate_shadow_dataset

lt = bp.light


@pytest.fixture(scope="module")
def homogeneous():
    return lt.HomogeneousPlasma()


@pytest.fixture(scope="module")
def power_law():
    return lt.PowerLawPlasma()


# ----------------------------------------------------------------------------- vacuum limit


@pytest.mark.slow
def test_zero_plasma_reproduces_vacuum(schwarzschild, rn, bardeen, kerr, homogeneous, power_law):
    """omega_p -> 0 gives the vacuum results to 1e-12 (mandatory)."""
    for plasma in (homogeneous, power_law):
        for metric, params in ((schwarzschild, {}), (rn, {"Q": 0.7}), (bardeen, {"q": 0.5})):
            vacuum = float(lt.shadow_radius(metric, **params))
            assert float(lt.shadow_in_plasma(metric, plasma, **params)) == pytest.approx(
                vacuum, rel=1e-12
            )
            for r_obs in (8.0, 50.0):
                assert float(lt.shadow_in_plasma(metric, plasma, r_obs, **params)) == pytest.approx(
                    float(lt.shadow_radius(metric, r_obs, **params)), rel=1e-12
                )
                assert float(
                    lt.shadow_angular_radius_in_plasma(metric, plasma, r_obs, **params)
                ) == pytest.approx(
                    float(lt.shadow_angular_radius(metric, r_obs, **params)), rel=1e-12
                )
            assert float(lt.photon_sphere_in_plasma(metric, plasma, **params)) == pytest.approx(
                float(lt.photon_sphere(metric, **params)), rel=1e-12
            )


@pytest.mark.slow
def test_kerr_separable_plasma_reduces_to_kerr_in_vacuum(kerr, homogeneous):
    """The Kerr separable-plasma case reduces to Kerr in vacuum (mandatory)."""
    separable = lt.SeparablePlasma(
        lambda r, c=0.0: c * r**1.5, lambda theta, c=0.0: c * (1.0 + 2.0 * jnp.sin(theta) ** 2)
    )
    for plasma in (homogeneous, separable):
        view = lt.with_plasma(kerr, plasma)
        assert view.is_separable and view.param_names[0] == "a"
        for a, inc in ((0.5, 1.2), (0.95, math.pi / 2)):
            vacuum = lt.shadow_contour(kerr, inc, a=a, n=64)
            contour = lt.shadow_contour(view, inc, a=a, n=64)
            np.testing.assert_allclose(contour.alpha, vacuum.alpha, rtol=1e-12, atol=1e-12)
            # beta = sqrt(Theta) vanishes at the two ends of the contour, where the square
            # root amplifies round-off; beta^2 is the well-conditioned quantity.
            np.testing.assert_allclose(contour.beta**2, vacuum.beta**2, rtol=1e-12, atol=1e-12)
            finite = lt.shadow_observables(view, inc, 40.0, a=a)
            reference = lt.shadow_observables(kerr, inc, 40.0, a=a)
            assert float(finite.mean_radius) == pytest.approx(
                float(reference.mean_radius), rel=1e-12
            )
            assert float(lt.shadow_in_plasma(kerr, plasma, inclination=inc, a=a)) == pytest.approx(
                float(lt.shadow_radius(kerr, inclination=inc, a=a)), rel=1e-12
            )


# ----------------------------------------------------------------------------- Schwarzschild


def test_schwarzschild_homogeneous_plasma(schwarzschild, homogeneous):
    """Photon sphere from PTB Eq. (33) and shadow from PTB Eq. (46)."""
    for omega_p in (0.1, 0.3, 0.6, 0.9):
        x = omega_p**2
        # PTB Eq. (33) with constant omega_p: r (r - 3) / (r - 2)^2 = x, solved for r.
        r_ph = (3.0 - 4.0 * x + math.sqrt(9.0 - 8.0 * x)) / (2.0 * (1.0 - x))
        assert r_ph * (r_ph - 3.0) / (r_ph - 2.0) ** 2 == pytest.approx(x, rel=1e-12)
        assert float(
            lt.photon_sphere_in_plasma(schwarzschild, homogeneous, omega_p=omega_p)
        ) == pytest.approx(r_ph, rel=1e-11)
        (sphere,) = lt.photon_spheres_in_plasma(schwarzschild, homogeneous, omega_p=omega_p)
        assert not sphere.stable and sphere.radius == pytest.approx(r_ph, rel=1e-11)

        def h2(r, x=x):  # PTB Eq. (22)
            return r**2 * (r / (r - 2.0) - x)

        assert sphere.impact_parameter == pytest.approx(math.sqrt(h2(r_ph)), rel=1e-11)
        for r_obs in (6.0, 15.0, 400.0):
            expected = math.sqrt(h2(r_ph) / h2(r_obs))  # PTB Eq. (46)
            assert math.sin(
                float(
                    lt.shadow_angular_radius_in_plasma(
                        schwarzschild, homogeneous, r_obs, omega_p=omega_p
                    )
                )
            ) == pytest.approx(expected, rel=1e-11)
            assert float(
                lt.shadow_in_plasma(schwarzschild, homogeneous, r_obs, omega_p=omega_p)
            ) == pytest.approx(r_obs * expected, rel=1e-11)
        # Observer at infinity inside the plasma: b_c / n, the limit of r_O sin(alpha).
        at_infinity = float(lt.shadow_in_plasma(schwarzschild, homogeneous, omega_p=omega_p))
        assert at_infinity == pytest.approx(math.sqrt(h2(r_ph) / (1.0 - x)), rel=1e-11)
        far = float(lt.shadow_in_plasma(schwarzschild, homogeneous, 1e7, omega_p=omega_p))
        assert far == pytest.approx(at_infinity, rel=1e-6)
    # PT, text after Eq. (78): a homogeneous plasma enlarges the shadow for every observer.
    sizes = lt.shadow_in_plasma(schwarzschild, homogeneous, 10.0, omega_p=np.linspace(0, 0.8, 9))
    assert np.all(np.diff(np.asarray(sizes)) > 0.0)


@pytest.mark.slow
def test_schwarzschild_power_law_plasma(schwarzschild):
    """PTB Eqs. (57)-(60): exact k = 2 case and the low-density expansion."""
    # k = 2: the photon sphere stays at 3M exactly, so b_c^2 = h_pl(3)^2 = 27 - beta_0.
    plasma = lt.PowerLawPlasma(k=2.0)
    for beta0 in (0.5, 3.0, 12.0):
        omega_p = math.sqrt(beta0)
        assert float(
            lt.photon_sphere_in_plasma(schwarzschild, plasma, omega_p=omega_p)
        ) == pytest.approx(3.0, rel=1e-11)
        assert float(lt.shadow_in_plasma(schwarzschild, plasma, omega_p=omega_p)) == pytest.approx(
            math.sqrt(27.0 - beta0), rel=1e-11
        )
    # General k: first-order formulas, error of second order in beta_0.
    for k in (1.0, 1.5, 3.0):
        plasma = lt.PowerLawPlasma(k=k)
        errors = []
        for beta0 in (1e-2, 1e-3):
            omega_p = math.sqrt(beta0)
            r_ph = float(lt.photon_sphere_in_plasma(schwarzschild, plasma, omega_p=omega_p))
            first_order = 3.0 + beta0 / 3.0 ** (k + 1.0) * (1.0 - 0.5 * k)  # PTB Eq. (58)
            errors.append(abs(r_ph - first_order))
            r_obs = 25.0
            sin2 = (
                27.0
                / r_obs**2
                * (1.0 - 2.0 / r_obs)
                * (1.0 - beta0 / 3.0 ** (k + 1.0) + (1.0 - 2.0 / r_obs) * beta0 / r_obs**k)
            )  # PTB Eq. (59)
            alpha = float(
                lt.shadow_angular_radius_in_plasma(schwarzschild, plasma, r_obs, omega_p=omega_p)
            )
            assert math.sin(alpha) ** 2 == pytest.approx(sin2, rel=3.0 * beta0**2 + 1e-12)
        assert errors[0] < 0.3 * 1e-4 and errors[1] < 0.02 * errors[0] + 1e-13  # ~ beta_0^2
    # PT Eq. (78) and Fig. 1: for k = 1 the plasma enlarges the shadow for 3 < r_O < 6 and
    # shrinks it for r_O > 6 (to first order in the density).
    plasma = lt.PowerLawPlasma(k=1.0)
    for r_obs, sign in ((4.5, +1.0), (5.5, +1.0), (6.5, -1.0), (30.0, -1.0)):
        change = float(
            lt.shadow_angular_radius_in_plasma(schwarzschild, plasma, r_obs, omega_p=0.02)
        ) - float(lt.shadow_angular_radius(schwarzschild, r_obs))
        assert sign * change > 0.0, r_obs


def test_schwarzschild_theta_dependent_plasma(schwarzschild):
    """PT Eqs. (69), (70), (72): omega_p^2 = (f_r + f_theta) / r^2 on Schwarzschild."""
    c, d, k = 0.9, 0.6, 1.5
    plasma = lt.SeparablePlasma(
        lambda r, c=c, d=d: c * r**k,
        lambda theta, c=c, d=d: d * (1.0 + 2.0 * jnp.sin(theta) ** 2),
        parameters={"c": c, "d": d},
    )
    f_r = lambda r: c * r**k
    f_th = lambda theta: d * (1.0 + 2.0 * math.sin(theta) ** 2)
    view = lt.with_plasma(schwarzschild, plasma)
    assert view.param_names == ("c", "d")
    for theta_obs in (0.4, 1.1, math.pi / 2):
        for r_obs in (12.0, 60.0):
            contour = lt.shadow_contour(view, theta_obs, r_obs, n=32)
            r_p = float(contour.r_p[0])
            # PT Eq. (69): (r - 3) 2 = (1 - 2/r)^2 f_r'(r).
            assert 2.0 * (r_p - 3.0) == pytest.approx(
                (1.0 - 2.0 / r_p) ** 2 * c * k * r_p ** (k - 1.0), rel=1e-10
            )
            carter = r_p**3 / (r_p - 2.0) - f_r(r_p)  # PT Eq. (70)
            sin_theta = math.sqrt(
                (carter - f_th(theta_obs))
                / (r_obs**3 / (r_obs - 2.0) - f_r(r_obs) - f_th(theta_obs))
            )  # PT Eq. (71) with (70), i.e. Eq. (72)
            radius = np.hypot(contour.alpha, contour.beta)
            np.testing.assert_allclose(radius, r_obs * sin_theta, rtol=1e-10)  # a circle
            assert float(
                lt.shadow_in_plasma(schwarzschild, plasma, r_obs, inclination=theta_obs)
            ) == pytest.approx(r_obs * sin_theta, rel=1e-10)
    # The shadow is circular but its size depends on the polar angle of the observer.
    sizes = [
        float(lt.shadow_in_plasma(schwarzschild, plasma, 12.0, inclination=t)) for t in (0.4, 1.5)
    ]
    assert abs(sizes[0] - sizes[1]) > 0.02


# ----------------------------------------------------------------------------- Kerr


def test_kerr_spherical_light_rays_in_a_plasma(kerr):
    """Constants of motion of spherical light rays, PT Eqs. (39) and (40) (upper sign)."""
    c, k, d = 0.7, 1.5, 0.4
    plasma = lt.SeparablePlasma(
        lambda r, c=c, d=d: c * r**k,
        lambda theta, c=c, d=d: d * jnp.sin(theta) ** 2,
        parameters={"c": c, "d": d},
    )
    view = lt.with_plasma(kerr, plasma)
    for a in (0.3, 0.9):
        for r in (2.2, 3.0, 3.6):
            delta = r**2 - 2.0 * r + a**2
            root = math.sqrt(1.0 - c * k * r ** (k - 1.0) * (r - 1.0) / (2.0 * r**2))
            a_p_phi = (a**2 - r**2 + r * delta * root) / (r - 1.0)  # Eq. (39), omega_0 = m = 1
            carter = r**2 * delta / (r - 1.0) ** 2 * (1.0 + root) ** 2 - c * r**k  # Eq. (40)
            critical = lt.critical_impact_parameters(view, r, a=a)
            xi, eta = float(critical.xi), float(critical.eta)
            # PT use p_t = +omega_0, so xi = L/E = -p_phi / omega_0; their Carter constant
            # is K = eta + (xi - a)^2 + f_theta(pi/2) in BlackPy's conventions.
            assert a * xi == pytest.approx(-a_p_phi, rel=1e-10)
            assert eta + (xi - a) ** 2 + d == pytest.approx(carter, rel=1e-10)
    # Homogeneous plasma on Kerr is the separable case of PT Eq. (36).
    homogeneous = lt.HomogeneousPlasma()
    a, wc = 0.8, 0.35
    explicit = lt.SeparablePlasma(
        lambda r: wc**2 * r**2, lambda theta: wc**2 * a**2 * jnp.cos(theta) ** 2, parameters=[]
    )
    first = lt.shadow_observables(lt.with_plasma(kerr, homogeneous), 1.0, a=a, omega_p=wc)
    second = lt.shadow_observables(lt.with_plasma(kerr, explicit), 1.0, a=a)
    for x, y in zip(first, second, strict=True):
        assert float(x) == pytest.approx(float(y), rel=1e-11, abs=1e-12)
    vacuum = lt.shadow_observables(kerr, 1.0, a=a)
    assert float(first.mean_radius) > float(vacuum.mean_radius)  # n < 1 at the observer


@pytest.mark.slow
def test_general_path_reproduces_the_static_formulas(rn, hayward, homogeneous, power_law):
    """Static metrics sent through the rotating (separable) code with the plasma hooks."""
    cases = (
        (rn, {"Q": 0.6}, power_law, {"omega_p": 0.9, "k": 1.0}),
        (rn, {"Q": 0.3}, homogeneous, {"omega_p": 0.5}),
        (hayward, {"l": 0.5}, power_law, {"omega_p": 1.4, "k": 2.5}),
    )
    for metric, params, plasma, plasma_params in cases:
        general = lt.with_plasma(metric, plasma).as_axisymmetric()
        assert general.is_rotating_class and general.plasma is plasma
        for r_obs in (None, 9.0, 70.0):
            static = float(lt.shadow_in_plasma(metric, plasma, r_obs, **params, **plasma_params))
            for inc in (0.5, 1.4):
                observables = lt.shadow_observables(general, inc, r_obs, **params, **plasma_params)
                assert float(observables.mean_radius) == pytest.approx(static, rel=1e-11)
                assert float(observables.delta_c) == 0.0


# ----------------------------------------------------------------------------- refusals


def test_refusals(schwarzschild, rn, sds, kerr, homogeneous, power_law):
    # A radial power law does not separate on a rotating metric (PT Eq. (26)).
    with pytest.raises(NotSeparableError, match=r"Eq\. \(26\)"):
        lt.shadow_in_plasma(kerr, power_law, inclination=1.0, a=0.5, omega_p=0.3)
    with pytest.raises(NotSeparableError):
        lt.with_plasma(kerr, lt.PlasmaProfile(lambda r, x0=0.2: x0 * jnp.exp(-r / 5.0)))
    # Opaque plasma: omega_p >= omega_0 at infinity, or a dense plasma further in.
    with pytest.raises(PlasmaPropagationError, match="cannot propagate"):
        lt.shadow_in_plasma(schwarzschild, homogeneous, omega_p=1.2)
    with pytest.raises(PlasmaPropagationError):
        lt.shadow_in_plasma(schwarzschild, lt.PowerLawPlasma(k=0.5), omega_p=3.0)
    with pytest.raises(PlasmaPropagationError):
        lt.shadow_in_plasma(kerr, homogeneous, inclination=1.0, a=0.5, omega_p=1.1)
    # Vectorised calls return numbers where light propagates.
    values = lt.shadow_in_plasma(schwarzschild, homogeneous, omega_p=np.array([0.0, 0.5]))
    assert values.shape == (2,) and np.all(np.isfinite(np.asarray(values)))
    # The vacuum functions never silently ignore a plasma.
    view = lt.with_plasma(rn, power_law)
    assert lt.with_plasma(rn, power_law) is view  # cached
    with pytest.raises(TypeError, match="carries a plasma"):
        lt.photon_sphere(view, Q=0.2)
    with pytest.raises(TypeError, match="carries a plasma"):
        lt.critical_impact_parameter(view, Q=0.2)
    assert float(lt.shadow_radius(view, Q=0.2, omega_p=0.7)) == pytest.approx(
        float(lt.shadow_in_plasma(rn, power_law, Q=0.2, omega_p=0.7)), rel=1e-14
    )
    assert float(lt.shadow_radius(view, Q=0.2, omega_p=0.7)) < float(lt.shadow_radius(rn, Q=0.2))
    # Misuse.
    with pytest.raises(ValueError, match="same name"):
        lt.with_plasma(rn, lt.PlasmaProfile(lambda r, Q=0.1: Q / r))
    with pytest.raises(TypeError, match="PlasmaModel"):
        lt.with_plasma(rn, "plasma")
    with pytest.raises(TypeError, match="already carries"):
        lt.with_plasma(view, power_law)
    with pytest.raises(TypeError):
        lt.PlasmaModel()
    with pytest.raises(NotAsymptoticallyFlatError):
        lt.shadow_in_plasma(sds, power_law, Lambda=0.01, omega_p=0.2)
    with pytest.raises(TypeError, match="inclination"):
        lt.shadow_in_plasma(kerr, homogeneous, a=0.5, omega_p=0.2)
    with pytest.raises(TypeError, match="static metrics"):
        lt.photon_spheres_in_plasma(kerr, homogeneous)
    with pytest.raises(ValueError, match="observer"):
        lt.shadow_in_plasma(schwarzschild, power_law, 2.5, omega_p=0.2)  # inside r_ph


def test_user_profile(schwarzschild):
    plasma = lt.PlasmaProfile(lambda r, x0=0.3, scale=6.0: x0 * jnp.exp(-r / scale))
    assert plasma.param_names == ("x0", "scale")
    size = float(lt.shadow_in_plasma(schwarzschild, plasma))
    (sphere,) = lt.photon_spheres_in_plasma(schwarzschild, plasma)
    x = 0.3 * math.exp(-sphere.radius / 6.0)
    assert size == pytest.approx(
        math.sqrt(sphere.radius**2 * (sphere.radius / (sphere.radius - 2.0) - x)), rel=1e-11
    )
    assert size < 3.0 * math.sqrt(3.0)
    sizes = lt.shadow_in_plasma(schwarzschild, plasma, x0=np.array([0.0, 0.2, 0.4]))
    assert np.all(np.diff(np.asarray(sizes)) < 0.0)


# ----------------------------------------------------------------------------- inference


def test_shadow_likelihood_with_plasma(rn, kerr, power_law, homogeneous):
    view = lt.with_plasma(rn, power_law)
    data = simulate_shadow_dataset(view, name="img", sigma=0.2, Q=0.5, omega_p=0.8, k=1.0)
    centre = data.gaussian()[0]
    assert centre == pytest.approx(
        float(lt.shadow_in_plasma(rn, power_law, Q=0.5, omega_p=0.8, k=1.0)), rel=1e-14
    )
    problem = build_problem(
        rn,
        data,
        priors={"Q": Uniform(0.0, 0.95), "omega_p": Uniform(0.0, 1.5)},
        fixed={"k": 1.0},
        plasma=power_law,
    )
    assert problem.param_names == ["Q", "omega_p"] and problem.kinds == ["metric", "plasma"]
    assert problem.description["plasma"]["model"] == "PowerLawPlasma"
    assert problem.fixed == {"k": 1.0}
    for charge, omega_p in ((0.5, 0.8), (0.2, 0.4), (0.9, 1.3)):
        model = float(lt.shadow_in_plasma(rn, power_law, Q=charge, omega_p=omega_p, k=1.0))
        expected = -0.5 * ((model - centre) / 0.2) ** 2 - math.log(0.2 * math.sqrt(2 * math.pi))
        assert float(problem.log_likelihood(np.array([charge, omega_p]))) == pytest.approx(
            expected, rel=1e-10
        )
    # Opaque plasma: zero likelihood, not an error.
    opaque = build_problem(rn, data, priors={"omega_p": Uniform(0.0, 3.0)}, plasma=homogeneous)
    assert float(opaque.log_likelihood(np.array([1.5]))) == -np.inf
    # Rotating metric with a separable plasma: inclination is still a nuisance parameter.
    kerr_data = simulate_shadow_dataset(
        lt.with_plasma(kerr, homogeneous),
        name="k",
        sigma=0.1,
        inclination_deg=40.0,
        a=0.6,
        omega_p=0.3,
    )
    rotating = build_problem(
        kerr,
        kerr_data,
        priors={"a": Uniform(0.0, 0.99), "omega_p": Uniform(0.0, 0.8)},
        plasma=homogeneous,
    )
    assert rotating.param_names == ["a", "omega_p", "inclination_deg"]
    truth = float(rotating.log_likelihood(np.array([0.6, 0.3, 40.0])))
    assert truth == pytest.approx(-math.log(0.1 * math.sqrt(2 * math.pi)), rel=1e-9)
    with pytest.raises(ValueError, match="no shadow dataset"):
        build_problem(
            rn,
            bp.infer.simulate_qpo_dataset(
                rn, "RP", name="s", mass=6.0, r=7.0, sigma_nu=2.0, mass_sigma=0.3
            ),
            "RP",
            {"r": Uniform(5.0, 9.0)},
            plasma=power_law,
        )
    with pytest.raises(NotSeparableError):
        build_problem(kerr, kerr_data, priors={"a": Uniform(0.0, 0.9)}, plasma=power_law)


@pytest.mark.slow
def test_plasma_parameter_is_covered_by_the_integrity_guards(rn):
    """Plasma and charge both shrink the shadow: degenerate, or prior dominated (mandatory)."""
    plasma = lt.PowerLawPlasma(k=1.0)
    view = lt.with_plasma(rn, plasma)
    priors = {"Q": Uniform(0.0, 0.95), "omega_p": Uniform(0.0, 1.5)}
    # A precise shadow size constrains one combination: strong degeneracy.
    tight = simulate_shadow_dataset(view, name="img", sigma=0.03, Q=0.5, omega_p=0.8)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = Fit(rn, tight, priors=priors, plasma=plasma, nsteps=3000, max_steps=12000).run(
            seed=1
        )
    assert any(issubclass(w.category, DegeneracyWarning) for w in caught)
    record = result.diagnostics["plasma_correlations"]["omega_p, Q"]
    assert record["degenerate"] and record["spearman"] < -0.9
    assert "strongly degenerate" in str(result)
    assert result.config["parameter_kinds"]["omega_p"] == "plasma"
    # An imprecise one constrains neither: both are flagged as prior dominated.
    loose = simulate_shadow_dataset(view, name="img", sigma=0.6, Q=0.5, omega_p=0.8)
    with pytest.warns(PriorDominatedWarning, match="omega_p"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DegeneracyWarning)
            result = Fit(rn, loose, priors=priors, plasma=plasma, nsteps=3000).run(seed=2)
    assert result.diagnostics["prior_dominated"]["omega_p"]
    assert result.summary()["omega_p"]["prior_dominated"]
    assert not isinstance(result, BlackPyWarning)
