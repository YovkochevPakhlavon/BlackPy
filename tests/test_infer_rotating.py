"""Inference with rotating metrics: RPM triplets, shadow size, new guards (B5)."""

import math
import warnings

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy.errors import (
    BlackPyWarning,
    DegeneracyWarning,
    IgnoredDataWarning,
    NodalPrecessionUnavailableError,
    NotSeparableError,
    ReflectionSymmetryError,
    SpinIgnoredWarning,
    UnverifiedDataWarning,
)
from blackpy.infer import (
    Fit,
    FitResult,
    Gaussian,
    Isotropic,
    Uniform,
    build_problem,
    prior_from_dict,
    simulate_qpo_dataset,
)
from blackpy.metric import StationaryAxisymmetric

TRUTH = dict(a=0.3, mass=6.0, r=5.8)
SIGMAS = dict(sigma_nu=(2.0, 3.0), sigma_low=0.1, mass_sigma=0.3)


def kerr_triplet(kerr, rng=None, name="src", **overrides):
    truth = {**TRUTH, **overrides}
    return simulate_qpo_dataset(
        kerr, "RP", name=name, mass=truth["mass"], r=truth["r"], rng=rng, a=truth["a"], **SIGMAS
    )


def gauss(x, mu, sigma):
    return -0.5 * ((x - mu) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))


# ----------------------------------------------------------------------------- priors, data


def test_isotropic_prior():
    prior = Isotropic(0.0, 90.0)
    x = np.linspace(0.0, 90.0, 20001)
    trapezoid = getattr(np, "trapezoid", None) or np.trapz  # noqa: NPY201 (NumPy < 2.0)
    assert trapezoid(np.exp(prior.logpdf(x)), x) == pytest.approx(1.0, rel=1e-8)
    u = np.linspace(0.0, 1.0, 11)
    np.testing.assert_allclose(prior.cdf(prior.ppf(u)), u, atol=1e-12)
    assert float(prior.ppf(0.5)) == pytest.approx(60.0, rel=1e-12)  # cos(i) = 1/2
    assert prior.logpdf(95.0) == -np.inf and prior.logpdf(-1.0) == -np.inf
    assert prior_from_dict(prior.to_dict()) == prior
    narrow = Isotropic(10.0, 30.0)
    assert float(narrow.cdf(30.0)) == 1.0 and float(narrow.cdf(10.0)) == 0.0
    samples = prior.sample(np.random.default_rng(0), 20000)
    assert np.mean(np.cos(np.radians(samples))) == pytest.approx(0.5, abs=0.01)  # uniform in cos i
    with pytest.raises(ValueError, match="0 <= low < high <= 180"):
        Isotropic(50.0, 20.0)


def test_simulated_triplet(kerr, rn):
    data = kerr_triplet(kerr)
    assert data.data_type == "qpo_triplet" and data.is_triplet and data.spin is None
    nu_u, nu_l = bp.orbits.QPO_MODELS["RP"].frequencies(kerr, 5.8, 6.0, a=0.3)
    nu_low = bp.orbits.nodal_precession_frequency(kerr, 5.8, 6.0, a=0.3)
    assert (data.upper.value, data.lower.value) == (float(nu_u), float(nu_l))
    assert data.low.value == pytest.approx(float(nu_low), rel=1e-13) and data.low.err_plus == 0.1
    noisy = kerr_triplet(kerr, rng=np.random.default_rng(1))
    assert noisy.low.value != data.low.value
    with pytest.raises(NodalPrecessionUnavailableError):
        simulate_qpo_dataset(
            rn, "RP", name="x", mass=6.0, r=7.0, sigma_nu=2.0, mass_sigma=0.3, sigma_low=0.1
        )


def test_registry_triplets_carry_published_spin():
    """Schema only (no numerical value is tested): spin and published fit are documented."""
    for dataset_id in ("qpo/gro_j1655_40", "qpo/xte_j1859_226"):
        data = bp.data.load(dataset_id)
        assert data.is_triplet and data.spin is not None and data.spin.is_complete
        assert data.spin.independent is False and data.spin.source in data.references
        assert "RPM" in data.published_fit["mass"].note and data.spin.location
        assert set(data.published_fit) == {"mass", "radius"} and not data.verified_by_human
    assert bp.data.load("qpo/xte_j1550_564").spin is None


# ----------------------------------------------------------------------------- likelihood


def test_triplet_likelihood_matches_hand_computation(kerr):
    data = kerr_triplet(kerr)
    problem = build_problem(kerr, data, "RP", {"a": Uniform(0, 0.99), "r": Uniform(2, 20)})
    assert problem.param_names == ["a", "M_src", "r_src"]
    assert problem.description["fitted_low_frequency_qpos"] == ["synthetic/src"]
    assert problem.description["metric"]["spin_parameter"] == "a"
    assert problem.description["metric"]["class"] == "StationaryAxisymmetric"
    at_truth = float(problem.log_likelihood(np.array([0.3, 6.0, 5.8])))
    assert at_truth == pytest.approx(
        gauss(0, 0, 2.0) + gauss(0, 0, 3.0) + gauss(0, 0, 0.1), rel=1e-10
    )
    # Away from the truth, every term from the Kerr closed forms.
    a, mass, r = 0.25, 6.3, 6.1
    omega = 1 / (r**1.5 + a)
    nu = lambda w: float(bp.constants.omega_to_hz(w, mass))
    nu_phi = nu(omega)
    nu_per = nu_phi - nu(omega * math.sqrt(1 - 6 / r + 8 * a / r**1.5 - 3 * a**2 / r**2))
    nu_nod = nu_phi - nu(omega * math.sqrt(1 - 4 * a / r**1.5 + 3 * a**2 / r**2))
    expected = (
        gauss(nu_phi, data.upper.value, 2.0)
        + gauss(nu_per, data.lower.value, 3.0)
        + gauss(nu_nod, data.low.value, 0.1)
    )
    assert float(problem.log_likelihood(np.array([a, mass, r]))) == pytest.approx(
        expected, rel=1e-9
    )
    # Without the nodal QPO: two terms and a warning.
    with pytest.warns(IgnoredDataWarning, match="use_low_frequency=False"):
        twin = build_problem(kerr, data, "RP", {"a": Uniform(0, 0.99)}, use_low_frequency=False)
    assert float(twin.log_likelihood(np.array([0.3, 6.0, 5.8]))) == pytest.approx(
        gauss(0, 0, 2.0) + gauss(0, 0, 3.0), rel=1e-10
    )
    # Retrograde orbits of spin a are prograde orbits of spin -a.
    retro = build_problem(kerr, data, "RP", {"a": Uniform(-0.99, 0.99)}, direction="retrograde")
    assert float(retro.log_likelihood(np.array([-0.3, 6.0, 5.8]))) == pytest.approx(
        at_truth, rel=1e-9
    )
    assert retro.description["direction"] == "retrograde"


def test_spin_is_bounded_by_the_black_hole_domain(kerr, kerr_newman, johannsen):
    data = kerr_triplet(kerr, r=9.0)
    problem = build_problem(kerr, data, "RP", {"a": Uniform(-2, 2)})
    assert problem.description["require_black_hole"] is True
    values = [
        float(problem.log_likelihood(np.array([a, 6.0, 9.0]))) for a in (0.999, 1.001, -1.001)
    ]
    assert np.isfinite(values[0]) and values[1] == -np.inf and values[2] == -np.inf
    # Naked singularities can be allowed on request.
    relaxed = build_problem(kerr, data, "RP", {"a": Uniform(-2, 2)}, require_black_hole=False)
    assert np.isfinite(float(relaxed.log_likelihood(np.array([1.001, 6.0, 9.0]))))
    # The bound depends on the metric: a^2 + Q^2 <= 1 for Kerr-Newman ...
    charged = build_problem(kerr_newman, data, "RP", {"a": Uniform(0, 1), "Q": Uniform(0, 1)})
    assert np.isfinite(float(charged.log_likelihood(np.array([0.6, 0.79, 6.0, 9.0]))))
    assert float(charged.log_likelihood(np.array([0.6, 0.81, 6.0, 9.0]))) == -np.inf
    # ... and Johannsen's regularity bound alpha13 > -(1 + sqrt(1 - a^2))^3 (= -5.832 at a = 0.6).
    deformed = build_problem(johannsen, data, "RP", {"a": Uniform(0, 1), "alpha13": Uniform(-8, 8)})
    assert np.isfinite(float(deformed.log_likelihood(np.array([0.6, -5.8, 6.0, 9.0]))))
    assert float(deformed.log_likelihood(np.array([0.6, -5.9, 6.0, 9.0]))) == -np.inf


def test_static_metric_for_a_rotating_source_warns(kerr, rn):
    data = kerr_triplet(kerr)
    with pytest.warns(SpinIgnoredWarning, match="rotating_from_static") as record:
        problem = build_problem(rn, data, "RP", {"Q": Uniform(0, 1)})
    assert any(issubclass(w.category, IgnoredDataWarning) for w in record)
    assert problem.description["static_metric_for_rotating_sources"] == ["synthetic/src"]
    assert problem.description["fitted_low_frequency_qpos"] == []
    # A published spin alone (twin peaks only) triggers the warning as well.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        with pytest.warns(SpinIgnoredWarning, match="qpo/gro_j1655_40"):
            build_problem(rn, "qpo/gro_j1655_40", "RP", {"Q": Uniform(0, 1)})
    # No warning for a rotating metric, nor for a static source.
    with warnings.catch_warnings():
        warnings.simplefilter("error", BlackPyWarning)
        build_problem(kerr, data, "RP", {"a": Uniform(0, 0.99)})
        pair = simulate_qpo_dataset(
            rn, "RP", name="pair", mass=6.0, r=7.0, sigma_nu=2.0, mass_sigma=0.3
        )
        build_problem(rn, pair, "RP", {"Q": Uniform(0, 1)})
    with pytest.raises(NodalPrecessionUnavailableError, match="needs a rotating metric"):
        build_problem(rn, data, "RP", {"Q": Uniform(0, 1)}, use_low_frequency=True)


@pytest.mark.slow
def test_rotating_shadow_likelihood(kerr, schwarzschild, rn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        problem = build_problem(kerr, "eht/sgra", priors={"a": Uniform(0, 0.998)})
        fixed = build_problem(
            kerr, "eht/sgra", priors={"a": Uniform(0, 0.998)}, fixed={"inclination_deg": 30.0}
        )
        by_area = build_problem(
            kerr,
            "eht/sgra",
            priors={"a": Uniform(0, 0.998)},
            shadow_size="area",
            fixed={"inclination_deg": 30.0},
        )
        mean, sigma = bp.data.load("eht/sgra").gaussian()
    # The inclination (degrees) is a nuisance parameter with an isotropic default prior.
    assert problem.param_names == ["a", "inclination_deg"] and problem.kinds == [
        "metric",
        "nuisance",
    ]
    assert problem.priors[1] == Isotropic(0.0, 90.0)
    assert problem.description["inclination_deg"] == "sampled"
    assert problem.description["shadow_size"] == "mean radius"
    obs = bp.light.shadow_observables(kerr, math.radians(30.0), n=128, a=0.9)
    expected = gauss(float(obs.mean_radius), mean, sigma)
    assert float(problem.log_likelihood(np.array([0.9, 30.0]))) == pytest.approx(expected, rel=1e-9)
    assert fixed.param_names == ["a"] and fixed.description["inclination_deg"] == 30.0
    assert float(fixed.log_likelihood(np.array([0.9]))) == pytest.approx(expected, rel=1e-9)
    assert float(by_area.log_likelihood(np.array([0.9]))) == pytest.approx(
        gauss(float(obs.area_radius), mean, sigma), rel=1e-9
    )
    # a = 0 reproduces the static likelihood; over-extremal spins are excluded.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        static = build_problem(rn, "eht/sgra", priors={"Q": Uniform(0, 1)})
        with pytest.raises(ValueError, match="circle for every observer"):
            build_problem(
                rn, "eht/sgra", priors={"Q": Uniform(0, 1), "inclination_deg": Uniform(0, 90)}
            )
        with pytest.raises(ValueError, match="both a prior and a fixed value"):
            build_problem(
                kerr,
                "eht/sgra",
                priors={"a": Uniform(0, 1), "inclination_deg": Uniform(0, 90)},
                fixed={"inclination_deg": 30.0},
            )
        with pytest.raises(ValueError, match="shadow_size"):
            build_problem(kerr, "eht/sgra", priors={"a": Uniform(0, 1)}, shadow_size="max")
    assert float(problem.log_likelihood(np.array([0.0, 50.0]))) == pytest.approx(
        float(static.log_likelihood(np.array([0.0]))), rel=1e-12
    )
    assert float(problem.log_likelihood(np.array([1.05, 50.0]))) == -np.inf
    # QPO-only problems have no inclination.
    with pytest.raises(ValueError, match="only a parameter when"):
        build_problem(
            kerr, kerr_triplet(kerr), "RP", {"a": Uniform(0, 1), "inclination_deg": Uniform(0, 90)}
        )


def test_unsuitable_rotating_metrics_are_refused(kerr):
    s = kerr.symbolic
    data = kerr_triplet(kerr)
    skewed = StationaryAxisymmetric.from_sympy(
        s.g_tt * (1 + sp.Rational(1, 10) * sp.cos(s.theta) / s.r**2),
        s.g_tphi,
        s.g_phiphi,
        s.g_rr,
        s.g_thth,
        r=s.r,
        theta=s.theta,
        parameters={"a": 0.3},
    )
    with pytest.raises(ReflectionSymmetryError):
        build_problem(skewed, data, "RP", {"a": Uniform(0, 0.9)})
    non_separable = StationaryAxisymmetric.from_sympy(
        s.g_tt * (1 + sp.cos(s.theta) ** 2 / (5 * s.r**3)),
        s.g_tphi,
        s.g_phiphi,
        s.g_rr,
        s.g_thth,
        r=s.r,
        theta=s.theta,
        parameters={"a": 0.3},
        asymptotically_flat=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        with pytest.raises(NotSeparableError, match=r"Gradus\.jl"):
            build_problem(non_separable, "eht/sgra", priors={"a": Uniform(0, 0.9)})
    # The same metric can still be fitted to QPOs: orbits do not need separability.
    assert build_problem(non_separable, data, "RP", {"a": Uniform(0, 0.9)}).ndim == 3


# ----------------------------------------------------------------------------- sampling


@pytest.fixture(scope="module")
def kerr_fits(kerr):
    data = kerr_triplet(kerr)
    priors = {"a": Uniform(0.0, 0.998), "M": Uniform(2.0, 20.0), "r": Uniform(2.0, 20.0)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        emcee_fit = Fit(kerr, data, "RP", priors, sampler="emcee", nsteps=4000, max_steps=16000)
        emcee_fit.run(seed=11)
        dynesty_fit = Fit(kerr, data, "RP", priors, sampler="dynesty", nlive=300)
        dynesty_fit.run(seed=12)
    return data, emcee_fit, dynesty_fit


@pytest.mark.slow
def test_samplers_recover_spin_mass_and_radius_from_a_triplet(kerr_fits):
    """RPM triplet likelihood with a flat mass prior: three frequencies fix (a, M, r)."""
    data, emcee_fit, dynesty_fit = kerr_fits
    truth = {"a": TRUTH["a"], "M_src": TRUTH["mass"], "r_src": TRUTH["r"]}
    analytic = bp.orbits.rpm_kerr_solution(data.low.value, data.lower.value, data.upper.value)
    assert float(analytic.spin) == pytest.approx(TRUTH["a"], rel=1e-8)
    summaries = []
    for fit in (emcee_fit, dynesty_fit):
        result = fit.result
        summary = result.summary()
        summaries.append(summary)
        assert result.diagnostics["converged"]
        for name, value in truth.items():
            assert abs(summary[name]["median"] - value) < 0.35 * summary[name]["std"], name
            assert not result.diagnostics["prior_dominated"][name]
        assert summary["a"]["std"] < 0.01 and summary["M_src"]["std"] < 0.3
        # One metric parameter only: no spin-deformation correlation to report.
        assert result.diagnostics["spin_correlations"] == {}
    for name in truth:
        ratio = summaries[0][name]["std"] / summaries[1][name]["std"]
        assert 0.75 < ratio < 1.35, name


@pytest.mark.slow
def test_spin_deformation_degeneracy_is_reported(kerr, johannsen, tmp_path):
    """A triplet cannot separate the spin from a Johannsen deformation parameter."""
    data = kerr_triplet(kerr)
    priors = {"a": Uniform(0.0, 0.99), "alpha13": Uniform(-2.0, 2.0), "r": Uniform(3.0, 12.0)}
    fit = Fit(johannsen, data, "RP", priors, sampler="emcee", nsteps=6000, max_steps=48000)
    with pytest.warns(DegeneracyWarning, match="strongly degenerate") as record:
        result = fit.run(seed=5)
    assert not any(issubclass(w.category, bp.errors.ConvergenceWarning) for w in record)
    correlation = result.diagnostics["spin_correlations"]["alpha13"]
    assert result.diagnostics["spin_parameter"] == "a"
    assert result.diagnostics["spin_degenerate"]["alpha13"]
    assert max(abs(correlation["pearson"]), abs(correlation["spearman"])) > 0.9
    assert result.correlation("a", "alpha13") == pytest.approx(correlation["pearson"], rel=1e-12)
    assert result.correlation("a", "alpha13", "spearman") == pytest.approx(correlation["spearman"])
    assert "strongly degenerate" in str(result)
    # The truth (Kerr: alpha13 = 0, a = 0.3) lies inside the degenerate band.
    summary = result.summary(level=0.95)
    assert summary["alpha13"]["lower"] < 0.0 < summary["alpha13"]["upper"]
    assert summary["a"]["lower"] < 0.3 < summary["a"]["upper"]
    # The marginal spin uncertainty is much larger than in the Kerr fit (0.003).
    assert result.summary()["a"]["std"] > 0.01
    # Reproducibility bundle keeps the new diagnostics and options.
    loaded = FitResult.load(result.save(tmp_path / "johannsen.npz"))
    assert loaded.diagnostics["spin_correlations"] == result.diagnostics["spin_correlations"]
    assert loaded.config["fitted_low_frequency_qpos"] == ["synthetic/src"]
    assert loaded.config["metric"]["spin_parameter"] == "a"


@pytest.mark.slow
def test_shadow_fit_with_inclination_nuisance(kerr):
    """Sgr A*-like bound: the shadow size barely constrains the Kerr spin (prior dominated)."""
    data = bp.data.load("eht/sgra")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        fit = Fit(
            kerr,
            data,
            priors={"a": Uniform(0.0, 0.998)},
            sampler="emcee",
            nsteps=3000,
            max_steps=12000,
        )
        result = fit.run(seed=3)
    assert result.param_names == ["a", "inclination_deg"] and result.diagnostics["converged"]
    # The isotropic prior of the nuisance parameter is recovered (the data do not see it).
    assert np.mean(np.cos(np.radians(result["inclination_deg"]))) == pytest.approx(0.5, abs=0.06)
    assert result.priors["inclination_deg"] == Isotropic(0.0, 90.0)
    assert result.summary()["a"]["std"] > 0.2  # essentially the prior width (0.29)


def test_gaussian_spin_prior_and_fixed_spin(kerr):
    """Spin can be given an informative prior or be fixed, like any metric parameter."""
    data = kerr_triplet(kerr)
    fixed = build_problem(kerr, data, "RP", {"r": Uniform(2, 20)}, fixed={"a": 0.3})
    assert fixed.param_names == ["M_src", "r_src"] and fixed.fixed == {"a": 0.3}
    assert np.isfinite(float(fixed.log_likelihood(np.array([6.0, 5.8]))))
    informed = build_problem(kerr, data, "RP", {"a": Gaussian(0.3, 0.05, low=0.0, high=0.998)})
    assert informed.priors[0].to_dict()["type"] == "Gaussian"
    assert float(jnp.asarray(informed.log_likelihood(np.array([0.3, 6.0, 5.8])))) > -10
