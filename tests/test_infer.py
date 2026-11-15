"""Priors, likelihoods, samplers, Bayes factors, integrity guards (item 7)."""

import math
import warnings

import numpy as np
import pytest
from scipy import integrate

import blackpy as bp
from blackpy.errors import (
    BlackPyError,
    BlackPyWarning,
    CircularInferenceWarning,
    ConvergenceWarning,
    IgnoredDataWarning,
    NotAsymptoticallyFlatError,
    PriorDominatedWarning,
    UnverifiedDataWarning,
)
from blackpy.infer import (
    BayesFactor,
    Fit,
    FitResult,
    Gaussian,
    LogUniform,
    Uniform,
    build_problem,
    prior_from_dict,
    simulate_qpo_dataset,
)

Q_TRUE = 0.6
PRIORS = {"Q": Uniform(0.0, 1.0), "r": Uniform(4.0, 12.0)}


def synthetic_sources(rn, q, noise_rng=None):
    """Two sources with noise-free ("Asimov") data unless a generator is given."""
    radii = (5.6, 6.2) if q > 0.3 else (7.0, 7.5)
    a = simulate_qpo_dataset(
        rn,
        "RP",
        name="src_a",
        mass=6.0,
        r=radii[0],
        sigma_nu=2.0,
        mass_sigma=0.1,
        rng=noise_rng,
        Q=q,
    )
    b = simulate_qpo_dataset(
        rn,
        "RP",
        name="src_b",
        mass=10.0,
        r=radii[1],
        sigma_nu=1.5,
        mass_sigma=0.15,
        rng=noise_rng,
        Q=q,
    )
    return [a, b]


# ----------------------------------------------------------------------------- priors


@pytest.mark.parametrize(
    "prior",
    [
        Uniform(-1.0, 3.0),
        LogUniform(0.1, 50.0),
        Gaussian(2.0, 0.7),
        Gaussian(1.0, 2.0, low=0.0),
        Gaussian(0.0, 1.0, -0.5, 2.0),
    ],
)
def test_prior_consistency(prior):
    low, high = prior.support
    a, b = max(low, -30.0), min(high, 60.0)
    norm, _ = integrate.quad(lambda x: float(np.exp(prior.logpdf(x))), a, b, limit=200)
    assert math.isclose(norm, 1.0, rel_tol=1e-8)
    u = np.linspace(0.01, 0.99, 25)
    np.testing.assert_allclose(prior.cdf(prior.ppf(u)), u, rtol=1e-10, atol=1e-12)
    x0 = float(prior.ppf(0.3))
    partial, _ = integrate.quad(lambda x: float(np.exp(prior.logpdf(x))), a, x0, limit=200)
    assert math.isclose(partial, 0.3, rel_tol=1e-7)
    if math.isfinite(low):
        assert prior.logpdf(low - 1e-3) == -np.inf and prior.is_on_boundary(low)
    assert prior_from_dict(prior.to_dict()) == prior
    draws = prior.sample(np.random.default_rng(0), 2000)
    assert np.all(np.isfinite(prior.logpdf(draws)))


def test_invalid_priors_are_rejected():
    for bad in (
        lambda: Uniform(1.0, 1.0),
        lambda: LogUniform(0.0, 1.0),
        lambda: Gaussian(0.0, -1.0),
        lambda: Gaussian(0.0, 1.0, 50.0, 60.0),
        lambda: prior_from_dict({"type": "Cauchy"}),
    ):
        with pytest.raises(ValueError):
            bad()


# ----------------------------------------------------------------------------- likelihood


def test_likelihood_peaks_at_truth_and_matches_hand_computation(rn):
    data = synthetic_sources(rn, Q_TRUE)
    fit = Fit(rn, data, "RP", PRIORS)
    assert fit.param_names == ["Q", "M_src_a", "r_src_a", "M_src_b", "r_src_b"]
    truth = dict(Q=Q_TRUE, M_src_a=6.0, r_src_a=5.6, M_src_b=10.0, r_src_b=6.2)
    # At the truth all residuals vanish: only the Gaussian normalisations remain.
    expected = -2 * math.log(2.0 * math.sqrt(2 * math.pi)) - 2 * math.log(
        1.5 * math.sqrt(2 * math.pi)
    )
    assert math.isclose(fit.log_likelihood(**truth), expected, rel_tol=1e-9)
    assert fit.log_likelihood(**{**truth, "Q": 0.3}) < expected
    # One-sigma shift of one frequency lowers log L by 1/2 (other frequency refitted by hand).
    nu_u, nu_l = bp.orbits.QPO_MODELS["RP"].frequencies(rn, 5.6, 6.0 * 1.001, Q=Q_TRUE)
    shifted = fit.log_likelihood(**{**truth, "M_src_a": 6.0 * 1.001})
    chi2 = ((float(nu_u) - data[0].upper.value) / 2.0) ** 2 + (
        (float(nu_l) - data[0].lower.value) / 2.0
    ) ** 2
    assert math.isclose(shifted, expected - 0.5 * chi2, rel_tol=1e-9)
    # Inside the ISCO the model is undefined: the likelihood is zero, not an error.
    assert fit.log_likelihood(**{**truth, "r_src_a": 4.5}) == -np.inf
    with pytest.raises(TypeError, match="Missing parameters"):
        fit.log_likelihood(Q=0.5)


def test_asymmetric_errors_use_split_normal(rn):
    raw = {
        "schema_version": 1,
        "id": "synthetic/asym",
        "object": "asym",
        "data_type": "qpo_pair",
        "dataset_version": "t",
        "verified_by_human": True,
        "frequencies": {
            "upper": {"value": 300.0, "err_minus": 2.0, "err_plus": 6.0, "unit": "Hz"},
            "lower": {"value": 200.0, "err_minus": 3.0, "err_plus": 3.0, "unit": "Hz"},
        },
        "mass_prior": {
            "value": 8.0,
            "err_minus": 0.5,
            "err_plus": 1.5,
            "unit": "Msun",
            "independent": True,
        },
        "references": {"x": {"citation": "test"}},
    }
    data = bp.data.parse_dataset(raw)
    problem = build_problem(rn, data, "RP", {"Q": Uniform(0, 1)})
    assert problem.priors[1] == Gaussian(8.0, 1.0, low=0.0)  # symmetrised mass error
    model = bp.orbits.QPO_MODELS["RP"]

    def log_like(mass):
        return float(problem.log_likelihood(np.array([0.2, mass, 7.0])))

    def expected(mass):
        nu_u, nu_l = (float(v) for v in model.frequencies(rn, 7.0, mass, Q=0.2))
        sigma_u = 6.0 if nu_u >= 300.0 else 2.0
        return (
            math.log(math.sqrt(2 / math.pi) / 8.0)
            - 0.5 * ((nu_u - 300.0) / sigma_u) ** 2
            + math.log(math.sqrt(2 / math.pi) / 6.0)
            - 0.5 * ((nu_l - 200.0) / 3.0) ** 2
        )

    for mass in (7.0, 7.8, 8.6):
        assert math.isclose(log_like(mass), expected(mass), rel_tol=1e-10)


def test_shadow_likelihoods(rn, sds):
    sgra = bp.data.load("eht/sgra")
    mean, sigma = sgra.gaussian()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        gauss = Fit(rn, sgra, priors={"Q": Uniform(0, 1)})
        interval = Fit(
            rn,
            sgra,
            priors={"Q": Uniform(0, 1)},
            shadow_likelihood="interval",
            shadow_level="2sigma",
        )
        with pytest.raises(NotAsymptoticallyFlatError):
            Fit(sds, sgra, priors={"Lambda": Uniform(1e-4, 1e-3)})
    r_sh = float(bp.light.shadow_radius(rn, Q=0.4))
    expected = -0.5 * ((r_sh - mean) / sigma) ** 2 - math.log(sigma * math.sqrt(2 * math.pi))
    assert math.isclose(gauss.log_likelihood(Q=0.4), expected, rel_tol=1e-10)
    lo, hi = sgra.bounds["2sigma"]
    assert math.isclose(interval.log_likelihood(Q=0.4), -math.log(hi - lo), rel_tol=1e-12)
    # Extremal RN has r_sh = 4, below the 2-sigma lower bound: excluded.
    assert float(bp.light.shadow_radius(rn, Q=1.0)) < lo
    assert interval.log_likelihood(Q=1.0) == -np.inf


def test_problem_construction_errors(rn, schwarzschild):
    data = synthetic_sources(rn, Q_TRUE)
    with pytest.raises(ValueError, match="choose a QPO model"):
        Fit(rn, data, priors=PRIORS)
    with pytest.raises(ValueError, match="unknown parameters"):
        Fit(rn, data, "RP", {**PRIORS, "spin": Uniform(0, 1)})
    with pytest.raises(ValueError, match="both a prior and a fixed value"):
        Fit(rn, data, "RP", PRIORS, fixed={"Q": 0.1})
    with pytest.raises(ValueError, match="sampler must be"):
        Fit(rn, data, "RP", PRIORS, sampler="multinest")
    with pytest.raises(ValueError, match="unique"):
        Fit(rn, [data[0], data[0]], "RP", PRIORS)
    with pytest.raises(TypeError, match="Unknown emcee options"):
        Fit(rn, data, "RP", PRIORS, nlive=100).run(seed=0)
    # Fixed metric parameters are not sampled.
    fit = Fit(rn, data, "RP", {"r": Uniform(4, 12)}, fixed={"Q": 0.3})
    assert "Q" not in fit.param_names and fit.problem.fixed == {"Q": 0.3}
    assert Fit(schwarzschild, data, "RP", {"r": Uniform(6, 12)}).param_names[0] == "M_src_a"


# ----------------------------------------------------------------------------- guards


def test_integrity_guards_on_registry_data(rn):
    with pytest.warns(CircularInferenceWarning, match="circular"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnverifiedDataWarning)
            Fit(rn, "qpo/m82_x1", "RP", {"Q": Uniform(0, 1)})
    with pytest.warns(IgnoredDataWarning, match="nodal precession"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UnverifiedDataWarning)
            Fit(rn, "qpo/gro_j1655_40", "RP", {"Q": Uniform(0, 1)})
    with pytest.warns(UnverifiedDataWarning, match="verified_by_human"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", IgnoredDataWarning)
            Fit(rn, "qpo/gro_j1655_40", "RP", {"Q": Uniform(0, 1)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        # A null mass in the registry is an explicit error with the remedy...
        with pytest.raises(BlackPyError, match="no usable mass measurement"):
            Fit(rn, "qpo/h1743_322", "RP", {"Q": Uniform(0, 1)})
        # ...and an explicit user prior resolves it.
        fit = Fit(rn, "qpo/h1743_322", "RP", {"Q": Uniform(0, 1), "M_h1743_322": Uniform(5, 20)})
        assert fit.param_names == ["Q", "M_h1743_322", "r_h1743_322"]
    # A user-supplied mass prior replaces the circular registry prior: no warning.
    with warnings.catch_warnings():
        warnings.simplefilter("error", CircularInferenceWarning)
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        Fit(rn, "qpo/m82_x1", "RP", {"Q": Uniform(0, 1), "M": LogUniform(10, 5000)})


def test_prior_dominated_parameter_is_flagged(rn):
    """With a free mass and radius per source, one source cannot constrain Q
    if the mass prior is uninformative: the posterior of Q must be flagged."""
    data = simulate_qpo_dataset(
        rn, "RP", name="lonely", mass=8.0, r=7.0, sigma_nu=3.0, mass_sigma=0.5, Q=0.3
    )
    fit = Fit(
        rn,
        data,
        "RP",
        {"Q": Uniform(0.0, 0.6), "M": Uniform(2.0, 30.0), "r": Uniform(4.0, 20.0)},
        nwalkers=32,
        nsteps=6000,
    )
    with pytest.warns(PriorDominatedWarning, match="do not constrain"):
        result = fit.run(seed=3)
    assert result.diagnostics["prior_dominated"]["Q"]
    assert result.summary()["Q"]["prior_dominated"]
    assert "NOT a constraint" in str(result)
    assert r"\dagger" in bp.plot.latex_table(result)


def test_unconverged_chain_warns(rn):
    fit = Fit(rn, synthetic_sources(rn, Q_TRUE), "RP", PRIORS, nwalkers=32, nsteps=60)
    with pytest.warns(ConvergenceWarning, match="not converged"):
        result = fit.run(seed=0)
    assert result.diagnostics["converged"] is False


# ----------------------------------------------------------------------------- samplers


@pytest.fixture(scope="module")
def emcee_fit(rn):
    fit = Fit(
        rn, synthetic_sources(rn, Q_TRUE), "RP", PRIORS, sampler="emcee", nwalkers=32, nsteps=5000
    )
    fit.run(seed=1)
    return fit


@pytest.fixture(scope="module")
def dynesty_fit(rn):
    fit = Fit(rn, synthetic_sources(rn, Q_TRUE), "RP", PRIORS, sampler="dynesty", nlive=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        fit.run(seed=2)
    return fit


@pytest.mark.slow
@pytest.mark.parametrize("which", ["emcee_fit", "dynesty_fit"])
def test_samplers_recover_true_charge(which, request):
    """Validation item 7: both samplers recover the true RN charge within 2 sigma.

    The data are noise-free, so the posterior is centred on the truth and the
    test does not depend on a lucky noise realisation.
    """
    result = request.getfixturevalue(which).result
    truth = dict(Q=Q_TRUE, M_src_a=6.0, r_src_a=5.6, M_src_b=10.0, r_src_b=6.2)
    summary = result.summary()
    for name, value in truth.items():
        assert abs(summary[name]["mean"] - value) < 2.0 * summary[name]["std"], name
    assert summary["Q"]["std"] < 0.08  # an actual constraint, not the prior width (0.29)
    assert not result.diagnostics["prior_dominated"]["Q"]
    assert result.diagnostics["converged"]
    assert result.samples.shape[1] == 5 and result.samples.shape[0] > 1000
    if which == "emcee_fit":
        assert result.diagnostics["steps_over_tau"] > 50
        assert result.diagnostics["burn_in"] > 0 and result.diagnostics["thin"] >= 1
        assert result.log_evidence is None
    else:
        assert np.isfinite(result.log_evidence) and 0 < result.log_evidence_err < 1


@pytest.mark.slow
def test_samplers_agree(emcee_fit, dynesty_fit):
    a, b = emcee_fit.result.summary(), dynesty_fit.result.summary()
    for name in a:
        assert abs(a[name]["mean"] - b[name]["mean"]) < 0.5 * a[name]["std"], name
        assert 0.7 < a[name]["std"] / b[name]["std"] < 1.4, name


@pytest.mark.slow
def test_fit_quality_is_the_same_for_both_samplers(emcee_fit, dynesty_fit):
    """Goodness of fit and posterior predictive check from emcee and dynesty samples."""
    first, second = emcee_fit.result.goodness_of_fit(), dynesty_fit.result.goodness_of_fit()
    # 4 frequencies + 2 Gaussian mass priors - 5 parameters = 1 degree of freedom.
    assert first.dof == second.dof == 1 and first.labels == second.labels
    assert (first.n_data, first.n_gaussian_priors, first.n_parameters) == (4, 2, 5)
    # Noise-free data: the best fit reproduces them exactly, from either set of samples.
    assert first.chi2 < 1e-6 and second.chi2 < 1e-6
    for name in first.point:
        assert first.point[name] == pytest.approx(second.point[name], rel=1e-3), name
    assert first.point["Q"] == pytest.approx(Q_TRUE, abs=1e-3)
    a = emcee_fit.result.posterior_predictive(4000, seed=1)
    b = dynesty_fit.result.posterior_predictive(4000, seed=2)
    # Sampling error of each p-value is at most 0.5 / sqrt(4000) = 0.008; allow 5 sigma
    # of the difference plus the sampler-to-sampler difference of the posteriors.
    assert abs(a.p_value - b.p_value) < 0.06 and abs(a.p_value - a.p_value_simulated) < 0.04
    np.testing.assert_allclose(a.datum_p_values, b.datum_p_values, atol=0.06)
    assert not a.poor_fit and not b.poor_fit and 0.35 < a.p_value < 0.75


@pytest.mark.slow
def test_bayes_factor_prefers_true_model_dynesty(dynesty_fit):
    """Validation item 7: evidence ratio, null model fitted automatically."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        bf = dynesty_fit.bayes_factor(null={"Q": 0.0})
    assert isinstance(bf, BayesFactor) and bf.method == "evidence ratio"
    assert bf.log_bf > 5.0 and bf.log_bf_err < 1.0
    assert "full model" in bf.interpretation
    # The favoured (full) model passes the fit-quality check, and the result says so.
    assert bf.favoured_fit_is_poor is False and "fit quality of the favoured" in str(bf)
    with pytest.raises(ValueError, match="sampled metric parameters"):
        dynesty_fit.bayes_factor(null={"M_src_a": 6.0})


@pytest.mark.slow
def test_bayes_factor_prefers_true_model_savage_dickey(emcee_fit):
    """Validation item 7 for emcee: Savage-Dickey ratio, null on the prior boundary."""
    with pytest.warns(BlackPyWarning, match="prior boundary"):
        bf = emcee_fit.bayes_factor(null={"Q": 0.0})
    assert bf.method == "Savage-Dickey" and bf.log_bf > 3.0
    # The posterior has no support at Q = 0, so only a lower limit can be given: it is set
    # by the number of samples (the density one sample at the null value would contribute).
    assert bf.is_lower_limit and ">=" in str(bf)
    with pytest.raises(BlackPyError, match="nested sampling"):
        emcee_fit.bayes_factor(null={"Q": 0.0}, method="evidence")


@pytest.mark.slow
def test_bayes_factors_prefer_null_when_null_is_true(rn):
    """Data generated with Q = 0: neither method may favour the charged model,
    and the two estimates of ln B must agree."""
    data = synthetic_sources(rn, 0.0)
    emcee_fit = Fit(rn, data, "RP", PRIORS, nwalkers=32, nsteps=6000)
    emcee_fit.run(seed=5)
    with pytest.warns(BlackPyWarning, match="prior boundary"):
        sd = emcee_fit.bayes_factor(null={"Q": 0.0})
    nested = Fit(rn, data, "RP", PRIORS, sampler="dynesty", nlive=300)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        nested.run(seed=6)
        ev = nested.bayes_factor(null={"Q": 0.0})
    assert sd.log_bf < 0 and not sd.is_lower_limit
    assert ev.log_bf < 0.5
    assert abs(sd.log_bf - ev.log_bf) < 4 * ev.log_bf_err + 0.3


@pytest.mark.slow
def test_credible_interval_coverage_with_noisy_data(rn):
    """Frequentist coverage of the 68% credible interval over 100 noisy datasets.

    For each dataset the true charge, mass and radius are drawn from the
    priors used in the fit, and Gaussian noise is added to both frequencies
    and to the mass measurement. Under these conditions a correct posterior
    contains the truth in its central 68% interval in 68% of the cases, so
    the number of hits is binomial(100, 0.68): mean 68, standard deviation
    4.7. The test accepts three standard deviations and requires every chain
    to be converged. Observed with the seeds below: 73 / 100.
    """
    n_data, level = 100, 0.68
    radius_prior = (6.5, 10.0)  # outside the ISCO for every charge in [0, 1]
    rng = np.random.default_rng(20261115)
    hits, converged = 0, 0
    for i in range(n_data):
        charge, mass, radius = rng.uniform(0, 1), rng.uniform(6, 14), rng.uniform(*radius_prior)
        data = simulate_qpo_dataset(
            rn,
            "RP",
            name="src",
            mass=mass,
            r=radius,
            sigma_nu=2.0,
            mass_sigma=0.3,
            rng=rng,
            Q=charge,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", BlackPyWarning)
            fit = Fit(
                rn,
                [data],
                qpo_model="RP",
                priors={"Q": Uniform(0.0, 1.0), "r": Uniform(*radius_prior)},
                sampler="emcee",
                nwalkers=32,
                nsteps=3000,
                max_steps=12000,
            )
            result = fit.run(seed=1000 + i)
        low, high = np.percentile(result["Q"], [50 * (1 - level), 50 * (1 + level)])
        hits += int(low <= charge <= high)
        converged += int(result.diagnostics["converged"])
    print(f"68% credible-interval coverage: {hits} / {n_data}")
    assert converged == n_data
    sigma = math.sqrt(n_data * level * (1 - level))
    assert abs(hits - n_data * level) <= 3.0 * sigma, hits


def test_savage_dickey_against_analytic_density():
    """Gaussian posterior and uniform prior: B_01 = N(theta_0; mu, s) * (b - a)."""
    rng = np.random.default_rng(0)
    samples = rng.normal(0.25, 0.1, size=(40000, 1))
    samples = samples[(samples[:, 0] > 0) & (samples[:, 0] < 1)]
    result = FitResult(
        samples=samples,
        log_prob=np.zeros(len(samples)),
        param_names=["a"],
        param_labels=["a"],
        priors={"a": Uniform(0.0, 1.0)},
        sampler="emcee",
    )
    exact = lambda x: -(-0.5 * ((x - 0.25) / 0.1) ** 2 - math.log(0.1 * math.sqrt(2 * math.pi)))
    interior = result.savage_dickey({"a": 0.3})
    assert abs(interior.log_bf - exact(0.3)) < 0.05 and interior.log_bf_err < 0.05
    with pytest.warns(BlackPyWarning, match="prior boundary"):
        edge = result.savage_dickey({"a": 0.0})
    # Truncating the Gaussian at zero renormalises it by 1/0.9938.
    assert abs(edge.log_bf - (exact(0.0) + math.log(0.99379))) < 0.15
    with pytest.raises(ValueError, match="outside the prior"):
        result.savage_dickey({"a": 2.0})
    with pytest.raises(KeyError):
        result.savage_dickey({"b": 0.0})


@pytest.mark.slow
def test_result_is_reproducible_and_round_trips(rn, emcee_fit, tmp_path):
    result = emcee_fit.result
    assert result.metadata["seed"] == 1
    assert result.metadata["versions"]["blackpy"] == bp.__version__
    assert result.metadata["sampler_settings"]["nwalkers"] == 32
    assert result.config["qpo_model"] == "RP" and result.config["metric"]["name"] == rn.name
    assert [d["id"] for d in result.config["datasets"]] == ["synthetic/src_a", "synthetic/src_b"]
    assert result.config["priors"]["Q"] == {"type": "Uniform", "low": 0.0, "high": 1.0}
    path = result.save(tmp_path / "fit")
    loaded = FitResult.load(path)
    np.testing.assert_array_equal(loaded.samples, result.samples)
    assert loaded.param_names == result.param_names and loaded.priors == result.priors
    assert loaded.config == result.config and loaded.metadata["seed"] == 1
    assert loaded.summary() == result.summary()
    with pytest.warns(BlackPyWarning):
        assert loaded.savage_dickey({"Q": 0.0}).log_bf == result.savage_dickey({"Q": 0.0}).log_bf
    # Same seed, same settings: identical chain.
    again = Fit(
        rn, synthetic_sources(rn, Q_TRUE), "RP", PRIORS, sampler="emcee", nwalkers=32, nsteps=5000
    ).run(seed=1)
    np.testing.assert_array_equal(again.samples, result.samples)
    with pytest.raises(KeyError):
        _ = result["nope"]


def test_registry_dataset_versions_are_recorded(rn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        problem = build_problem(rn, ["qpo/gro_j1655_40", "eht/sgra"], "RP", {"Q": Uniform(0, 1)})
    recorded = {d["id"]: d for d in problem.description["datasets"]}
    assert recorded["qpo/gro_j1655_40"]["sha256"] == bp.data.load("qpo/gro_j1655_40").sha256
    assert recorded["eht/sgra"]["dataset_version"] == "1"
    assert problem.description["ignored_low_frequency_qpos"] == ["qpo/gro_j1655_40"]
