"""Goodness of fit and posterior predictive checks (B1)."""

import warnings

import numpy as np
import pytest

import blackpy as bp
from blackpy.errors import BlackPyWarning, PoorFitWarning
from blackpy.infer import (
    Fit,
    FitResult,
    Gaussian,
    Uniform,
    simulate_qpo_dataset,
    simulate_shadow_dataset,
)
from blackpy.infer.quality import UNDEFINED, draw_split_normal


def test_pulls_and_chi2_match_hand_computation(schwarzschild):
    data = simulate_qpo_dataset(
        schwarzschild, "RP", name="s", mass=8.0, r=7.0, sigma_nu=(2.0, 4.0), mass_sigma=0.5
    )
    fit = Fit(schwarzschild, data, "RP", {"r": Uniform(6.5, 12.0)})
    assert fit.param_names == ["M_s", "r_s"]  # the mass prior is Gaussian(8, 0.5)
    mass, radius = 8.6, 7.4
    freq = bp.orbits.frequencies_hz(schwarzschild, radius, mass)
    nu_upper, nu_lower = float(freq.nu_phi), float(freq.nu_phi - freq.nu_r)
    expected = np.array(
        [
            (nu_upper - data.upper.value) / 2.0,
            (nu_lower - data.lower.value) / 4.0,
            (mass - 8.0) / 0.5,
        ]
    )
    quality = fit.goodness_of_fit(M_s=mass, r_s=radius)
    assert quality.labels == ["nu_upper[s]", "nu_lower[s]", "prior[M_s]"]
    np.testing.assert_allclose(quality.pulls, expected, rtol=1e-10)
    assert quality.chi2 == pytest.approx(float(np.sum(expected**2)), rel=1e-10)
    # 2 data + 1 Gaussian prior - 2 parameters = 1 degree of freedom.
    assert (quality.n_data, quality.n_gaussian_priors, quality.n_parameters) == (2, 1, 2)
    assert quality.dof == 1 and quality.defined
    assert quality.p_value is None and not quality.is_best_fit  # not the minimum
    assert "given point" in str(quality)
    with pytest.raises(TypeError, match="sampled parameters"):
        fit.goodness_of_fit(M_s=mass)


def test_two_frequency_fit_has_no_degrees_of_freedom(rn, tmp_path):
    data = simulate_qpo_dataset(
        rn, "RP", name="s", mass=6.0, r=6.0, sigma_nu=2.0, mass_sigma=0.1, Q=0.5
    )
    fit = Fit(rn, data, "RP", {"Q": Uniform(0.0, 1.0), "r": Uniform(4.0, 12.0)}, nsteps=1500)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PoorFitWarning)  # must not fire: the check is undefined
        warnings.simplefilter("ignore", BlackPyWarning)
        warnings.simplefilter("error", PoorFitWarning)
        result = fit.run(seed=0)
        quality = result.goodness_of_fit()
        check = result.posterior_predictive(500, seed=1)
    assert quality.dof == 0 and not quality.defined and quality.p_value is None
    assert quality.reduced_chi2 is None and quality.chi2 < 1e-6  # exact fit of noise-free data
    assert check.dof == 0 and check.p_value is None and not check.poor_fit
    assert UNDEFINED in str(result) and UNDEFINED in str(quality) and UNDEFINED in str(check)
    assert check.replicated.shape == (500, 3) and check.datum_p_values.shape == (3,)
    # The record survives a round trip through the reproducibility bundle.
    loaded = FitResult.load(result.save(tmp_path / "fit"))
    assert loaded.goodness_of_fit().dof == 0
    np.testing.assert_allclose(loaded.predictions, result.predictions)
    repeat = loaded.posterior_predictive(500, seed=1)
    np.testing.assert_allclose(repeat.replicated, check.replicated)
    old = FitResult(
        result.samples,
        result.log_prob,
        result.param_names,
        result.param_labels,
        result.priors,
        "emcee",
    )
    with pytest.raises(bp.errors.BlackPyError, match="0.3"):
        old.goodness_of_fit()
    with pytest.raises(bp.errors.BlackPyError, match="0.3"):
        old.posterior_predictive()


def test_wrong_model_triggers_poor_fit_warning(schwarzschild, kerr):
    """A static metric fitted to data simulated from Kerr with a tight mass prior."""
    data = simulate_qpo_dataset(
        kerr, "RP", name="s", mass=6.0, r=5.5, sigma_nu=2.0, mass_sigma=0.05, sigma_low=0.1, a=0.9
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        fit = Fit(schwarzschild, data, "RP", {"r": Uniform(6.1, 20.0)}, nsteps=2000)
    with pytest.warns(PoorFitWarning, match="Poor fit"):
        result = fit.run(seed=3)
    quality = result.goodness_of_fit()
    assert quality.dof == 1 and quality.chi2 > 100 and quality.p_value < 1e-10
    assert result.diagnostics["posterior_predictive"]["poor_fit"] and "POOR FIT" in str(result)
    with pytest.warns(PoorFitWarning):
        check = result.posterior_predictive(1000, seed=0)
    assert check.poor_fit and check.p_value < 1e-10 and check.p_value_simulated == 0.0
    # The true model on the same (noise-free) data fits and does not warn.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        warnings.simplefilter("error", PoorFitWarning)
        good = Fit(
            kerr, data, "RP", {"a": Uniform(0.0, 0.99), "r": Uniform(2.0, 20.0)}, nsteps=3000
        )
        good_result = good.run(seed=4)
    assert good_result.goodness_of_fit().chi2 < 1e-4
    assert good_result.posterior_predictive(2000, seed=0).p_value > 0.3


def test_split_normal_noise_model():
    rng = np.random.default_rng(0)
    draws = draw_split_normal(rng, np.zeros(200000), 1.0, 3.0)
    # Below the centre with probability err_plus / (err_minus + err_plus), width err_plus.
    assert np.mean(draws < 0) == pytest.approx(0.75, abs=0.005)
    assert np.std(draws[draws < 0] / 3.0) == pytest.approx(np.std(draws[draws > 0]), rel=0.03)
    assert isinstance(Gaussian(0.0, 1.0), bp.infer.Prior)


def test_simulated_shadow_dataset_and_shadow_pulls(rn, kerr):
    import math

    truth = float(bp.light.shadow_radius(rn, Q=0.6))
    data = simulate_shadow_dataset(rn, name="img", sigma=0.2, Q=0.6)
    assert data.id == "synthetic/img" and data.verified_by_human
    mean, width = data.gaussian()
    assert mean == pytest.approx(truth, rel=1e-12) and width == pytest.approx(0.2, rel=1e-12)
    assert data.bounds["2sigma"] == pytest.approx((truth - 0.4, truth + 0.4))
    noisy = simulate_shadow_dataset(rn, name="img", sigma=0.2, Q=0.6, rng=np.random.default_rng(1))
    assert noisy.gaussian()[0] != mean and abs(noisy.gaussian()[0] - mean) < 1.0
    rotating = simulate_shadow_dataset(kerr, name="k", sigma=0.1, inclination=60.0, a=0.8)
    expected = float(bp.light.shadow_radius(kerr, inclination=math.radians(60.0), a=0.8))
    assert rotating.gaussian()[0] == pytest.approx(expected, rel=1e-12)
    with pytest.raises(TypeError, match="inclination"):
        simulate_shadow_dataset(kerr, name="k", sigma=0.1, a=0.8)
    with pytest.raises(ValueError, match="sigma"):
        simulate_shadow_dataset(rn, name="img", sigma=0.0)

    # One shadow datum, one Gaussian prior on Q, one parameter: one degree of freedom.
    fit = Fit(rn, data, priors={"Q": Gaussian(0.3, 0.1, low=0.0, high=1.0)})
    quality = fit.goodness_of_fit(Q=0.45)
    model = float(bp.light.shadow_radius(rn, Q=0.45))
    np.testing.assert_allclose(quality.pulls, [(model - truth) / 0.2, (0.45 - 0.3) / 0.1])
    assert quality.labels == ["r_sh[img]", "prior[Q]"] and quality.dof == 1
    # The interval likelihood has no pull: it is not counted, and this is stated.
    interval = Fit(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shadow_likelihood="interval")
    quality = interval.goodness_of_fit(Q=0.45)
    assert quality.n_data == 0 and quality.dof == -1 and "top-hat" in quality.notes[0]


@pytest.mark.slow
def test_residual_plot(rn):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    data = simulate_shadow_dataset(rn, name="img", sigma=0.2, Q=0.6)
    fit = Fit(rn, data, priors={"Q": Gaussian(0.3, 0.1, low=0.0, high=1.0)}, nsteps=1500)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        result = fit.run(seed=2)
    ax = bp.plot.residuals(result, n=500)
    labels = [tick.get_text() for tick in ax.get_xticklabels()]
    assert labels == ["r_sh[img]", "prior[Q]"] and "dof" in ax.get_title()
    np.testing.assert_allclose(ax.get_lines()[-1].get_ydata(), result.goodness_of_fit().pulls)
    plt.close(ax.figure)


@pytest.mark.slow
def test_poor_fit_warning_is_rare_under_the_true_model(kerr):
    """False-alarm rate of PoorFitWarning over repeated simulations of the true model.

    Truth drawn from the priors, Gaussian noise on the three frequencies and on the mass
    measurement; one degree of freedom (3 data + 1 Gaussian prior - 3 parameters). The
    posterior predictive p-value is conservative, so the rate must stay below the nominal
    threshold of 5%. Observed with these seeds: 1 of 100 in a longer run of the same
    set-up; here 60 fits, at most 3 alarms allowed.
    """
    rng = np.random.default_rng(20261116)
    n_fits, alarms, p_values = 60, 0, []
    for i in range(n_fits):
        a, mass, radius = rng.uniform(0.05, 0.9), rng.uniform(5.0, 12.0), rng.uniform(6.2, 9.0)
        data = simulate_qpo_dataset(
            kerr,
            "RP",
            name="s",
            mass=mass,
            r=radius,
            sigma_nu=2.0,
            mass_sigma=0.3,
            sigma_low=0.1,
            rng=rng,
            a=a,
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            fit = Fit(
                kerr,
                data,
                "RP",
                {"a": Uniform(0.0, 0.95), "r": Uniform(6.1, 9.5)},
                nwalkers=32,
                nsteps=2000,
                max_steps=8000,
            )
            result = fit.run(seed=100 + i)
        alarms += any(issubclass(w.category, PoorFitWarning) for w in caught)
        p_values.append(result.diagnostics["posterior_predictive"]["p_value"])
    p_values = np.asarray(p_values)
    print(f"PoorFitWarning under the true model: {alarms} of {n_fits}")
    assert alarms <= bp.infer.POOR_FIT_P_VALUE * n_fits, alarms
    # Not uniform: concentrated around 0.5, as the literature says.
    assert 0.4 < np.median(p_values) < 0.65 and np.percentile(p_values, 95) < 0.75


@pytest.mark.slow
def test_bayes_factor_warns_when_the_favoured_model_fails(kerr):
    """Inconsistent data (mass measurement 10 sigma off): whichever model wins, it fails."""
    import dataclasses

    data = simulate_qpo_dataset(
        kerr, "RP", name="s", mass=7.0, r=6.5, sigma_nu=2.0, mass_sigma=0.05, sigma_low=0.1, a=0.3
    )
    shifted = dataclasses.replace(data.mass_prior, value=data.mass_prior.value + 0.5)
    data = dataclasses.replace(data, mass_prior=shifted)
    fit = Fit(kerr, data, "RP", {"a": Uniform(0.0, 0.95), "r": Uniform(5.0, 12.0)}, nsteps=4000)
    with pytest.warns(PoorFitWarning):
        result = fit.run(seed=11)
    assert result.goodness_of_fit().dof == 1 and result.goodness_of_fit().chi2 > 10
    null_value = float(np.median(result["a"]))
    with pytest.warns(PoorFitWarning, match="Bayes factor favours"):
        factor = fit.bayes_factor(null={"a": null_value})
    assert factor.favoured_fit_is_poor is True
    assert any("FAILS THE FIT-QUALITY CHECK" in note for note in factor.notes)
