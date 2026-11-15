"""Goodness of fit and posterior predictive checks (B1)."""

import warnings

import numpy as np
import pytest

import blackpy as bp
from blackpy.errors import BlackPyWarning, PoorFitWarning
from blackpy.infer import Fit, FitResult, Gaussian, Uniform, simulate_qpo_dataset
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
