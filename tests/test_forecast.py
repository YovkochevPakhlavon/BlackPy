"""Fisher-matrix forecasts (B5)."""

import math
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackpy.errors import BlackPyError, BlackPyWarning, PriorDominatedWarning
from blackpy.infer import (
    Fit,
    Forecast,
    ForecastWarning,
    Gaussian,
    Problem,
    Uniform,
    build_problem,
    forecast,
    forecast_from_problem,
    simulate_qpo_dataset,
    simulate_shadow_dataset,
)


def linear_problem(matrix, sigma, priors, names):
    """A linear Gaussian model m = A theta, built by hand."""
    matrix = jnp.asarray(matrix)
    table = [
        {"label": f"y{k}", "dataset": "toy", "kind": "qpo", "value": 0.0,
         "err_minus": float(s), "err_plus": float(s), "unit": ""}
        for k, s in enumerate(sigma)
    ]  # fmt: skip
    predict = lambda theta: matrix @ theta
    return Problem(
        param_names=list(names),
        param_labels=list(names),
        priors=list(priors),
        kinds=["metric"] * len(names),
        fixed={},
        log_likelihood=None,
        log_likelihood_batch=None,
        predict=predict,
        predict_batch=jax.vmap(predict),
        data_table=table,
    )


def test_linear_gaussian_problem_is_exact():
    """Fisher covariance of a linear Gaussian model, exact to 1e-10 (mandatory)."""
    rng = np.random.default_rng(5)
    matrix = rng.normal(size=(6, 3))
    sigma = rng.uniform(0.5, 2.0, size=6)
    priors = [Uniform(-50.0, 50.0), Gaussian(0.3, 0.7), Gaussian(-1.0, 2.5)]
    problem = linear_problem(matrix, sigma, priors, ["x", "y", "z"])
    result = forecast_from_problem(problem, [0.1, 0.2, -0.4])
    assert isinstance(result, Forecast) and result.param_names == ["x", "y", "z"]
    exact_fisher = matrix.T @ np.diag(1.0 / sigma**2) @ matrix + np.diag([0.0, 0.7**-2, 2.5**-2])
    exact_covariance = np.linalg.inv(exact_fisher)
    np.testing.assert_allclose(result.fisher, exact_fisher, rtol=1e-12)
    np.testing.assert_allclose(result.covariance, exact_covariance, rtol=1e-10, atol=1e-14)
    np.testing.assert_allclose(
        [result.sigma[n] for n in "xyz"], np.sqrt(np.diag(exact_covariance)), rtol=1e-10
    )
    data_only = np.linalg.inv(matrix.T @ np.diag(1.0 / sigma**2) @ matrix)
    np.testing.assert_allclose(
        [result.sigma_data[n] for n in "xyz"], np.sqrt(np.diag(data_only)), rtol=1e-10
    )
    np.testing.assert_allclose(np.diag(result.correlation), 1.0, rtol=1e-12)
    assert not result.is_singular and math.isfinite(result.condition_number)
    assert not any(result.prior_dominated.values()) and not any(result.near_boundary.values())
    assert "least constrained direction" in str(result)
    # The answer does not depend on the fiducial point for a linear model.
    other = forecast_from_problem(problem, {"x": 3.0, "y": -2.0, "z": 5.0})
    np.testing.assert_allclose(other.covariance, result.covariance, rtol=1e-12)
    with pytest.raises(TypeError, match="lacks values"):
        forecast_from_problem(problem, {"x": 0.0})
    with pytest.raises(BlackPyError, match="outside the support"):
        forecast_from_problem(problem, [80.0, 0.0, 0.0])


def test_singular_fisher_matrix_reports_the_degenerate_direction():
    """Two parameters that enter only through their sum: a result, not an error."""
    matrix = np.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0], [0.0, 0.0, 1.0]])
    priors = [Uniform(-4.0, 4.0), Uniform(-4.0, 4.0), Uniform(-4.0, 4.0)]
    problem = linear_problem(matrix, [0.1, 0.1, 0.1], priors, ["p", "q", "s"])
    with pytest.warns(PriorDominatedWarning, match=r"\['p', 'q'\]"):
        result = forecast_from_problem(problem, [0.5, -0.5, 1.0])
    assert result.is_singular and result.condition_number == math.inf
    (item,) = result.degenerate_directions
    direction = item["direction"]
    assert direction["p"] == pytest.approx(-direction["q"], rel=1e-9)  # p - q is unconstrained
    assert abs(direction["s"]) < 1e-9 and abs(direction["p"]) == pytest.approx(math.sqrt(0.5))
    assert result.sigma["p"] == math.inf and result.sigma["q"] == math.inf
    assert result.sigma["s"] == pytest.approx(0.1, rel=1e-10)
    assert np.isnan(result.covariance[0, 1]) and result.covariance[2, 2] == pytest.approx(0.01)
    assert result.prior_dominated == {"p": True, "q": True, "s": False}
    assert "degenerate direction" in str(result) and "NOT a constraint" in str(result)
    # A Gaussian prior on one of the two lifts the degeneracy.
    priors[0] = Gaussian(0.0, 1.0)
    lifted = forecast_from_problem(
        linear_problem(matrix, [0.1, 0.1, 0.1], priors, ["p", "q", "s"]),
        [0.5, -0.5, 1.0],
        warn=False,
    )
    assert not lifted.is_singular and lifted.sigma["p"] == pytest.approx(1.0, rel=1e-3)
    assert lifted.correlation[0, 1] < -0.99 and lifted.prior_dominated["q"]
    # An interval likelihood carries no curvature and is reported in the notes.
    problem.data_table[2] = {"label": "bound", "dataset": "toy", "kind": "shadow_interval",
                             "lower": 0.0, "upper": 2.0, "unit": ""}  # fmt: skip
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        ignored = forecast_from_problem(problem, [0.5, -0.5, 1.0])
    assert ignored.sigma["s"] == math.inf and "no curvature" in ignored.notes[0]


@pytest.mark.parametrize(
    "case",
    [
        "qpo pair",
        "qpo triplet",
        pytest.param("shadow static", marks=pytest.mark.slow),
        pytest.param("shadow rotating", marks=pytest.mark.slow),
    ],
)
def test_log_likelihood_is_twice_differentiable_end_to_end(case, rn, kerr, kerr_newman):
    """Minus the autodiff Hessian of ln L at the model point equals J^T J / sigma^2."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        if case == "qpo pair":
            data = simulate_qpo_dataset(
                rn, "RP", name="s", mass=8.0, r=7.0, sigma_nu=2.0, mass_sigma=0.3, Q=0.5
            )
            problem = build_problem(rn, data, "RP", {"Q": Uniform(0, 1), "r": Uniform(5, 12)})
            theta = [0.5, 8.0, 7.0]
        elif case == "qpo triplet":
            data = simulate_qpo_dataset(
                kerr, "RP", name="s", mass=8.0, r=7.0, sigma_nu=2.0, mass_sigma=0.3,
                sigma_low=0.1, a=0.4,
            )  # fmt: skip
            problem = build_problem(kerr, data, "RP", {"a": Uniform(0, 0.99), "r": Uniform(4, 12)})
            theta = [0.4, 8.0, 7.0]
        elif case == "shadow static":
            data = simulate_shadow_dataset(rn, name="i", sigma=0.1, Q=0.5)
            problem = build_problem(rn, data, priors={"Q": Uniform(0, 1)})
            theta = [0.5]
        else:
            data = simulate_shadow_dataset(
                kerr_newman, name="i", sigma=0.1, inclination=60.0, a=0.6, Q=0.4
            )
            problem = build_problem(
                kerr_newman, data, priors={"a": Uniform(0, 0.99), "Q": Uniform(0, 0.9)}
            )
            theta = [0.6, 0.4, 60.0]
    theta = jnp.asarray(theta)
    hessian = -np.asarray(jax.jacfwd(jax.jacfwd(problem.log_likelihood))(theta))
    result = forecast_from_problem(problem, theta, warn=False)
    assert np.all(np.isfinite(hessian))
    scale = np.max(np.abs(result.fisher_data))
    np.testing.assert_allclose(hessian, result.fisher_data, rtol=1e-8, atol=1e-9 * scale)


@pytest.mark.slow
def test_forecast_interface_and_boundary_warning(rn):
    data = simulate_shadow_dataset(rn, name="img", sigma=0.05, Q=0.6)
    with warnings.catch_warnings():
        warnings.simplefilter("error", BlackPyWarning)
        result = forecast(rn, data, {"Q": 0.6}, priors={"Q": Uniform(0.0, 1.0)})
    # d r_sh / dQ by hand: sigma_Q = sigma / |d r_sh / dQ|.
    # Forward mode: reverse-mode autodiff cannot pass through the root finder's while loop.
    from blackpy.light.shadow import shadow_radius_core

    slope = float(jax.jacfwd(lambda q: shadow_radius_core(rn, jnp.stack([q])))(0.6))
    with pytest.raises(ValueError, match="Reverse-mode"):
        jax.grad(lambda q: shadow_radius_core(rn, jnp.stack([q])))(0.6)
    assert result.sigma["Q"] == pytest.approx(0.05 / abs(slope), rel=1e-9)
    assert not result.prior_dominated["Q"] and not result.near_boundary["Q"]
    # Close to Q = 0 the shadow hardly depends on Q: sigma exceeds the distance to the boundary.
    with pytest.warns(ForecastWarning, match="prior boundary"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PriorDominatedWarning)
            near = forecast(rn, data, {"Q": 0.05}, priors={"Q": Uniform(0.0, 1.0)})
    assert near.near_boundary["Q"] and near.sigma["Q"] > 0.05
    assert "within 1 sigma of a prior boundary" in str(near)
    with pytest.raises(BlackPyError, match="not finite"):
        forecast(rn, data, {"Q": 1.5}, priors={"Q": Uniform(0.0, 2.0)}, warn=False)  # no shadow


@pytest.mark.slow
def test_prior_dominated_bounded_parameter_is_warned_about(kerr):
    """The shadow size hardly depends on the Kerr spin: the forecast says so (mandatory)."""
    data = simulate_shadow_dataset(kerr, name="img", sigma=0.3, inclination=30.0, a=0.5)
    with pytest.warns(PriorDominatedWarning, match="do not report"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", ForecastWarning)
            result = forecast(
                kerr,
                data,
                {"a": 0.5},
                priors={"a": Uniform(0.0, 0.998)},
                fixed={"inclination": 30.0},
            )
    assert result.prior_dominated["a"] and result.sigma["a"] > 0.29  # wider than the prior
    assert result.near_boundary["a"]


@pytest.mark.slow
def test_forecast_agrees_with_the_emcee_posterior(kerr):
    """Well-constrained simulated triplet: forecast sigma = posterior sigma (mandatory)."""
    truth = {"a": 0.4, "M_s": 8.0, "r_s": 6.5}
    data = simulate_qpo_dataset(
        kerr, "RP", name="s", mass=8.0, r=6.5, sigma_nu=2.0, mass_sigma=0.3, sigma_low=0.1, a=0.4
    )
    priors = {"a": Uniform(0.0, 0.99), "r": Uniform(4.0, 12.0)}
    predicted = forecast(kerr, data, truth, "RP", priors)
    assert not predicted.is_singular and not any(predicted.prior_dominated.values())
    fit = Fit(kerr, data, "RP", priors, nwalkers=32, nsteps=6000, max_steps=24000)
    result = fit.run(seed=3)
    assert result.diagnostics["converged"]
    summary = result.summary()
    n_independent = result.diagnostics["n_independent_estimate"]
    tolerance = (
        5.0 / math.sqrt(2.0 * n_independent) + 0.03
    )  # sampling error of a std + non-Gaussianity
    for name in predicted.param_names:
        ratio = summary[name]["std"] / predicted.sigma[name]
        assert abs(ratio - 1.0) < tolerance, (name, ratio, tolerance)
    # The correlation structure agrees as well.
    for first, second in (("a", "M_s"), ("a", "r_s"), ("M_s", "r_s")):
        i, j = predicted.param_names.index(first), predicted.param_names.index(second)
        assert result.correlation(first, second) == pytest.approx(
            predicted.correlation[i, j], abs=0.05
        )
