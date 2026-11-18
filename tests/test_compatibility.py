"""Backward compatibility within 1.x: result bundles, the inclination rename, unit guards.

The bundle fixtures in ``tests/data/`` were written by BlackPy 0.3.0 (format 0) and by
1.0 (format 1); see ``tests/data/make_bundles.py``. They guard loading for all of 1.x.
"""

import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

import blackpy as bp
from blackpy.errors import BlackPyError, UnverifiedDataWarning
from blackpy.infer import (
    Fit,
    FitResult,
    Isotropic,
    Uniform,
    build_problem,
    forecast,
    simulate_shadow_dataset,
)
from blackpy.infer.result import BUNDLE_FORMAT

DATA = Path(__file__).parent / "data"


# ---------------------------------------------------------------------- result bundles


def test_bundle_of_the_current_format_loads():
    result = FitResult.load(DATA / "bundle_format_1.npz")
    assert result.param_names == ["a", "inclination_deg"]
    assert result.samples.shape == (300, 2)
    assert result.predictions.shape == (300, 1)
    assert result.priors["inclination_deg"] == Isotropic(0.0, 90.0)
    assert result.config["datasets"][0]["id"] == "synthetic/img"
    assert np.all((result["a"] >= 0.0) & (result["a"] <= 0.99))
    assert result.summary()["inclination_deg"]["median"] > 0.0
    assert result.goodness_of_fit().n_data == 1


def test_pre_1_0_bundle_loads_with_the_inclination_renamed():
    with np.load(DATA / "bundle_format_0.npz") as bundle:
        raw = json.loads(str(bundle["record"]))
    assert "bundle_format" not in raw and raw["param_names"] == ["a", "inclination"]
    assert raw["metadata"]["versions"]["blackpy"] == "0.3.0"
    result = FitResult.load(DATA / "bundle_format_0.npz")
    assert result.param_names == ["a", "inclination_deg"]
    assert set(result.priors) == {"a", "inclination_deg"}
    assert result.config["parameter_kinds"] == {"a": "metric", "inclination_deg": "nuisance"}
    assert result.config["parameter_base"]["inclination_deg"] == "inclination_deg"
    assert result.metadata["loaded_from_bundle_format"] == 0
    assert result.summary()["inclination_deg"]["median"] > 0.0
    with pytest.warns(DeprecationWarning, match="inclination_deg"):
        old = result["inclination"]
    np.testing.assert_array_equal(old, result["inclination_deg"])


def test_save_writes_the_format_and_round_trips(tmp_path):
    result = FitResult.load(DATA / "bundle_format_0.npz")
    path = result.save(tmp_path / "again")
    with np.load(path) as bundle:
        assert json.loads(str(bundle["record"]))["bundle_format"] == BUNDLE_FORMAT == 1
    again = FitResult.load(path)
    assert again.param_names == result.param_names
    np.testing.assert_array_equal(again.samples, result.samples)


def test_newer_bundle_format_is_refused(tmp_path):
    with np.load(DATA / "bundle_format_1.npz") as bundle:
        record = json.loads(str(bundle["record"]))
        arrays = {k: np.array(bundle[k]) for k in bundle.files if k != "record"}
    record["bundle_format"] = 99
    record["metadata"]["versions"]["blackpy"] = "7.0.0"
    path = tmp_path / "future.npz"
    np.savez(path, record=np.array(json.dumps(record)), **arrays)
    with pytest.raises(BlackPyError, match=r"format 99.*7\.0\.0"):
        FitResult.load(path)


# ---------------------------------------------------------------------- inclination rename


@pytest.fixture(scope="module")
def shadow_data():
    return simulate_shadow_dataset(
        bp.metric.Kerr(), name="img", sigma=0.3, inclination_deg=60.0, a=0.5
    )


def test_old_inclination_name_is_accepted_with_a_warning(kerr, shadow_data):
    with pytest.warns(DeprecationWarning, match="'inclination_deg'"):
        old = build_problem(
            kerr, shadow_data, priors={"a": Uniform(0, 0.99), "inclination": Isotropic(0, 80)}
        )
    new = build_problem(
        kerr, shadow_data, priors={"a": Uniform(0, 0.99), "inclination_deg": Isotropic(0, 80)}
    )
    assert old.param_names == new.param_names == ["a", "inclination_deg"]
    assert old.priors == new.priors
    theta = np.array([0.4, 50.0])
    assert float(old.log_likelihood(theta)) == float(new.log_likelihood(theta))
    with pytest.warns(DeprecationWarning, match="in fixed"):
        fixed = build_problem(
            kerr, shadow_data, priors={"a": Uniform(0, 0.99)}, fixed={"inclination": 30.0}
        )
    assert fixed.description["inclination_deg"] == 30.0


def test_old_per_source_inclination_names_are_renamed(kerr):
    data = [
        simulate_shadow_dataset(kerr, name=n, sigma=0.1, inclination_deg=40.0, a=0.5)
        for n in ("one", "two")
    ]
    with pytest.warns(DeprecationWarning, match="inclination_deg_one"):
        problem = build_problem(
            kerr,
            data,
            priors={"a": Uniform(0, 0.99)},
            fixed={"inclination_one": 40.0, "inclination_deg_two": 30.0},
        )
    assert problem.description["inclinations_deg"] == {
        "inclination_deg_one": 40.0,
        "inclination_deg_two": 30.0,
    }


def test_inclination_given_twice_is_an_error(kerr, shadow_data):
    with pytest.raises(ValueError, match="twice"):
        build_problem(
            kerr,
            shadow_data,
            priors={
                "a": Uniform(0, 0.99),
                "inclination": Isotropic(0, 90),
                "inclination_deg": Isotropic(0, 90),
            },
        )
    with pytest.raises(ValueError, match="both 'inclination'"):
        simulate_shadow_dataset(kerr, name="x", sigma=0.1, inclination=30.0, inclination_deg=30.0)


def test_old_keywords_of_fit_simulation_and_forecast(kerr, shadow_data):
    with pytest.warns(DeprecationWarning, match="inclination_deg=<degrees>"):
        old = simulate_shadow_dataset(kerr, name="img", sigma=0.3, inclination=60.0, a=0.5)
    assert old.gaussian() == shadow_data.gaussian()
    with pytest.warns(DeprecationWarning):
        fit = Fit(
            kerr, shadow_data, priors={"a": Uniform(0, 0.99), "inclination": Isotropic(0, 90)}
        )
    assert fit.param_names == ["a", "inclination_deg"]
    assert "inclination_deg" in fit.priors and "inclination" not in fit.priors
    with pytest.warns(DeprecationWarning):
        old_point = fit.log_likelihood(a=0.4, inclination=50.0)
    assert old_point == fit.log_likelihood(a=0.4, inclination_deg=50.0)
    with pytest.warns(DeprecationWarning):
        old_quality = fit.goodness_of_fit(a=0.4, inclination=50.0)
    assert old_quality.chi2 == fit.goodness_of_fit(a=0.4, inclination_deg=50.0).chi2
    priors = {"a": Uniform(0, 0.99)}
    with pytest.warns(DeprecationWarning):
        old_forecast = forecast(
            kerr, shadow_data, {"a": 0.5, "inclination": 60.0}, None, priors, warn=False
        )
    new_forecast = forecast(
        kerr, shadow_data, {"a": 0.5, "inclination_deg": 60.0}, None, priors, warn=False
    )
    assert old_forecast.sigma == new_forecast.sigma


def test_new_names_do_not_warn(kerr, shadow_data):
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        problem = build_problem(
            kerr,
            shadow_data,
            priors={"a": Uniform(0, 0.99), "inclination_deg": Isotropic(0, 90)},
        )
    assert problem.param_labels[-1] == r"i\ [{}^\circ]"


def test_fixed_inclination_in_radians_is_caught(kerr, shadow_data):
    # Degrees are expected; an out-of-range value says so.
    with pytest.raises(ValueError, match="degrees"):
        build_problem(kerr, shadow_data, priors={"a": Uniform(0, 1)}, fixed={"inclination_deg": -1})
    with pytest.raises(ValueError, match="degrees"):
        simulate_shadow_dataset(kerr, name="x", sigma=0.1, inclination_deg=200.0, a=0.5)


# ---------------------------------------------------------------------- radian guards


@pytest.mark.parametrize(
    "call",
    [
        lambda m: bp.light.shadow_contour(m, 60.0, a=0.5),
        lambda m: bp.light.shadow_observables(m, np.array([0.5, 60.0]), a=0.5),
        lambda m: bp.light.shadow_radius(m, inclination=60.0, a=0.5),
        lambda m: bp.light.deviation_from_circularity(m, -0.1, a=0.5),
        lambda m: bp.light.photon_region(m, 4.0, a=0.5),
        lambda m: bp.light.shadow_in_plasma(m, bp.light.HomogeneousPlasma(), inclination=90, a=0.5),
        lambda m: bp.plot.shadow(m, [0.3, 45.0], a=0.5),
    ],
)
def test_inclination_outside_zero_pi_is_refused(kerr, call):
    with pytest.raises(ValueError, match=r"radians.*math\.radians"):
        call(kerr)


def test_inclination_in_range_passes_the_guard(kerr):
    assert math.isfinite(float(bp.light.shadow_radius(kerr, inclination=math.pi, a=0.5)))
    assert math.isfinite(float(bp.light.shadow_radius(kerr, inclination=0.0, a=0.5)))


def test_registry_inclination_stays_in_degrees_with_a_unit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnverifiedDataWarning)
        m87 = bp.data.load("eht/m87")
    assert m87.inclination.unit == "deg"
