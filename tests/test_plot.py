"""Plotting helpers."""

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.infer import FitResult, Uniform  # noqa: E402


def test_profile_single_metric(rn):
    q = np.linspace(0.0, 1.0, 30)
    ax = bp.plot.profile(rn, "isco", Q=q)
    (line,) = ax.get_lines()
    np.testing.assert_allclose(line.get_ydata(), bp.orbits.isco(rn, Q=q))
    assert ax.get_xlabel() == "$Q$" and "ISCO" in ax.get_ylabel()
    plt.close(ax.figure)


def test_profile_several_metrics_and_quantities(rn, bardeen, hayward):
    grid = np.linspace(0.0, 0.7, 20)
    fig, ax = plt.subplots()
    for quantity in ("photon_sphere", "b_c", "shadow_radius", "event_horizon", "nu_phi_isco"):
        bp.plot.profile(
            [(rn, {"Q": grid}), (bardeen, {"q": grid}), (hayward, {"l": grid})], quantity, ax=ax
        )
    assert len(ax.get_lines()) == 15 and ax.get_legend() is not None
    assert ax.get_xlabel() == "metric parameter"
    plt.close(fig)
    ax = bp.plot.profile(rn, "nu_r", r=8.0, mass=10.0, Q=grid)
    assert "Hz" in ax.get_ylabel()
    expected = bp.orbits.frequencies_hz(rn, 8.0, 10.0, Q=grid).nu_r
    np.testing.assert_allclose(ax.get_lines()[0].get_ydata(), expected)
    plt.close(ax.figure)
    ax = bp.plot.profile(rn, lambda m, **p: bp.orbits.specific_energy(m, 10.0, **p), Q=grid)
    assert len(ax.get_lines()) == 1
    plt.close(ax.figure)


def test_profile_errors(rn, bardeen):
    with pytest.raises(ValueError, match="exactly one array-valued"):
        bp.plot.profile(rn, "isco", Q=0.5)
    with pytest.raises(ValueError, match="Unknown quantity"):
        bp.plot.profile(rn, "temperature", Q=np.linspace(0, 1, 5))
    with pytest.raises(ValueError, match="pass r="):
        bp.plot.profile(rn, "nu_phi", Q=np.linspace(0, 1, 5))
    with pytest.raises(ValueError, match="has no parameter"):
        bp.plot.profile([(bardeen, {"Q": np.linspace(0, 1, 5)})], "isco")


def _fake_result():
    rng = np.random.default_rng(1)
    samples = np.column_stack([rng.normal(0.5, 0.05, 4000), rng.normal(6.0, 0.2, 4000)])
    return FitResult(
        samples=samples,
        log_prob=np.zeros(4000),
        param_names=["Q", "M_src"],
        param_labels=["Q", r"M_{\rm src}/M_\odot"],
        priors={"Q": Uniform(0, 1), "M_src": Uniform(0, 20)},
        sampler="dynesty",
        log_evidence=-12.3,
        log_evidence_err=0.2,
        diagnostics={"prior_dominated": {"Q": False, "M_src": True}},
    )


def test_corner_and_latex_table():
    result = _fake_result()
    fig = bp.plot.corner(result, truths={"Q": 0.5})
    assert len(fig.axes) == 4
    plt.close(fig)
    fig = bp.plot.corner(result, parameters=["Q"])
    assert len(fig.axes) == 1
    plt.close(fig)
    table = bp.plot.latex_table({"RN": result, "RN again": result}, caption="Test.")
    assert r"\begin{tabular}{lcc}" in table and "$Q$ & $0." in table
    assert r"\ln Z" in table and "-12.30 \\pm 0.20" in table
    assert table.count(r"$^\dagger$") == 3 and "not a constraint" in table
    assert table.strip().endswith(r"\end{table}")
