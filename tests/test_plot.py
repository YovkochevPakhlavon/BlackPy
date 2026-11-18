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


@pytest.mark.slow
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


def test_shadow_contours(kerr, kerr_newman, rn):
    import math

    spins = [0.0, 0.5, 0.9]
    ax = bp.plot.shadow(kerr, math.pi / 2, a=spins, reference_circle=True)
    lines = ax.get_lines()
    assert len(lines) == 4 and ax.get_aspect() == 1.0  # three contours and the circle
    contour = bp.light.shadow_contour(kerr, math.pi / 2, a=0.9)
    np.testing.assert_allclose(lines[2].get_xdata()[:-1], contour.alpha)
    np.testing.assert_allclose(lines[2].get_ydata()[:-1], contour.beta)
    assert lines[2].get_xdata()[-1] == lines[2].get_xdata()[0]  # closed curve
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert labels[:3] == ["$a = 0$", "$a = 0.5$", "$a = 0.9$"] and "Schwarzschild" in labels[3]
    np.testing.assert_allclose(np.hypot(*lines[3].get_data()), 3.0 * math.sqrt(3.0))
    plt.close(ax.figure)

    # Several inclinations, recentred, with the Delta C definition named explicitly.
    ax = bp.plot.shadow(
        kerr, np.radians([17.0, 90.0]), a=0.95, recentre=True, annotate_delta_c="bambi2019"
    )
    labels = [text.get_text() for text in ax.get_legend().get_texts()]
    assert "i = 17" in labels[0] and "B19" in labels[0] and "0.134" in labels[1]
    centroid = float(bp.light.shadow_observables(kerr, math.pi / 2, a=0.95).centroid)
    contour = bp.light.shadow_contour(kerr, math.pi / 2, a=0.95)
    np.testing.assert_allclose(ax.get_lines()[1].get_xdata()[:-1], contour.alpha - centroid)
    assert "recentred" in ax.get_xlabel()
    plt.close(ax.figure)

    # A metric parameter scanned together with the spin; a static metric gives circles;
    # an observational band from a shadow dataset.
    ax = bp.plot.shadow(
        kerr_newman, math.pi / 3, a=[0.2, 0.4], Q=[0.1, 0.6], band="eht/sgra", labels=["A", "B"]
    )
    assert len(ax.get_lines()) == 2 and len(ax.patches) == 1
    assert [t.get_text() for t in ax.get_legend().get_texts()][-2:] == ["A", "B"]
    bp.plot.shadow(rn, 1.0, Q=[0.0, 0.5], r_obs=50.0, ax=ax)
    radius = np.hypot(*ax.get_lines()[-1].get_data())
    np.testing.assert_allclose(radius, float(bp.light.shadow_radius(rn, 50.0, Q=0.5)), rtol=1e-9)
    plt.close(ax.figure)

    with pytest.raises(ValueError, match="same length"):
        bp.plot.shadow(kerr_newman, 1.0, a=[0.1, 0.2], Q=[0.1, 0.2, 0.3])
    with pytest.raises(ValueError, match="fractional_rms"):
        bp.plot.shadow(kerr, 1.0, a=0.5, annotate_delta_c="default")
    with pytest.raises(ValueError, match="labels"):
        bp.plot.shadow(kerr, 1.0, a=[0.1, 0.2], labels=["one"])
    with pytest.raises(ValueError, match="3sigma"):
        bp.plot.shadow(kerr, 1.0, a=0.5, band="eht/sgra", band_level="3sigma")
    plt.close("all")


# ---------------------------------------------------------------------- prediction plots (1.0)


def test_deflection_plot_draws_the_lensing_functions(schwarzschild):
    ax = bp.plot.deflection(schwarzschild)
    exact, weak, strong = ax.get_lines()[:3]
    b_c = float(bp.lensing.strong_deflection_limit(schwarzschild).b_c)
    b = b_c * (1.0 + np.geomspace(1e-6, 30.0, 300))  # the default impact parameters
    np.testing.assert_allclose(exact.get_xdata(), b / b_c - 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        exact.get_ydata(), bp.lensing.deflection_angle(schwarzschild, b), rtol=1e-12
    )
    b_far = b[b >= 2.0 * b_c]  # the weak-field estimate only where it applies
    np.testing.assert_allclose(weak.get_xdata(), b_far / b_c - 1.0, rtol=1e-12)
    np.testing.assert_allclose(
        weak.get_ydata(), bp.lensing.weak_deflection_angle(schwarzschild, b_far), rtol=1e-12
    )
    np.testing.assert_allclose(
        strong.get_ydata(), bp.lensing.strong_deflection_angle(schwarzschild, b), rtol=1e-12
    )
    assert ax.get_ylabel() == "deflection angle [rad]" and "b_c" in ax.get_xlabel()
    with pytest.raises(ValueError, match="b_c"):
        bp.plot.deflection(schwarzschild, b=[4.0, 10.0])
    plt.close("all")


@pytest.mark.slow
def test_disk_flux_plot_draws_page_thorne(kerr, rn):
    ax = bp.plot.disk_flux(kerr, a=[0.0, 0.9], newtonian=True)
    lines = ax.get_lines()
    assert [line.get_label() for line in lines[:2]] == ["Kerr, a = 0", "Kerr, a = 0.9"]
    profile = bp.disk.page_thorne_flux(kerr, a=0.9)
    np.testing.assert_allclose(lines[1].get_xdata(), profile.r)
    np.testing.assert_allclose(lines[1].get_ydata(), np.clip(profile.flux, 1e-14, None))
    assert "dot M" in ax.get_ylabel() and ax.get_xlabel() == "$r/M$"
    ax2 = bp.plot.disk_flux([(kerr, {"a": 0.5}), (rn, {"Q": 0.8}, "RN")], quantity="temperature")
    labels = [line.get_label() for line in ax2.get_lines()]
    assert labels == ["Kerr a = 0.5", "RN"]
    np.testing.assert_allclose(
        ax2.get_lines()[1].get_ydata(), bp.disk.page_thorne_flux(rn, Q=0.8).temperature
    )
    assert "T_{\\rm max}" in ax2.get_ylabel()
    with pytest.raises(ValueError, match="quantity"):
        bp.plot.disk_flux(kerr, quantity="spectrum", a=0.5)
    plt.close("all")


@pytest.mark.slow
def test_forecast_plot_ellipses_match_the_covariance(kerr):
    from blackpy.infer import simulate_qpo_dataset

    data = simulate_qpo_dataset(
        kerr, "RP", name="src", mass=6.0, r=5.7, sigma_nu=1.0, mass_sigma=0.3, sigma_low=0.1, a=0.3
    )
    fiducial = {"a": 0.3, "M_src": 6.0, "r_src": 5.7}
    priors = {"a": Uniform(0.0, 0.998), "r": Uniform(2.0, 20.0)}
    prediction = bp.infer.forecast(kerr, data, fiducial, "RP", priors, warn=False)
    figure = bp.plot.forecast(prediction, levels=(0.68,))
    axes = np.array(figure.axes).reshape(3, 3)
    # The 68% ellipse of (M_src, r_src): (x - m)^T C^-1 (x - m) = -2 ln(1 - 0.68) on the curve.
    ellipse = axes[2, 1].get_lines()[0]
    points = np.stack([ellipse.get_xdata() - 6.0, ellipse.get_ydata() - 5.7])
    block = prediction.covariance[np.ix_([1, 2], [1, 2])]
    q = np.einsum("in,ij,jn->n", points, np.linalg.inv(block), points)
    np.testing.assert_allclose(q, -2.0 * np.log(1.0 - 0.68), rtol=1e-8)
    # Diagonal: a Gaussian with the forecast sigma.
    curve = axes[0, 0].get_lines()[0]
    peak = curve.get_xdata()[np.argmax(curve.get_ydata())]
    assert peak == pytest.approx(0.3, abs=prediction.sigma["a"] / 50)
    assert axes[2, 1].get_xlabel().startswith("M_src") and "odot" in axes[2, 1].get_xlabel()
    assert axes[2, 0].get_ylabel() == "r_src [M]"
    assert not axes[0, 2].get_visible()
    # On top of an existing K x K figure (a corner plot has this layout).
    import matplotlib.pyplot as plt_

    corner_like, _ = plt_.subplots(2, 2)
    bp.plot.forecast(prediction, parameters=["a", "M_src"], figure=corner_like)
    assert len(corner_like.axes[2].get_lines()) >= 2
    with pytest.raises(ValueError, match="not 3 x 3"):
        bp.plot.forecast(prediction, figure=corner_like)
    plt.close("all")
