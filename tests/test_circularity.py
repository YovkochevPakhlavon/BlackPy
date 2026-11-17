"""Deviation from circularity: the three definitions and the Bambi et al. (2019) figure (A1)."""

import math
import warnings

import numpy as np
import pytest
from scipy.optimize import brentq

import blackpy as bp
from blackpy.errors import CircularityDefinitionWarning
from blackpy.light.rotating import BAMBI2019_FACTOR, warn_if_definition_mismatch

# Exclusion curve of Fig. 1 of Bambi, Freese, Vagnozzi & Visinelli (2019, arXiv:1904.12983),
# read from the pixels of the published figure (ar5iv rendering, 434 x 534 px; plot frame
# x = 77..416 px for a = 0..1 and y = 371..30 px for 0..90 deg; one pixel is 0.003 in a and
# 0.26 deg). Columns: spin, observation angle of the black curve in degrees.
BAMBI_FIG1_CURVE = ((0.92, 64.0), (0.95, 54.1), (0.97, 48.1), (0.99, 42.4))
#: Spin at which the curve reaches the top of the figure (89.2 deg).
BAMBI_FIG1_EDGE_ON_SPIN = 0.8805


def delta_c(kerr, a, degrees, definition):
    return float(
        bp.light.deviation_from_circularity(kerr, math.radians(degrees), a=a, definition=definition)
    )


@pytest.mark.slow
def test_definitions_are_related_as_documented(kerr):
    a, inc = 0.9, math.radians(70.0)
    observables = bp.light.shadow_observables(kerr, inc, a=a)
    fractional = bp.light.deviation_from_circularity(kerr, inc, a=a)  # default definition
    assert float(fractional) == pytest.approx(float(observables.delta_c), rel=1e-14)
    assert bp.light.DELTA_C_DEFINITIONS == ("fractional_rms", "jp_asymmetry", "bambi2019")
    bambi = bp.light.deviation_from_circularity(kerr, inc, a=a, definition="bambi2019")
    assert BAMBI2019_FACTOR == 4.0
    assert float(bambi) == pytest.approx(4.0 * float(fractional), rel=1e-14)
    # Vectorised over spin and inclination, zero without rotation, for every definition.
    for definition in bp.light.DELTA_C_DEFINITIONS:
        grid = bp.light.deviation_from_circularity(
            kerr, np.radians([[20.0], [80.0]]), a=np.array([0.0, 0.3, 0.9]), definition=definition
        )
        assert grid.shape == (2, 3)
        np.testing.assert_allclose(np.asarray(grid)[:, 0], 0.0, atol=1e-12)
        assert np.all(np.diff(np.asarray(grid), axis=1) > 0.0)  # grows with spin
        assert np.all(np.asarray(grid)[1] >= np.asarray(grid)[0])  # and with inclination
    with pytest.raises(ValueError, match="fractional_rms"):
        bp.light.deviation_from_circularity(kerr, inc, a=a, definition="eht")


@pytest.mark.parametrize(
    "a, degrees",
    [
        pytest.param(0.6, 90.0, marks=pytest.mark.slow),
        (0.9, 90.0),
        pytest.param(0.95, 50.0, marks=pytest.mark.slow),
        pytest.param(0.998, 75.0, marks=pytest.mark.slow),
    ],
)
def test_jp_asymmetry_against_polygon_computation(kerr, a, degrees):
    """Johannsen & Psaltis (2010), Eqs. (3)-(8), evaluated directly on a fine polygon."""
    contour = bp.light.shadow_contour(kerr, math.radians(degrees), a=a, n=6000)
    displacement = 0.5 * (contour.alpha.max() + contour.alpha.min())  # Eq. (3), signed
    radius = np.hypot(contour.alpha - displacement, contour.beta)  # Eq. (5)
    angle = np.unwrap(np.arctan2(contour.beta, contour.alpha - displacement))  # Eq. (6)
    weight = np.abs(np.gradient(angle))
    mean = np.sum(radius * weight) / np.sum(weight)  # Eq. (4)
    asymmetry = 2.0 * math.sqrt(np.sum((radius - mean) ** 2 * weight) / np.sum(weight))  # (8)
    value = delta_c(kerr, a, degrees, "jp_asymmetry")
    assert value == pytest.approx(asymmetry / mean, rel=2e-5)
    # Roughly twice the fractional rms (different centre and mean, same idea).
    assert value == pytest.approx(2.0 * delta_c(kerr, a, degrees, "fractional_rms"), rel=0.15)


def test_jp_asymmetry_agrees_with_the_fit_formula_of_the_paper(kerr):
    """Johannsen & Psaltis (2010), Eq. (10): A/M = 0.36 a^3 sin^(3/2) i for Kerr (a fit)."""
    for a, degrees in ((0.6, 90.0), (0.8, 60.0), (0.9, 90.0)):
        inc = math.radians(degrees)
        contour = bp.light.shadow_contour(kerr, inc, a=a, n=2000)
        displacement = 0.5 * (contour.alpha.max() + contour.alpha.min())
        radius = np.hypot(contour.alpha - displacement, contour.beta)
        angle = np.unwrap(np.arctan2(contour.beta, contour.alpha - displacement))
        weight = np.abs(np.gradient(angle))
        mean = np.sum(radius * weight) / np.sum(weight)
        asymmetry = delta_c(kerr, a, degrees, "jp_asymmetry") * mean  # in units of M
        assert asymmetry == pytest.approx(0.36 * a**3 * math.sin(inc) ** 1.5, rel=0.15)


def test_bambi2019_reproduces_the_published_exclusion_curve(kerr):
    """The 0.1 contour of the 'bambi2019' definition is the black curve of their Fig. 1.

    Tolerances: 1.5 degrees (six pixels of the published figure) on the angle and 0.01 (three
    pixels) on the spin where the curve is nearly vertical.
    """
    for a, published in BAMBI_FIG1_CURVE:
        angle = brentq(lambda x, a=a: delta_c(kerr, a, x, "bambi2019") - 0.1, 5.0, 90.0)
        assert angle == pytest.approx(published, abs=1.5), (a, angle, published)
        # The value on the published curve is 0.1 within the reading accuracy.
        assert delta_c(kerr, a, published, "bambi2019") == pytest.approx(0.1, rel=0.03)
    edge_on = brentq(lambda a: delta_c(kerr, a, 89.2, "bambi2019") - 0.1, 0.5, 0.99)
    assert edge_on == pytest.approx(BAMBI_FIG1_EDGE_ON_SPIN, abs=0.01)
    # Their abstract: "the observation angle is unbounded for a <~ 0.9" ...
    for a in (0.5, 0.8, 0.87):
        assert delta_c(kerr, a, 90.0, "bambi2019") < 0.1
    # ... and Kerr black holes with 0.95 <~ a <= 1 are excluded above an angle that
    # decreases from 54 deg (a = 0.95) to 40 deg (a -> 1); the abstract rounds this to 45.
    assert delta_c(kerr, 0.95, 60.0, "bambi2019") > 0.1 > delta_c(kerr, 0.95, 45.0, "bambi2019")
    assert delta_c(kerr, 0.998, 45.0, "bambi2019") > 0.1 > delta_c(kerr, 0.998, 35.0, "bambi2019")


def test_other_normalisations_do_not_reproduce_the_published_curve(kerr):
    """Eq. (5) as printed, twice it, and the unnormalised rms all miss the figure."""
    # (i) fractional rms (their Eq. (5) as printed): never reaches 0.1 for a Kerr black hole.
    spins = np.linspace(0.0, 0.998, 40)[:, None]
    angles = np.radians(np.linspace(1.0, 90.0, 30))[None, :]
    fractional = np.asarray(bp.light.deviation_from_circularity(kerr, angles, a=spins))
    assert 0.049 < fractional.max() < 0.051
    # (ii) twice the fractional rms: excludes nothing below a = 0.99.
    assert 2.0 * fractional[spins[:, 0] < 0.99].max() < 0.1
    # (iii) unnormalised rms in units of M: the values on the published curve are 0.12, not
    # 0.1, and its own 0.1 contour lies 9 to 13 degrees below the published curve.
    for a, published in BAMBI_FIG1_CURVE[:3]:
        observables = bp.light.shadow_observables(kerr, math.radians(published), a=a)
        rms = float(observables.delta_c * observables.area_radius)
        assert 0.118 < rms < 0.13

        def unnormalised(x, a=a):
            obs = bp.light.shadow_observables(kerr, math.radians(x), a=a)
            return float(obs.delta_c * obs.area_radius) - 0.1

        assert published - brentq(unnormalised, 5.0, 90.0) > 7.0


def test_bound_with_another_definition_warns():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert not warn_if_definition_mismatch("bambi2019", "bambi2019", "The bound")
    with pytest.warns(CircularityDefinitionWarning, match="factors of 2 to 4"):
        assert warn_if_definition_mismatch("fractional_rms", "bambi2019", "The bound")
    with pytest.warns(CircularityDefinitionWarning, match="none of the definitions"):
        assert warn_if_definition_mismatch("fractional_rms", "eht2019_diameter_spread", "It")
    with pytest.raises(ValueError):
        warn_if_definition_mismatch("unknown", "bambi2019", "The bound")
