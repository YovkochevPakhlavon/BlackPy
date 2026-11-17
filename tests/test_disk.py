"""Thin-disk observables (B7): efficiency and the Page-Thorne flux.

Source: Page & Thorne (1974), Astrophys. J. 191, 499 (read from the scanned journal
version at the NASA ADS).
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import blackpy as bp
from blackpy.errors import BlackPyError, ReflectionSymmetryError

dk = bp.disk


def page_thorne_kerr(r, a):
    r"""Page & Thorne (1974), Eqs. (14), (15k)-(15n) and (11b), with M = 1.

    The prefactor of Eq. (15n) is 3/(2M); the "3" is lost in the scanned text and was
    fixed by the Newtonian limit F -> 3 / (8 pi r^3) (a test below checks it separately).
    """
    x = math.sqrt(r)
    third = math.acos(a) / 3.0
    x1, x2, x3 = (
        2 * math.cos(third - math.pi / 3),
        2 * math.cos(third + math.pi / 3),
        -2 * math.cos(third),
    )
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = math.sqrt(3 * a * a + z1 * z1)
    x0 = math.sqrt(3 + z2 - math.copysign(1.0, a) * math.sqrt((3 - z1) * (3 + z1 + 2 * z2)))
    bracket = x - x0 - 1.5 * a * math.log(x / x0)
    for xi, xj, xk in ((x1, x2, x3), (x2, x1, x3), (x3, x1, x2)):
        bracket -= 3 * (xi - a) ** 2 / (xi * (xi - xj) * (xi - xk)) * math.log((x - xi) / (x0 - xi))
    f = 1.5 / (x * x * (x**3 - 3 * x + 2 * a)) * bracket
    return f / (4 * math.pi * x * x), x0 * x0  # e^(nu + psi + mu) = r, Eq. (15d)


def kerr_energy(r, a):
    """Page & Thorne Eq. (15g): E = (1 - 2/r + a r^-3/2) / sqrt(1 - 3/r + 2 a r^-3/2)."""
    return (1 - 2 / r + a * r**-1.5) / math.sqrt(1 - 3 / r + 2 * a * r**-1.5)


def test_radiative_efficiency(schwarzschild, kerr, rn):
    """Schwarzschild 1 - sqrt(8/9); Kerr from E(r_ISCO); retrograde = opposite spin."""
    assert float(dk.radiative_efficiency(schwarzschild)) == pytest.approx(
        1.0 - math.sqrt(8.0 / 9.0), rel=1e-12
    )
    for a in (0.3, 0.9, 0.99, 0.9999):
        r_isco = float(bp.orbits.isco(kerr, a=a))
        expected = 1.0 - kerr_energy(r_isco, a)
        assert float(dk.radiative_efficiency(kerr, a=a)) == pytest.approx(expected, rel=1e-10)
        assert float(dk.radiative_efficiency(kerr, a=a, direction="retrograde")) == pytest.approx(
            float(dk.radiative_efficiency(kerr, a=-a)), rel=1e-12
        )
    # Extremal limit 1 - 1/sqrt(3), approached from below (evaluated at high, not extremal spin:
    # the closed form at the BPT ISCO for a -> 1).
    spins = [0.99, 0.999, 0.9999, 0.99999]
    values = [float(dk.radiative_efficiency(kerr, a=a)) for a in spins]
    assert all(later > earlier for earlier, later in zip(values, values[1:], strict=False))
    assert values[-1] < 1.0 - 1.0 / math.sqrt(3.0) < values[-1] + 0.05
    # Explicit inner edge, vectorised over parameters; RN at Q = 0 equals Schwarzschild.
    np.testing.assert_allclose(
        dk.radiative_efficiency(rn, r_in=8.0, Q=np.array([0.0, 0.5])),
        [1.0 - kerr_energy(8.0, 0.0), 1.0 - float(bp.orbits.specific_energy(rn, 8.0, Q=0.5))],
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "a",
    [
        0.0,
        0.9,
        pytest.param(0.5, marks=pytest.mark.slow),
        pytest.param(0.998, marks=pytest.mark.slow),
        pytest.param(-0.7, marks=pytest.mark.slow),
    ],
)
def test_flux_matches_the_page_thorne_kerr_closed_form(kerr, a):
    """Over a grid of radii and spins (mandatory)."""
    r_isco = page_thorne_kerr(10.0, a)[1]
    radii = r_isco * np.array([1.001, 1.05, 1.3, 2.0, 5.0, 40.0, 1.0e3])
    profile = dk.page_thorne_flux(kerr, radii, a=a)
    assert profile.r_in == pytest.approx(r_isco, rel=1e-11)
    expected = np.array([page_thorne_kerr(r, a)[0] for r in radii])
    # The closed form loses digits next to x0 (logarithms of nearly equal numbers).
    np.testing.assert_allclose(profile.flux, expected, rtol=1e-9)
    # Maximum of the closed form, found independently on a fine grid.
    grid = r_isco * np.linspace(1.01, 4.0, 4000)
    closed = np.array([page_thorne_kerr(r, a)[0] for r in grid])
    assert profile.r_max == pytest.approx(grid[np.argmax(closed)], rel=2e-3)
    assert profile.flux_max == pytest.approx(closed.max(), rel=1e-6)
    assert profile.efficiency == pytest.approx(1.0 - kerr_energy(r_isco, a), rel=1e-10)
    assert profile.temperature.max() <= 1.0 + 1e-12


def test_flux_vanishes_at_the_inner_edge_and_falls_as_r_cubed(schwarzschild, kerr):
    profile = dk.page_thorne_flux(schwarzschild)
    assert profile.r_in == pytest.approx(6.0, rel=1e-12) and profile.flux[0] == 0.0
    assert profile.r[-1] == pytest.approx(6.0e3) and np.all(profile.flux[1:] > 0.0)
    # Newtonian limit F -> 3 / (8 pi r^3) (independent of the closed form). The relative
    # correction falls like r^(-1/2), as the inner-edge term 1 - sqrt(r_in / r) of the
    # Newtonian disk; its coefficient (3.9 here, not sqrt(6)) contains the relativistic
    # angular momentum and energy at the ISCO.
    radii = np.array([1.0e4, 1.0e6, 1.0e8])
    newtonian = 3.0 / (8.0 * math.pi * radii**3)
    ratio = np.asarray(dk.page_thorne_flux(schwarzschild, radii).flux) / newtonian
    scaled = (1.0 - ratio) * np.sqrt(radii)
    assert np.all(ratio < 1.0) and ratio[-1] > 0.999
    np.testing.assert_allclose(scaled, scaled[-1], rtol=0.02)
    # Just outside the inner edge the flux grows linearly from zero in (r - r_in)^... it is
    # continuous: tiny steps give tiny fluxes.
    near = dk.page_thorne_flux(
        kerr, np.array([1.0 + 1e-8, 1.0 + 1e-6]) * float(bp.orbits.isco(kerr, a=0.7)), a=0.7
    ).flux
    assert 0.0 < near[0] < near[1] < 1e-8
    # Inside the inner edge the disk emits nothing.
    assert float(dk.page_thorne_flux(schwarzschild, np.array([5.0])).flux[0]) == 0.0


@pytest.mark.slow
def test_general_path_reproduces_the_static_one(rn, hayward):
    """Static metrics through as_axisymmetric() (CLAUDE.md: every general formula)."""
    for metric, params in ((rn, {"Q": 0.6}), (hayward, {"l": 0.5})):
        radii = np.array([7.0, 9.0, 20.0, 150.0])
        static = dk.page_thorne_flux(metric, radii, **params)
        general = dk.page_thorne_flux(metric.as_axisymmetric(), radii, **params)
        np.testing.assert_allclose(general.flux, static.flux, rtol=1e-10)
        assert general.r_in == pytest.approx(static.r_in, rel=1e-10)
        assert general.r_max == pytest.approx(static.r_max, rel=1e-6)
        assert general.efficiency == pytest.approx(static.efficiency, rel=1e-11)
    # A charge moves the inner edge in and raises the peak flux.
    assert dk.page_thorne_flux(rn, Q=0.6).flux_max > dk.page_thorne_flux(rn, Q=0.0).flux_max


def test_refusals(rn, kerr):
    with pytest.raises(BlackPyError, match="no innermost stable circular orbit"):
        dk.radiative_efficiency(rn, Q=1.2)  # stable orbits down to the singularity region
    two = bp.StaticSpherical(lambda r, A=-0.05: 1 - 2 / r + A * jnp.exp(-(((r - 15.0) / 1.5) ** 2)))
    with pytest.raises(BlackPyError, match="several marginally stable orbits"):
        dk.page_thorne_flux(two)
    profile = dk.page_thorne_flux(two, np.array([25.0, 40.0]), r_in=19.3)  # explicit inner edge
    assert profile.r_in == 19.3 and np.all(profile.flux > 0.0)
    with pytest.raises(BlackPyError, match="No circular orbit"):
        dk.page_thorne_flux(rn, r_in=2.0, Q=0.3)
    tilted = bp.StationaryAxisymmetric(
        lambda r, th, a=0.0: -(1 - 2 / r) + 0.01 * jnp.cos(th) / r**2,
        lambda r, th, a=0.0: 0.0 * r,
        lambda r, th, a=0.0: r**2 * jnp.sin(th) ** 2,
        lambda r, th, a=0.0: 1 / (1 - 2 / r),
        lambda r, th, a=0.0: r**2,
    )
    with pytest.raises(ReflectionSymmetryError):
        dk.radiative_efficiency(tilted)
    # No continuum-fitting likelihood is offered; the module says why.
    assert "no likelihood is offered" in " ".join(dk.__doc__.split())
