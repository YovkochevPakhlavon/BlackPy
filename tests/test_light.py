"""Photon spheres and shadows (items 1, 2)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest

import blackpy as bp
from blackpy import light as lt
from blackpy.errors import NotAsymptoticallyFlatError


def test_schwarzschild_photon_sphere_and_shadow(schwarzschild):
    """Validation item 1: r_ph = 3, b_c = 3 sqrt(3)."""
    assert math.isclose(float(lt.photon_sphere(schwarzschild)), 3.0, rel_tol=1e-14)
    assert math.isclose(
        float(lt.critical_impact_parameter(schwarzschild)), 3 * math.sqrt(3), rel_tol=1e-14
    )
    assert math.isclose(float(lt.shadow_radius(schwarzschild)), 3 * math.sqrt(3), rel_tol=1e-14)
    (ps,) = lt.photon_spheres(schwarzschild)
    assert not ps.stable and math.isclose(ps.impact_parameter, 3 * math.sqrt(3), rel_tol=1e-14)


def test_rn_photon_sphere(rn):
    """Validation item 2: r_ph = (3 + sqrt(9 - 8 Q^2)) / 2."""
    q = jnp.linspace(0.0, 1.0, 21)
    expected = (3 + jnp.sqrt(9 - 8 * q**2)) / 2
    np.testing.assert_allclose(lt.photon_sphere(rn, Q=q), expected, rtol=1e-13)
    f_ph = 1 - 2 / expected + q**2 / expected**2
    np.testing.assert_allclose(
        lt.critical_impact_parameter(rn, Q=q), expected / jnp.sqrt(f_ph), rtol=1e-13
    )


def test_naked_rn_has_stable_and_unstable_photon_sphere(rn):
    spheres = lt.photon_spheres(rn, Q=1.05)
    disc = math.sqrt(9 - 8 * 1.05**2)
    assert [s.stable for s in spheres] == [True, False]
    assert math.isclose(spheres[0].radius, (3 - disc) / 2, rel_tol=1e-12)
    assert math.isclose(spheres[1].radius, (3 + disc) / 2, rel_tol=1e-12)
    # Q^2 > 9/8: no photon sphere at all, reported as NaN / empty list.
    assert lt.photon_spheres(rn, Q=1.2) == [] and np.isnan(lt.photon_sphere(rn, Q=1.2))


def test_static_observer_at_finite_radius(schwarzschild):
    """Synge's formula: sin^2(alpha) = 27 (1 - 2/r) / r^2."""
    r_obs = jnp.array([3.5, 5.0, 10.0, 100.0])
    expected = jnp.arcsin(jnp.sqrt(27 * (1 - 2 / r_obs) / r_obs**2))
    np.testing.assert_allclose(lt.shadow_angular_radius(schwarzschild, r_obs), expected, rtol=1e-12)
    # At the photon sphere the shadow covers exactly half of the sky.
    assert math.isclose(
        float(lt.shadow_angular_radius(schwarzschild, 3.0)), math.pi / 2, rel_tol=1e-7
    )
    # Inside the photon sphere it covers more than half.
    inside = float(lt.shadow_angular_radius(schwarzschild, 2.5))
    assert math.isclose(
        inside, math.pi - math.asin(math.sqrt(27 * (1 - 2 / 2.5) / 2.5**2)), rel_tol=1e-12
    )
    # Far away the finite-distance shadow radius tends to b_c.
    assert math.isclose(float(lt.shadow_radius(schwarzschild, 1e7)), 3 * math.sqrt(3), rel_tol=1e-6)
    assert np.isnan(lt.shadow_angular_radius(schwarzschild, 1.5))  # inside the horizon


def test_sds_shadow_requires_finite_static_observer(sds):
    lam = 0.01
    assert math.isclose(float(lt.photon_sphere(sds, Lambda=lam)), 3.0, rel_tol=1e-13)
    b_c = float(lt.critical_impact_parameter(sds, Lambda=lam))
    assert math.isclose(b_c, 3 * math.sqrt(3) / math.sqrt(1 - 9 * lam), rel_tol=1e-13)
    with pytest.raises(NotAsymptoticallyFlatError, match="r_obs"):
        lt.shadow_radius(sds, Lambda=lam)
    r_obs = 10.0
    f_obs = 1 - 2 / r_obs - lam * r_obs**2 / 3
    alpha = float(lt.shadow_angular_radius(sds, r_obs, Lambda=lam))
    assert math.isclose(math.sin(alpha), b_c * math.sqrt(f_obs) / r_obs, rel_tol=1e-12)
    assert math.isclose(
        float(lt.shadow_radius(sds, r_obs, Lambda=lam)), b_c * math.sqrt(f_obs), rel_tol=1e-12
    )
    with pytest.raises(ValueError, match="static region"):
        lt.shadow_radius(sds, 30.0, Lambda=lam)  # beyond the cosmological horizon
    # Lambda = 0 is flat, so the shadow at infinity is defined again.
    assert math.isclose(float(lt.shadow_radius(sds, Lambda=0.0)), 3 * math.sqrt(3), rel_tol=1e-13)


def test_shadow_shrinks_with_charge(rn, bardeen, hayward):
    for metric, name, top in ((rn, "Q", 1.0), (bardeen, "q", 0.76), (hayward, "l", 0.76)):
        values = np.asarray(lt.shadow_radius(metric, **{name: jnp.linspace(0.0, top, 30)}))
        assert np.all(np.diff(values) < 0) and math.isclose(
            values[0], 3 * math.sqrt(3), rel_tol=1e-12
        )
    assert math.isclose(
        float(lt.shadow_radius(rn, Q=1.0)), 4.0, rel_tol=1e-12
    )  # extremal RN: b_c = 4


def test_future_light_stubs_raise():
    with pytest.raises(NotImplementedError, match="MVP"):
        lt.deflection_angle(None, 5.0)
    with pytest.raises(NotImplementedError):
        lt.shadow_in_plasma(None, None)
    # Out of scope on purpose (v0.3 roadmap): the messages must not promise a release.
    with pytest.raises(NotImplementedError, match="not scheduled") as info:
        bp.qnm.quasinormal_modes(None)
    assert "roadmap" in str(info.value) and "v0.3" not in str(info.value)
    for stub in (bp.thermo.hawking_temperature, bp.thermo.entropy, bp.thermo.heat_capacity):
        with pytest.raises(NotImplementedError, match="not scheduled") as info:
            stub(None)
        assert "roadmap" in str(info.value) and "v0.3" not in str(info.value)
    assert "not scheduled" in bp.qnm.__doc__ and "not scheduled" in bp.thermo.__doc__
    with pytest.raises(TypeError):
        lt.PlasmaModel()
