"""Metric construction, parameters, and autodiff vs symbolic derivatives (items 3, 4)."""

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy._numerics import derivative
from blackpy.errors import SymbolicUnavailableError
from blackpy.metric import CATALOGUE, Parameter, StaticSpherical

EXTRA = {
    "ReissnerNordstrom": {"Q": 0.7},
    "SchwarzschildDeSitter": {"Lambda": 3e-4},
    "Bardeen": {"q": 0.5},
    "Hayward": {"l": 0.6},
    "Schwarzschild": {},
}


def test_callable_constructor_infers_parameters():
    m = StaticSpherical(lambda r, alpha, beta=0.5: 1 - 2 / r + alpha / r**2 + beta / r**3)
    assert m.param_names == ("alpha", "beta")
    assert m.parameters[0].default is None and m.parameters[1].default == 0.5
    assert not m.is_symbolic and m.g_equals_f and m.h_is_r_squared
    assert np.isclose(m.f(2.0, alpha=1.0), 1 - 1 + 0.25 + 0.5 / 8)
    with pytest.raises(TypeError, match="no default"):
        m.f(2.0)
    with pytest.raises(TypeError, match="no parameter"):
        m.f(2.0, alpha=1.0, gamma=3.0)


def test_parameter_bounds_are_enforced():
    m = StaticSpherical(
        lambda r, a: 1 - 2 / r + a / r**2, parameters=[Parameter("a", 0.0, (0.0, 1.0))]
    )
    with pytest.raises(ValueError, match="must lie in"):
        m.f(3.0, a=1.5)
    with pytest.raises(ValueError):
        Parameter("a", 2.0, (0.0, 1.0))
    with pytest.raises(ValueError):
        StaticSpherical(lambda r, a: 1 - 2 / r, r_range=(0.0, 10.0))


def test_from_sympy_general_f_g_h():
    r, a = sp.symbols("r a", positive=True)
    m = StaticSpherical.from_sympy(
        1 - 2 / r, g=(1 - 2 / r) * (1 - a / r), h=r**2 + a**2, parameters={"a": 0.1}
    )
    assert m.is_symbolic and not m.g_equals_f and not m.h_is_r_squared
    assert np.isclose(m.g(4.0), 0.5 * (1 - 0.1 / 4))
    assert np.isclose(m.h(4.0, a=0.3), 16.09)
    assert "dt^2" in m.latex()
    with pytest.raises(ValueError, match="not listed"):
        StaticSpherical.from_sympy(1 - 2 / r + a / r**2, parameters=())


def test_symbolic_unavailable_for_callable_metric():
    m = StaticSpherical(lambda r: 1 - 2 / r)
    with pytest.raises(SymbolicUnavailableError, match="from_sympy"):
        _ = m.symbolic
    with pytest.raises(SymbolicUnavailableError):
        bp.geometry.stress_energy_symbolic(m)


def test_broadcasting_over_r_and_parameters(rn):
    r = jnp.linspace(3.0, 10.0, 5)
    q = jnp.array([[0.1], [0.5]])
    out = rn.f(r, Q=q)
    assert out.shape == (2, 5)
    np.testing.assert_allclose(out, 1 - 2 / r + q**2 / r**2, rtol=1e-15)


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_autodiff_matches_symbolic_derivatives(name):
    """Validation item 4: autodiff vs sympy derivatives agree to 1e-12 on a grid."""
    metric = CATALOGUE[name]()
    sym = metric.symbolic
    values = EXTRA[name]
    p = metric.pack(**values)
    subs = {s: values[s.name] for s in sym.params}
    grid = np.geomspace(2.5, 60.0, 40)
    f_num = lambda x: metric.f_array(x, p)
    for order in (1, 2, 3):
        exact = sp.lambdify(sym.r, sp.diff(sym.f, sym.r, order).subs(subs), "mpmath")
        expected = np.array([float(exact(sp.Float(x, 30))) for x in grid])
        got = np.array([float(derivative(f_num, order)(x)) for x in grid])
        scale = np.maximum(np.abs(expected), 1e-300)
        assert np.max(np.abs(got - expected) / scale) < 1e-12, (name, order)


@pytest.mark.parametrize("name", list(CATALOGUE))
def test_catalogue_entries_have_reference_and_flat_limit(name):
    metric = CATALOGUE[name]()
    assert metric.reference and metric.is_symbolic
    zero = {p.name: 0.0 for p in metric.parameters}
    np.testing.assert_allclose(metric.f(jnp.array([3.0, 10.0]), **zero), [1 / 3, 0.8], rtol=1e-15)


def test_asymptotic_flatness(schwarzschild, sds):
    assert schwarzschild.is_asymptotically_flat()
    assert sds.is_asymptotically_flat(Lambda=0.0)
    assert not sds.is_asymptotically_flat(Lambda=1e-4)
    deficit = StaticSpherical(lambda r, a=0.1: 1 - a - 2 / r)
    assert not deficit.is_asymptotically_flat()


def test_interpolated_metric_matches_closed_form():
    """Numeric path: a metric given only on a grid (no closed form)."""
    grid = np.geomspace(1.5, 400.0, 4000)
    fn = bp.metric.interpolated.spline(grid, 1 - 2 / grid)
    m = StaticSpherical(
        lambda r: fn(r), name="tabulated", r_range=(1.5, 400.0), asymptotically_flat=True
    )
    assert np.isclose(bp.geometry.event_horizon(m), 2.0, rtol=1e-9)
    assert np.isclose(bp.light.photon_sphere(m), 3.0, rtol=1e-8)
    assert np.isclose(bp.orbits.isco(m), 6.0, rtol=1e-6)
    assert np.isclose(bp.light.shadow_radius(m), 3 * np.sqrt(3), rtol=1e-8)


def test_future_metric_stubs_raise():
    with pytest.raises(NotImplementedError, match="MVP"):
        bp.metric.StationaryAxisymmetric()
    with pytest.raises(NotImplementedError):
        bp.metric.future.newman_janis(None)
