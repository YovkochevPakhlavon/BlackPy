"""StationaryAxisymmetric, the rotating catalogue and the rotating generator (B1)."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy.errors import SymbolicUnavailableError
from blackpy.metric import StationaryAxisymmetric, rotating_from_static

R = np.array([2.5, 4.0, 9.0])[:, None]
THETA = np.array([0.4, 1.1, math.pi / 2, 2.3])[None, :]
COMPONENTS = ("g_tt", "g_tphi", "g_phiphi", "g_rr", "g_thth")


def components(metric, **params):
    return np.stack([np.asarray(getattr(metric, c)(R, THETA, **params)) for c in COMPONENTS])


def kerr_callables():
    def sigma(r, theta, a):
        return r**2 + a**2 * jnp.cos(theta) ** 2

    return (
        lambda r, theta, a=0.0: -(1 - 2 * r / sigma(r, theta, a)),
        lambda r, theta, a=0.0: -2 * a * r * jnp.sin(theta) ** 2 / sigma(r, theta, a),
        lambda r, theta, a=0.0: (
            (r**2 + a**2 + 2 * a**2 * r * jnp.sin(theta) ** 2 / sigma(r, theta, a))
            * jnp.sin(theta) ** 2
        ),
        lambda r, theta, a=0.0: sigma(r, theta, a) / (r**2 - 2 * r + a**2),
        lambda r, theta, a=0.0: sigma(r, theta, a),
    )


def test_callable_constructor_matches_catalogue_kerr(kerr):
    numeric = StationaryAxisymmetric(*kerr_callables(), name="Kerr (callables)")
    assert numeric.param_names == ("a",) and numeric.parameters[0].default == 0.0
    assert not numeric.is_symbolic and not numeric.is_spherically_symmetric
    np.testing.assert_allclose(components(numeric, a=0.7), components(kerr, a=0.7), rtol=1e-14)
    with pytest.raises(SymbolicUnavailableError, match="from_sympy"):
        _ = numeric.symbolic
    # Numerical checks replace the symbolic ones.
    assert numeric.is_reflection_symmetric and numeric.is_separable
    assert numeric.is_asymptotically_flat(a=0.9)
    assert "numeric" in repr(numeric) and "symbolic" in repr(kerr)


def test_constructor_errors():
    fns = kerr_callables()
    with pytest.raises(TypeError, match="from_sympy"):
        StationaryAxisymmetric(sp.Symbol("x"), *fns[1:])
    with pytest.raises(ValueError, match="reserved"):
        StationaryAxisymmetric(*fns, parameters=["theta"])
    with pytest.raises(ValueError, match="r_range"):
        StationaryAxisymmetric(*fns, r_range=(2.0, 1.0))
    r, a = sp.symbols("r a")
    with pytest.raises(ValueError, match="theta"):
        StationaryAxisymmetric.from_sympy(-1 + a / r, 0, r**2, 1, r**2)
    theta, b = sp.symbols("theta b")
    with pytest.raises(ValueError, match="not listed"):
        StationaryAxisymmetric.from_sympy(
            -1 + b / r, 0, r**2 * sp.sin(theta) ** 2, 1, r**2, parameters=["a"]
        )


def test_from_sympy_latex_and_parameter_handling(kerr):
    assert kerr.is_symbolic and kerr.spin_parameter == "a"
    assert kerr.latex().startswith("ds^2 = ") and r"d\phi^2" in kerr.latex()
    with pytest.raises(TypeError, match="no parameter"):
        kerr.g_tt(3.0, 1.0, Q=0.1)
    assert kerr.g_tt(R, THETA, a=np.array(0.3)).shape == (3, 4)
    assert np.asarray(kerr.g_tt(3.0, 1.0, a=np.linspace(0, 0.9, 5))).shape == (5,)


def test_kerr_matches_independent_form(kerr):
    """Kerr in the 'Delta / Sigma' form of the line element, written independently."""
    a = 0.8
    sigma = R**2 + a**2 * np.cos(THETA) ** 2
    delta = R**2 - 2 * R + a**2
    s2 = np.sin(THETA) ** 2
    # ds^2 = -(Delta/Sigma)(dt - a s2 dphi)^2 + (s2/Sigma)((r^2+a^2) dphi - a dt)^2 + ...
    expected = np.stack(
        [
            -(delta - a**2 * s2) / sigma,
            (delta * a * s2 - a * s2 * (R**2 + a**2)) / sigma,
            (-delta * a**2 * s2**2 + s2 * (R**2 + a**2) ** 2) / sigma,
            sigma / delta,
            sigma + 0 * R,
        ]
    )
    np.testing.assert_allclose(components(kerr, a=a), expected, rtol=1e-13)


def test_kerr_newman_limits(kerr, kerr_newman, rn):
    """Kerr-Newman reduces to Kerr for Q = 0 and to Reissner-Nordstrom for a = 0."""
    np.testing.assert_allclose(
        components(kerr_newman, a=0.6, Q=0.0), components(kerr, a=0.6), rtol=1e-14
    )
    np.testing.assert_allclose(
        components(kerr_newman, a=0.0, Q=0.5),
        components(rn.as_axisymmetric(), Q=0.5),
        rtol=1e-14,
        atol=1e-300,
    )
    s = kerr_newman.symbolic
    a_sym, q_sym = s.params
    for expr, target in zip(s.components, bp.metric.Kerr().symbolic.components, strict=True):
        assert sp.simplify(expr.subs(q_sym, 0) - target) == 0


def test_johannsen_reduces_to_kerr_and_has_documented_domain(kerr, johannsen):
    assert johannsen.param_names == ("a", "alpha13", "alpha22", "alpha52", "epsilon3")
    assert [p.role for p in johannsen.parameters] == ["spin"] + ["deformation"] * 4
    np.testing.assert_allclose(components(johannsen, a=0.7), components(kerr, a=0.7), rtol=1e-13)
    assert not np.allclose(components(johannsen, a=0.7, alpha13=0.5), components(kerr, a=0.7))
    # Lower bounds of Eqs. (74), (75), (77), (78): -(1 + sqrt(1 - a^2))^n with n = 3, 2, 2, 3.
    b = 1 + math.sqrt(1 - 0.6**2)
    for name, power in (("alpha13", 3), ("alpha22", 2), ("alpha52", 2), ("epsilon3", 3)):
        assert bool(johannsen.in_domain(a=0.6, **{name: -(b**power) + 1e-9}))
        assert not bool(johannsen.in_domain(a=0.6, **{name: -(b**power) - 1e-9}))
    assert not bool(johannsen.in_domain(a=1.01))
    # g_rr depends on alpha52 only through A_5: g_rr(Kerr) / g_rr = A_5 = 1 + alpha52 / r^2.
    ratio = kerr.g_rr(4.0, 1.0, a=0.3) / johannsen.g_rr(4.0, 1.0, a=0.3, alpha52=0.8)
    assert math.isclose(float(ratio), 1 + 0.8 / 16, rel_tol=1e-13)


@pytest.mark.parametrize("name", list(bp.metric.ROTATING_CATALOGUE))
def test_rotating_catalogue_entries(name):
    metric = bp.metric.ROTATING_CATALOGUE[name]()
    assert metric.reference and metric.is_symbolic and metric.is_rotating_class
    assert metric.spin_parameter == "a" and metric.is_asymptotically_flat()
    # The declared properties agree with the actual checks.
    assert metric.is_reflection_symmetric and metric.is_separable
    assert metric._reflection_symmetric_symbolic() is True
    assert metric.check_reflection_symmetry() and metric.check_separability()
    # Schwarzschild limit: all parameters at their defaults.
    schwarzschild = bp.metric.Schwarzschild().as_axisymmetric()
    np.testing.assert_allclose(
        components(metric), components(schwarzschild), rtol=1e-14, atol=1e-300
    )
    assert bool(metric.in_domain()) and not bool(metric.in_domain(a=1.5))


def test_reflection_symmetry_is_detected_symbolically_and_numerically(kerr):
    s = kerr.symbolic
    (a,) = s.params
    skewed = s.g_tt * (1 + sp.Rational(1, 10) * sp.cos(s.theta) / s.r**2)
    asymmetric = StationaryAxisymmetric.from_sympy(
        skewed, s.g_tphi, s.g_phiphi, s.g_rr, s.g_thth, r=s.r, theta=s.theta, parameters=[a.name]
    )
    assert asymmetric._reflection_symmetric_symbolic() is None  # sympy alone does not decide
    assert not asymmetric.check_reflection_symmetry(a=0.5)
    assert not asymmetric.is_reflection_symmetric
    fns = kerr_callables()
    numeric = StationaryAxisymmetric(
        lambda r, theta, a=0.0: fns[0](r, theta, a) * (1 + 0.1 * jnp.cos(theta) / r**2), *fns[1:]
    )
    assert not numeric.is_reflection_symmetric


def test_separability_check_rejects_non_separable_metric(kerr):
    fns = kerr_callables()
    deformed = StationaryAxisymmetric(
        lambda r, theta, a=0.0: fns[0](r, theta, a) * (1 + 0.2 * jnp.cos(theta) ** 2 / r**3),
        *fns[1:],
        name="non-separable test metric",
    )
    assert deformed.is_reflection_symmetric
    assert not deformed.check_separability(a=0.5) and not deformed.is_separable
    assert kerr.check_separability(a=0.99)


def test_static_metric_as_axisymmetric(rn, schwarzschild):
    ax = rn.as_axisymmetric()
    assert ax.param_names == rn.param_names and ax.is_symbolic and ax.is_rotating_class
    q, r, theta = 0.4, 5.0, 1.0
    f = float(rn.f(r, Q=q))
    assert math.isclose(float(ax.g_tt(r, theta, Q=q)), -f, rel_tol=1e-15)
    assert float(ax.g_tphi(r, theta, Q=q)) == 0.0
    assert math.isclose(float(ax.g_rr(r, theta, Q=q)), 1 / f, rel_tol=1e-15)
    assert math.isclose(float(ax.g_phiphi(r, theta, Q=q)), 25 * math.sin(theta) ** 2, rel_tol=1e-15)
    numeric = bp.StaticSpherical(lambda r, Q=0.0: 1 - 2 / r + Q**2 / r**2).as_axisymmetric()
    np.testing.assert_allclose(
        components(numeric, Q=q), components(ax, Q=q), rtol=1e-14, atol=1e-300
    )
    assert not numeric.is_symbolic


def test_rotating_from_static_gives_kerr_and_kerr_newman(schwarzschild, rn, kerr, kerr_newman):
    """Azreg-Ainou (2014): Schwarzschild -> Kerr and Reissner-Nordstrom -> Kerr-Newman, exactly."""
    for seed, target in ((schwarzschild, kerr), (rn, kerr_newman)):
        generated = rotating_from_static(seed)
        assert generated.param_names == target.param_names
        assert generated.parameters[0].role == "spin" and generated.is_separable
        for expr, expected in zip(
            generated.symbolic.components, target.symbolic.components, strict=True
        ):
            assert sp.simplify(expr - expected) == 0
        assert "Azreg-Ainou" in generated.reference and generated.seed is seed
    np.testing.assert_allclose(
        components(rotating_from_static(rn), a=0.5, Q=0.6),
        components(kerr_newman, a=0.5, Q=0.6),
        rtol=1e-13,
    )


def test_rotating_from_static_general_seeds(bardeen, schwarzschild):
    rotating = rotating_from_static(bardeen, a=0.3)
    assert rotating.param_names == ("a", "q") and rotating.parameters[0].default == 0.3
    assert rotating.is_reflection_symmetric and rotating.check_separability(a=0.6, q=0.4)
    # a = 0 gives back the static seed.
    np.testing.assert_allclose(
        components(rotating, a=0.0, q=0.5),
        components(bardeen.as_axisymmetric(), q=0.5),
        rtol=1e-13,
        atol=1e-300,
    )
    # Numeric seed (callables) and a user-supplied conformal factor.
    seed = bp.StaticSpherical(lambda r, Q=0.0: 1 - 2 / r + Q**2 / r**2)
    numeric = rotating_from_static(seed)
    assert not numeric.is_symbolic
    np.testing.assert_allclose(
        components(numeric, a=0.5, Q=0.6),
        components(bp.metric.KerrNewman(), a=0.5, Q=0.6),
        rtol=1e-13,
    )
    conformal = rotating_from_static(
        schwarzschild, psi=lambda r, theta, a: 2.0 * (r**2 + a**2 * jnp.cos(theta) ** 2)
    )
    np.testing.assert_allclose(
        components(conformal, a=0.5), 2.0 * components(bp.metric.Kerr(), a=0.5), rtol=1e-13
    )
    assert conformal.check_separability(a=0.5)  # null geodesics do not see the conformal factor
    # f != g seed: K = sqrt(g/f) h enters; the a = 0 limit must still be the seed.
    general = bp.StaticSpherical(
        lambda r: (1 - 2 / r) * (1 + 1 / r**2), lambda r: 1 - 2 / r, name="f != g"
    )
    np.testing.assert_allclose(
        components(rotating_from_static(general), a=0.0),
        components(general.as_axisymmetric()),
        rtol=1e-13,
        atol=1e-300,
    )
    with pytest.raises(TypeError, match="StaticSpherical"):
        rotating_from_static(bp.metric.Kerr())
    with pytest.raises(ValueError, match="already has a parameter"):
        rotating_from_static(bp.StaticSpherical(lambda r, a=0.0: 1 - 2 / r + a / r**2))


def test_remaining_metric_stubs_raise():
    with pytest.raises(NotImplementedError, match="rotating_from_static"):
        bp.metric.future.newman_janis(None)
    # The RZ and KRZ parametrisations are implemented since v0.3 (tests/test_parametrized.py);
    # their reserved names are deprecated aliases of the implementations.
    with pytest.warns(DeprecationWarning, match="RezzollaZhidenko"):
        assert bp.metric.future.rezzolla_zhidenko is bp.metric.RezzollaZhidenko
    with pytest.warns(DeprecationWarning, match="KRZ"):
        assert bp.metric.future.konoplya_rezzolla_zhidenko is bp.metric.KRZ
