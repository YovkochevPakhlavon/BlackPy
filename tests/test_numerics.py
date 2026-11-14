"""Root finding, derivatives and quadrature (validation item 9, toolbox part)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from blackpy._numerics import derivative, find_roots, gauss_legendre, refine_root


def test_derivative_orders():
    fn = lambda x: jnp.sin(x) * x**3
    x = 0.7
    assert np.isclose(derivative(fn, 0)(x), np.sin(x) * x**3, rtol=1e-15)
    exact3 = -np.cos(x) * x**3 - 9 * np.sin(x) * x**2 + 18 * np.cos(x) * x + 6 * np.sin(x)
    assert np.isclose(derivative(fn, 3)(x), exact3, rtol=1e-13)
    with pytest.raises(ValueError):
        derivative(fn, -1)


def test_refine_root_simple_and_invalid_bracket():
    fn = lambda x: x**3 - 2.0
    assert np.isclose(refine_root(fn, 1.0, 2.0), 2.0 ** (1 / 3), rtol=1e-15)
    assert np.isnan(refine_root(fn, 2.0, 3.0))  # no sign change
    assert float(refine_root(lambda x: x - 1.0, 1.0, 2.0)) == 1.0  # root at the end point


def test_find_all_roots_of_polynomial():
    roots_true = np.array([1.0, 2.0, 3.0, 5.0, 7.0])
    fn = lambda x: jnp.prod(x - jnp.asarray(roots_true))
    roots, degenerate = find_roots(fn, 0.1, 100.0, max_roots=6)
    np.testing.assert_allclose(roots[:5], roots_true, rtol=1e-13)
    assert np.isnan(roots[5]) and not degenerate.any()


def test_close_pair_inside_one_scan_cell():
    """Two roots inside one scan cell are invisible to a sign-change scan but must be found."""
    fn = lambda x: (x - 3.0) * (x - 3.001)
    plain, _ = find_roots(fn, 0.1, 100.0, detect_extrema=False)
    assert np.isnan(plain).all()
    roots, degenerate = find_roots(fn, 0.1, 100.0)
    np.testing.assert_allclose(roots[:2], [3.0, 3.001], rtol=1e-12)
    assert not degenerate[:2].any() and np.isnan(roots[2:]).all()


def test_pair_closer_than_tolerance_is_reported_as_degenerate():
    """|fn| <= degenerate_atol at the extremum: the pair is reported as one double root."""
    roots, degenerate = find_roots(lambda x: (x - 3.0) * (x - 3.000001), 0.1, 100.0)
    assert np.isclose(roots[0], 3.0000005, rtol=1e-9) and bool(degenerate[0])
    assert np.isnan(roots[1:]).all()


@pytest.mark.parametrize("sign", [1.0, -1.0])
def test_degenerate_root(sign):
    roots, degenerate = find_roots(lambda x: sign * (x - 3.0) ** 2, 0.1, 100.0)
    assert np.isclose(roots[0], 3.0, rtol=1e-12) and bool(degenerate[0])
    assert np.isnan(roots[1:]).all()


def test_no_roots_and_truncation_keeps_outermost():
    roots, _ = find_roots(lambda x: x**2 + 1.0, 0.1, 100.0)
    assert np.isnan(roots).all()
    roots, _ = find_roots(jnp.sin, 1.0, 20.0, log_spacing=False, max_roots=3)
    np.testing.assert_allclose(roots, np.pi * np.array([4, 5, 6]), rtol=1e-13)


def test_find_roots_is_jit_and_vmap_compatible():
    solve = jax.jit(jax.vmap(lambda c: find_roots(lambda x: x**2 - c, 0.1, 100.0)[0]))
    c = jnp.array([1.0, 4.0, 9.0])
    np.testing.assert_allclose(solve(c)[:, 0], [1.0, 2.0, 3.0], rtol=1e-14)


def test_gauss_legendre():
    assert np.isclose(gauss_legendre(jnp.sin, 0.0, np.pi), 2.0, rtol=1e-14)
    assert np.isclose(gauss_legendre(lambda x: x**5, 0.0, 2.0, n=3), 64.0 / 6.0, rtol=1e-14)
