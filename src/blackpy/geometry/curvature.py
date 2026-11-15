r"""Curvature and effective stress-energy.

Closed-form expressions for ``ds^2 = -f dt^2 + dr^2/g + h dOmega^2``.
With primes denoting d/dr, define the four independent orthonormal-frame
curvature components

.. math::

    E_1 &= g\left(\frac{f''}{2f} - \frac{f'^2}{4f^2}\right) + \frac{g' f'}{4f}, \\
    E_2 &= \frac{g f' h'}{4 f h}, \\
    E_3 &= g\left(\frac{h''}{2h} - \frac{h'^2}{4h^2}\right) + \frac{g' h'}{4h}, \\
    E_4 &= \frac{1}{h} - \frac{g h'^2}{4 h^2}.

Then

.. math::

    G^t{}_t = 2E_3 - E_4, \quad G^r{}_r = 2E_2 - E_4, \quad
    G^\theta{}_\theta = G^\phi{}_\phi = E_1 + E_2 + E_3,

    R = -2E_1 - 4E_2 - 4E_3 + 2E_4, \qquad
    K = R_{abcd}R^{abcd} = 4E_1^2 + 8E_2^2 + 8E_3^2 + 4E_4^2 .

These were derived for BlackPy with sympy from the full tensor definitions
(:mod:`blackpy.geometry.symbolic`) and are re-verified against them, for
arbitrary undefined functions f, g, h, in the test-suite.

Stress-energy conventions
-------------------------
Signature (-,+,+,+) and Einstein equations ``G^mu_nu = 8 pi T^mu_nu`` (G = c
= 1). In the static region (f > 0, g > 0), where t is the time coordinate,

    rho = -T^t_t,   p_r = T^r_r,   p_t = T^theta_theta = T^phi_phi.

For Reissner-Nordstrom this gives ``rho = -p_r = p_t = Q^2 / (8 pi r^4)``,
the energy density of the Coulomb field in Gaussian-geometrized units.

All derivatives in the numerical functions are automatic (forward-mode ``jax.jvp``).
"""

from __future__ import annotations

import math
from typing import NamedTuple

import jax

from blackpy._numerics import derivative
from blackpy.errors import SymbolicTimeoutError
from blackpy.geometry import axisymmetric as _axi
from blackpy.geometry.symbolic import axisymmetric_tensors, line_element_tensors

#: Default time limit (seconds) of the full 4D symbolic computation for rotating metrics.
SYMBOLIC_TIMEOUT = 120.0

_EIGHT_PI = 8.0 * math.pi


class StressEnergy(NamedTuple):
    """Effective anisotropic-fluid stress-energy, in units of 1/M^2."""

    rho: jax.Array
    p_r: jax.Array
    p_t: jax.Array


def curvature_components_core(metric, p, r):
    """Orthonormal-frame curvature components ``(E1, E2, E3, E4)`` at radius ``r``."""
    f_fn = lambda x: metric.f_array(x, p)
    g_fn = lambda x: metric.g_array(x, p)
    h_fn = lambda x: metric.h_array(x, p)
    f, f1, f2 = f_fn(r), derivative(f_fn)(r), derivative(f_fn, 2)(r)
    g, g1 = g_fn(r), derivative(g_fn)(r)
    h, h1, h2 = h_fn(r), derivative(h_fn)(r), derivative(h_fn, 2)(r)
    e1 = g * (f2 / (2.0 * f) - f1**2 / (4.0 * f**2)) + g1 * f1 / (4.0 * f)
    e2 = g * f1 * h1 / (4.0 * f * h)
    e3 = g * (h2 / (2.0 * h) - h1**2 / (4.0 * h**2)) + g1 * h1 / (4.0 * h)
    e4 = 1.0 / h - g * h1**2 / (4.0 * h**2)
    return e1, e2, e3, e4


def _kretschmann_core(metric, p, r):
    e1, e2, e3, e4 = curvature_components_core(metric, p, r)
    return 4.0 * e1**2 + 8.0 * e2**2 + 8.0 * e3**2 + 4.0 * e4**2


def _ricci_scalar_core(metric, p, r):
    e1, e2, e3, e4 = curvature_components_core(metric, p, r)
    return -2.0 * e1 - 4.0 * e2 - 4.0 * e3 + 2.0 * e4


def stress_energy_core(metric, p, r):
    """``(rho, p_r, p_t)`` at radius ``r`` (jit/vmap-compatible)."""
    e1, e2, e3, e4 = curvature_components_core(metric, p, r)
    rho = -(2.0 * e3 - e4) / _EIGHT_PI
    p_r = (2.0 * e2 - e4) / _EIGHT_PI
    p_t = (e1 + e2 + e3) / _EIGHT_PI
    return StressEnergy(rho, p_r, p_t)


def _theta(theta):
    return _axi.HALF_PI if theta is None else theta


def kretschmann(metric, r, theta=None, **params):
    r"""Kretschmann scalar ``K = R_abcd R^abcd`` (numerical, autodiff).

    Works for symbolic and numeric metrics alike.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    r : float or array
        Radius in units of M.
    theta : float or array, optional
        Polar angle; used for rotating metrics only (default: the equator).
        ``r`` and ``theta`` broadcast, so ``r[:, None]`` and
        ``theta[None, :]`` give an ``(r, theta)`` grid.
    **params
        Metric parameters.

    Returns
    -------
    jax.Array
        ``K`` in units of 1/M^4. For Schwarzschild ``K = 48 / r^6``; for
        Kerr :math:`K = 48 (r^2 - a^2x^2)[(r^2 + a^2x^2)^2 - 16a^2r^2x^2] /
        (r^2 + a^2x^2)^6` with :math:`x = \cos\theta` (derived with sympy
        in the test-suite).

    Notes
    -----
    Static metrics use closed forms in ``f, g, h``; rotating metrics use the
    full tensor computation of :mod:`blackpy.geometry.axisymmetric`.
    """
    if metric.is_rotating_class:
        return metric.evaluate(_axi.kretschmann_core, params, r, _theta(theta))
    return metric.evaluate(_kretschmann_core, params, r)


def ricci_scalar(metric, r, theta=None, **params):
    """Ricci scalar ``R`` (numerical, autodiff), in units of 1/M^2.

    ``theta`` is used for rotating metrics only (default: the equator).
    """
    if metric.is_rotating_class:
        return metric.evaluate(_axi.ricci_scalar_core, params, r, _theta(theta))
    return metric.evaluate(_ricci_scalar_core, params, r)


def einstein_tensor(metric, r, theta=None, **params):
    """Mixed Einstein tensor ``G^mu_nu`` (numerical, full 4D autodiff computation).

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
        A static metric is converted with ``as_axisymmetric()``.
    r, theta : float or array
        Position (``theta`` defaults to the equator).
    **params
        Metric parameters.

    Returns
    -------
    jax.Array
        Shape ``(..., 4, 4)``, coordinates ``(t, r, theta, phi)``, in units
        of 1/M^2.
    """
    if not metric.is_rotating_class:
        metric = metric.as_axisymmetric()
    return metric.evaluate(_axi.einstein_core, params, r, _theta(theta))


def stress_energy(metric, r, theta=None, **params):
    """Effective stress-energy from ``G^mu_nu = 8 pi T^mu_nu`` (numerical).

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    r : float or array
        Radius in units of M. For static metrics the result is meaningful
        as (rho, p_r, p_t) only in the static region, i.e. outside the event
        horizon.
    theta : float or array, optional
        Polar angle; rotating metrics only (default: the equator).
    **params
        Metric parameters.

    Returns
    -------
    StressEnergy or jax.Array
        Static metrics: ``rho = -T^t_t``, ``p_r = T^r_r``,
        ``p_t = T^theta_theta`` in units of 1/M^2. Rotating metrics: the
        mixed tensor ``T^mu_nu`` of shape ``(..., 4, 4)``, which is not
        diagonal in general; use :func:`blackpy.geometry.effective_source`
        for the energy density, principal pressures and source type.
    """
    if metric.is_rotating_class:
        return einstein_tensor(metric, r, theta, **params) / _EIGHT_PI
    return metric.evaluate(stress_energy_core, params, r)


# ----------------------------------------------------------------------------- symbolic


def _symbolic_tensors(metric, timeout: float | None = SYMBOLIC_TIMEOUT) -> dict:
    cache = metric._symbolic_cache
    if "tensors" not in cache:
        s = metric.symbolic
        if metric.is_rotating_class:
            try:
                cache["tensors"] = axisymmetric_tensors(
                    *s.components, s.r, s.theta, timeout=timeout
                )
            except TimeoutError:
                raise SymbolicTimeoutError(
                    f"The symbolic 4D curvature computation for '{metric.name}' did not finish "
                    f"within {timeout:g} s. Use the numerical functions instead "
                    "(blackpy.geometry.kretschmann, einstein_tensor, effective_source): they "
                    "give the same quantities on an (r, theta) grid by automatic "
                    "differentiation. To keep trying symbolically, pass a larger timeout=... "
                    "(or timeout=None for no limit)."
                ) from None
        else:
            cache["tensors"] = line_element_tensors(s.f, s.g, s.h, s.r)
    return cache["tensors"]


def einstein_tensor_symbolic(metric, timeout: float | None = SYMBOLIC_TIMEOUT):
    """Mixed Einstein tensor ``G^mu_nu`` as a 4x4 sympy matrix, coordinates (t, r, theta, phi).

    For rotating metrics this is the full 4D symbolic computation, which can
    be slow (about 10 s for Kerr, 30 s for Kerr-Newman, much longer for
    deformed metrics); it is limited to ``timeout`` seconds. The result is
    cached on the metric.

    Raises
    ------
    SymbolicUnavailableError
        If the metric was defined with numerical callables.
    SymbolicTimeoutError
        If the time limit is exceeded (rotating metrics).
    """
    return _symbolic_tensors(metric, timeout)["einstein_mixed"]


def kretschmann_symbolic(metric, timeout: float | None = SYMBOLIC_TIMEOUT):
    """Kretschmann scalar as a sympy expression (symbolic metrics only)."""
    return _symbolic_tensors(metric, timeout)["kretschmann"]


def ricci_scalar_symbolic(metric, timeout: float | None = SYMBOLIC_TIMEOUT):
    """Ricci scalar as a sympy expression (symbolic metrics only)."""
    return _symbolic_tensors(metric, timeout)["ricci_scalar"]


def stress_energy_symbolic(metric, timeout: float | None = SYMBOLIC_TIMEOUT):
    """Effective stress-energy as sympy expressions (symbolic metrics only).

    Returns
    -------
    dict or sympy.Matrix
        Static metrics: a dict with keys ``"rho"``, ``"p_r"``, ``"p_t"``,
        where ``rho = -G^t_t / (8 pi)``, ``p_r = G^r_r / (8 pi)``,
        ``p_t = G^theta_theta / (8 pi)``. Rotating metrics: the 4x4 matrix
        ``T^mu_nu = G^mu_nu / (8 pi)``.
    """
    import sympy as sp

    einstein = einstein_tensor_symbolic(metric, timeout)
    if metric.is_rotating_class:
        return einstein / (8 * sp.pi)
    return {
        "rho": sp.simplify(-einstein[0, 0] / (8 * sp.pi)),
        "p_r": sp.simplify(einstein[1, 1] / (8 * sp.pi)),
        "p_t": sp.simplify(einstein[2, 2] / (8 * sp.pi)),
    }


def is_vacuum_symbolic(metric, timeout: float | None = SYMBOLIC_TIMEOUT) -> bool:
    """True if the metric solves the vacuum Einstein equations ``G^mu_nu = 0`` exactly."""
    import sympy as sp

    einstein = einstein_tensor_symbolic(metric, timeout)
    if metric.is_rotating_class:
        return all(sp.simplify(component) == 0 for component in einstein)
    return all(sp.simplify(einstein[i, i]) == 0 for i in range(4))
