r"""Rotating metrics generated from static ones (Azreg-Ainou 2014).

Reference
---------
M. Azreg-Ainou, "Generating rotating regular black hole solutions without
complexification", Phys. Rev. D 90 (2014) 064041, arXiv:1405.2569.

The procedure (transcribed from the arXiv version)
--------------------------------------------------
Start from the static metric of Eq. (1) of the paper, which in BlackPy's
notation and signature is

.. math::

    ds^2 = -f\,dt^2 + \frac{dr^2}{g} + h\,d\Omega^2
    \qquad (G_{\rm AA} = f,\; F_{\rm AA} = g,\; H_{\rm AA} = h).

The modified Newman-Janis algorithm, which drops the complexification of
``r``, gives in Boyer-Lindquist coordinates (Eqs. (16) and (18), written
here with signature (-,+,+,+))

.. math::

    ds^2 = \frac{\Psi}{\rho^2}\left[-\left(1 - \frac{2\tilde f}{\rho^2}\right)dt^2
           + \frac{\rho^2}{\Delta}\,dr^2
           - \frac{4a\tilde f\sin^2\theta}{\rho^2}\,dt\,d\phi
           + \rho^2 d\theta^2 + \frac{\Sigma\sin^2\theta}{\rho^2}\,d\phi^2\right],

.. math::

    K = \sqrt{g/f}\;h, \quad \rho^2 = K + a^2\cos^2\theta, \quad
    2\tilde f = K - g h, \quad \Delta = g h + a^2, \quad
    \Sigma = (K + a^2)^2 - a^2\Delta\sin^2\theta .

:math:`\Psi(r, \theta, a)` is not fixed by the algorithm. For ``g = f`` and
``h = r^2`` the paper shows (Eq. (23)) that
:math:`\Psi = r^2 + a^2\cos^2\theta = \rho^2` solves the two equations that
make the source an imperfect fluid, which gives its Eq. (24). BlackPy uses

.. math::

    \Psi = h + a^2\cos^2\theta

by default: it reproduces Eq. (23) in that case and satisfies the paper's
condition :math:`\lim_{a\to0}\Psi = h` (Eq. (7)) in general. For seeds with
``g != f`` or ``h != r^2`` this default is a *choice* that the paper does not
make; pass ``psi=`` to use another conformal factor. :math:`\Psi` does not
affect null geodesics (shadows) but it does affect timelike orbits and the
effective source.
"""

from __future__ import annotations

import math
from collections.abc import Callable

import jax.numpy as jnp

from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = "Azreg-Ainou (2014), Phys. Rev. D 90, 064041 (arXiv:1405.2569), Eqs. (16), (18)"


def _components(f, g, h, a, cos2, sin2, psi, sqrt):
    """Metric components from the seed functions (works for sympy and JAX objects)."""
    k = sqrt(g / f) * h
    rho2 = k + a**2 * cos2
    two_f = k - g * h
    delta = g * h + a**2
    sigma = (k + a**2) ** 2 - a**2 * delta * sin2
    conformal = psi / rho2
    return (
        -conformal * (1 - two_f / rho2),
        -conformal * a * two_f * sin2 / rho2,
        conformal * sigma * sin2 / rho2,
        psi / delta,
        psi,
    )


def rotating_from_static(
    static_metric: StaticSpherical,
    a: float | None = 0.0,
    *,
    psi: Callable | None = None,
    name: str | None = None,
) -> StationaryAxisymmetric:
    r"""Rotating counterpart of a static metric (Azreg-Ainou 2014, no complexification).

    .. warning::

        The result is a **model**, not a solution of any field equations. The
        algorithm guarantees neither that the rotating metric solves the
        equations that the static seed solves, nor that its source is
        physical. Check the effective source with :mod:`blackpy.geometry`
        (``stress_energy``, ``energy_conditions``, ``report``); do not assume
        it. Known exact cases, used as tests: Schwarzschild gives Kerr and
        Reissner-Nordstrom gives Kerr-Newman.

    Parameters
    ----------
    static_metric : StaticSpherical
        The seed, :math:`ds^2 = -f dt^2 + dr^2/g + h d\Omega^2`. Its
        parameters are kept; a spin parameter ``a`` is prepended.
    a : float or None
        Default value of the new spin parameter ``a`` (in units of M).
        ``None`` means "no default".
    psi : callable, optional
        Conformal factor :math:`\Psi` as ``psi(r, theta, a, *seed_params)``
        (JAX-traceable). Default: :math:`h(r) + a^2\cos^2\theta`, see the
        module docstring. With a user-supplied ``psi`` the metric is numeric
        (no symbolic form).
    name : str, optional
        Display name. Default: ``"rotating <seed name>"``.

    Returns
    -------
    StationaryAxisymmetric
        With parameters ``(a, *seed parameters)``. It has a symbolic form if
        the seed has one and ``psi`` is not given. Null geodesics are
        separable by construction, for any :math:`\Psi`.

    Raises
    ------
    TypeError
        If the seed is not a :class:`StaticSpherical` metric.
    ValueError
        If the seed already has a parameter named ``a``.
    """
    if not isinstance(static_metric, StaticSpherical):
        raise TypeError(
            "rotating_from_static needs a StaticSpherical seed metric, got "
            f"{type(static_metric).__name__}."
        )
    if "a" in static_metric.param_names:
        raise ValueError(
            "The seed metric already has a parameter called 'a'; rename it before generating "
            "the rotating metric, which uses 'a' for the spin."
        )
    spin = Parameter(
        "a",
        None if a is None else float(a),
        (-math.inf, math.inf),
        latex="a",
        description="spin a/M",
        role="spin",
    )
    parameters = [spin, *static_metric.parameters]
    options = {
        "parameters": parameters,
        "name": name or f"rotating {static_metric.name}",
        "asymptotically_flat": static_metric.asymptotically_flat,
        "r_range": static_metric.r_range,
        "reference": f"{REFERENCE}; seed: {static_metric.reference or static_metric.name}",
    }

    if static_metric.is_symbolic and psi is None:
        import sympy as sp

        s = static_metric.symbolic
        theta = sp.Symbol("theta", real=True)
        a_sym = sp.Symbol("a", real=True)
        cos2, sin2 = sp.cos(theta) ** 2, sp.sin(theta) ** 2
        exprs = _components(s.f, s.g, s.h, a_sym, cos2, sin2, s.h + a_sym**2 * cos2, sp.sqrt)
        metric = StationaryAxisymmetric.from_sympy(
            *exprs,
            r=s.r,
            theta=theta,
            **{**options, "parameters": parameters},
        )
        # from_sympy orders the symbols by the parameter list: (a, *seed parameters).
    else:
        f_fn, g_fn, h_fn = static_metric._f_user, static_metric._g_user, static_metric._h_user

        def component(index):
            def fn(r, theta, a_value, *seed):
                cos2, sin2 = jnp.cos(theta) ** 2, jnp.sin(theta) ** 2
                h = h_fn(r, *seed)
                factor = h + a_value**2 * cos2 if psi is None else psi(r, theta, a_value, *seed)
                return _components(
                    f_fn(r, *seed), g_fn(r, *seed), h, a_value, cos2, sin2, factor, jnp.sqrt
                )[index]

            return fn

        metric = StationaryAxisymmetric(*[component(i) for i in range(5)], **options)
    metric._declared["separable"] = True
    metric.seed = static_metric
    return metric
