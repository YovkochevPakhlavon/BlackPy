r"""Deflection of light by a static, spherically symmetric metric.

Exact deflection angle
----------------------
For :math:`ds^2 = -f\,dt^2 + dr^2/g + h\,d\Omega^2` a light ray with closest
approach :math:`r_0` is deflected by (Bozza 2002, Phys. Rev. D 66, 103001,
Eqs. (6)-(7); with a plasma: Perlick, Tsupko & Bisnovatyi-Kogan 2015,
Eq. (20))

.. math::

    \alpha(r_0) = 2\int_{r_0}^{\infty}
        \frac{dr}{\sqrt{g\,h\,\bigl[H(r)/H(r_0) - 1\bigr]}} - \pi,
    \qquad H = \frac{h}{f}\,\Bigl(1 - f\,\frac{\omega_p^2}{\omega_0^2}\Bigr),

where :math:`H = h/f` in vacuum and :math:`H(r_0) = (L/E)^2`.

Numerical treatment (after Tsukamoto 2017, Phys. Rev. D 95, 064035,
Eqs. (28)-(44)): with :math:`z = 1 - r_0/r` the integral is
:math:`\int_0^1 2r_0\,G^{-1/2}dz`,
:math:`G(z) = [H/H_0 - 1]\,g\,h\,(1-z)^4 = c_1z + c_2z^2 + \dots`. The
integrand has an integrable :math:`z^{-1/2}` singularity at the turning
point, which turns into a logarithmic divergence as :math:`r_0` approaches
the photon sphere (:math:`c_1 \to 0`). The part
:math:`2r_0/\sqrt{c_1z + c_2z^2}` is integrated in closed form and only the
bounded remainder is integrated numerically (Gauss-Legendre in
:math:`s = \sqrt z`). The result is accurate from the weak-deflection
regime down to turning points very close to the photon sphere. The
coefficients :math:`c_1, c_2` are obtained by automatic differentiation.

Weak deflection: Gauss-Bonnet
-----------------------------
Gibbons & Werner (2008), Class. Quantum Grav. 25, 235009: the deflection
angle is minus the integral of the Gaussian curvature :math:`K` of the
optical metric :math:`dt^2 = dr^2/(fg) + (h/f)\,d\phi^2` over the region
outside the ray (their Eq. (5)); to leading order the ray is the straight
line :math:`r = b/\sin\phi` (their Eq. (14)). For a metric of the form
:math:`E\,dr^2 + \mathcal{G}\,d\phi^2`,
:math:`K\,dS = -\partial_r\bigl[E^{-1/2}\partial_r\sqrt{\mathcal{G}}\bigr]dr\,d\phi`,
so the radial integral is exact and

.. math::

    \alpha_{\rm GB}(b) = \int_0^\pi\Bigl[1 - \sqrt{fg}\;
        \frac{d}{dr}\sqrt{\frac{h}{f}}\Bigr]_{r = b/\sin\phi} d\phi ,

which is :math:`4M/b` for Schwarzschild. It is a first-order result: it
agrees with the exact angle up to terms of order :math:`(M/b)^2`.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import numpy as np

from blackpy._links import ROADMAP
from blackpy._numerics import derivative, gauss_legendre, refine_root, value_and_derivative
from blackpy.errors import BlackPyError, NotAsymptoticallyFlatError

HALF_PI = 0.5 * math.pi


def _static_only(metric, what: str) -> None:
    if metric.is_rotating_class:
        raise TypeError(
            f"{what} is implemented for static, spherically symmetric metrics only. Lensing by "
            f"the rotating metric '{metric.name}' (even in its equatorial plane) is not "
            "implemented in this version of BlackPy. Use a static metric (for example the "
            "a = 0 member of the family), or see " + ROADMAP + "."
        )


def _view(metric, plasma):
    """The metric itself, or the metric with plasma."""
    if plasma is None:
        return metric
    from blackpy.light.plasma import with_plasma

    return with_plasma(metric, plasma)


def effective_h2(view, p, r):
    r""":math:`H(r)`: :math:`h/f` in vacuum, :math:`h_{\rm pl}^2` with a plasma."""
    if hasattr(view, "plasma"):
        from blackpy.light.plasma import h_plasma_squared

        return h_plasma_squared(view, p, r)
    return view.h_array(r, p) / view.f_array(r, p)


def refractive_index_squared_at_infinity(view, p):
    """``n^2`` at spatial infinity (1 in vacuum or if the plasma thins out)."""
    if hasattr(view, "plasma"):
        return 1.0 - view.plasma_infinity_array(jnp.asarray(HALF_PI), p)
    return jnp.asarray(1.0)


def _g_function(view, p, r0):
    """``G(z) = [H/H_0 - 1] g h (1 - z)^4`` as a function of ``z = 1 - r_0/r``."""
    h2_0 = effective_h2(view, p, r0)

    def big_g(z):
        r = r0 / (1.0 - z)
        ratio = effective_h2(view, p, r) / h2_0 - 1.0
        return ratio * view.g_array(r, p) * view.h_array(r, p) * (1.0 - z) ** 4

    return big_g


def expansion_coefficients(view, p, r0):
    """``(c_1, c_2)`` of ``G(z) = c_1 z + c_2 z^2 + ...`` at closest approach ``r_0``."""
    c1, c2, _, _ = taylor_coefficients(view, p, r0)
    return c1, c2


def taylor_coefficients(view, p, r0):
    """``(c_1, c_2, c_3, c_4)`` of ``G(z)`` at ``z = 0``, by nested forward-mode autodiff."""
    big_g = _g_function(view, p, r0)
    zero = jnp.asarray(0.0)
    return (
        derivative(big_g, 1)(zero),
        derivative(big_g, 2)(zero) / 2.0,
        derivative(big_g, 3)(zero) / 6.0,
        derivative(big_g, 4)(zero) / 24.0,
    )


#: Below this ``z`` the difference between ``G`` and its two leading terms is taken from
#: the Taylor series: evaluated directly it is lost to cancellation, because
#: ``H(r)/H(r_0) - 1`` is the difference of two nearly equal numbers there.
_Z_SERIES = 1.0e-3


def regular_integral(big_g, c1, c2, c3, c4, n=128):
    r"""Regular part of the deflection integral.

    :math:`\int_0^1[G^{-1/2} - (c_1z + c_2z^2)^{-1/2}]dz`, by Gauss-Legendre
    quadrature in :math:`s = \sqrt z`.
    """

    def integrand(s):
        z = s**2
        small = z < _Z_SERIES
        model = c1 * z + c2 * z**2
        tail = c3 * z**3 + c4 * z**4
        full = jnp.where(small, model + tail, big_g(jnp.where(small, 0.5, z)))
        gap = jnp.where(small, -tail, model - full)  # model - full
        # 1/sqrt(full) - 1/sqrt(model), written without a second cancellation.
        return 2.0 * s * gap / (jnp.sqrt(full * model) * (jnp.sqrt(full) + jnp.sqrt(model)))

    return gauss_legendre(integrand, 0.0, 1.0, n)


def _divergent_integral(c1, c2):
    r""":math:`\int_0^1 dz/\sqrt{c_1z + c_2z^2}` for :math:`c_1 > 0` and either sign of ``c_2``."""
    k = c2 / c1
    positive = jnp.where(k > 0.0, k, 1.0)
    negative = jnp.where(k < 0.0, -k, 0.5)
    grow = jnp.arcsinh(jnp.sqrt(positive)) / jnp.sqrt(positive)
    shrink = jnp.arcsin(jnp.sqrt(jnp.clip(negative, 0.0, 1.0))) / jnp.sqrt(negative)
    series = 1.0 - k / 6.0 + 3.0 * k**2 / 40.0
    shape = jnp.where(jnp.abs(k) < 1e-4, series, jnp.where(k > 0.0, grow, shrink))
    return 2.0 * shape / jnp.sqrt(c1)


def deflection_at_core(view, p, r0, n=128):
    """Deflection angle (radians) for closest approach ``r0`` (NaN if not a turning point)."""
    big_g = _g_function(view, p, r0)
    c1, c2, c3, c4 = taylor_coefficients(view, p, r0)
    ok = (c1 > 0.0) & jnp.isfinite(c1) & jnp.isfinite(c2)
    safe_c1 = jnp.where(ok, c1, 1.0)
    safe_c2 = jnp.where(ok, c2, 0.0)
    regular = regular_integral(big_g, safe_c1, safe_c2, c3, c4, n)
    total = 2.0 * r0 * (_divergent_integral(safe_c1, safe_c2) + regular)
    return jnp.where(ok, total - math.pi, jnp.nan)


def closest_approach_core(view, p, b, n_scan=256):
    """Closest approach of the ray with geometric impact parameter ``b`` (NaN if captured)."""
    if hasattr(view, "plasma"):
        from blackpy.light.plasma import static_shadow_core

        shadow = static_shadow_core(view, p, jnp.asarray(jnp.inf), n_scan=n_scan, finite=False)
        r_ph, b_c = shadow.r_ph, shadow.radius
    else:
        from blackpy.light.photon_sphere import critical_impact_parameter_core, photon_sphere_core

        r_ph = photon_sphere_core(view, p, n_scan=n_scan)
        b_c = critical_impact_parameter_core(view, p, n_scan=n_scan)
    n2 = refractive_index_squared_at_infinity(view, p)
    target = n2 * b**2  # (L/E)^2
    ok = jnp.isfinite(r_ph) & (b > b_c) & (n2 > 0.0)
    lower = jnp.where(ok, r_ph, 1.0)
    upper = jnp.where(ok, 2.0 * b + 10.0 * lower, 2.0)
    root = refine_root(lambda r: effective_h2(view, p, r) - target, lower, upper)
    return jnp.where(ok, root, jnp.nan)


def deflection_core(view, p, b, n=128, n_scan=256):
    """Deflection angle (radians) as a function of the geometric impact parameter ``b``."""
    r0 = closest_approach_core(view, p, b, n_scan=n_scan)
    safe = jnp.where(jnp.isfinite(r0), r0, 10.0)
    return jnp.where(jnp.isfinite(r0), deflection_at_core(view, p, safe, n=n), jnp.nan)


def weak_deflection_core(metric, p, b, n=64):
    """Gauss-Bonnet weak-deflection angle for impact parameter ``b`` (straight-line boundary)."""
    f = lambda r: metric.f_array(r, p)

    def slope(r):
        radius = lambda x: jnp.sqrt(metric.h_array(x, p) / f(x))
        return jnp.sqrt(f(r) * metric.g_array(r, p)) * value_and_derivative(radius, r)[1]

    return gauss_legendre(lambda phi: 1.0 - slope(b / jnp.sin(phi)), 0.0, math.pi, n)


# ----------------------------------------------------------------------------- public API


def _require_flat(metric, params, plasma):
    names = metric.param_names
    if not metric.is_asymptotically_flat(**{k: v for k, v in params.items() if k in names}):
        raise NotAsymptoticallyFlatError(
            f"A deflection angle needs light rays that come from and go to infinity, but "
            f"metric '{metric.name}' is not asymptotically flat for these parameters."
        )


def closest_approach(metric, b, *, plasma=None, n_scan: int = 256, **params):
    r"""Radius of closest approach of a light ray with impact parameter ``b``.

    The outermost solution of :math:`H(r_0) = n_\infty^2 b^2` (see
    :func:`deflection_angle` for the meaning of ``b`` in a plasma).

    Returns
    -------
    jax.Array
        In units of M; NaN if the ray is captured (``b`` below the critical
        impact parameter) or there is no photon sphere.
    """
    _static_only(metric, "closest_approach")
    _require_flat(metric, params, plasma)
    return _view(metric, plasma).evaluate(closest_approach_core, params, b, n_scan=n_scan)


def deflection_angle_at(metric, r0, *, plasma=None, n: int = 128, **params):
    """Exact deflection angle (radians) of the ray whose closest approach is ``r0``.

    See :func:`deflection_angle`. ``r0`` must lie outside the photon sphere;
    NaN is returned otherwise.
    """
    _static_only(metric, "deflection_angle_at")
    _require_flat(metric, params, plasma)
    return _view(metric, plasma).evaluate(deflection_at_core, params, r0, n=n)


def deflection_angle(metric, b, *, plasma=None, n: int = 128, n_scan: int = 256, **params):
    r"""Exact deflection angle of a light ray, from weak to strong deflection.

    Parameters
    ----------
    metric : StaticSpherical
        Asymptotically flat.
    b : float or array
        Impact parameter in units of M: the distance of the incoming ray's
        asymptote from the centre. In vacuum (and in a plasma that thins
        out) this is :math:`L/E`. In a plasma that extends to infinity it is
        :math:`L/(n_\infty E)` with :math:`n_\infty` the refractive index
        there, which is the geometric impact parameter used in the lensing
        literature (Bisnovatyi-Kogan & Tsupko 2010).
    plasma : blackpy.light.PlasmaModel, optional
        A spherically symmetric plasma around the lens; its parameters are
        passed as keywords with the metric parameters.
    n : int
        Number of Gauss-Legendre nodes for the regular part of the integral.
    **params
        Metric (and plasma) parameters.

    Returns
    -------
    jax.Array
        Deflection angle in radians. It exceeds :math:`2\pi` for rays that
        loop around the black hole and diverges logarithmically at the
        critical impact parameter; NaN below it (the ray is captured).

    Notes
    -----
    Checks (tests): the Schwarzschild series
    :math:`4M/b + \tfrac{15\pi}{4}(M/b)^2 + \tfrac{128}{3}(M/b)^3` and its
    Reissner-Nordstrom counterpart (Keeton & Petters 2005, Phys. Rev. D 72,
    104006, Eqs. (24)-(25), (52)); the strong deflection limit as
    :math:`b \to b_c` (:func:`blackpy.lensing.strong_deflection_limit`); the
    weak-field formula in a homogeneous plasma (Bisnovatyi-Kogan & Tsupko
    2010, Eq. (38)).

    For rotating metrics lensing is not implemented.
    """
    _static_only(metric, "deflection_angle")
    _require_flat(metric, params, plasma)
    view = _view(metric, plasma)
    out = view.evaluate(deflection_core, params, b, n=n, n_scan=n_scan)
    if np.all(np.isnan(np.asarray(out))):
        raise BlackPyError(
            f"deflection_angle: every ray with the given impact parameter(s) is captured by "
            f"'{metric.name}' (b is not above the critical impact parameter), or the metric has "
            "no photon sphere for these parameters. See blackpy.light.shadow_radius(...) for "
            "the critical impact parameter."
        )
    return out


def weak_deflection_angle(metric, b, *, n: int = 64, **params):
    r"""Weak-deflection angle from the Gauss-Bonnet theorem (Gibbons & Werner 2008).

    .. math::

        \alpha_{\rm GB}(b) = \int_0^\pi\Bigl[1 - \sqrt{fg}\,
            \frac{d}{dr}\sqrt{h/f}\Bigr]_{r = b/\sin\phi}d\phi

    (see the module docstring for the derivation). The integration region
    is bounded by the undeflected straight line, so the result is correct
    to first order in the deflection: :math:`4M/b` for Schwarzschild, and it
    differs from the exact :func:`deflection_angle` at order
    :math:`(M/b)^2`. It needs no photon sphere and no root finding, and is
    meant as an independent check of the exact integral in the weak regime.

    Parameters
    ----------
    metric : StaticSpherical
    b : float or array
        Impact parameter in units of M; must be well outside the horizon.
    **params
        Metric parameters.
    """
    _static_only(metric, "weak_deflection_angle")
    _require_flat(metric, params, None)
    return metric.evaluate(weak_deflection_core, params, b, n=n)


__all__ = [
    "closest_approach",
    "deflection_angle",
    "deflection_angle_at",
    "weak_deflection_angle",
]
