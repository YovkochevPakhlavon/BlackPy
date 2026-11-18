"""Horizons of static, spherically symmetric metrics.

A Killing horizon of ``ds^2 = -f dt^2 + dr^2/g + h dOmega^2`` sits where
``g(r) = 0`` and ``f(r) = 0`` simultaneously. All roots of ``g`` on the scan
range are located by sign-change scanning plus extremum detection (see
:func:`blackpy._numerics.find_roots`), then classified:

``event``
    ``g`` changes sign from negative to positive with increasing ``r``
    (static region outside). The outermost such root is *the* event horizon:
    event horizon = outermost black hole horizon, not the cosmological
    horizon.
``extremal``
    Degenerate (double) root with ``g >= 0`` on both sides, e.g. extremal
    Reissner-Nordstrom. Counts as an event horizon.
``inner``
    ``g`` goes from positive to negative, located inside the event horizon
    (Cauchy horizon).
``cosmological``
    ``g`` goes from positive to negative, located outside the event horizon
    (or with no event horizon at all, as in pure de Sitter space).
``degenerate``
    Double root with ``g <= 0`` on both sides (e.g. the Nariai limit); there
    is no static region next to it.
``not a horizon``
    ``g = 0`` but ``f != 0`` (e.g. a wormhole throat in these coordinates).

Note that "outermost root of g" is *not* a safe definition of the event
horizon: in Schwarzschild-de Sitter the outermost root is the cosmological
horizon.

Rotating metrics
----------------
For a :class:`blackpy.metric.StationaryAxisymmetric` metric the same code is
used with ``g -> g^rr = 1 / g_rr`` and ``f -> N^2 = rho2 / g_phiphi`` (the
squared lapse, which vanishes on a Killing horizon), evaluated at a polar
angle ``theta`` (default: the equator). For Kerr-like metrics
``g^rr = Delta / Sigma``, so the roots are those of ``Delta``.

The surface ``g^rr = 0`` is the event horizon only if its radius does not
depend on ``theta``. In general the horizon ``r = H(theta)`` of a stationary,
axisymmetric metric solves ``g^rr + g^thth (dH/dtheta)^2 = 0`` (Johannsen
2013, Phys. Rev. D 88, 044002, Eq. (58)); :func:`horizon_profile` reports
whether the roots are theta-independent, and BlackPy does not solve the
general equation.

The ergosurface (static limit) is the outermost root of ``g_tt = 0``
(:func:`ergosurface`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from blackpy._numerics import derivative, find_roots, nanmax_or_nan, nanmin_or_nan

HALF_PI = 0.5 * math.pi

#: |f| below this value at a root of g counts as "f vanishes too".
F_ZERO_ATOL = 1e-8

#: Default tolerance on |g| at a minimum of g for an extremal (double) horizon.
DEGENERATE_ATOL = 1e-10

#: Relative spread of the roots of g^rr over theta below which they count as equal.
THETA_INDEPENDENCE_RTOL = 1e-6

#: Relative offset of the points used to read the direction of a sign change of g^rr.
_SIDE = 1e-7

# Integer codes used inside jit-compiled code.
_EVENT, _EXTREMAL, _OUTGOING, _DEGENERATE, _NOT_HORIZON = 1, 2, -1, -2, 0


def _horizon_functions(metric, p, theta):
    """``(g, f)`` as functions of r: the horizon function and the lapse-like function."""
    if metric.is_rotating_class:
        return (
            lambda r: metric.grr_inverse_array(r, theta, p),
            lambda r: metric.lapse2_array(r, theta, p),
        )
    return (lambda r: metric.g_array(r, p)), (lambda r: metric.f_array(r, p))


def horizon_roots_core(
    metric, p, n_scan=256, max_roots=6, degenerate_atol=DEGENERATE_ATOL, theta=HALF_PI
):
    """Roots of ``g`` (static) or ``g^rr`` (rotating) with classification codes.

    Returns
    -------
    roots : array (max_roots,)
        Ascending, NaN padded.
    codes : array of int (max_roots,)
        ``1`` event-type, ``2`` extremal, ``-1`` inner/cosmological-type,
        ``-2`` degenerate without static side, ``0`` not a horizon / padding.
    """
    r_min, r_max = metric.r_range
    g, f = _horizon_functions(metric, p, theta)
    roots, degenerate = find_roots(
        g, r_min, r_max, n_scan=n_scan, max_roots=max_roots, degenerate_atol=degenerate_atol
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    if metric.is_rotating_class:
        # g^rr = 1 / g_rr: at an exact root g_rr is infinite and its autodiff derivative
        # is undefined, so the direction of the sign change is read from the sign of
        # g^rr just outside and just inside the root (a sign test, not a derivative).
        outside = jax.vmap(g)(safe * (1.0 + _SIDE))
        inside = jax.vmap(g)(safe * (1.0 - _SIDE))
        slope = jnp.where(outside > inside, 1.0, -1.0)
        curvature = jnp.where((outside > 0.0) & (inside > 0.0), 1.0, -1.0)
    else:
        slope = jax.vmap(derivative(g))(safe)
        curvature = jax.vmap(derivative(g, 2))(safe)
    f_at_root = jax.vmap(f)(safe)
    f_vanishes = jnp.abs(f_at_root) <= F_ZERO_ATOL
    valid = jnp.isfinite(roots)
    codes = jnp.where(
        degenerate,
        jnp.where(curvature > 0.0, _EXTREMAL, _DEGENERATE),
        jnp.where(slope > 0.0, _EVENT, _OUTGOING),
    )
    codes = jnp.where(valid & f_vanishes, codes, _NOT_HORIZON)
    return roots, codes


def static_domain_core(
    metric, p, n_scan=256, max_roots=3, degenerate_atol=DEGENERATE_ATOL, theta=HALF_PI
):
    """Event horizon, cosmological horizon and the static domain between them.

    For rotating metrics this is the domain of outer communication at polar
    angle ``theta`` (default: the equator).

    Only the ``max_roots`` outermost roots of ``g`` are examined (default 3:
    inner, event and cosmological horizon), which keeps the vectorised fast
    path cheap. Pass a larger value for metrics with more horizons.

    Returns
    -------
    r_event : float
        Event horizon radius (outermost black hole horizon, not the
        cosmological horizon); NaN if the configuration is horizonless.
    r_cosmo : float
        Cosmological horizon radius, NaN if there is none.
    r_in, r_out : float
        Inner and outer edge of the domain searched for orbits and photon
        spheres: ``r_in`` is the event horizon (or the lower end of the scan
        range if horizonless), ``r_out`` the cosmological horizon (or the
        upper end of the scan range).
    """
    r_min, r_max = metric.r_range
    roots, codes = horizon_roots_core(
        metric,
        p,
        n_scan=n_scan,
        max_roots=max_roots,
        degenerate_atol=degenerate_atol,
        theta=theta,
    )
    r_event = nanmax_or_nan(roots, (codes == _EVENT) | (codes == _EXTREMAL))
    has_event = jnp.isfinite(r_event)
    outside = jnp.where(has_event, roots > r_event, True)
    r_cosmo = nanmin_or_nan(roots, (codes == _OUTGOING) & outside)
    r_in = jnp.where(has_event, r_event, r_min)
    r_out = jnp.where(jnp.isfinite(r_cosmo), r_cosmo, r_max)
    return r_event, r_cosmo, r_in, r_out


def _event_horizon_core(metric, p, n_scan=256, degenerate_atol=DEGENERATE_ATOL):
    return static_domain_core(metric, p, n_scan=n_scan, degenerate_atol=degenerate_atol)[0]


def _event_horizon_at_core(metric, p, theta, n_scan=256, degenerate_atol=DEGENERATE_ATOL):
    return static_domain_core(
        metric, p, n_scan=n_scan, degenerate_atol=degenerate_atol, theta=theta
    )[0]


def _cosmological_horizon_core(metric, p, n_scan=256):
    return static_domain_core(metric, p, n_scan=n_scan)[1]


def _ergosurface_core(metric, p, theta, n_scan=256):
    """Outermost root of ``g_tt(r, theta) = 0`` with ``-g_tt`` increasing outward."""
    r_min, r_max = metric.r_range
    minus_gtt = lambda r: -metric.gtt_array(r, theta, p)
    roots, degenerate = find_roots(minus_gtt, r_min, r_max, n_scan=n_scan, max_roots=3)
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    slope = jax.vmap(derivative(minus_gtt))(safe)
    return nanmax_or_nan(roots, (slope > 0.0) | degenerate)


def event_horizon(
    metric,
    theta=None,
    *,
    n_scan: int = 256,
    degenerate_atol: float = DEGENERATE_ATOL,
    **params,
):
    """Event horizon radius (vectorised over parameters).

    The event horizon is the outermost *black hole* horizon: the outermost
    root of ``g`` across which ``g`` goes from negative to positive (or an
    extremal double root). It is not simply the outermost root of ``g``,
    which in de Sitter-like spacetimes is the cosmological horizon.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    theta : float or array, optional
        Rotating metrics only: polar angle at which ``g^rr = 0`` is solved
        (default: the equator). The result is the event horizon only if it
        does not depend on ``theta``; see :func:`horizon_profile`.
    n_scan : int
        Number of scan points on the metric's ``r_range``.
    degenerate_atol : float
        A minimum of ``g`` with ``|g| <= degenerate_atol`` counts as an
        extremal (double) horizon. Lower it to resolve configurations closer
        than about 1e-10 to extremality.
    **params
        Metric parameters (scalars or arrays).

    Returns
    -------
    jax.Array
        Radius in units of M; NaN where the configuration is horizonless
        (e.g. Reissner-Nordstrom with ``Q > 1``).
    """
    options = {"n_scan": n_scan, "degenerate_atol": float(degenerate_atol)}
    if theta is not None and metric.is_rotating_class:
        return metric.evaluate(_event_horizon_at_core, params, theta, **options)
    return metric.evaluate(_event_horizon_core, params, **options)


def ergosurface(metric, theta=HALF_PI, *, n_scan: int = 256, **params):
    r"""Radius of the ergosurface (static limit), the outermost root of ``g_tt = 0``.

    Parameters
    ----------
    metric : StationaryAxisymmetric
    theta : float or array
        Polar angle (default: the equator).
    **params
        Metric parameters (scalars or arrays).

    Returns
    -------
    jax.Array
        :math:`r_E(\theta)` in units of M; NaN if ``g_tt`` has no root. For
        Kerr :math:`r_E = 1 + \sqrt{1 - a^2\cos^2\theta}`. For a
        non-rotating metric the ergosurface coincides with the horizon.
    """
    if not metric.is_rotating_class:
        metric = metric.as_axisymmetric()
    return metric.evaluate(_ergosurface_core, params, theta, n_scan=n_scan)


@dataclass(frozen=True)
class HorizonProfile:
    """Roots of ``g^rr = 0`` as a function of the polar angle.

    Attributes
    ----------
    theta : numpy.ndarray
        Polar angles.
    radius : numpy.ndarray
        Outermost event-type root of ``g^rr`` at each angle (NaN if none).
    theta_independent : bool
        True if the radius is the same at all angles (relative spread below
        1e-6; an extremal double root is located to about 1e-8). Only then is
        the surface ``g^rr = 0`` the event horizon.
    message : str
        Plain-language statement of the result.
    """

    theta: np.ndarray
    radius: np.ndarray
    theta_independent: bool
    message: str

    def __str__(self) -> str:
        return self.message


def horizon_profile(metric, theta=None, *, n_scan: int = 256, **params) -> HorizonProfile:
    """Horizon radius as a function of ``theta`` and whether it is theta-independent.

    Parameters
    ----------
    metric : StationaryAxisymmetric
    theta : array, optional
        Polar angles strictly between 0 and pi. Default: 17 angles from
        0.05 to pi - 0.05.
    **params
        Scalar metric parameters.

    Returns
    -------
    HorizonProfile
    """
    if not metric.is_rotating_class:
        metric = metric.as_axisymmetric()
    theta = np.linspace(0.05, math.pi - 0.05, 17) if theta is None else np.atleast_1d(theta)
    if np.any(theta <= 0.0) or np.any(theta >= math.pi):
        raise ValueError(
            "horizon_profile: theta (radians) must lie strictly between 0 and pi; to probe the "
            "poles use values such as 1e-6 and pi - 1e-6."
        )
    radius = np.asarray(metric.evaluate(_event_horizon_at_core, params, theta, n_scan=n_scan))
    finite = np.isfinite(radius)
    if not finite.any():
        independent = True
        message = f"{metric.name}: g^rr has no event-type root at any polar angle (horizonless)."
    elif not finite.all():
        independent = False
        message = (
            f"{metric.name}: g^rr = 0 has a root only at some polar angles; there is no "
            "closed surface g^rr = 0."
        )
    else:
        spread = float(np.max(radius) - np.min(radius))
        independent = spread <= THETA_INDEPENDENCE_RTOL * float(np.mean(radius))
        if independent:
            message = (
                f"{metric.name}: the horizon radius r = {float(np.mean(radius)):.12g} is "
                "independent of theta."
            )
        else:
            message = (
                f"{metric.name}: the root of g^rr depends on theta (from "
                f"{float(np.min(radius)):.8g} to {float(np.max(radius)):.8g}). The surface "
                "g^rr = 0 is then not the event horizon, which solves "
                "g^rr + g^thth (dH/dtheta)^2 = 0; BlackPy does not solve that equation."
            )
    return HorizonProfile(
        theta=theta, radius=radius, theta_independent=independent, message=message
    )


def cosmological_horizon(metric, *, n_scan: int = 256, **params):
    """Cosmological horizon radius (vectorised); NaN if there is none."""
    return metric.evaluate(_cosmological_horizon_core, params, n_scan=n_scan)


@dataclass(frozen=True)
class Horizon:
    """One root of ``g(r) = 0``.

    Attributes
    ----------
    radius : float
        Radius in units of M.
    kind : str
        One of ``"event"``, ``"extremal"``, ``"inner"``, ``"cosmological"``,
        ``"degenerate"``, ``"not a horizon"``.
    degenerate : bool
        True for a double root.
    f_vanishes : bool
        Whether ``f`` also vanishes there (required for a Killing horizon).
    """

    radius: float
    kind: str
    degenerate: bool
    f_vanishes: bool


@dataclass(frozen=True)
class HorizonResult:
    """All horizons of a metric for one set of parameters.

    Attributes
    ----------
    horizons : tuple of Horizon
        Every root of ``g`` on the scan range, in ascending order.
    event_horizon : float or None
        Radius of the event horizon, i.e. of the outermost black hole
        horizon (not the cosmological horizon); ``None`` if horizonless.
    cosmological_horizon : float or None
    horizonless : bool
        True if there is no event horizon on the scan range.
    r_range : tuple of float
        Radial range that was scanned.
    theta_independent : bool or None
        Rotating metrics: whether the roots of ``g^rr`` are the same at all
        polar angles (see :func:`horizon_profile`). ``None`` for static
        metrics.
    """

    horizons: tuple[Horizon, ...]
    event_horizon: float | None
    cosmological_horizon: float | None
    horizonless: bool
    r_range: tuple[float, float]
    message: str = field(default="")
    theta_independent: bool | None = None

    @property
    def radii(self) -> tuple[float, ...]:
        """Radii of all genuine horizons, ascending."""
        return tuple(hz.radius for hz in self.horizons if hz.kind != "not a horizon")

    def __str__(self) -> str:
        lines = [self.message]
        lines += [f"  r = {hz.radius:.12g}  ({hz.kind})" for hz in self.horizons]
        return "\n".join(lines)


def horizons(
    metric,
    *,
    n_scan: int = 256,
    max_roots: int = 6,
    degenerate_atol: float = DEGENERATE_ATOL,
    **params,
) -> HorizonResult:
    """Find and classify all horizons for one set of (scalar) parameters.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
        For a rotating metric the roots of ``g^rr`` on the equatorial plane
        are classified, and the result records whether they are
        theta-independent.
    n_scan : int
        Number of scan points on ``metric.r_range``.
    max_roots : int
        Maximum number of roots reported.
    degenerate_atol : float
        Tolerance on ``|g|`` at a minimum for an extremal (double) horizon.
    **params
        Scalar metric parameters.

    Returns
    -------
    HorizonResult
        A horizonless configuration (e.g. a naked singularity) is reported
        through ``horizonless=True`` and ``event_horizon=None``; it is not an
        error.
    """
    options = {"n_scan": n_scan, "max_roots": max_roots, "degenerate_atol": float(degenerate_atol)}
    roots, codes = metric.evaluate(horizon_roots_core, params, **options)
    roots, codes = np.asarray(roots), np.asarray(codes)
    if roots.ndim != 1:
        raise ValueError(
            "horizons() classifies one configuration at a time; pass scalar parameters, or use "
            "event_horizon() for arrays."
        )
    r_event, r_cosmo, _, _ = (
        float(v) for v in metric.evaluate(static_domain_core, params, **options)
    )
    found = []
    for radius, code in zip(roots, codes, strict=True):
        if not np.isfinite(radius):
            continue
        if code == _NOT_HORIZON:
            kind = "not a horizon"
        elif code == _EXTREMAL:
            kind = "extremal"
        elif code == _DEGENERATE:
            kind = "degenerate"
        elif code == _EVENT:
            kind = "event" if radius == r_event else "inner"
        elif np.isfinite(r_event) and radius < r_event:
            kind = "inner"
        else:
            kind = "cosmological"
        found.append(
            Horizon(float(radius), kind, code in (_EXTREMAL, _DEGENERATE), code != _NOT_HORIZON)
        )
    horizonless = not np.isfinite(r_event)
    independent, profile_note = None, ""
    if metric.is_rotating_class:
        profile = horizon_profile(metric, n_scan=n_scan, **params)
        independent = profile.theta_independent
        profile_note = "\n" + profile.message + " (Radii below: equatorial plane.)"
    if horizonless:
        message = (
            f"{metric.name}: no event horizon found for r in [{metric.r_range[0]:g}, "
            f"{metric.r_range[1]:g}] (horizonless configuration)."
        )
    else:
        message = f"{metric.name}: event horizon at r = {r_event:.12g}."
    return HorizonResult(
        horizons=tuple(found),
        event_horizon=None if horizonless else r_event,
        cosmological_horizon=r_cosmo if np.isfinite(r_cosmo) else None,
        horizonless=horizonless,
        r_range=metric.r_range,
        message=message + profile_note,
        theta_independent=independent,
    )
