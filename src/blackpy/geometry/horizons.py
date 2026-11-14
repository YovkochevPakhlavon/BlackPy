"""Horizons of static, spherically symmetric metrics.

A Killing horizon of ``ds^2 = -f dt^2 + dr^2/g + h dOmega^2`` sits where
``g(r) = 0`` and ``f(r) = 0`` simultaneously. All roots of ``g`` on the scan
range are located by sign-change scanning plus extremum detection (see
:func:`blackpy._numerics.find_roots`), then classified:

``event``
    ``g`` changes sign from negative to positive with increasing ``r``
    (static region outside). The outermost such root is *the* event horizon.
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
"""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from blackpy._numerics import derivative, find_roots, nanmax_or_nan, nanmin_or_nan

#: |f| below this value at a root of g counts as "f vanishes too".
F_ZERO_ATOL = 1e-8

# Integer codes used inside jit-compiled code.
_EVENT, _EXTREMAL, _OUTGOING, _DEGENERATE, _NOT_HORIZON = 1, 2, -1, -2, 0


def horizon_roots_core(metric, p, n_scan=256, max_roots=6):
    """Roots of ``g`` with classification codes (jit/vmap-compatible).

    Returns
    -------
    roots : array (max_roots,)
        Ascending, NaN padded.
    codes : array of int (max_roots,)
        ``1`` event-type, ``2`` extremal, ``-1`` inner/cosmological-type,
        ``-2`` degenerate without static side, ``0`` not a horizon / padding.
    """
    r_min, r_max = metric.r_range
    g = lambda r: metric.g_array(r, p)
    roots, degenerate = find_roots(g, r_min, r_max, n_scan=n_scan, max_roots=max_roots)
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    slope = jax.vmap(derivative(g))(safe)
    curvature = jax.vmap(derivative(g, 2))(safe)
    f_at_root = jax.vmap(lambda r: metric.f_array(r, p))(safe)
    f_vanishes = jnp.abs(f_at_root) <= F_ZERO_ATOL
    valid = jnp.isfinite(roots)
    codes = jnp.where(
        degenerate,
        jnp.where(curvature > 0.0, _EXTREMAL, _DEGENERATE),
        jnp.where(slope > 0.0, _EVENT, _OUTGOING),
    )
    codes = jnp.where(valid & f_vanishes, codes, _NOT_HORIZON)
    return roots, codes


def static_domain_core(metric, p, n_scan=256, max_roots=3):
    """Event horizon, cosmological horizon and the static domain between them.

    Only the ``max_roots`` outermost roots of ``g`` are examined (default 3:
    inner, event and cosmological horizon), which keeps the vectorised fast
    path cheap. Pass a larger value for metrics with more horizons.

    Returns
    -------
    r_event : float
        Event horizon radius, NaN if the configuration is horizonless.
    r_cosmo : float
        Cosmological horizon radius, NaN if there is none.
    r_in, r_out : float
        Inner and outer edge of the domain searched for orbits and photon
        spheres: ``r_in`` is the event horizon (or the lower end of the scan
        range if horizonless), ``r_out`` the cosmological horizon (or the
        upper end of the scan range).
    """
    r_min, r_max = metric.r_range
    roots, codes = horizon_roots_core(metric, p, n_scan=n_scan, max_roots=max_roots)
    r_event = nanmax_or_nan(roots, (codes == _EVENT) | (codes == _EXTREMAL))
    has_event = jnp.isfinite(r_event)
    outside = jnp.where(has_event, roots > r_event, True)
    r_cosmo = nanmin_or_nan(roots, (codes == _OUTGOING) & outside)
    r_in = jnp.where(has_event, r_event, r_min)
    r_out = jnp.where(jnp.isfinite(r_cosmo), r_cosmo, r_max)
    return r_event, r_cosmo, r_in, r_out


def _event_horizon_core(metric, p, n_scan=256):
    return static_domain_core(metric, p, n_scan=n_scan)[0]


def _cosmological_horizon_core(metric, p, n_scan=256):
    return static_domain_core(metric, p, n_scan=n_scan)[1]


def event_horizon(metric, *, n_scan: int = 256, **params):
    """Event horizon radius (vectorised over parameters).

    Parameters
    ----------
    metric : StaticSpherical
    n_scan : int
        Number of scan points on the metric's ``r_range``.
    **params
        Metric parameters (scalars or arrays).

    Returns
    -------
    jax.Array
        Radius in units of M; NaN where the configuration is horizonless
        (e.g. Reissner-Nordstrom with ``Q > 1``).
    """
    return metric.evaluate(_event_horizon_core, params, n_scan=n_scan)


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
        Radius of the event horizon; ``None`` if horizonless.
    cosmological_horizon : float or None
    horizonless : bool
        True if there is no event horizon on the scan range.
    r_range : tuple of float
        Radial range that was scanned.
    """

    horizons: tuple[Horizon, ...]
    event_horizon: float | None
    cosmological_horizon: float | None
    horizonless: bool
    r_range: tuple[float, float]
    message: str = field(default="")

    @property
    def radii(self) -> tuple[float, ...]:
        """Radii of all genuine horizons, ascending."""
        return tuple(hz.radius for hz in self.horizons if hz.kind != "not a horizon")

    def __str__(self) -> str:
        lines = [self.message]
        lines += [f"  r = {hz.radius:.12g}  ({hz.kind})" for hz in self.horizons]
        return "\n".join(lines)


def horizons(metric, *, n_scan: int = 256, max_roots: int = 6, **params) -> HorizonResult:
    """Find and classify all horizons for one set of (scalar) parameters.

    Parameters
    ----------
    metric : StaticSpherical
    n_scan : int
        Number of scan points on ``metric.r_range``.
    max_roots : int
        Maximum number of roots reported.
    **params
        Scalar metric parameters.

    Returns
    -------
    HorizonResult
        A horizonless configuration (e.g. a naked singularity) is reported
        through ``horizonless=True`` and ``event_horizon=None``; it is not an
        error.
    """
    roots, codes = metric.evaluate(horizon_roots_core, params, n_scan=n_scan, max_roots=max_roots)
    roots, codes = np.asarray(roots), np.asarray(codes)
    if roots.ndim != 1:
        raise ValueError(
            "horizons() classifies one configuration at a time; pass scalar parameters, or use "
            "event_horizon() for arrays."
        )
    r_event, r_cosmo, _, _ = (
        float(v)
        for v in metric.evaluate(static_domain_core, params, n_scan=n_scan, max_roots=max_roots)
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
        message=message,
    )
