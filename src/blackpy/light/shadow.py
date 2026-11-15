r"""Black hole shadow for static observers.

For a static observer at radius :math:`r_O` the angular radius
:math:`\alpha_{\rm sh}` of the shadow is given by (Synge 1966; Perlick &
Tsupko 2022, Phys. Rep. 947, 1, their Eq. (22) for this metric form)

.. math::

    \sin\alpha_{\rm sh} = b_c \sqrt{\frac{f(r_O)}{h(r_O)}},
    \qquad b_c = \sqrt{\frac{h(r_{\rm ph})}{f(r_{\rm ph})}} .

For an observer outside the photon sphere the shadow covers less than half
of the sky, :math:`\alpha_{\rm sh} = \arcsin(\ldots)`; for an observer
between the horizon and the photon sphere it covers more than half,
:math:`\alpha_{\rm sh} = \pi - \arcsin(\ldots)`.

In an asymptotically flat spacetime the shadow *radius* seen from infinity
is :math:`r_{\rm sh} = \lim r_O \alpha_{\rm sh} = b_c`.

Spacetimes that are not asymptotically flat (e.g. Schwarzschild-de Sitter)
have no observer at infinity: the shadow is only defined for a finite
``r_obs`` inside the static region, and the result applies to a *static*
observer. For observers comoving with the cosmic expansion see Perlick,
Tsupko & Bisnovatyi-Kogan (2018), Phys. Rev. D 97, 104062.

Rotating metrics have non-circular shadows; they are handled by
:mod:`blackpy.light.rotating`. :func:`shadow_radius` forwards to it and
returns the mean radius of the contour.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from blackpy.errors import NotAsymptoticallyFlatError
from blackpy.geometry.horizons import static_domain_core
from blackpy.light.photon_sphere import critical_impact_parameter_core, photon_sphere_core


def shadow_angular_radius_core(metric, p, r_obs, n_scan=256):
    """Angular shadow radius (rad) for a static observer at ``r_obs`` (NaN if undefined)."""
    _, r_cosmo, r_in, _ = static_domain_core(metric, p, n_scan=n_scan)
    r_ph = photon_sphere_core(metric, p, n_scan=n_scan)
    b_c = critical_impact_parameter_core(metric, p, n_scan=n_scan)
    f_obs = metric.f_array(r_obs, p)
    h_obs = metric.h_array(r_obs, p)
    # The outer limit is the cosmological horizon, if any; the scan range does not
    # restrict the observer position.
    inside_cosmo = jnp.where(jnp.isfinite(r_cosmo), r_obs < r_cosmo, True)
    in_static_region = (r_obs > r_in) & inside_cosmo & (f_obs > 0.0)
    ok = in_static_region & jnp.isfinite(b_c)
    sin_alpha = jnp.clip(b_c * jnp.sqrt(jnp.where(ok, f_obs / h_obs, 0.0)), 0.0, 1.0)
    alpha = jnp.arcsin(sin_alpha)
    alpha = jnp.where(r_obs >= r_ph, alpha, jnp.pi - alpha)
    return jnp.where(ok, alpha, jnp.nan)


def shadow_radius_core(metric, p, n_scan=256):
    """Shadow radius seen from infinity, ``r_sh = b_c`` (asymptotically flat metrics)."""
    return critical_impact_parameter_core(metric, p, n_scan=n_scan)


def shadow_radius_finite_core(metric, p, r_obs, n_scan=256):
    """``sqrt(h(r_obs)) * sin(alpha_sh)`` for a static observer at ``r_obs``."""
    alpha = shadow_angular_radius_core(metric, p, r_obs, n_scan=n_scan)
    return jnp.sqrt(metric.h_array(r_obs, p)) * jnp.sin(alpha)


def _static_only(metric, what: str, instead: str) -> None:
    if metric.is_rotating_class:
        raise TypeError(
            f"{what} is defined for static, spherically symmetric metrics; '{metric.name}' is a "
            f"StationaryAxisymmetric metric, whose shadow is not a circle. Use {instead}."
        )


def shadow_angular_radius(metric, r_obs, *, n_scan: int = 256, **params):
    r"""Angular radius of the shadow for a static observer at radius ``r_obs``.

    Parameters
    ----------
    metric : StaticSpherical
    r_obs : float or array
        Observer radius in units of M. Must lie in the static region
        (outside the event horizon and inside any cosmological horizon).
    **params
        Metric parameters.

    Returns
    -------
    jax.Array
        :math:`\alpha_{\rm sh}` in radians; NaN if the observer is not in
        the static region or there is no unstable photon sphere.
    """
    _static_only(
        metric, "shadow_angular_radius", "blackpy.light.shadow_contour(metric, inclination, r_obs)"
    )
    if hasattr(metric, "plasma"):  # a metric with plasma: never fall back to vacuum
        from blackpy.light.plasma import shadow_angular_radius_in_plasma

        return shadow_angular_radius_in_plasma(
            metric.metric, metric.plasma, r_obs, n_scan=n_scan, **params
        )
    return metric.evaluate(shadow_angular_radius_core, params, r_obs, n_scan=n_scan)


def shadow_radius(
    metric, r_obs=None, *, inclination=None, kind: str = "mean", n_scan: int = 256, **params
):
    r"""Shadow radius in units of M.

    Rotating metrics (:class:`blackpy.metric.StationaryAxisymmetric`): the
    shadow is not a circle, so ``inclination`` (radians) is required and the
    result is the mean radius of the contour about its centre
    (``kind="mean"``, default) or the area-equivalent radius
    (``kind="area"``); see :func:`blackpy.light.shadow_observables`.

    Static metrics (``inclination`` and ``kind`` are ignored):

    * ``r_obs=None``: observer at infinity, :math:`r_{\rm sh} = b_c`. Only
      defined for asymptotically flat metrics.
    * finite ``r_obs``: :math:`\sqrt{h(r_O)}\,\sin\alpha_{\rm sh}` for a
      static observer, i.e. the areal radius of the observer times the sine
      of the angular shadow radius. It tends to ``b_c`` as ``r_obs`` goes to
      infinity in an asymptotically flat spacetime.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    r_obs : float or array, optional
        Observer radius in units of M.
    inclination : float or array, optional
        Rotating metrics: polar angle of the observer in radians.
    kind : {"mean", "area"}
        Rotating metrics: which radius to return.
    **params
        Metric parameters.

    Returns
    -------
    jax.Array

    Raises
    ------
    NotAsymptoticallyFlatError
        If ``r_obs`` is None and the metric is not asymptotically flat.
    NotSeparableError
        Rotating metrics whose null geodesics do not separate.
    """
    if hasattr(metric, "plasma") and not metric.is_rotating_class:
        # A static metric with plasma: never fall back to the vacuum formula.
        from blackpy.light.plasma import shadow_in_plasma

        return shadow_in_plasma(
            metric.metric,
            metric.plasma,
            r_obs,
            inclination=inclination,
            kind=kind,
            n_scan=n_scan,
            **params,
        )
    if metric.is_rotating_class:
        from blackpy.light.rotating import shadow_observables

        if inclination is None:
            raise TypeError(
                f"The shadow of the rotating metric '{metric.name}' depends on the observer's "
                "inclination: pass inclination=<polar angle in radians> (pi/2 is edge-on)."
            )
        if kind not in ("mean", "area"):
            raise ValueError("kind must be 'mean' or 'area'.")
        observables = shadow_observables(metric, inclination, r_obs, n_scan=n_scan, **params)
        return observables.mean_radius if kind == "mean" else observables.area_radius
    if r_obs is None:
        if not metric.is_asymptotically_flat(**params):
            raise NotAsymptoticallyFlatError(
                f"Metric '{metric.name}' is not asymptotically flat for these parameters, so the "
                "shadow seen 'from infinity' is not defined. Pass r_obs=<radius of a static "
                "observer inside the static region>. If the metric is flat but approaches "
                "Minkowski slowly, declare asymptotically_flat=True when constructing it."
            )
        return metric.evaluate(shadow_radius_core, params, n_scan=n_scan)
    out = metric.evaluate(shadow_radius_finite_core, params, r_obs, n_scan=n_scan)
    if np.all(np.isnan(np.asarray(out))):
        raise ValueError(
            f"shadow_radius: the observer radius r_obs={r_obs} is not inside the static region "
            f"of '{metric.name}' (or there is no unstable photon sphere) for these parameters. "
            "Check blackpy.geometry.horizons(...) and blackpy.light.photon_spheres(...)."
        )
    return out
