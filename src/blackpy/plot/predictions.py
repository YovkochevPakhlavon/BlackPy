"""Plots of predictions: deflection angles, thin-disk profiles, Fisher-forecast ellipses.

Like the other plot helpers, each function draws on an optional ``ax`` (or
figure), returns it, and changes no global matplotlib state.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from blackpy.metric.base import Metric

#: Units shown in the axis labels of forecast plots, by parameter-name prefix.
_UNITS = (("inclination_deg", r"[${}^\circ$]"), ("M_", r"[$M_\odot$]"), ("r_", "[M]"))


def _new_axes(ax, figsize):
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    return ax


def deflection(
    metric: Metric,
    b=None,
    *,
    weak: bool = True,
    strong: bool = True,
    plasma=None,
    label: str | None = None,
    ax=None,
    **params,
):
    r"""Exact, weak-field and strong-deflection angles against the impact parameter.

    Draws :func:`blackpy.lensing.deflection_angle` and, optionally,
    :func:`blackpy.lensing.weak_deflection_angle` (Gauss–Bonnet) and
    :func:`blackpy.lensing.strong_deflection_angle`, against
    :math:`b/b_c - 1` on logarithmic axes, with a line at :math:`2\pi` (rays
    that loop once around the black hole). Static, asymptotically flat metrics
    only.

    Parameters
    ----------
    metric : StaticSpherical
    b : array, optional
        Impact parameters in units of M, all above the critical one :math:`b_c`.
        Default: 300 values with :math:`b/b_c - 1` from 1e-6 to 30.
    weak, strong : bool
        Also draw the weak-field and the strong-deflection approximations. The
        weak-field (Gauss–Bonnet) estimate is drawn only for :math:`b \ge 2 b_c`,
        where it is meant to apply, and not with a plasma (it is for vacuum).
    plasma : blackpy.light.PlasmaModel, optional
        Deflection in a plasma (exact and strong limit).
    label : str, optional
        Legend label of the exact curve (default: the metric name).
    ax : matplotlib axes, optional
    **params
        Scalar metric (and plasma) parameters.

    Returns
    -------
    matplotlib.axes.Axes
        x: :math:`b/b_c - 1`; y: deflection angle in radians.
    """
    from blackpy import lensing

    limit = lensing.strong_deflection_limit(metric, plasma=plasma, **params)
    b_c = float(limit.b_c)
    if b is None:
        b = b_c * (1.0 + np.geomspace(1e-6, 30.0, 300))
    b = np.asarray(b, dtype=float)
    if np.any(b <= b_c):
        raise ValueError(
            f"Every impact parameter must exceed the critical one, b_c = {b_c:.6f} M (rays with "
            "b <= b_c are captured): pass b = b_c * (1 + x) with x > 0."
        )
    x = b / b_c - 1.0
    ax = _new_axes(ax, (6.4, 4.4))
    exact = np.asarray(lensing.deflection_angle(metric, b, plasma=plasma, **params))
    ax.loglog(x, exact, label=label or f"{metric.name}, exact")
    far = b >= 2.0 * b_c
    if weak and plasma is None and np.any(far):
        weak_angle = lensing.weak_deflection_angle(metric, b[far], **params)
        ax.loglog(x[far], np.asarray(weak_angle), "--", label="weak field (Gauss–Bonnet)")
    if strong:
        approx = lensing.strong_deflection_angle(metric, b, plasma=plasma, **params)
        ax.loglog(x, np.asarray(approx), ":", label="strong deflection limit")
    ax.axhline(2.0 * math.pi, color="0.6", lw=0.8)
    ax.set_xlabel(r"$b/b_c - 1$")
    ax.set_ylabel("deflection angle [rad]")
    ax.legend()
    return ax


def _cases(model, params):
    """``[(metric, params, label)]`` from a metric with (sequence) parameters or a list."""
    if isinstance(model, Metric):
        sweeps = {k: v for k, v in params.items() if np.ndim(v) > 0}
        if len(sweeps) > 1:
            raise ValueError(
                f"Sweep one parameter at a time, got sequences for {sorted(sweeps)}; for "
                "several metrics or combinations pass [(metric, params), ...]."
            )
        if not sweeps:
            label = ", ".join(f"{k} = {v:g}" for k, v in params.items()) or model.name
            return [(model, dict(params), label)]
        ((name, values),) = sweeps.items()
        return [
            (model, {**params, name: float(v)}, f"{model.name}, {name} = {float(v):g}")
            for v in values
        ]
    out = []
    for item in model:
        metric, item_params = item[0], dict(item[1])
        default = f"{metric.name} " + ", ".join(f"{k} = {v:g}" for k, v in item_params.items())
        out.append((metric, item_params, item[2] if len(item) > 2 else default.strip()))
    return out


def disk_flux(
    model,
    *,
    quantity: str = "flux",
    r=None,
    direction: str = "prograde",
    newtonian: bool = False,
    labels: Sequence[str] | None = None,
    ax=None,
    **params,
):
    r"""Page–Thorne flux or temperature profiles of thin disks.

    Draws :func:`blackpy.disk.page_thorne_flux` for several spins or metric
    parameters. Predictions only (see the disk module).

    Parameters
    ----------
    model : StaticSpherical or StationaryAxisymmetric, or sequence
        A metric, with one parameter in ``**params`` given as a sequence (one
        curve per value), or a sequence of ``(metric, params)`` or
        ``(metric, params, label)`` tuples.
    quantity : {"flux", "temperature"}
        Flux per unit accretion rate :math:`F M^2/\dot M` (log-log), or the
        effective temperature normalised to its maximum (log radius).
    r : array, optional
        Radii in units of M (default: those of :func:`page_thorne_flux`).
    direction : {"prograde", "retrograde"}
        Rotating metrics: orbit direction of the disk.
    newtonian : bool
        Also draw the large-radius limit :math:`3/(8\pi r^3)` (flux only).
    labels : sequence of str, optional
    ax : matplotlib axes, optional
    **params
        Metric parameters (one of them may be a sequence).

    Returns
    -------
    matplotlib.axes.Axes
        x: :math:`r/M`; y: :math:`F M^2/\dot M` or :math:`T_{\rm eff}/T_{\rm max}`.
    """
    from blackpy import disk

    if quantity not in ("flux", "temperature"):
        raise ValueError(f"quantity must be 'flux' or 'temperature', got {quantity!r}.")
    cases = _cases(model, params)
    if labels is not None and len(labels) != len(cases):
        raise ValueError(f"labels has {len(labels)} entries for {len(cases)} curves.")
    ax = _new_axes(ax, (6.4, 4.4))
    largest = 0.0
    for k, (metric, item_params, label) in enumerate(cases):
        options = {"direction": direction} if metric.is_rotating_class else {}
        profile = disk.page_thorne_flux(metric, r=r, **options, **item_params)
        text = labels[k] if labels is not None else label
        if quantity == "flux":
            ax.loglog(profile.r, np.clip(profile.flux, 1e-14, None), label=text)
            largest = max(largest, float(np.nanmax(profile.flux)))
        else:
            ax.semilogx(profile.r, profile.temperature, label=text)
    if quantity == "flux":
        if newtonian:
            radii = np.geomspace(10.0, float(np.max(ax.get_xlim())), 50)
            ax.loglog(radii, 3.0 / (8.0 * math.pi * radii**3), "k:", label=r"$3/(8\pi r^3)$")
        ax.set_ylim(largest * 1e-8, largest * 3.0)
        ax.set_ylabel(r"$F\,M^2/\dot M$")
    else:
        ax.set_ylabel(r"$T_{\rm eff}/T_{\rm max}$")
    ax.set_xlabel(r"$r/M$")
    ax.legend(fontsize="small")
    return ax


def _axis_label(name: str) -> str:
    for prefix, unit in _UNITS:
        if name.startswith(prefix):
            return f"{name} {unit}"
    return name


def _ellipse(mean, cov, level: float, n: int = 200):
    """Points of the ``level`` contour of a 2-D Gaussian (``n`` points)."""
    radius = math.sqrt(-2.0 * math.log1p(-level))  # chi-square quantile for 2 dof
    values, vectors = np.linalg.eigh(cov)
    t = np.linspace(0.0, 2.0 * math.pi, n)
    circle = np.stack([np.cos(t), np.sin(t)])
    points = vectors @ (np.sqrt(np.clip(values, 0.0, None))[:, None] * circle) * radius
    return mean[0] + points[0], mean[1] + points[1]


def forecast(
    prediction,
    *,
    parameters: Sequence[str] | None = None,
    figure=None,
    levels: Sequence[float] = (0.68, 0.95),
    color: str = "C1",
    label: str = "Fisher forecast",
):
    """Fisher-forecast ellipses, alone or on top of a corner plot.

    Draws the 68% and 95% ellipses of the forecast covariance
    (:class:`blackpy.infer.Forecast`) for every pair of parameters, centred on
    the fiducial point, and the 1-D Gaussians on the diagonal. Pass
    ``figure=bp.plot.corner(result, parameters=...)`` to overlay them on the
    posterior of a fit with the same parameters (in the same order).

    Parameters
    ----------
    prediction : Forecast
    parameters : sequence of str, optional
        Parameters to draw, in order (default: all of the forecast). With
        ``figure``, the parameters of the corner plot.
    figure : matplotlib.figure.Figure, optional
        A corner plot (``K x K`` axes); a new figure otherwise.
    levels : sequence of float
        Probability contents of the 2-D ellipses.
    color : str
    label : str
        Legend label (shown in the top-right panel).

    Returns
    -------
    matplotlib.figure.Figure

    Notes
    -----
    Directions with an infinite forecast uncertainty (degenerate, see
    ``prediction.degenerate_directions``) have no ellipse; the panel says so.
    The ellipses are the Gaussian approximation of the forecast: see its
    limits in :func:`blackpy.infer.forecast`.
    """
    import matplotlib.pyplot as plt

    names = list(parameters or prediction.param_names)
    unknown = [n for n in names if n not in prediction.param_names]
    if unknown:
        raise ValueError(
            f"{unknown} are not parameters of the forecast; its parameters: "
            f"{prediction.param_names}."
        )
    index = [prediction.param_names.index(n) for n in names]
    k = len(names)
    if figure is None:
        figure, axes = plt.subplots(k, k, figsize=(2.4 * k + 0.6, 2.4 * k + 0.6), squeeze=False)
        own = True
    else:
        if len(figure.axes) != k * k:
            raise ValueError(
                f"The figure has {len(figure.axes)} axes, not {k} x {k}: pass the parameters of "
                "the corner plot, in its order, as parameters=[...]."
            )
        axes = np.array(figure.axes).reshape(k, k)
        own = False
    mean = np.array([prediction.fiducial[n] for n in names])
    cov = np.asarray(prediction.covariance)[np.ix_(index, index)]
    with np.errstate(invalid="ignore"):
        sigmas = np.sqrt(np.diag(cov))
    spans = [
        (m - 4.0 * s, m + 4.0 * s) if np.isfinite(s) and s > 0 else None
        for m, s in zip(mean, sigmas, strict=True)
    ]
    for i in range(k):
        for j in range(k):
            ax = axes[i, j]
            if j > i:
                if own:
                    ax.set_visible(False)
                continue
            if i == j:
                sigma = math.sqrt(cov[i, i]) if np.isfinite(cov[i, i]) else math.inf
                if math.isfinite(sigma) and sigma > 0:
                    span = (mean[i] - 4.0 * sigma, mean[i] + 4.0 * sigma)
                    lo, hi = span if own else ax.get_xlim()
                    x = np.linspace(lo, hi, 400)
                    y = np.exp(-0.5 * ((x - mean[i]) / sigma) ** 2)
                    scale = 0.9 * ax.get_ylim()[1] if not own else 1.0
                    ax.plot(x, scale * y, color=color)
                else:
                    ax.text(0.5, 0.5, "unconstrained", transform=ax.transAxes, ha="center")
                if own:
                    ax.set_yticks([])
            else:
                block = cov[np.ix_([j, i], [j, i])]
                if np.all(np.isfinite(block)):
                    for level in levels:
                        ex, ey = _ellipse(mean[[j, i]], block, level)
                        ax.plot(ex, ey, color=color, lw=1.2)
                else:
                    ax.text(0.5, 0.5, "degenerate", transform=ax.transAxes, ha="center")
                ax.plot(mean[j], mean[i], "+", color=color)
            if own:
                from matplotlib.ticker import MaxNLocator

                if spans[j] is not None:
                    ax.set_xlim(*spans[j])
                if i != j and spans[i] is not None:
                    ax.set_ylim(*spans[i])
                ax.xaxis.set_major_locator(MaxNLocator(3))
                ax.yaxis.set_major_locator(MaxNLocator(3))
                ax.tick_params(axis="x", labelrotation=45)
                if i == k - 1:
                    ax.set_xlabel(_axis_label(names[j]))
                else:
                    ax.set_xticklabels([])
                if j == 0 and i > 0:
                    ax.set_ylabel(_axis_label(names[i]))
                elif i != j:
                    ax.set_yticklabels([])
                if i == j:
                    ax.set_yticks([])
    axes[0, 0].plot([], [], color=color, label=label)
    axes[0, 0].legend(
        loc="upper left", bbox_to_anchor=(1.05, 1.0) if k > 1 else None, fontsize="small"
    )
    if own:
        figure.tight_layout()
    return figure


__all__ = ["deflection", "disk_flux", "forecast"]
