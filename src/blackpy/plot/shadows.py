"""Shadow contour plots on celestial coordinates."""

from __future__ import annotations

import math

import numpy as np

from blackpy import light
from blackpy.data import ShadowDataset, load
from blackpy.light.rotating import check_delta_c_definition, warn_if_definition_mismatch
from blackpy.metric.base import Metric

_DELTA_C_SYMBOL = {
    "fractional_rms": r"\Delta C",
    "jp_asymmetry": r"A/\langle R\rangle",
    "bambi2019": r"\Delta C_{\rm B19}",
}


def _scan(inclination, params: dict) -> list[tuple[float, dict]]:
    """Expand sequence-valued inclination/parameters into a list of curves."""
    values = {"inclination": inclination, **params}
    lengths = {k: len(np.atleast_1d(v)) for k, v in values.items() if np.ndim(v) > 0}
    if len(set(lengths.values())) > 1:
        raise ValueError(
            f"Sequence-valued arguments must have the same length, got {lengths}. Each element "
            "defines one contour."
        )
    n = next(iter(lengths.values()), 1)
    curves = []
    for k in range(n):
        point = {
            key: float(np.atleast_1d(v)[k if np.ndim(v) > 0 else 0]) for key, v in values.items()
        }
        curves.append((point.pop("inclination"), point))
    return curves


def shadow(
    metric: Metric,
    inclination,
    *,
    r_obs: float | None = None,
    labels=None,
    reference_circle: bool | float = False,
    band: ShadowDataset | str | None = None,
    band_level: str = "1sigma",
    recentre: bool = False,
    annotate_delta_c: str | None = None,
    circularity_bound_definition: str | None = None,
    n: int = 256,
    ax=None,
    **params,
):
    r"""Plot one or several shadow contours on celestial coordinates.

    Any of ``inclination`` and the metric parameters may be a sequence; each
    element then defines one contour (sequences must have equal length, the
    other arguments are shared). Examples::

        bp.plot.shadow(kerr, np.pi / 2, a=[0.0, 0.5, 0.9, 0.998])      # several spins
        bp.plot.shadow(kerr, np.radians([17, 45, 90]), a=0.9)          # several inclinations
        bp.plot.shadow(kn, np.pi / 3, a=0.5, Q=[0.0, 0.4, 0.8])        # a metric parameter

    Parameters
    ----------
    metric : StationaryAxisymmetric or StaticSpherical
        Must have separable null geodesics (a static metric gives circles).
    inclination : float or sequence
        Polar angle of the observer in **radians** (``pi/2`` is edge-on).
    r_obs : float, optional
        Observer radius; default: observer at infinity (see
        :func:`blackpy.light.shadow_contour` for the screen coordinates).
    labels : sequence of str, optional
        Legend labels, one per contour. Default: built from the scanned
        quantities.
    reference_circle : bool or float
        Draw a dashed circle centred on the origin: ``True`` for the
        Schwarzschild radius :math:`3\sqrt3`, or a radius in units of M.
    band : ShadowDataset or registry id, optional
        Shade the interval of the shadow radius allowed by a shadow dataset
        (an annulus centred on the origin between the bounds of
        ``band_level``). The published bounds refer to a radius, not to a
        position, so combine this with ``recentre=True``.
    band_level : str
        Which bound of the dataset to draw (e.g. ``"1sigma"``).
    recentre : bool
        Shift every contour so that its geometric centre is at the origin.
    annotate_delta_c : str, optional
        If given, the deviation from circularity is added to each legend
        entry. The value is the **name of the definition**
        (``"fractional_rms"``, ``"jp_asymmetry"`` or ``"bambi2019"``); there
        is deliberately no default, because the definitions differ by
        factors of 2 to 4 (:func:`blackpy.light.deviation_from_circularity`).
    circularity_bound_definition : str, optional
        Definition for which the circularity bound of ``band`` was derived,
        if it differs from the one recorded in the dataset. Only used to
        decide whether a :class:`blackpy.errors.CircularityDefinitionWarning`
        is due when ``annotate_delta_c`` is combined with a dataset that
        carries a circularity bound.
    n : int
        Points per half contour.
    ax : matplotlib axes, optional
    **params
        Metric parameters (scalars or sequences).

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if annotate_delta_c is not None:
        check_delta_c_definition(annotate_delta_c)
    curves = _scan(inclination, params)
    if labels is not None and len(labels) != len(curves):
        raise ValueError(f"labels has {len(labels)} entries for {len(curves)} contours.")
    scanned = [
        k
        for k in ("inclination", *params)
        if np.ndim({"inclination": inclination, **params}[k]) > 0
    ]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.2, 5.2))

    if band is not None:
        dataset = load(band) if isinstance(band, str) else band
        if not isinstance(dataset, ShadowDataset):
            raise TypeError("band must be a ShadowDataset or the registry id of one.")
        if band_level not in dataset.bounds or None in dataset.bounds[band_level]:
            raise ValueError(
                f"Dataset '{dataset.id}' has no complete '{band_level}' bound. "
                f"Available: {list(dataset.bounds)}."
            )
        lower, upper = dataset.bounds[band_level]
        angle = np.linspace(0.0, 2.0 * math.pi, 361)
        level = band_level.replace("sigma", r"$\sigma$")
        ax.fill(
            np.concatenate([upper * np.cos(angle), lower * np.cos(angle[::-1])]),
            np.concatenate([upper * np.sin(angle), lower * np.sin(angle[::-1])]),
            color="0.85",
            lw=0,
            zorder=0,
            label=f"{dataset.object} {level}: {lower:g} to {upper:g}",
        )
        bound = getattr(dataset, "circularity", None)
        if annotate_delta_c is not None and bound is not None:
            warn_if_definition_mismatch(
                annotate_delta_c,
                circularity_bound_definition or bound.definition,
                f"The circularity bound of '{dataset.id}'",
            )

    for k, (inc, point) in enumerate(curves):
        contour = light.shadow_contour(metric, inc, r_obs, n=n, **point)
        alpha, beta = np.asarray(contour.alpha), np.asarray(contour.beta)
        if recentre:
            centroid = float(light.shadow_observables(metric, inc, r_obs, n=n, **point).centroid)
            alpha = alpha - centroid
        if labels is not None:
            label = labels[k]
        else:
            parts = []
            for key in scanned:
                if key == "inclination":
                    parts.append(rf"$i = {math.degrees(inc):.3g}^\circ$")
                else:
                    symbol = next(p.symbol for p in metric.parameters if p.name == key)
                    parts.append(rf"${symbol} = {point[key]:.4g}$")
            label = ", ".join(parts) or metric.name
        if annotate_delta_c is not None:
            value = float(
                light.deviation_from_circularity(
                    metric, inc, r_obs, definition=annotate_delta_c, n=n, **point
                )
            )
            label += rf" (${_DELTA_C_SYMBOL[annotate_delta_c]} = {value:.3f}$)"
        ax.plot(np.append(alpha, alpha[0]), np.append(beta, beta[0]), label=label)

    if reference_circle:
        radius = 3.0 * math.sqrt(3.0) if reference_circle is True else float(reference_circle)
        angle = np.linspace(0.0, 2.0 * math.pi, 361)
        name = r"$3\sqrt{3}$ (Schwarzschild)" if reference_circle is True else f"r = {radius:g}"
        ax.plot(radius * np.cos(angle), radius * np.sin(angle), "k--", lw=0.8, label=name)

    ax.set_aspect("equal")
    unit = "M" if r_obs is None else r"M\ {\rm (screen)}"
    ax.set_xlabel(rf"$\alpha / {unit}$" + (" (recentred)" if recentre else ""))
    ax.set_ylabel(rf"$\beta / {unit}$")
    ax.grid(alpha=0.3)
    ax.legend(fontsize="small")
    return ax
