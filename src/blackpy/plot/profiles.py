"""Profile plots: an observable as a function of a metric parameter."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np

from blackpy import geometry, light, orbits
from blackpy.metric.base import Metric


def _isco_frequency(kind: str):
    def quantity(metric, mass=None, **params):
        r_isco = orbits.isco(metric, **params)
        if mass is None:
            omega = {"phi": orbits.angular_velocity, "r": orbits.radial_epicyclic_frequency}[kind]
            return omega(metric, r_isco, **params)
        return getattr(orbits.frequencies_hz(metric, r_isco, mass, **params), f"nu_{kind}")

    return quantity


def _frequency(kind: str):
    def quantity(metric, r=None, mass=None, **params):
        if r is None:
            raise ValueError(f"Quantity 'nu_{kind}' needs the orbit radius: pass r=<radius in M>.")
        if mass is None:
            fn = {
                "phi": orbits.angular_velocity,
                "r": orbits.radial_epicyclic_frequency,
                "theta": orbits.vertical_epicyclic_frequency,
            }[kind]
            return fn(metric, r, **params)
        return getattr(orbits.frequencies_hz(metric, r, mass, **params), f"nu_{kind}")

    return quantity


def _simple(fn):
    return lambda metric, r=None, mass=None, **params: fn(metric, **params)


#: Built-in quantities: name -> (function, axis label in geometric units).
QUANTITIES: dict[str, tuple[Callable, str]] = {
    "event_horizon": (_simple(geometry.event_horizon), r"$r_{\rm h}/M$"),
    "isco": (_simple(orbits.isco), r"$r_{\rm ISCO}/M$"),
    "marginally_bound": (_simple(orbits.marginally_bound_orbit), r"$r_{\rm mb}/M$"),
    "photon_sphere": (_simple(light.photon_sphere), r"$r_{\rm ph}/M$"),
    "b_c": (_simple(light.critical_impact_parameter), r"$b_{\rm c}/M$"),
    "shadow_radius": (_simple(light.shadow_radius), r"$r_{\rm sh}/M$"),
    "nu_phi": (_frequency("phi"), r"$\Omega_\phi M$"),
    "nu_r": (_frequency("r"), r"$\Omega_r M$"),
    "nu_theta": (_frequency("theta"), r"$\Omega_\theta M$"),
    "nu_phi_isco": (
        lambda metric, r=None, mass=None, **p: _isco_frequency("phi")(metric, mass, **p),
        r"$\Omega_\phi(r_{\rm ISCO})\,M$",
    ),
}
_ALIASES = {
    "r_isco": "isco",
    "r_ph": "photon_sphere",
    "r_h": "event_horizon",
    "critical_impact_parameter": "b_c",
    "r_sh": "shadow_radius",
    "r_mb": "marginally_bound",
}
_HZ_LABELS = {
    "nu_phi": r"$\nu_\phi$ [Hz]",
    "nu_r": r"$\nu_r$ [Hz]",
    "nu_theta": r"$\nu_\theta$ [Hz]",
    "nu_phi_isco": r"$\nu_\phi(r_{\rm ISCO})$ [Hz]",
}


def _split_grid(metric, param_grid: Mapping) -> tuple[str, np.ndarray, dict]:
    sweeps = {k: np.asarray(v, dtype=float) for k, v in param_grid.items() if np.ndim(v) > 0}
    if len(sweeps) != 1:
        raise ValueError(
            f"profile() needs exactly one array-valued parameter to sweep for metric "
            f"'{metric.name}', got {sorted(sweeps) or 'none'}. Example: profile(metric, 'isco', "
            f"{(metric.param_names or ('param',))[0]}=np.linspace(0, 1, 100))."
        )
    ((name, values),) = sweeps.items()
    if name not in metric.param_names:
        raise ValueError(
            f"Metric '{metric.name}' has no parameter '{name}'. Its parameters: "
            f"{list(metric.param_names)}."
        )
    scalars = {k: v for k, v in param_grid.items() if k != name}
    return name, values, scalars


def profile(
    model,
    quantity: str | Callable,
    *,
    ax=None,
    r: float | None = None,
    mass: float | None = None,
    labels: Sequence[str] | None = None,
    **param_grid,
):
    """Plot an observable against a metric parameter.

    Parameters
    ----------
    model : StaticSpherical, or sequence of ``(metric, param_grid)`` pairs
        A single metric (swept with ``**param_grid``), or several metrics on
        one figure, each with its own grid, e.g.
        ``[(rn, {"Q": q}), (bardeen, {"q": q})]``.
    quantity : str or callable
        One of ``"event_horizon"``, ``"isco"``, ``"marginally_bound"``,
        ``"photon_sphere"``, ``"b_c"``, ``"shadow_radius"``, ``"nu_phi"``,
        ``"nu_r"``, ``"nu_theta"`` (these three need ``r``),
        ``"nu_phi_isco"``; or a callable ``fn(metric, **params) -> array``.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on (default: a new figure).
    r : float, optional
        Orbit radius in units of M for the frequency quantities.
    mass : float, optional
        Black hole mass in solar masses. If given, frequencies are in Hz;
        otherwise dimensionless angular frequencies are plotted.
    labels : sequence of str, optional
        Legend labels (default: metric names).
    **param_grid
        Exactly one array-valued metric parameter (the abscissa); further
        scalar parameters are held fixed.

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt

    if isinstance(model, Metric):
        entries = [(model, param_grid)]
    else:
        if param_grid:
            raise ValueError(
                "With several metrics, give each its own grid: "
                "profile([(metric1, {'a': grid}), (metric2, {'b': grid})], quantity)."
            )
        entries = [(m, dict(g)) for m, g in model]
    if callable(quantity):
        fn = lambda metric, r=None, mass=None, **p: quantity(metric, **p)
        ylabel = getattr(quantity, "__name__", "quantity")
        key = None
    else:
        key = _ALIASES.get(quantity, quantity)
        if key not in QUANTITIES:
            raise ValueError(f"Unknown quantity '{quantity}'. Available: {sorted(QUANTITIES)}.")
        fn, ylabel = QUANTITIES[key]
        if mass is not None and key in _HZ_LABELS:
            ylabel = _HZ_LABELS[key]
    if ax is None:
        _, ax = plt.subplots(figsize=(5.5, 4.0))
    symbols = []
    for i, (metric, grid) in enumerate(entries):
        name, values, scalars = _split_grid(metric, grid)
        y = np.asarray(fn(metric, r=r, mass=mass, **{name: values}, **scalars))
        label = labels[i] if labels is not None else metric.name
        ax.plot(values, y, label=label)
        symbols.append(metric.parameters[metric.param_names.index(name)].symbol)
    xlabel = f"${symbols[0]}$" if len(set(symbols)) == 1 else "metric parameter"
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if len(entries) > 1 or labels is not None:
        ax.legend()
    return ax
