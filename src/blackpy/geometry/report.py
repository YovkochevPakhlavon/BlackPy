"""One-call geometry report for a metric."""

from __future__ import annotations

import math

from blackpy.errors import SymbolicTimeoutError
from blackpy.geometry.axisymmetric import effective_source
from blackpy.geometry.curvature import is_vacuum_symbolic, stress_energy_symbolic
from blackpy.geometry.energy_conditions import energy_conditions
from blackpy.geometry.horizons import ergosurface, horizons


def report(metric, *, symbolic: bool | None = None, timeout: float = 60.0, **params) -> str:
    """Run the geometry checks and return a plain-text report.

    Static metrics: symbolic checks (exact stress-energy, vacuum test) are
    run when the metric has a symbolic form; otherwise they are skipped with
    an explicit note and only the numerical checks are reported.

    Rotating metrics: horizons (with the theta-independence check),
    ergosurface, reflection symmetry, separability, and the effective source
    with its type and energy conditions from the numerical (autodiff)
    computation. The symbolic 4D vacuum test is optional because it can be
    slow.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    symbolic : bool, optional
        Run the symbolic checks. Default: True for static symbolic metrics,
        False for rotating metrics.
    timeout : float
        Time limit in seconds of the symbolic 4D computation (rotating
        metrics).
    **params
        Scalar metric parameters.

    Returns
    -------
    str
    """
    if metric.is_rotating_class:
        return _report_rotating(metric, bool(symbolic), timeout, params)
    lines = [
        f"Geometry report: {metric.name}",
        f"  parameters: {dict(zip(metric.param_names, metric.resolve(params), strict=True))}",
    ]
    hz = horizons(metric, **params)
    lines.append(str(hz))
    flat = metric.is_asymptotically_flat(**params)
    lines.append(f"Asymptotically flat: {flat}")
    if metric.is_symbolic and symbolic is not False:
        import sympy as sp

        lines.append(f"Line element: {metric.latex()}")
        lines.append(f"Vacuum solution (G^mu_nu = 0): {is_vacuum_symbolic(metric)}")
        for key, expr in stress_energy_symbolic(metric).items():
            lines.append(f"  {key} = {sp.sstr(expr)}")
    elif metric.is_symbolic:
        lines.append("Symbolic checks skipped on request (symbolic=False).")
    else:
        lines.append(
            "Symbolic checks skipped: the metric was defined with numerical callables. "
            "Numerical stress-energy (autodiff) is used below."
        )
    if hz.horizonless:
        lines.append(
            "Energy conditions: evaluated from the lower end of the scan range (no horizon)."
        )
    lines.append(energy_conditions(metric, **params).summary())
    return "\n".join(lines)


def _report_rotating(metric, symbolic: bool, timeout: float, params: dict) -> str:
    lines = [
        f"Geometry report: {metric.name} (stationary, axisymmetric)",
        f"  parameters: {dict(zip(metric.param_names, metric.resolve(params), strict=True))}",
        f"  inside the declared black hole domain: {bool(metric.in_domain(**params))}",
    ]
    hz = horizons(metric, **params)
    lines.append(str(hz))
    r_ergo = float(ergosurface(metric, **params))
    lines.append(
        f"Ergosurface (g_tt = 0) on the equator: r = {r_ergo:.12g}"
        if math.isfinite(r_ergo)
        else "Ergosurface: g_tt has no root on the equator."
    )
    lines.append(f"Asymptotically flat: {metric.is_asymptotically_flat(**params)}")
    lines.append(
        f"Reflection symmetric (theta -> pi - theta): {metric.check_reflection_symmetry(**params)}"
    )
    lines.append(f"Null geodesics separable: {metric.check_separability(**params)}")
    if symbolic and not metric.is_symbolic:
        lines.append("Symbolic checks skipped: the metric was defined with numerical callables.")
    elif symbolic:
        try:
            lines.append(
                "Vacuum solution (G^mu_nu = 0, symbolic 4D computation): "
                f"{is_vacuum_symbolic(metric, timeout)}"
            )
        except SymbolicTimeoutError as error:
            lines.append(f"Symbolic vacuum test not completed: {error}")
    else:
        lines.append(
            "Symbolic 4D checks not run (they can be slow); pass symbolic=True to run them. "
            "The numerical source analysis follows."
        )
    lines.append(effective_source(metric, **params).summary())
    return "\n".join(lines)
