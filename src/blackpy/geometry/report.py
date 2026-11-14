"""One-call geometry report for a metric."""

from __future__ import annotations

from blackpy.geometry.curvature import is_vacuum_symbolic, stress_energy_symbolic
from blackpy.geometry.energy_conditions import energy_conditions
from blackpy.geometry.horizons import horizons


def report(metric, **params) -> str:
    """Run the geometry checks and return a plain-text report.

    Symbolic checks (exact stress-energy, vacuum test) are run when the
    metric has a symbolic form; otherwise they are skipped with an explicit
    note and only the numerical checks are reported.

    Parameters
    ----------
    metric : StaticSpherical
    **params
        Scalar metric parameters.

    Returns
    -------
    str
    """
    lines = [
        f"Geometry report: {metric.name}",
        f"  parameters: {dict(zip(metric.param_names, metric.resolve(params), strict=True))}",
    ]
    hz = horizons(metric, **params)
    lines.append(str(hz))
    flat = metric.is_asymptotically_flat(**params)
    lines.append(f"Asymptotically flat: {flat}")
    if metric.is_symbolic:
        import sympy as sp

        lines.append(f"Line element: {metric.latex()}")
        lines.append(f"Vacuum solution (G^mu_nu = 0): {is_vacuum_symbolic(metric)}")
        for key, expr in stress_energy_symbolic(metric).items():
            lines.append(f"  {key} = {sp.sstr(expr)}")
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
