r"""Point-wise energy conditions for the effective stress-energy.

For an anisotropic fluid ``T^mu_nu = diag(-rho, p_r, p_t, p_t)`` (Hawking &
Ellis type I) the conditions are, with ``p_i`` running over ``p_r, p_t``:

* NEC (null):      ``rho + p_i >= 0``
* WEC (weak):      NEC and ``rho >= 0``
* SEC (strong):    NEC and ``rho + p_r + 2 p_t >= 0``
* DEC (dominant):  ``rho >= |p_i|``

References: S. W. Hawking & G. F. R. Ellis, *The Large Scale Structure of
Space-Time* (Cambridge University Press, 1973), section 4.3; E. Curiel, "A
primer on energy conditions", Einstein Stud. 13 (2017) 43, arXiv:1405.0403.

They are evaluated only in the static region outside the event horizon,
where ``rho = -T^t_t`` is the energy density seen by static observers.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from blackpy.geometry.curvature import stress_energy
from blackpy.geometry.horizons import static_domain_core

CONDITIONS = ("NEC", "WEC", "SEC", "DEC")


@dataclass(frozen=True)
class EnergyConditionResult:
    """Energy conditions on a radial grid.

    Attributes
    ----------
    r : numpy.ndarray
        Radial grid (units of M), outside the event horizon.
    rho, p_r, p_t : numpy.ndarray
        Effective density and pressures on the grid (units of 1/M^2).
    satisfied : dict of str -> numpy.ndarray of bool
        For each of NEC, WEC, SEC, DEC: True where the condition holds.
    rtol : float
        Relative tolerance used; a combination ``x`` counts as non-negative
        if ``x >= -rtol / h(r)``, which absorbs round-off in vacuum.
    """

    r: np.ndarray
    rho: np.ndarray
    p_r: np.ndarray
    p_t: np.ndarray
    satisfied: dict
    rtol: float

    def holds(self, condition: str) -> bool:
        """True if ``condition`` holds on the whole grid."""
        return bool(np.all(self.satisfied[condition.upper()]))

    def violated(self, condition: str) -> np.ndarray:
        """Grid radii at which ``condition`` is violated."""
        return self.r[~self.satisfied[condition.upper()]]

    def violation_intervals(self, condition: str) -> list[tuple[float, float]]:
        """Contiguous radial intervals ``(r_start, r_end)`` of violation (grid resolution)."""
        bad = ~self.satisfied[condition.upper()]
        intervals, start = [], None
        for i, flag in enumerate(bad):
            if flag and start is None:
                start = i
            if start is not None and (not flag or i == len(bad) - 1):
                end = i if flag else i - 1
                intervals.append((float(self.r[start]), float(self.r[end])))
                start = None
        return intervals

    def summary(self) -> str:
        """Human-readable summary of where each condition is violated."""
        lines = []
        for name in CONDITIONS:
            if self.holds(name):
                lines.append(f"{name}: satisfied on [{self.r[0]:.6g}, {self.r[-1]:.6g}]")
            else:
                spans = ", ".join(f"[{a:.6g}, {b:.6g}]" for a, b in self.violation_intervals(name))
                lines.append(f"{name}: VIOLATED for r in {spans}")
        return "\n".join(lines)


def energy_conditions(
    metric, *, r=None, r_max: float | None = None, n: int = 400, rtol: float = 1e-10, **params
) -> EnergyConditionResult:
    """Evaluate NEC, WEC, SEC and DEC on a radial grid outside the event horizon.

    Parameters
    ----------
    metric : StaticSpherical
    r : array, optional
        Explicit radial grid. Default: ``n`` logarithmically spaced points
        from just outside the event horizon to ``r_max``.
    r_max : float, optional
        Outer edge of the default grid. Default: 100 times the inner edge,
        capped at the cosmological horizon if there is one.
    n : int
        Number of grid points of the default grid.
    rtol : float
        Round-off tolerance, see :class:`EnergyConditionResult`.
    **params
        Scalar metric parameters.

    Returns
    -------
    EnergyConditionResult
    """
    if r is None:
        _, r_cosmo, r_in, r_out = (float(v) for v in metric.evaluate(static_domain_core, params))
        lo = r_in * (1.0 + 1e-6)
        hi = 100.0 * lo if r_max is None else float(r_max)
        if np.isfinite(r_cosmo):
            hi = min(hi, r_out * (1.0 - 1e-6))
        r = np.geomspace(lo, hi, n)
    r = np.asarray(r, dtype=float)
    rho, p_r, p_t = (np.asarray(v) for v in stress_energy(metric, r, **params))
    tol = rtol / np.asarray(metric.h(r, **params))

    nonneg = lambda x: x >= -tol
    nec = nonneg(rho + p_r) & nonneg(rho + p_t)
    wec = nec & nonneg(rho)
    sec = nec & nonneg(rho + p_r + 2.0 * p_t)
    dec = nonneg(rho - np.abs(p_r)) & nonneg(rho - np.abs(p_t))
    satisfied = {"NEC": nec, "WEC": wec, "SEC": sec, "DEC": dec}
    return EnergyConditionResult(r=r, rho=rho, p_r=p_r, p_t=p_t, satisfied=satisfied, rtol=rtol)
