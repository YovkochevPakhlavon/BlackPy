"""Bardeen regular black hole.

References
----------
J. M. Bardeen, "Non-singular general-relativistic gravitational collapse",
in Proceedings of the International Conference GR5, Tbilisi, USSR (1968),
p. 174.

E. Ayon-Beato and A. Garcia, "The Bardeen model as a nonlinear magnetic
monopole", Phys. Lett. B 493 (2000) 149-152 (interpretation of the parameter
as a magnetic monopole charge of nonlinear electrodynamics).
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = (
    "Bardeen (1968), Proc. GR5 Tbilisi, 174; Ayon-Beato & Garcia (2000), Phys. Lett. B 493, 149"
)


def Bardeen() -> StaticSpherical:
    r"""Bardeen regular black hole, :math:`f = g = 1 - 2 r^2 / (r^2 + q^2)^{3/2}`.

    Units: G = c = M = 1.

    Parameters of the metric
    ------------------------
    q : float, default 0
        Magnetic monopole charge in units of M (often written ``g`` in the
        literature; renamed here to avoid confusion with the metric function
        ``g``). ``q >= 0``; horizons exist for ``q <= 4 / (3 sqrt(3))``,
        approximately 0.7698.

    Returns
    -------
    StaticSpherical
    """
    r = sp.Symbol("r", positive=True)
    q = sp.Symbol("q", real=True)
    return StaticSpherical.from_sympy(
        1 - 2 * r**2 / (r**2 + q**2) ** sp.Rational(3, 2),
        r=r,
        parameters=[
            Parameter("q", 0.0, (0.0, math.inf), latex="q", description="magnetic charge q/M")
        ],
        name="Bardeen",
        asymptotically_flat=True,
        reference=REFERENCE,
    )
