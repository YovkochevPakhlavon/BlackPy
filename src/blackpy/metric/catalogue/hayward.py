"""Hayward regular black hole.

Reference
---------
S. A. Hayward, "Formation and evaporation of nonsingular black holes",
Phys. Rev. Lett. 96 (2006) 031103.
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = "Hayward (2006), Phys. Rev. Lett. 96, 031103"


def Hayward() -> StaticSpherical:
    r"""Hayward regular black hole, :math:`f = g = 1 - 2 r^2 / (r^3 + 2 l^2)`.

    Units: G = c = M = 1.

    Parameters of the metric
    ------------------------
    l : float, default 0
        Length scale of the regular core in units of M, ``l >= 0``. Horizons
        exist for ``l <= 4 / (3 sqrt(3))``, approximately 0.7698.

    Returns
    -------
    StaticSpherical
    """
    r = sp.Symbol("r", positive=True)
    ell = sp.Symbol("l", real=True)
    return StaticSpherical.from_sympy(
        1 - 2 * r**2 / (r**3 + 2 * ell**2),
        r=r,
        parameters=[
            Parameter("l", 0.0, (0.0, math.inf), latex=r"\ell", description="core length scale l/M")
        ],
        name="Hayward",
        asymptotically_flat=True,
        reference=REFERENCE,
    )
