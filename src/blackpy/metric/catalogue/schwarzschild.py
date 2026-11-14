"""Schwarzschild metric.

Reference
---------
K. Schwarzschild, "Ueber das Gravitationsfeld eines Massenpunktes nach der
Einsteinschen Theorie", Sitzungsber. Preuss. Akad. Wiss. Berlin (Math. Phys.)
1916, 189-196.
"""

from __future__ import annotations

import sympy as sp

from blackpy.metric.base import StaticSpherical

REFERENCE = "Schwarzschild (1916), Sitzungsber. Preuss. Akad. Wiss. Berlin, 189"


def Schwarzschild() -> StaticSpherical:
    r"""Schwarzschild black hole, :math:`f = g = 1 - 2/r`, :math:`h = r^2`.

    Units: G = c = M = 1. The metric has no free parameters.

    Known results (used as tests): event horizon ``r_h = 2``, photon sphere
    ``r_ph = 3``, critical impact parameter ``b_c = 3 sqrt(3)``,
    ``r_ISCO = 6``, marginally bound orbit ``r_mb = 4``.

    Returns
    -------
    StaticSpherical
    """
    r = sp.Symbol("r", positive=True)
    return StaticSpherical.from_sympy(
        1 - 2 / r,
        r=r,
        parameters=(),
        name="Schwarzschild",
        asymptotically_flat=True,
        reference=REFERENCE,
    )
