"""Reissner-Nordstrom metric.

References
----------
H. Reissner, "Ueber die Eigengravitation des elektrischen Feldes nach der
Einsteinschen Theorie", Ann. Phys. 355 (1916) 106-120.

G. Nordstrom, "On the energy of the gravitational field in Einstein's
theory", Proc. Kon. Ned. Akad. Wet. 20 (1918) 1238-1245.
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = (
    "Reissner (1916), Ann. Phys. 355, 106; Nordstrom (1918), Proc. Kon. Ned. Akad. Wet. 20, 1238"
)


def ReissnerNordstrom() -> StaticSpherical:
    r"""Reissner-Nordstrom black hole, :math:`f = g = 1 - 2/r + Q^2/r^2`.

    Units: G = c = M = 1 and Gaussian-geometrized charge, so ``Q`` is the
    charge-to-mass ratio Q/M.

    Parameters of the metric
    ------------------------
    Q : float, default 0
        Charge in units of M, ``Q >= 0`` (only ``Q^2`` enters). A black hole
        exists for ``Q <= 1``; ``Q > 1`` is a naked singularity, which
        BlackPy reports as horizonless.

    Known results (used as tests): horizons ``r_pm = 1 +- sqrt(1 - Q^2)``,
    photon sphere ``r_ph = (3 + sqrt(9 - 8 Q^2)) / 2``, extremal ISCO
    ``r_ISCO(Q = 1) = 4``, electromagnetic energy density
    ``rho = Q^2 / (8 pi r^4)``.

    Returns
    -------
    StaticSpherical
    """
    r = sp.Symbol("r", positive=True)
    q = sp.Symbol("Q", real=True)
    return StaticSpherical.from_sympy(
        1 - 2 / r + q**2 / r**2,
        r=r,
        parameters=[
            Parameter("Q", 0.0, (0.0, math.inf), latex="Q", description="electric charge Q/M")
        ],
        name="Reissner-Nordstrom",
        asymptotically_flat=True,
        reference=REFERENCE,
    )
