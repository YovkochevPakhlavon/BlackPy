"""Gibbons-Maeda-Garfinkle-Horowitz-Strominger (GMGHS) dilaton black hole.

References
----------
G. W. Gibbons & K. Maeda, Nucl. Phys. B 298 (1988) 741; D. Garfinkle, G. T.
Horowitz & A. Strominger, Phys. Rev. D 43 (1991) 3140. Gibbons & Maeda was
read later: their Eqs. (4.15)-(4.19) with D = 4, g_2 = 1 give this line
element (see the theory page); Garfinkle, Horowitz & Strominger was not
accessed. The metric was originally taken from the static
(a = 0) limit of the Einstein-frame metric of A. Sen, Phys. Rev. Lett. 69
(1992) 1006, hep-th/9204046 (read), whose static limit is the GMGHS solution
(Kocherlakota et al. 2021, Phys. Rev. D 103, 104047, footnote 2, read). The
same line element is written by Rezzolla & Zhidenko (2014), Phys. Rev. D 90,
084009, section IV (read), whose results are used as tests.
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = (
    "Gibbons & Maeda (1988), Nucl. Phys. B 298, 741; Garfinkle, Horowitz & Strominger (1991), "
    "Phys. Rev. D 43, 3140; metric as the a = 0 limit of Sen (1992), Phys. Rev. Lett. 69, 1006"
)


def GMGHS() -> StaticSpherical:
    r"""GMGHS charged dilaton black hole (Einstein frame, dilaton coupling 1, G = c = M = 1).

    With :math:`b = Q^2/2` and :math:`m = 1 - b` (Sen 1992, Eq. (18) at
    :math:`a = 0`: :math:`m = M - Q^2/2M`, :math:`m\sinh^2(\alpha/2) = M - m`),

    .. math::

        ds^2 = -\frac{\rho - 2m}{\rho + 2b}dt^2 + \frac{\rho + 2b}{\rho - 2m}d\rho^2
               + \rho(\rho + 2b)\,d\Omega^2 .

    The radial coordinate :math:`\rho` is **not** the areal radius:
    :math:`h(\rho) = \rho^2 + 2b\rho \ne \rho^2`, so this entry exercises
    the general areal function. All radii returned by BlackPy for this
    metric are values of :math:`\rho`; the areal radius is
    :math:`\sqrt{h(\rho)}`.

    Parameters of the metric
    ------------------------
    Q : float, default 0
        Electric charge in units of M (the :math:`Q` of Sen 1992). A black
        hole exists for :math:`Q < \sqrt2`; the event horizon is at
        :math:`\rho = 2m = 2 - Q^2`, with areal radius
        :math:`2\sqrt{1 - Q^2/2}`.

    Known results (tests): Schwarzschild at Q = 0; the photon-orbit impact
    parameter and the ISCO frequency of Rezzolla & Zhidenko (2014),
    Eqs. (37) and (43), with :math:`\mu = m`, :math:`b = Q^2/2`.

    Returns
    -------
    StaticSpherical
    """
    rho = sp.Symbol("r", positive=True)
    charge = sp.Symbol("Q", real=True)
    b = charge**2 / 2
    m = 1 - b
    f = (rho - 2 * m) / (rho + 2 * b)
    return StaticSpherical.from_sympy(
        f,
        f,
        rho * (rho + 2 * b),
        r=rho,
        parameters=[
            Parameter(
                "Q",
                0.0,
                (0.0, math.sqrt(2.0) - 1e-12),
                latex="Q",
                description="electric charge Q/M (dilaton black hole)",
                role="charge",
            )
        ],
        name="GMGHS",
        asymptotically_flat=True,
        reference=REFERENCE,
    )
