"""Simpson-Visser black bounce / wormhole family.

Reference
---------
A. Simpson & M. Visser, "Black-bounce to traversable wormhole", J. Cosmol.
Astropart. Phys. 02 (2019) 042, arXiv:1812.07114 (read; Eq. (2.1) and
section 7).
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = "Simpson & Visser (2019), JCAP 02, 042, arXiv:1812.07114"


def SimpsonVisser() -> StaticSpherical:
    r"""Simpson-Visser metric (their Eq. (2.1)), G = c = M = 1.

    .. math::

        ds^2 = -\Bigl(1 - \frac{2}{\sqrt{r^2 + \ell^2}}\Bigr)dt^2
               + \frac{dr^2}{1 - 2/\sqrt{r^2 + \ell^2}} + (r^2 + \ell^2)\,d\Omega^2 .

    One family, four geometries (their section 1): Schwarzschild
    (:math:`\ell = 0`); a regular black hole with a one-way spacelike throat
    (:math:`0 < \ell < 2`); a one-way wormhole with an extremal null throat
    (:math:`\ell = 2`); a traversable wormhole (:math:`\ell > 2`). BlackPy
    covers the region :math:`r > 0` (our universe); the throat is at
    :math:`r = 0`. A horizonless configuration is a result, not an error:
    for :math:`\ell > 2` ``horizons`` reports none.

    Parameters of the metric
    ------------------------
    ell : float, default 0
        The length scale written :math:`a` in the paper, in units of M,
        renamed so that :func:`blackpy.metric.rotating_from_static` can add a
        spin ``a``.

    Known results (their section 7; tests): event horizon
    :math:`\sqrt{4 - \ell^2}`, photon sphere :math:`\sqrt{9 - \ell^2}` (none
    in :math:`r > 0` for :math:`\ell > 3`), ISCO :math:`\sqrt{36 - \ell^2}`
    (none in :math:`r > 0` for :math:`\ell > 6`). Derived from these: the
    shadow radius is :math:`3\sqrt3` for every :math:`\ell < 3`.

    Returns
    -------
    StaticSpherical
    """
    r = sp.Symbol("r", positive=True)
    ell = sp.Symbol("ell", real=True)
    f = 1 - 2 / sp.sqrt(r**2 + ell**2)
    return StaticSpherical.from_sympy(
        f,
        f,
        r**2 + ell**2,
        r=r,
        parameters=[
            Parameter(
                "ell",
                0.0,
                (0.0, math.inf),
                latex=r"\ell",
                description="bounce / throat length scale (the paper's a), in units of M",
                role="deformation",
            )
        ],
        name="Simpson-Visser",
        asymptotically_flat=True,
        reference=REFERENCE,
    )
