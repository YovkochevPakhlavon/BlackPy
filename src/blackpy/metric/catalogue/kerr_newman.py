"""Kerr-Newman metric.

References
----------
E. T. Newman, E. Couch, K. Chinnapared, A. Exton, A. Prakash and
R. Torrence, "Metric of a rotating, charged mass", J. Math. Phys. 6 (1965)
918-919.

The Boyer-Lindquist components below are the Kerr components with
``2 r -> 2 r - Q^2``. They are not copied from a paper: the test-suite
verifies that they (i) reduce to Kerr for ``Q = 0`` and to
Reissner-Nordstrom for ``a = 0``, (ii) coincide with the metric that the
Azreg-Ainou (2014) procedure generates from Reissner-Nordstrom, and
(iii) have a traceless source (``R = 0``) of electromagnetic type.
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.base import Parameter
from blackpy.metric.catalogue.kerr import SPIN

REFERENCE = "Newman et al. (1965), J. Math. Phys. 6, 918"


def KerrNewman() -> StationaryAxisymmetric:
    r"""Kerr-Newman black hole in Boyer-Lindquist coordinates (G = c = M = 1).

    With :math:`\Sigma = r^2 + a^2\cos^2\theta` and
    :math:`\Delta = r^2 - 2r + a^2 + Q^2`:

    .. math::

        g_{tt} = -\left(1 - \frac{2r - Q^2}{\Sigma}\right), \quad
        g_{t\phi} = -\frac{a(2r - Q^2)\sin^2\theta}{\Sigma}, \quad
        g_{rr} = \frac{\Sigma}{\Delta}, \quad g_{\theta\theta} = \Sigma, \quad
        g_{\phi\phi} = \frac{(r^2 + a^2)^2 - a^2\Delta\sin^2\theta}{\Sigma}\,\sin^2\theta .

    Parameters of the metric
    ------------------------
    a : float, default 0
        Spin a/M.
    Q : float, default 0
        Charge Q/M in Gaussian-geometrized units, ``Q >= 0``.

    A black hole exists for ``a^2 + Q^2 <= 1`` (the declared domain), with
    horizons :math:`r_\pm = 1 \pm \sqrt{1 - a^2 - Q^2}`.

    Returns
    -------
    StationaryAxisymmetric
    """
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", real=True)
    a = sp.Symbol("a", real=True)
    q = sp.Symbol("Q", real=True)
    sigma = r**2 + a**2 * sp.cos(theta) ** 2
    delta = r**2 - 2 * r + a**2 + q**2
    sin2 = sp.sin(theta) ** 2
    two_f = 2 * r - q**2
    metric = StationaryAxisymmetric.from_sympy(
        -(1 - two_f / sigma),
        -a * two_f * sin2 / sigma,
        ((r**2 + a**2) ** 2 - a**2 * delta * sin2) * sin2 / sigma,
        sigma / delta,
        sigma,
        r=r,
        theta=theta,
        parameters=[
            SPIN,
            Parameter(
                "Q",
                0.0,
                (0.0, math.inf),
                latex="Q",
                description="electric charge Q/M",
                role="charge",
            ),
        ],
        name="Kerr-Newman",
        asymptotically_flat=True,
        reference=REFERENCE,
        domain=lambda a, Q: a**2 + Q**2 <= 1.0,
    )
    metric._declared.update(reflection=True, separable=True)
    return metric
