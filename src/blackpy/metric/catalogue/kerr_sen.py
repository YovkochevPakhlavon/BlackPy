"""Kerr-Sen rotating charged dilaton black hole.

Reference
---------
A. Sen, "Rotating charged black hole solution in heterotic string theory",
Phys. Rev. Lett. 69 (1992) 1006, hep-th/9204046 (read; Einstein-frame metric
Eq. (16), parameters Eqs. (17)-(18), horizons Eq. (20) of the arXiv version).
The same metric in the form of Konoplya, Rezzolla & Zhidenko (2016), Phys.
Rev. D 93, 064015, Eq. (6) (read) is used as an independent test.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import sympy as sp

from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.base import Parameter
from blackpy.metric.catalogue.kerr import SPIN

REFERENCE = "Sen (1992), Phys. Rev. Lett. 69, 1006, hep-th/9204046"


def KerrSen() -> StationaryAxisymmetric:
    r"""Kerr-Sen black hole in the Einstein frame (G = c = M = 1).

    Sen's Einstein-frame metric with :math:`M = 1`. From his Eqs. (17)-(18),
    :math:`m = 1 - Q^2/2` and :math:`m\sinh^2(\alpha/2) = M - m = b`,
    :math:`m\cosh^2(\alpha/2) = M = 1`, with :math:`b = Q^2/2`. Writing
    :math:`\Sigma = \rho^2 + a^2\cos^2\theta + 2b\rho`:

    .. math::

        g_{tt} = -\frac{\rho^2 + a^2\cos^2\theta - 2m\rho}{\Sigma}, \quad
        g_{t\phi} = -\frac{2\rho a\sin^2\theta}{\Sigma}, \quad
        g_{\rho\rho} = \frac{\Sigma}{\rho^2 + a^2 - 2m\rho}, \quad
        g_{\theta\theta} = \Sigma,

        g_{\phi\phi} = \frac{\sin^2\theta}{\Sigma}\bigl[(\rho^2 + a^2)
            (\rho^2 + a^2\cos^2\theta) + 2m\rho a^2\sin^2\theta
            + 4b\rho(\rho^2 + a^2) + 4b^2\rho^2\bigr].

    It is an exact solution of heterotic string theory (Einstein-Maxwell-
    dilaton-axion), not a model. Separability of null geodesics is not
    assumed: it is tested numerically (``is_separable``).

    Parameters of the metric
    ------------------------
    a : float, default 0
        Spin :math:`J/M^2` (Sen's :math:`a = J/M`).
    Q : float, default 0
        Electric charge in units of M.

    Horizons (Sen Eq. (20)): :math:`\rho_\pm = m \pm \sqrt{m^2 - a^2}`; a black
    hole exists for :math:`|a| \le 1 - Q^2/2` (the declared domain). Kerr at
    :math:`Q = 0`; the static GMGHS metric (:func:`GMGHS`) at :math:`a = 0`.

    Returns
    -------
    StationaryAxisymmetric
    """
    rho = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", real=True)
    a = sp.Symbol("a", real=True)
    charge = sp.Symbol("Q", real=True)
    b = charge**2 / 2
    m = 1 - b
    cos2, sin2 = sp.cos(theta) ** 2, sp.sin(theta) ** 2
    sigma = rho**2 + a**2 * cos2 + 2 * b * rho
    g_phiphi = (
        (
            (rho**2 + a**2) * (rho**2 + a**2 * cos2)
            + 2 * m * rho * a**2 * sin2
            + 4 * b * rho * (rho**2 + a**2)
            + 4 * b**2 * rho**2
        )
        * sin2
        / sigma
    )
    return StationaryAxisymmetric.from_sympy(
        -(rho**2 + a**2 * cos2 - 2 * m * rho) / sigma,
        -2 * rho * a * sin2 / sigma,
        g_phiphi,
        sigma / (rho**2 + a**2 - 2 * m * rho),
        sigma,
        r=rho,
        theta=theta,
        parameters=[
            SPIN,
            Parameter(
                "Q",
                0.0,
                (0.0, math.sqrt(2.0) - 1e-12),
                latex="Q",
                description="electric charge Q/M",
                role="charge",
            ),
        ],
        name="Kerr-Sen",
        asymptotically_flat=True,
        reference=REFERENCE,
        domain=lambda a, q: jnp.abs(a) <= 1.0 - q**2 / 2.0,
    )
