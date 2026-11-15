"""Kerr metric.

References
----------
R. P. Kerr, "Gravitational field of a spinning mass as an example of
algebraically special metrics", Phys. Rev. Lett. 11 (1963) 237-238.

The Boyer-Lindquist form used here was transcribed from Eq. (1) of
T. Johannsen, Phys. Rev. D 88 (2013) 044002 (arXiv:1501.02809), and is
verified in the test-suite to solve the vacuum Einstein equations exactly.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import sympy as sp

from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.base import Parameter

REFERENCE = "Kerr (1963), Phys. Rev. Lett. 11, 237"

SPIN = Parameter(
    "a", 0.0, (-math.inf, math.inf), latex="a", description="spin a/M = J/M^2", role="spin"
)


def Kerr() -> StationaryAxisymmetric:
    r"""Kerr black hole in Boyer-Lindquist coordinates (G = c = M = 1).

    With :math:`\Sigma = r^2 + a^2\cos^2\theta` and
    :math:`\Delta = r^2 - 2r + a^2`:

    .. math::

        g_{tt} = -\left(1 - \frac{2r}{\Sigma}\right), \quad
        g_{t\phi} = -\frac{2ar\sin^2\theta}{\Sigma}, \quad
        g_{rr} = \frac{\Sigma}{\Delta}, \quad g_{\theta\theta} = \Sigma, \quad
        g_{\phi\phi} = \left(r^2 + a^2 + \frac{2a^2 r\sin^2\theta}{\Sigma}\right)\sin^2\theta .

    Parameters of the metric
    ------------------------
    a : float, default 0
        Spin :math:`a/M = J/M^2`. A black hole exists for ``|a| <= 1``
        (the declared domain); ``|a| > 1`` is a naked singularity, which
        BlackPy reports as horizonless. Negative ``a`` describes a
        counter-rotating hole: prograde orbits (``direction="prograde"``)
        then move against the spin.

    Known results (used as tests): horizons
    :math:`r_\pm = 1 \pm \sqrt{1 - a^2}`, ergosurface
    :math:`r_E = 1 + \sqrt{1 - a^2\cos^2\theta}`, the
    Bardeen-Press-Teukolsky ISCO, and the closed-form orbital and epicyclic
    frequencies.

    Returns
    -------
    StationaryAxisymmetric
    """
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", real=True)
    a = sp.Symbol("a", real=True)
    sigma = r**2 + a**2 * sp.cos(theta) ** 2
    delta = r**2 - 2 * r + a**2
    sin2 = sp.sin(theta) ** 2
    metric = StationaryAxisymmetric.from_sympy(
        -(1 - 2 * r / sigma),
        -2 * a * r * sin2 / sigma,
        (r**2 + a**2 + 2 * a**2 * r * sin2 / sigma) * sin2,
        sigma / delta,
        sigma,
        r=r,
        theta=theta,
        parameters=[SPIN],
        name="Kerr",
        asymptotically_flat=True,
        reference=REFERENCE,
        domain=lambda a: jnp.abs(a) <= 1.0,
    )
    metric._declared.update(reflection=True, separable=True)
    return metric
