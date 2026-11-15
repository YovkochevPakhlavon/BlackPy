"""Johannsen (2013) Kerr-like metric with three constants of motion.

Reference
---------
T. Johannsen, "Regular black hole metric with three constants of motion",
Phys. Rev. D 88 (2013) 044002, arXiv:1501.02809.

The metric components are transcribed from Eq. (51) of the arXiv version,
with the deviation functions of Eqs. (52)-(56) truncated at the lowest
non-vanishing order, as recommended below Eq. (56). The parameter names
(``alpha13``, ``alpha22``, ``alpha52``, ``epsilon3``) follow the paper.
"""

from __future__ import annotations

import math

import jax.numpy as jnp
import sympy as sp

from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.base import Parameter
from blackpy.metric.catalogue.kerr import SPIN

REFERENCE = "Johannsen (2013), Phys. Rev. D 88, 044002 (arXiv:1501.02809), Eqs. (51)-(56)"


def _deformation(name: str, latex: str, description: str) -> Parameter:
    return Parameter(
        name,
        0.0,
        (-math.inf, math.inf),
        latex=latex,
        description=description,
        role="deformation",
    )


def _domain(a, alpha13, alpha22, alpha52, epsilon3):
    """Regular exterior: |a| <= 1 and Eqs. (74), (75), (77), (78) of the paper."""
    b = 1.0 + jnp.sqrt(jnp.clip(1.0 - a**2, 0.0, None))  # Kerr horizon radius r_+ / M
    return (
        (jnp.abs(a) <= 1.0)
        & (alpha13 > -(b**3))
        & (alpha22 > -(b**2))
        & (alpha52 > -(b**2))
        & (epsilon3 > -(b**3))
    )


def Johannsen() -> StationaryAxisymmetric:
    r"""Johannsen metric at lowest order in the deviation parameters (G = c = M = 1).

    With :math:`\Delta = r^2 - 2r + a^2`,
    :math:`\tilde\Sigma = r^2 + a^2\cos^2\theta + f(r)` and the deviation
    functions (Eqs. (52)-(56) of the paper at lowest order)

    .. math::

        A_1 = 1 + \frac{\alpha_{13}}{r^3}, \quad
        A_2 = 1 + \frac{\alpha_{22}}{r^2}, \quad
        A_5 = 1 + \frac{\alpha_{52}}{r^2}, \quad
        f = \frac{\epsilon_3}{r},

    the components are (Eq. (51))

    .. math::

        g_{tt} &= -\frac{\tilde\Sigma\,[\Delta - a^2 A_2^2 \sin^2\theta]}{D^2}, \qquad
        g_{t\phi} = -\frac{a\,[(r^2 + a^2) A_1 A_2 - \Delta]\,\tilde\Sigma \sin^2\theta}{D^2}, \\
        g_{rr} &= \frac{\tilde\Sigma}{\Delta A_5}, \qquad
        g_{\theta\theta} = \tilde\Sigma, \qquad
        g_{\phi\phi} = \frac{\tilde\Sigma \sin^2\theta\,[(r^2 + a^2)^2 A_1^2
                       - a^2 \Delta \sin^2\theta]}{D^2},

    with :math:`D = (r^2 + a^2) A_1 - a^2 A_2 \sin^2\theta`. All deviation
    parameters equal to zero gives the Kerr metric.

    Parameters of the metric
    ------------------------
    a : float, default 0
        Spin a/M.
    alpha13, alpha22, alpha52, epsilon3 : float, default 0
        Deviation parameters :math:`\alpha_{13}, \alpha_{22}, \alpha_{52},
        \epsilon_3`. The exterior is regular for
        :math:`\alpha_{13}, \epsilon_3 > -(1 + \sqrt{1 - a^2})^3` and
        :math:`\alpha_{22}, \alpha_{52} > -(1 + \sqrt{1 - a^2})^2`
        (Eqs. (74), (75), (77), (78)); this is the declared domain, together
        with ``|a| <= 1``.

    Notes
    -----
    The event horizon coincides with the Kerr horizon
    :math:`r_+ = 1 + \sqrt{1 - a^2}` (Sec. III.1 of the paper). The metric
    is not a solution of the Einstein equations: its effective source must
    be inspected with :mod:`blackpy.geometry`, not assumed.

    Returns
    -------
    StationaryAxisymmetric
    """
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", real=True)
    a = sp.Symbol("a", real=True)
    a13, a22, a52, e3 = sp.symbols("alpha13 alpha22 alpha52 epsilon3", real=True)
    delta = r**2 - 2 * r + a**2
    sin2 = sp.sin(theta) ** 2
    big_a1 = 1 + a13 / r**3
    big_a2 = 1 + a22 / r**2
    big_a5 = 1 + a52 / r**2
    sigma = r**2 + a**2 * sp.cos(theta) ** 2 + e3 / r
    denom = ((r**2 + a**2) * big_a1 - a**2 * big_a2 * sin2) ** 2
    metric = StationaryAxisymmetric.from_sympy(
        -sigma * (delta - a**2 * big_a2**2 * sin2) / denom,
        -a * ((r**2 + a**2) * big_a1 * big_a2 - delta) * sigma * sin2 / denom,
        sigma * sin2 * ((r**2 + a**2) ** 2 * big_a1**2 - a**2 * delta * sin2) / denom,
        sigma / (delta * big_a5),
        sigma,
        r=r,
        theta=theta,
        parameters=[
            SPIN,
            _deformation("alpha13", r"\alpha_{13}", "deviation of A_1 (g_tt, g_tphi, g_phiphi)"),
            _deformation("alpha22", r"\alpha_{22}", "deviation of A_2 (frame dragging)"),
            _deformation("alpha52", r"\alpha_{52}", "deviation of A_5 (g_rr)"),
            _deformation("epsilon3", r"\epsilon_3", "deviation of the conformal factor"),
        ],
        name="Johannsen",
        asymptotically_flat=True,
        reference=REFERENCE,
        domain=_domain,
    )
    metric._declared.update(reflection=True, separable=True)
    return metric
