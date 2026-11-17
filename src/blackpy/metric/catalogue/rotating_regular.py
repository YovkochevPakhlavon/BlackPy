"""Rotating Bardeen and rotating Hayward black holes (generated models).

Reference
---------
C. Bambi & L. Modesto, "Rotating regular black holes", Phys. Lett. B 721
(2013) 329, arXiv:1302.6075 (read): the Kerr metric with the mass replaced
by :math:`m(r)` (their Eq. (14) with Eqs. (15) and (26)). BlackPy builds
them with :func:`blackpy.metric.rotating_from_static` (Azreg-Ainou 2014),
which for these seeds gives exactly that metric (tested).
"""

from __future__ import annotations

import jax.numpy as jnp

from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.catalogue.bardeen import Bardeen
from blackpy.metric.catalogue.hayward import Hayward
from blackpy.metric.rotating import rotating_from_static

REFERENCE = "Bambi & Modesto (2013), Phys. Lett. B 721, 329, arXiv:1302.6075"

_MODEL_NOTE = """
    .. warning::

        This is a **generated model**, not a solution of any field equations:
        the rotating metric is obtained from the static one by the
        Newman-Janis / Azreg-Ainou procedure. Its effective source has to be
        checked with :func:`blackpy.geometry.effective_source`; it is not
        assumed to be physical.
"""


def RotatingBardeen() -> StationaryAxisymmetric:
    r"""Rotating Bardeen black hole: Kerr with :math:`M \to m(r) = (r^2/(r^2 + q^2))^{3/2}`.

    Bambi & Modesto (2013), Eqs. (14) and (26), with their :math:`g` called
    ``q`` (as in :func:`blackpy.metric.Bardeen`). Parameters ``a`` (spin)
    and ``q``. At :math:`a = 0` it is the static Bardeen metric, at
    :math:`q = 0` Kerr.
    """
    metric = rotating_from_static(Bardeen(), name="rotating Bardeen")
    metric.reference = REFERENCE
    metric._domain_user = _necessary_spin_bound
    return metric


def RotatingHayward() -> StationaryAxisymmetric:
    r"""Rotating Hayward black hole: Kerr with :math:`M \to m(r) = r^3/(r^3 + 2l^2)`.

    Bambi & Modesto (2013), Eqs. (14) and (15), whose :math:`g` is related to
    the ``l`` of :func:`blackpy.metric.Hayward` by :math:`g^3 = 2l^2`.
    Parameters ``a`` and ``l``. At :math:`a = 0` the static Hayward metric,
    at :math:`l = 0` Kerr.
    """
    metric = rotating_from_static(Hayward(), name="rotating Hayward")
    metric.reference = REFERENCE
    metric._domain_user = _necessary_spin_bound
    return metric


def _necessary_spin_bound(a, *seed):
    """``|a| <= 1``: necessary for a horizon (``m(r) <= 1``, so ``Delta >= r^2 - 2r + a^2``).

    It is not sufficient (the black hole range of ``a`` shrinks with the regularisation
    parameter); the likelihood additionally requires an event horizon to exist.
    """
    return jnp.abs(a) <= 1.0


RotatingBardeen.__doc__ += _MODEL_NOTE
RotatingHayward.__doc__ += _MODEL_NOTE
