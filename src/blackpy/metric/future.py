"""Interfaces reserved for post-MVP metric classes.

Nothing here is implemented. The classes exist so that rotating and
parametrised metrics can be added later as subclasses of
:class:`blackpy.metric.Metric` without changing the public API of the
observable modules (``blackpy.orbits``, ``blackpy.light``, ...).
"""

from __future__ import annotations

from blackpy.metric.base import Metric

_ROADMAP = (
    "{what} is not part of the BlackPy MVP, which covers static, spherically symmetric "
    "metrics only. The interface is reserved; see DESIGN_NOTES.md (roadmap)."
)


class StationaryAxisymmetric(Metric):
    """Stationary, axisymmetric metric (Kerr-like). **Not implemented.**

    Planned interface: metric components ``g_tt, g_tphi, g_phiphi, g_rr,
    g_thetatheta`` as functions of ``(r, theta, *params)``. With rotation the
    vertical epicyclic frequency differs from the orbital one, which enables
    nodal-precession (low-frequency QPO) modelling.
    """

    def __init__(self, *args, **kwargs):
        raise NotImplementedError(_ROADMAP.format(what="StationaryAxisymmetric (rotating metrics)"))

    @property
    def is_spherically_symmetric(self) -> bool:
        return False


def newman_janis(metric, *args, **kwargs):
    """Rotating metric from a static one (Newman-Janis algorithm). **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="The Newman-Janis algorithm"))


def rezzolla_zhidenko(*args, **kwargs):
    """Rezzolla-Zhidenko parametrised spherical metric. **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="The Rezzolla-Zhidenko parametrisation"))


def konoplya_rezzolla_zhidenko(*args, **kwargs):
    """Konoplya-Rezzolla-Zhidenko parametrised axisymmetric metric. **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="The KRZ parametrisation"))


def johannsen(*args, **kwargs):
    """Johannsen parametrised Kerr-like metric. **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="The Johannsen parametrisation"))
