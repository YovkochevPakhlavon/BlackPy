"""Interfaces reserved for metric families that are not implemented yet.

Implemented since v0.2 (no longer here): rotating metrics
(:class:`blackpy.metric.StationaryAxisymmetric`), the Johannsen metric
(:func:`blackpy.metric.Johannsen`) and the rotating generator
(:func:`blackpy.metric.rotating_from_static`). Implemented since v0.3: the
Rezzolla-Zhidenko and KRZ parametrisations (:mod:`blackpy.metric.parametrized`); their
old reserved names below now forward to the implementations.
"""

from __future__ import annotations

from blackpy.metric.parametrized import KRZ as konoplya_rezzolla_zhidenko
from blackpy.metric.parametrized import RezzollaZhidenko as rezzolla_zhidenko

__all__ = ["konoplya_rezzolla_zhidenko", "newman_janis", "rezzolla_zhidenko"]

_ROADMAP = (
    "{what} is not implemented in this version of BlackPy. The interface is reserved; see "
    "DESIGN_NOTES.md (roadmap)."
)


def newman_janis(metric, *args, **kwargs):
    """Original Newman-Janis algorithm (complexification of r). **Not implemented.**

    Use :func:`blackpy.metric.rotating_from_static` instead: it implements
    the Azreg-Ainou (2014) version of the algorithm, which needs no
    complexification and always reaches Boyer-Lindquist coordinates.
    """
    raise NotImplementedError(
        _ROADMAP.format(what="The original Newman-Janis algorithm (with complexification)")
        + " Use blackpy.metric.rotating_from_static(static_metric, a) instead."
    )
