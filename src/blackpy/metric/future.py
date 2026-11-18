"""Reserved interface for the original Newman-Janis algorithm, and deprecated aliases.

Implemented since v0.2 (no longer here): rotating metrics
(:class:`blackpy.metric.StationaryAxisymmetric`), the Johannsen metric
(:func:`blackpy.metric.Johannsen`) and the rotating generator
(:func:`blackpy.metric.rotating_from_static`). Implemented since v0.3: the
Rezzolla-Zhidenko and KRZ parametrisations (:mod:`blackpy.metric.parametrized`).
Their old reserved names still work but emit a :class:`DeprecationWarning`
(since 1.0) and will be removed in 2.0:

* ``blackpy.metric.future.rezzolla_zhidenko`` -> :func:`blackpy.metric.RezzollaZhidenko`
* ``blackpy.metric.future.konoplya_rezzolla_zhidenko`` -> :func:`blackpy.metric.KRZ`
"""

from __future__ import annotations

from blackpy._deprecation import alias_getattr
from blackpy._links import ROADMAP

__all__ = ["newman_janis"]

__getattr__ = alias_getattr(
    __name__,
    {
        "rezzolla_zhidenko": (
            "blackpy.metric.parametrized",
            "RezzollaZhidenko",
            "blackpy.metric.RezzollaZhidenko",
        ),
        "konoplya_rezzolla_zhidenko": (
            "blackpy.metric.parametrized",
            "KRZ",
            "blackpy.metric.KRZ",
        ),
    },
)


def newman_janis(metric, *args, **kwargs):
    """Original Newman-Janis algorithm (complexification of r). **Not implemented.**

    Use :func:`blackpy.metric.rotating_from_static` instead: it implements
    the Azreg-Ainou (2014) version of the algorithm, which needs no
    complexification and always reaches Boyer-Lindquist coordinates.

    Raises
    ------
    NotImplementedError
        Always.
    """
    raise NotImplementedError(
        "The original Newman-Janis algorithm (with complexification of r) is not implemented "
        "and not planned: use blackpy.metric.rotating_from_static(static_metric) instead, which "
        "implements the Azreg-Ainou (2014) version and always reaches Boyer-Lindquist "
        "coordinates. See " + ROADMAP + "."
    )
