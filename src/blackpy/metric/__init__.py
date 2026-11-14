"""Metric definitions: the :class:`StaticSpherical` class and a built-in catalogue."""

from blackpy.metric import future, interpolated
from blackpy.metric.base import Metric, Parameter, StaticSpherical, SymbolicForm
from blackpy.metric.catalogue import (
    CATALOGUE,
    Bardeen,
    Hayward,
    ReissnerNordstrom,
    Schwarzschild,
    SchwarzschildDeSitter,
)
from blackpy.metric.future import StationaryAxisymmetric

__all__ = [
    "CATALOGUE",
    "Bardeen",
    "Hayward",
    "Metric",
    "Parameter",
    "ReissnerNordstrom",
    "Schwarzschild",
    "SchwarzschildDeSitter",
    "StaticSpherical",
    "StationaryAxisymmetric",
    "SymbolicForm",
    "future",
    "interpolated",
]
