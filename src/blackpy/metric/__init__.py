"""Metric definitions: :class:`StaticSpherical`, :class:`StationaryAxisymmetric`,
the built-in catalogue and the rotating-metric generator."""

from blackpy.metric import future, interpolated
from blackpy.metric.axisymmetric import StationaryAxisymmetric, SymbolicFormAxisymmetric
from blackpy.metric.base import Metric, Parameter, StaticSpherical, SymbolicForm
from blackpy.metric.catalogue import (
    CATALOGUE,
    ROTATING_CATALOGUE,
    Bardeen,
    Hayward,
    Johannsen,
    Kerr,
    KerrNewman,
    ReissnerNordstrom,
    Schwarzschild,
    SchwarzschildDeSitter,
)
from blackpy.metric.rotating import rotating_from_static

__all__ = [
    "CATALOGUE",
    "ROTATING_CATALOGUE",
    "Bardeen",
    "Hayward",
    "Johannsen",
    "Kerr",
    "KerrNewman",
    "Metric",
    "Parameter",
    "ReissnerNordstrom",
    "Schwarzschild",
    "SchwarzschildDeSitter",
    "StaticSpherical",
    "StationaryAxisymmetric",
    "SymbolicForm",
    "SymbolicFormAxisymmetric",
    "future",
    "interpolated",
    "rotating_from_static",
]
