"""Built-in catalogue of metrics.

Each entry lives in its own file with its literature reference and has at
least one dedicated test. Static, spherically symmetric entries return a
:class:`blackpy.metric.StaticSpherical`; rotating entries return a
:class:`blackpy.metric.StationaryAxisymmetric`.
"""

from blackpy.metric.catalogue.bardeen import Bardeen
from blackpy.metric.catalogue.gmghs import GMGHS
from blackpy.metric.catalogue.hayward import Hayward
from blackpy.metric.catalogue.johannsen import Johannsen
from blackpy.metric.catalogue.kerr import Kerr
from blackpy.metric.catalogue.kerr_newman import KerrNewman
from blackpy.metric.catalogue.kerr_sen import KerrSen
from blackpy.metric.catalogue.reissner_nordstrom import ReissnerNordstrom
from blackpy.metric.catalogue.rotating_regular import RotatingBardeen, RotatingHayward
from blackpy.metric.catalogue.schwarzschild import Schwarzschild
from blackpy.metric.catalogue.schwarzschild_de_sitter import SchwarzschildDeSitter
from blackpy.metric.catalogue.simpson_visser import SimpsonVisser

#: Static, spherically symmetric entries by name.
CATALOGUE = {
    "Schwarzschild": Schwarzschild,
    "ReissnerNordstrom": ReissnerNordstrom,
    "SchwarzschildDeSitter": SchwarzschildDeSitter,
    "Bardeen": Bardeen,
    "Hayward": Hayward,
    "GMGHS": GMGHS,
    "SimpsonVisser": SimpsonVisser,
}

#: Rotating (stationary, axisymmetric) entries by name.
ROTATING_CATALOGUE = {
    "Kerr": Kerr,
    "KerrNewman": KerrNewman,
    "Johannsen": Johannsen,
    "KerrSen": KerrSen,
    "RotatingBardeen": RotatingBardeen,
    "RotatingHayward": RotatingHayward,
}

__all__ = [
    "SimpsonVisser",
    "RotatingHayward",
    "RotatingBardeen",
    "KerrSen",
    "GMGHS",
    "CATALOGUE",
    "ROTATING_CATALOGUE",
    "Bardeen",
    "Hayward",
    "Johannsen",
    "Kerr",
    "KerrNewman",
    "ReissnerNordstrom",
    "Schwarzschild",
    "SchwarzschildDeSitter",
]
