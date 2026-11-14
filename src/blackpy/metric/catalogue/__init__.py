"""Built-in catalogue of static, spherically symmetric metrics.

Each entry lives in its own file with its literature reference and has at
least one dedicated test.
"""

from blackpy.metric.catalogue.bardeen import Bardeen
from blackpy.metric.catalogue.hayward import Hayward
from blackpy.metric.catalogue.reissner_nordstrom import ReissnerNordstrom
from blackpy.metric.catalogue.schwarzschild import Schwarzschild
from blackpy.metric.catalogue.schwarzschild_de_sitter import SchwarzschildDeSitter

#: Catalogue entries by name.
CATALOGUE = {
    "Schwarzschild": Schwarzschild,
    "ReissnerNordstrom": ReissnerNordstrom,
    "SchwarzschildDeSitter": SchwarzschildDeSitter,
    "Bardeen": Bardeen,
    "Hayward": Hayward,
}

__all__ = [
    "CATALOGUE",
    "Bardeen",
    "Hayward",
    "ReissnerNordstrom",
    "Schwarzschild",
    "SchwarzschildDeSitter",
]
