"""BlackPy: black hole metric in, observables and parameter constraints out.

Importing the package enables 64-bit floating point in JAX
(``jax_enable_x64 = True``); this is the only global state BlackPy touches.
"""

import importlib as _importlib

import jax as _jax

_jax.config.update("jax_enable_x64", True)

from blackpy import constants, errors, metric  # noqa: E402
from blackpy._version import __version__  # noqa: E402
from blackpy.metric import StaticSpherical, StationaryAxisymmetric  # noqa: E402

_LAZY = ("geometry", "orbits", "light", "data", "infer", "plot", "qnm", "thermo")

__all__ = [
    "StaticSpherical",
    "StationaryAxisymmetric",
    "__version__",
    "constants",
    "errors",
    "metric",
    *_LAZY,
]


def __getattr__(name):
    # Sub-packages are imported on first use so that ``import blackpy`` stays fast.
    if name in _LAZY:
        return _importlib.import_module(f"blackpy.{name}")
    raise AttributeError(f"module 'blackpy' has no attribute '{name}'")


def __dir__():
    return sorted(__all__)
