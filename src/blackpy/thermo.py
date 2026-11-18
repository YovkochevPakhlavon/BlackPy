"""Black hole thermodynamics. **Not implemented and not scheduled; interface only.**

Thermodynamics was removed from the roadmap of version 0.3 on purpose:
BlackPy is "metric in, constraints out", and there are no thermodynamic data
that a likelihood could use. The functions below reserve the names and raise
``NotImplementedError``. See the 'Roadmap' page of the documentation.
"""

from __future__ import annotations

from blackpy._links import ROADMAP

__all__ = ["entropy", "hawking_temperature", "heat_capacity"]

_MESSAGE = (
    "Black hole thermodynamics is not implemented in BlackPy and is not scheduled for a "
    "release: no observational data constrain it, so it would not feed a likelihood. The "
    "interface is reserved; see " + ROADMAP + "."
)


def hawking_temperature(metric, **params):
    """Hawking temperature of the event horizon. **Not implemented, not scheduled.**"""
    raise NotImplementedError(_MESSAGE)


def entropy(metric, **params):
    """Bekenstein-Hawking entropy. **Not implemented, not scheduled.**"""
    raise NotImplementedError(_MESSAGE)


def heat_capacity(metric, **params):
    """Heat capacity at fixed parameters. **Not implemented, not scheduled.**"""
    raise NotImplementedError(_MESSAGE)
