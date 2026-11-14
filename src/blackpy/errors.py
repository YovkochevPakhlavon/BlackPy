"""Exceptions and warnings used across BlackPy."""

from __future__ import annotations


class BlackPyError(Exception):
    """Base class for all BlackPy errors."""


class SymbolicUnavailableError(BlackPyError):
    """A symbolic result was requested for a metric defined by plain callables."""


class NotAsymptoticallyFlatError(BlackPyError):
    """A quantity defined at spatial infinity was requested for a metric that is
    not asymptotically flat."""


class NodalPrecessionUnavailableError(BlackPyError):
    """The nodal (Lense-Thirring) precession frequency was requested in a
    spherically symmetric spacetime, where it vanishes identically."""


class BlackPyWarning(UserWarning):
    """Base class for all BlackPy warnings."""


class CircularInferenceWarning(BlackPyWarning):
    """A mass prior was itself derived from a QPO fit in the Kerr or
    Schwarzschild metric, so using it to test another metric is circular."""


class PriorDominatedWarning(BlackPyWarning):
    """The posterior of a parameter is indistinguishable from its prior."""


class ConvergenceWarning(BlackPyWarning):
    """An MCMC chain is shorter than required for a reliable estimate."""


class UnverifiedDataWarning(BlackPyWarning):
    """A registry dataset has not been checked by a human against its source."""


class IgnoredDataWarning(BlackPyWarning):
    """Part of a dataset cannot be used by the current model and is ignored."""
