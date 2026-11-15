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
    """The nodal (Lense-Thirring) precession frequency was requested for a static,
    spherically symmetric metric, where it vanishes identically."""


class ReflectionSymmetryError(BlackPyError):
    """Equatorial circular orbits were requested for a metric that is not symmetric
    under theta -> pi - theta, so the equatorial plane is not a geodesic surface."""


class NotSeparableError(BlackPyError):
    """An analytic shadow was requested for a metric whose null geodesics do not
    separate; this needs numerical ray tracing, which BlackPy does not provide."""


class SymbolicTimeoutError(BlackPyError):
    """A symbolic computation exceeded its time limit."""


class BlackPyWarning(UserWarning):
    """Base class for all BlackPy warnings."""


class CircularInferenceWarning(BlackPyWarning):
    """A mass prior was itself derived from a QPO fit in the Kerr or
    Schwarzschild metric, so using it to test another metric is circular."""


class PriorDominatedWarning(BlackPyWarning):
    """The posterior of a parameter is indistinguishable from its prior."""


class ConvergenceWarning(BlackPyWarning):
    """An MCMC chain is shorter than required for a reliable estimate."""


class SpinIgnoredWarning(BlackPyWarning):
    """A non-rotating metric is fitted to a source that is known to rotate (it has a
    published spin measurement or a nodal-precession QPO)."""


class DegeneracyWarning(BlackPyWarning):
    """The spin and another metric parameter are strongly correlated in the posterior."""


class UnverifiedDataWarning(BlackPyWarning):
    """A registry dataset has not been checked by a human against its source."""


class IgnoredDataWarning(BlackPyWarning):
    """Part of a dataset cannot be used by the current model and is ignored."""


class CircularityDefinitionWarning(BlackPyWarning):
    """A bound on the deviation from circularity is combined with a definition of
    Delta C other than the one the bound was derived for."""


class PoorFitWarning(BlackPyWarning):
    """The best model does not describe the data: the posterior predictive p-value of
    the chi-square discrepancy is below the documented threshold."""
