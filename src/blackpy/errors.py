"""Exceptions and warnings used across BlackPy.

Every exception and warning class that BlackPy raises is defined here, so that
``from blackpy.errors import ...`` is the one place to import them from. Each
class is also explained, with what to do about it, in the "Integrity guards and
errors" section of the user guide.
"""

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


class PlasmaPropagationError(BlackPyError):
    """Light of the requested frequency cannot propagate through the plasma along the
    path from the photon sphere to the observer (plasma frequency >= photon frequency)."""


class SymbolicTimeoutError(BlackPyError):
    """A symbolic computation exceeded its time limit."""


class RegistryError(BlackPyError, ValueError):
    """A registry file is missing or does not follow the schema.

    Also a :class:`ValueError`, as in versions before 1.0.
    """


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


class ForecastWarning(BlackPyWarning):
    """A Fisher forecast is being used outside the range where it can be trusted
    (fiducial value near a prior boundary)."""


class PoorFitWarning(BlackPyWarning):
    """The best model does not describe the data: the posterior predictive p-value of
    the chi-square discrepancy is below the documented threshold."""


__all__ = [
    "BlackPyError",
    "BlackPyWarning",
    "CircularInferenceWarning",
    "CircularityDefinitionWarning",
    "ConvergenceWarning",
    "DegeneracyWarning",
    "ForecastWarning",
    "IgnoredDataWarning",
    "NodalPrecessionUnavailableError",
    "NotAsymptoticallyFlatError",
    "NotSeparableError",
    "PlasmaPropagationError",
    "PoorFitWarning",
    "PriorDominatedWarning",
    "ReflectionSymmetryError",
    "RegistryError",
    "SpinIgnoredWarning",
    "SymbolicTimeoutError",
    "SymbolicUnavailableError",
    "UnverifiedDataWarning",
]
