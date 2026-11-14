"""Prior distributions.

Every prior exposes ``logpdf`` (for MCMC), ``ppf`` (the inverse CDF, i.e. the
prior transform needed by nested sampling), ``cdf`` and ``sample``. All
methods are vectorised with NumPy.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod

import numpy as np
from scipy.special import ndtr, ndtri


class Prior(ABC):
    """Base class of one-dimensional priors."""

    #: Support of the distribution, ``(low, high)``.
    support: tuple[float, float]

    @abstractmethod
    def logpdf(self, x):
        """Natural log of the probability density; ``-inf`` outside the support."""

    @abstractmethod
    def cdf(self, x):
        """Cumulative distribution function."""

    @abstractmethod
    def ppf(self, u):
        """Inverse CDF: maps ``u`` in [0, 1] to the parameter value."""

    @abstractmethod
    def to_dict(self) -> dict:
        """JSON-serialisable description (see :func:`prior_from_dict`)."""

    def sample(self, rng: np.random.Generator, size=None):
        """Draw random values with the given NumPy generator."""
        return self.ppf(rng.uniform(size=size))

    def is_on_boundary(self, x: float, rtol: float = 1e-9) -> bool:
        """True if ``x`` coincides with a finite edge of the support."""
        low, high = self.support
        scale = max(abs(x), 1.0)
        return any(math.isfinite(b) and abs(x - b) <= rtol * scale for b in (low, high))

    def __repr__(self) -> str:
        args = ", ".join(f"{k}={v}" for k, v in self.to_dict().items() if k != "type")
        return f"{type(self).__name__}({args})"

    def __eq__(self, other) -> bool:
        return isinstance(other, Prior) and self.to_dict() == other.to_dict()

    def __hash__(self) -> int:
        return hash(tuple(sorted(self.to_dict().items())))


class Uniform(Prior):
    """Uniform prior on ``[low, high]``."""

    def __init__(self, low: float, high: float):
        if not (math.isfinite(low) and math.isfinite(high) and low < high):
            raise ValueError(f"Uniform prior needs finite low < high, got ({low}, {high}).")
        self.low, self.high = float(low), float(high)
        self.support = (self.low, self.high)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.low) & (x <= self.high)
        return np.where(inside, -math.log(self.high - self.low), -np.inf)

    def cdf(self, x):
        return np.clip((np.asarray(x, dtype=float) - self.low) / (self.high - self.low), 0.0, 1.0)

    def ppf(self, u):
        return self.low + (self.high - self.low) * np.asarray(u, dtype=float)

    def to_dict(self) -> dict:
        return {"type": "Uniform", "low": self.low, "high": self.high}


class LogUniform(Prior):
    """Log-uniform (Jeffreys) prior, density proportional to ``1/x`` on ``[low, high]``."""

    def __init__(self, low: float, high: float):
        if not (math.isfinite(low) and math.isfinite(high) and 0.0 < low < high):
            raise ValueError(f"LogUniform prior needs 0 < low < high, got ({low}, {high}).")
        self.low, self.high = float(low), float(high)
        self.support = (self.low, self.high)
        self._log_ratio = math.log(self.high / self.low)

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        inside = (x >= self.low) & (x <= self.high)
        safe = np.where(inside, x, 1.0)
        return np.where(inside, -np.log(safe) - math.log(self._log_ratio), -np.inf)

    def cdf(self, x):
        x = np.clip(np.asarray(x, dtype=float), self.low, self.high)
        return np.log(x / self.low) / self._log_ratio

    def ppf(self, u):
        return self.low * np.exp(self._log_ratio * np.asarray(u, dtype=float))

    def to_dict(self) -> dict:
        return {"type": "LogUniform", "low": self.low, "high": self.high}


class Gaussian(Prior):
    """Gaussian prior, optionally truncated to ``[low, high]``.

    Parameters
    ----------
    mean, sigma : float
        Mean and standard deviation of the untruncated distribution.
    low, high : float
        Truncation limits (default: none). The density is renormalised.
    """

    def __init__(self, mean: float, sigma: float, low: float = -math.inf, high: float = math.inf):
        if not sigma > 0:
            raise ValueError(f"Gaussian prior needs sigma > 0, got {sigma}.")
        if not low < high:
            raise ValueError(f"Gaussian prior needs low < high, got ({low}, {high}).")
        self.mean, self.sigma = float(mean), float(sigma)
        self.low, self.high = float(low), float(high)
        self.support = (self.low, self.high)
        self._cdf_low = float(ndtr((self.low - self.mean) / self.sigma))
        self._norm = float(ndtr((self.high - self.mean) / self.sigma)) - self._cdf_low
        if not self._norm > 0:
            raise ValueError("The truncation interval contains no probability mass.")

    def logpdf(self, x):
        x = np.asarray(x, dtype=float)
        z = (x - self.mean) / self.sigma
        log_density = -0.5 * z**2 - math.log(self.sigma * math.sqrt(2.0 * math.pi) * self._norm)
        return np.where((x >= self.low) & (x <= self.high), log_density, -np.inf)

    def cdf(self, x):
        x = np.clip(np.asarray(x, dtype=float), self.low, self.high)
        return np.clip((ndtr((x - self.mean) / self.sigma) - self._cdf_low) / self._norm, 0.0, 1.0)

    def ppf(self, u):
        u = np.asarray(u, dtype=float)
        return self.mean + self.sigma * ndtri(self._cdf_low + u * self._norm)

    def to_dict(self) -> dict:
        return {
            "type": "Gaussian",
            "mean": self.mean,
            "sigma": self.sigma,
            "low": self.low,
            "high": self.high,
        }


_PRIOR_TYPES = {"Uniform": Uniform, "LogUniform": LogUniform, "Gaussian": Gaussian}


def prior_from_dict(spec: dict) -> Prior:
    """Rebuild a prior from the output of :meth:`Prior.to_dict`."""
    spec = dict(spec)
    kind = spec.pop("type")
    if kind not in _PRIOR_TYPES:
        raise ValueError(f"Unknown prior type '{kind}'. Known types: {sorted(_PRIOR_TYPES)}.")
    return _PRIOR_TYPES[kind](**spec)
