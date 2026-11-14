"""Typed objects of the data registry.

The registry stores one YAML file per dataset. Loading a file produces one of
the frozen dataclasses defined here. Numerical values may be ``None``: the
data-integrity rules of the project forbid writing a number that was not read
from an accessible original source, so an unavailable value is stored as
``null`` together with a ``todo`` entry.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

SCHEMA_VERSION = 1
QPO_TYPES = ("qpo_pair", "qpo_triplet")
SHADOW_TYPES = ("shadow_bound",)


class RegistryError(ValueError):
    """A registry file is missing or does not follow the schema."""


@dataclass(frozen=True)
class Reference:
    """A literature reference.

    Attributes
    ----------
    key : str
        Short key used inside the dataset file.
    citation : str
        Human-readable citation.
    doi, arxiv : str or None
        Identifiers.
    role : str
        What the reference was used for.
    """

    key: str
    citation: str
    doi: str | None = None
    arxiv: str | None = None
    role: str = ""


@dataclass(frozen=True)
class Measurement:
    """A measured value with (possibly asymmetric) 1-sigma uncertainties.

    Attributes
    ----------
    value : float or None
        Central value; ``None`` if not verified from an accessible source.
    err_minus, err_plus : float or None
        Lower and upper 1-sigma uncertainties (both positive numbers).
    unit : str
        Unit of the value.
    source : str or None
        Key of the reference the value was read from.
    location : str
        Exact location in the source (table, figure, equation, section).
    note : str
        Free-text remark.
    """

    value: float | None
    err_minus: float | None = None
    err_plus: float | None = None
    unit: str = ""
    source: str | None = None
    location: str = ""
    note: str = ""

    @property
    def is_complete(self) -> bool:
        """True if value and both uncertainties are present."""
        return None not in (self.value, self.err_minus, self.err_plus)

    @property
    def sigma(self) -> float:
        """Symmetrised 1-sigma uncertainty, ``(err_minus + err_plus) / 2``."""
        if not self.is_complete:
            raise RegistryError("This measurement is incomplete (null value or uncertainty).")
        return 0.5 * (self.err_minus + self.err_plus)

    def __str__(self) -> str:
        if self.value is None:
            return "null"
        if self.err_minus is None or self.err_plus is None:
            return f"{self.value:g} {self.unit}".strip()
        if self.err_minus == self.err_plus:
            return f"{self.value:g} +- {self.err_plus:g} {self.unit}".strip()
        return f"{self.value:g} +{self.err_plus:g} -{self.err_minus:g} {self.unit}".strip()


@dataclass(frozen=True)
class MassPrior(Measurement):
    """Mass measurement used as a prior.

    Additional attributes
    ---------------------
    method : str
        How the mass was measured.
    independent : bool or None
        True if the measurement is independent of QPO modelling (e.g. a
        dynamical mass). False if it was derived from a QPO fit in the Kerr
        or Schwarzschild metric: using it to test another metric is circular
        and triggers a :class:`blackpy.errors.CircularInferenceWarning`.
    """

    method: str = ""
    independent: bool | None = None


@dataclass(frozen=True)
class Dataset:
    """Fields common to all datasets.

    Attributes
    ----------
    id : str
        Registry identifier, e.g. ``"qpo/gro_j1655_40"``.
    object : str
        Astronomical object.
    data_type : str
        ``"qpo_pair"``, ``"qpo_triplet"`` or ``"shadow_bound"``.
    dataset_version : str
        Version of this registry entry; bumped whenever a value changes.
    verified_by_human : bool
        Whether a human has checked every value against the source.
    references : dict of str -> Reference
    source_access : str
        How and when the sources were accessed.
    notes : str
    todo : tuple of str
        Open issues (unverified or missing values).
    sha256 : str
        Hash of the YAML file, stored in fit results for reproducibility.
    """

    id: str
    object: str
    data_type: str
    dataset_version: str
    verified_by_human: bool
    references: dict[str, Reference]
    source_access: str = ""
    description: str = ""
    notes: str = ""
    todo: tuple[str, ...] = ()
    sha256: str = ""

    @property
    def is_complete(self) -> bool:
        """True if the dataset has every value needed for inference."""
        raise NotImplementedError

    @property
    def status(self) -> str:
        """Short verification status shown by :func:`blackpy.data.list`."""
        state = "complete" if self.is_complete else "INCOMPLETE (null values)"
        human = "human-verified" if self.verified_by_human else "not human-verified"
        return f"{state}; {human}"


@dataclass(frozen=True)
class QPODataset(Dataset):
    """Twin-peak (and optionally low-frequency) QPO measurement of one source.

    Attributes
    ----------
    upper, lower : Measurement
        Upper and lower high-frequency QPO, in Hz.
    low : Measurement or None
        Low-frequency (type-C) QPO for triplets. It cannot be modelled in
        spherical symmetry and is ignored (with a warning) by the MVP.
    simultaneous : bool
        Whether the peaks were detected in the same data selection.
    mass_prior : MassPrior
        Mass measurement in solar masses.
    intended_models : tuple of str
        QPO models the data have been used with in the source papers.
    alternatives : tuple of dict
        Other published measurements of the same source, for reference.
    """

    upper: Measurement = field(default=None)  # type: ignore[assignment]
    lower: Measurement = field(default=None)  # type: ignore[assignment]
    low: Measurement | None = None
    simultaneous: bool = True
    mass_prior: MassPrior = field(default=None)  # type: ignore[assignment]
    intended_models: tuple[str, ...] = ()
    alternatives: tuple[dict[str, Any], ...] = ()

    @property
    def has_frequencies(self) -> bool:
        """True if both high-frequency QPOs have values and uncertainties."""
        return self.upper.is_complete and self.lower.is_complete

    @property
    def is_complete(self) -> bool:
        return self.has_frequencies and self.mass_prior.is_complete


@dataclass(frozen=True)
class ShadowDataset(Dataset):
    """Bounds on the shadow radius in units of the mass, ``r_sh / M``.

    Attributes
    ----------
    quantity : str
        Bounded quantity (``"r_sh/M"``).
    bounds : dict
        ``{"1sigma": (lower, upper), "2sigma": (lower, upper)}``.
    bounds_location : dict
        Location of each bound in the source.
    fractional_deviation : Measurement or None
        ``delta = r_sh / (3 sqrt(3) M) - 1``, if given by the source; its
        central value and uncertainty define the Gaussian likelihood.
    """

    quantity: str = "r_sh/M"
    bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    bounds_location: dict[str, str] = field(default_factory=dict)
    fractional_deviation: Measurement | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.bounds) and all(
            lo is not None and hi is not None for lo, hi in self.bounds.values()
        )

    def gaussian(self) -> tuple[float, float]:
        """Mean and standard deviation of ``r_sh / M`` for a Gaussian likelihood.

        If the source gives the fractional deviation ``delta`` with a Gaussian
        uncertainty, ``r_sh/M = 3 sqrt(3) (1 + delta)`` is used; otherwise the
        centre and half-width of the 1-sigma interval.
        """
        fd = self.fractional_deviation
        if fd is not None and fd.is_complete:
            scale = 3.0 * 3.0**0.5
            return scale * (1.0 + fd.value), scale * fd.sigma
        lo, hi = self.bounds["1sigma"]
        if lo is None or hi is None:
            raise RegistryError(f"Dataset '{self.id}' has no complete 1-sigma bound.")
        return 0.5 * (lo + hi), 0.5 * (hi - lo)
