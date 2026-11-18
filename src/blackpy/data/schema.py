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
    other_values : tuple of Measurement
        Other published values of the *same* quantity that disagree with
        ``value`` (for example a number in the text of a paper that differs
        from the one in its table), each with its own ``location``. The
        main entry is then the preferred one and ``preferred_reason`` says
        why. They are stored for the record and never used in a fit.
    preferred_reason : str
        Why ``value`` is preferred over ``other_values`` (empty if there are
        none).
    """

    value: float | None
    err_minus: float | None = None
    err_plus: float | None = None
    unit: str = ""
    source: str | None = None
    location: str = ""
    note: str = ""
    other_values: tuple = ()
    preferred_reason: str = ""

    @property
    def has_conflicting_values(self) -> bool:
        """True if the source gives more than one value for this quantity."""
        return bool(self.other_values)

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
        True if the measurement is independent of the data it is combined
        with: of QPO modelling for a QPO source (e.g. a dynamical mass), of
        the horizon-scale image for a shadow source (e.g. stellar dynamics).
        False if it was derived from those data themselves (a QPO fit in the
        Kerr metric; a mass obtained from the EHT ring): using it to test
        another metric is circular and triggers a
        :class:`blackpy.errors.CircularInferenceWarning`.
    theta_g : Measurement or None
        Shadow sources: the angular gravitational radius
        :math:`GM/(c^2 D)` implied by this mass measurement. It is the
        quantity a dynamical measurement actually determines (a mass scales
        with the assumed distance).
    fractional_deviation : Measurement or None
        Shadow sources: the published fractional deviation ``delta`` of the
        shadow size obtained with this mass prior.
    """

    method: str = ""
    independent: bool | None = None
    theta_g: Measurement | None = None
    fractional_deviation: Measurement | None = None


@dataclass(frozen=True)
class CircularityBound:
    """Upper bound on the deviation from circularity of an image.

    Attributes
    ----------
    upper : float or None
        The bound (dimensionless).
    definition : str
        The definition of the deviation from circularity for which the bound
        was derived: one of ``blackpy.light.DELTA_C_DEFINITIONS``, or another
        label if none of them matches. A likelihood that uses the bound with
        a different definition raises a
        :class:`blackpy.errors.CircularityDefinitionWarning`.
    applies_to : str
        What was measured (for example the emission ring, which is not the
        shadow boundary).
    source, location, note : str
        Provenance, as for :class:`Measurement`.
    """

    upper: float | None
    definition: str
    applies_to: str = ""
    source: str | None = None
    location: str = ""
    note: str = ""


@dataclass(frozen=True)
class SpinMeasurement(Measurement):
    """Published spin of the source (dimensionless, ``a = J c / (G M^2)``).

    Additional attributes
    ---------------------
    method : str
        How the spin was measured (e.g. a relativistic-precession-model fit
        of the QPO triplet in the Kerr metric).
    independent : bool or None
        True if the measurement does not rely on QPO modelling. A spin that
        was derived from the same QPOs in the Kerr metric must not be used
        as a prior when another metric is tested with those QPOs.
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
    open_questions : tuple of str
        Unresolved scientific questions about the published values (for
        example a published uncertainty that could not be reproduced). Unlike
        ``todo`` items they cannot be closed by checking the paper again.
    sha256 : str
        SHA-256 hash of the YAML file with its line endings normalised to LF,
        stored in fit results for reproducibility. The same file therefore has
        the same hash on every operating system, whatever line endings the
        checkout or an editor gave it.
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
    open_questions: tuple[str, ...] = ()

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
        Low-frequency (type-C) QPO for triplets. With a rotating metric it is
        fitted as the nodal precession frequency; with a static metric it
        cannot be modelled and is ignored with a warning.
    spin : SpinMeasurement or None
        Published spin of the source, if any. It marks the source as
        rotating (fitting it with a static metric triggers a
        :class:`blackpy.errors.SpinIgnoredWarning`) and is used by the
        examples that reproduce published results. It is never used as a
        prior automatically.
    published_fit : dict of str -> Measurement
        Other published results of the same fit as ``spin`` (e.g. ``"mass"``
        and ``"radius"`` of a relativistic-precession-model solution), for
        comparison only.
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
    spin: SpinMeasurement | None = None
    published_fit: dict[str, Measurement] = field(default_factory=dict)
    simultaneous: bool = True
    mass_prior: MassPrior = field(default=None)  # type: ignore[assignment]
    intended_models: tuple[str, ...] = ()
    alternatives: tuple[dict[str, Any], ...] = ()

    @property
    def is_triplet(self) -> bool:
        """True if a complete low-frequency QPO is available besides the twin peaks."""
        return self.low is not None and self.low.is_complete

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
    assumed_mass_priors : tuple of str
        Keys of ``mass_priors`` on which the default ``bounds`` and
        ``fractional_deviation`` rest (several keys: an average). A shadow
        size in units of M always assumes a mass-to-distance ratio.
    mass_priors : dict of str -> MassPrior
        Mass measurements of the source, each with its origin, whether it is
        independent of the horizon-scale image, and (where published) the
        angular gravitational radius and the fractional deviation obtained
        with it.
    measured : dict of str -> Measurement
        What the experiment measured (for example the diameter of the
        emission ring and the calibrated angular gravitational radius), for
        the record; the likelihood uses the fractional deviation.
    distance : Measurement or None
    inclination : Measurement or None
        A measured inclination (for example of the jet), in degrees. It is
        **not** used as a prior automatically; see
        :func:`blackpy.infer.inclination_prior`.
    circularity : CircularityBound or None
        Bound on the deviation from circularity, with its definition.
    """

    quantity: str = "r_sh/M"
    bounds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    bounds_location: dict[str, str] = field(default_factory=dict)
    fractional_deviation: Measurement | None = None
    assumed_mass_priors: tuple[str, ...] = ()
    mass_priors: dict[str, MassPrior] = field(default_factory=dict)
    measured: dict[str, Measurement] = field(default_factory=dict)
    distance: Measurement | None = None
    inclination: Measurement | None = None
    circularity: CircularityBound | None = None

    @property
    def is_complete(self) -> bool:
        return bool(self.bounds) and all(
            lo is not None and hi is not None for lo, hi in self.bounds.values()
        )

    def mass_prior_is_independent(self, mass_prior: str | None = None) -> bool | None:
        """Whether the mass prior behind the shadow bound is independent of the image.

        ``mass_prior=None`` refers to the default bound (``assumed_mass_priors``).
        Returns None if this is not recorded.
        """
        keys = self.assumed_mass_priors if mass_prior is None else (mass_prior,)
        if not keys:
            return None
        flags = [self._mass_prior(key).independent for key in keys]
        if any(flag is False for flag in flags):
            return False
        return True if all(flag is True for flag in flags) else None

    def _mass_prior(self, key: str) -> MassPrior:
        if key not in self.mass_priors:
            raise RegistryError(
                f"Dataset '{self.id}' has no mass prior '{key}'. Available: "
                f"{sorted(self.mass_priors) or 'none'}."
            )
        return self.mass_priors[key]

    def deviation(self, mass_prior: str | None = None) -> Measurement:
        r"""Fractional deviation ``delta`` of the shadow size for a choice of mass prior.

        ``mass_prior=None``: the default of the dataset. Otherwise the value
        published for that mass prior; if none was published it is computed
        as :math:`\delta = \theta_g/\theta_{\rm dyn} - 1` from the measured
        angular gravitational radius of the image (``measured["theta_g"]``)
        and that of the mass prior, with the two (symmetrised) relative
        uncertainties added in quadrature. This is the definition of the EHT
        Collaboration (2019, Paper VI, Eq. (31)).
        """
        if mass_prior is None:
            if self.fractional_deviation is None or not self.fractional_deviation.is_complete:
                raise RegistryError(f"Dataset '{self.id}' has no default fractional deviation.")
            return self.fractional_deviation
        prior = self._mass_prior(mass_prior)
        if prior.fractional_deviation is not None and prior.fractional_deviation.is_complete:
            return prior.fractional_deviation
        image = self.measured.get("theta_g")
        if image is None or not image.is_complete or prior.theta_g is None:
            raise RegistryError(
                f"Dataset '{self.id}': mass prior '{mass_prior}' has no published fractional "
                "deviation and the angular gravitational radii needed to compute one are "
                "missing."
            )
        if not prior.theta_g.is_complete:
            raise RegistryError(f"Dataset '{self.id}': theta_g of '{mass_prior}' is incomplete.")
        ratio = image.value / prior.theta_g.value
        sigma = (
            ratio
            * ((image.sigma / image.value) ** 2 + (prior.theta_g.sigma / prior.theta_g.value) ** 2)
            ** 0.5
        )
        return Measurement(
            value=ratio - 1.0,
            err_minus=sigma,
            err_plus=sigma,
            source=prior.source,
            location=f"computed by BlackPy from theta_g (image) and theta_g ('{mass_prior}')",
        )

    def gaussian(self, mass_prior: str | None = None) -> tuple[float, float]:
        """Mean and standard deviation of ``r_sh / M`` for a Gaussian likelihood.

        If the source gives the fractional deviation ``delta`` with a Gaussian
        uncertainty, ``r_sh/M = 3 sqrt(3) (1 + delta)`` is used; otherwise the
        centre and half-width of the 1-sigma interval. ``mass_prior`` selects
        the deviation obtained with another mass prior of the dataset
        (see :meth:`deviation`).
        """
        scale = 3.0 * 3.0**0.5
        if mass_prior is not None:
            fd = self.deviation(mass_prior)
            return scale * (1.0 + fd.value), scale * fd.sigma
        fd = self.fractional_deviation
        if fd is not None and fd.is_complete:
            return scale * (1.0 + fd.value), scale * fd.sigma
        lo, hi = self.bounds["1sigma"]
        if lo is None or hi is None:
            raise RegistryError(f"Dataset '{self.id}' has no complete 1-sigma bound.")
        return 0.5 * (lo + hi), 0.5 * (hi - lo)
