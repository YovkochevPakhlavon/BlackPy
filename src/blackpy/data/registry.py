"""Loading and listing of registry datasets."""

from __future__ import annotations

import builtins
import hashlib
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

from blackpy.data.schema import (
    QPO_TYPES,
    SCHEMA_VERSION,
    SHADOW_TYPES,
    CircularityBound,
    Dataset,
    MassPrior,
    Measurement,
    QPODataset,
    Reference,
    RegistryError,
    ShadowDataset,
    SpinMeasurement,
)

_REQUIRED = (
    "schema_version",
    "id",
    "object",
    "data_type",
    "dataset_version",
    "verified_by_human",
    "references",
)


def registry_path() -> Path:
    """Directory that holds the built-in registry files."""
    return Path(str(resources.files("blackpy.data") / "registry"))


def _number(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise RegistryError(f"{where}: expected a number or null, got {value!r}.")
    return float(value)


def _measurement(raw: Any, where: str, cls=Measurement, **extra) -> Measurement:
    if not isinstance(raw, dict):
        raise RegistryError(f"{where}: expected a mapping with 'value', 'err_minus', 'err_plus'.")
    for key in ("value", "err_minus", "err_plus"):
        if key not in raw:
            raise RegistryError(f"{where}: missing field '{key}' (use null if unknown).")
    for key in ("err_minus", "err_plus"):
        if raw[key] is not None and raw[key] < 0:
            raise RegistryError(f"{where}: '{key}' must be a positive number.")
    others_raw = raw.get("other_values") or ()
    if not isinstance(others_raw, list | tuple):
        raise RegistryError(f"{where}: 'other_values' must be a list of measurements.")
    others = []
    for k, item in enumerate(others_raw):
        if isinstance(item, dict) and item.get("other_values"):
            raise RegistryError(f"{where}: other_values[{k}] must not be nested.")
        if not isinstance(item, dict) or not str(item.get("location", "")).strip():
            raise RegistryError(
                f"{where}: other_values[{k}] needs a 'location' (where in the source the "
                "conflicting value is printed)."
            )
        others.append(
            _measurement(
                {"unit": raw.get("unit", ""), "source": raw.get("source"), **item},
                f"{where}: other_values[{k}]",
            )
        )
    reason = str(raw.get("preferred_reason", "")).strip()
    if others and not reason:
        raise RegistryError(
            f"{where}: 'other_values' are given, so 'preferred_reason' must say why the main "
            "value is preferred."
        )
    extra = {**extra, "other_values": tuple(others), "preferred_reason": reason}
    return cls(
        value=_number(raw["value"], where),
        err_minus=_number(raw["err_minus"], where),
        err_plus=_number(raw["err_plus"], where),
        unit=str(raw.get("unit", "")),
        source=raw.get("source"),
        location=str(raw.get("location", "")),
        note=str(raw.get("note", "")),
        **extra,
    )


def parse_dataset(raw: dict, *, sha256: str = "", origin: str = "<memory>") -> Dataset:
    """Validate a raw mapping (parsed YAML) and build the typed dataset.

    Raises
    ------
    RegistryError
        With a message naming the offending field.
    """
    if not isinstance(raw, dict):
        raise RegistryError(f"{origin}: the file must contain a YAML mapping.")
    for key in _REQUIRED:
        if key not in raw:
            raise RegistryError(f"{origin}: missing required field '{key}'.")
    if raw["schema_version"] != SCHEMA_VERSION:
        raise RegistryError(
            f"{origin}: schema_version {raw['schema_version']} is not supported "
            f"(this BlackPy reads version {SCHEMA_VERSION})."
        )
    if not isinstance(raw["verified_by_human"], bool):
        raise RegistryError(f"{origin}: 'verified_by_human' must be true or false.")
    refs_raw = raw["references"]
    if not isinstance(refs_raw, dict) or not refs_raw:
        raise RegistryError(f"{origin}: 'references' must be a non-empty mapping.")
    references = {}
    for key, ref in refs_raw.items():
        if not isinstance(ref, dict) or "citation" not in ref:
            raise RegistryError(f"{origin}: reference '{key}' needs at least a 'citation'.")
        references[key] = Reference(
            key=key,
            citation=str(ref["citation"]),
            doi=ref.get("doi"),
            arxiv=None if ref.get("arxiv") is None else str(ref["arxiv"]),
            role=str(ref.get("role", "")),
        )

    def check_source(m: Measurement | None, where: str):
        if m is not None and m.source is not None and m.source not in references:
            raise RegistryError(f"{origin}: {where} cites unknown reference '{m.source}'.")

    common = dict(
        id=str(raw["id"]),
        object=str(raw["object"]),
        data_type=str(raw["data_type"]),
        dataset_version=str(raw["dataset_version"]),
        verified_by_human=raw["verified_by_human"],
        references=references,
        source_access=str(raw.get("source_access", "")),
        description=str(raw.get("description", "")),
        notes=str(raw.get("notes", "")),
        todo=tuple(str(t) for t in (raw.get("todo") or ())),
        open_questions=tuple(str(t) for t in (raw.get("open_questions") or ())),
        sha256=sha256,
    )
    data_type = common["data_type"]
    if data_type in QPO_TYPES:
        freqs = raw.get("frequencies")
        if not isinstance(freqs, dict) or "upper" not in freqs or "lower" not in freqs:
            raise RegistryError(f"{origin}: 'frequencies' must contain 'upper' and 'lower'.")
        upper = _measurement(freqs["upper"], f"{origin}: frequencies.upper")
        lower = _measurement(freqs["lower"], f"{origin}: frequencies.lower")
        low = None
        if freqs.get("low") is not None:
            low = _measurement(freqs["low"], f"{origin}: frequencies.low")
        if data_type == "qpo_triplet" and low is None:
            raise RegistryError(f"{origin}: a qpo_triplet needs 'frequencies.low'.")
        if "mass_prior" not in raw:
            raise RegistryError(f"{origin}: missing 'mass_prior' (use null values if unknown).")
        mp = raw["mass_prior"]
        independent = mp.get("independent") if isinstance(mp, dict) else None
        if independent is not None and not isinstance(independent, bool):
            raise RegistryError(f"{origin}: mass_prior.independent must be true, false or null.")
        mass = _measurement(
            mp,
            f"{origin}: mass_prior",
            cls=MassPrior,
            method=str(mp.get("method", "")),
            independent=independent,
        )
        spin = None
        if raw.get("spin") is not None:
            sp_raw = raw["spin"]
            spin_independent = sp_raw.get("independent") if isinstance(sp_raw, dict) else None
            if spin_independent is not None and not isinstance(spin_independent, bool):
                raise RegistryError(f"{origin}: spin.independent must be true, false or null.")
            spin = _measurement(
                sp_raw,
                f"{origin}: spin",
                cls=SpinMeasurement,
                method=str(sp_raw.get("method", "")),
                independent=spin_independent,
            )
        published_raw = raw.get("published_fit") or {}
        if not isinstance(published_raw, dict):
            raise RegistryError(f"{origin}: 'published_fit' must be a mapping.")
        published = {
            str(key): _measurement(value, f"{origin}: published_fit.{key}")
            for key, value in published_raw.items()
        }
        for m, where in (
            (upper, "frequencies.upper"),
            (lower, "frequencies.lower"),
            (low, "frequencies.low"),
            (mass, "mass_prior"),
            (spin, "spin"),
            *[(m, f"published_fit.{key}") for key, m in published.items()],
        ):
            check_source(m, where)
        return QPODataset(
            **common,
            upper=upper,
            lower=lower,
            low=low,
            spin=spin,
            published_fit=published,
            simultaneous=bool(raw.get("simultaneous", True)),
            mass_prior=mass,
            intended_models=tuple(str(m) for m in (raw.get("intended_models") or ())),
            alternatives=tuple(raw.get("alternatives") or ()),
        )
    if data_type in SHADOW_TYPES:
        bounds_raw = raw.get("bounds")
        if not isinstance(bounds_raw, dict) or not bounds_raw:
            raise RegistryError(f"{origin}: 'bounds' must be a non-empty mapping.")
        bounds, locations = {}, {}
        for level, b in bounds_raw.items():
            if not isinstance(b, dict) or "lower" not in b or "upper" not in b:
                raise RegistryError(f"{origin}: bounds.{level} needs 'lower' and 'upper'.")
            bounds[str(level)] = (
                _number(b["lower"], f"{origin}: bounds.{level}.lower"),
                _number(b["upper"], f"{origin}: bounds.{level}.upper"),
            )
            locations[str(level)] = str(b.get("location", ""))
        deviation = None
        if raw.get("fractional_deviation") is not None:
            deviation = _measurement(raw["fractional_deviation"], f"{origin}: fractional_deviation")
            check_source(deviation, "fractional_deviation")

        def optional(key):
            if raw.get(key) is None:
                return None
            m = _measurement(raw[key], f"{origin}: {key}")
            check_source(m, key)
            return m

        priors_raw = raw.get("mass_priors") or {}
        if not isinstance(priors_raw, dict):
            raise RegistryError(f"{origin}: 'mass_priors' must be a mapping name -> measurement.")
        mass_priors = {}
        for key, item in priors_raw.items():
            where = f"{origin}: mass_priors.{key}"
            if not isinstance(item, dict):
                raise RegistryError(f"{where}: expected a mapping.")
            independent = item.get("independent")
            if not isinstance(independent, bool):
                raise RegistryError(
                    f"{where}: 'independent' must be true or false (is this mass independent of "
                    "the horizon-scale image?)."
                )
            extra = {}
            for sub in ("theta_g", "fractional_deviation"):
                if item.get(sub) is not None:
                    extra[sub] = _measurement(
                        {"source": item.get("source"), **item[sub]}, f"{where}.{sub}"
                    )
                    check_source(extra[sub], f"mass_priors.{key}.{sub}")
            mass_priors[str(key)] = _measurement(
                item,
                where,
                cls=MassPrior,
                method=str(item.get("method", "")),
                independent=independent,
                **extra,
            )
            check_source(mass_priors[str(key)], f"mass_priors.{key}")
        assumed = raw.get("assumed_mass_priors") or ()
        if isinstance(assumed, str):
            assumed = (assumed,)
        unknown = [key for key in assumed if key not in mass_priors]
        if unknown:
            raise RegistryError(
                f"{origin}: assumed_mass_priors {unknown} are not entries of 'mass_priors'."
            )
        measured_raw = raw.get("measured") or {}
        if not isinstance(measured_raw, dict):
            raise RegistryError(f"{origin}: 'measured' must be a mapping name -> measurement.")
        measured = {}
        for key, item in measured_raw.items():
            measured[str(key)] = _measurement(item, f"{origin}: measured.{key}")
            check_source(measured[str(key)], f"measured.{key}")
        circularity = None
        if raw.get("circularity") is not None:
            c = raw["circularity"]
            if not isinstance(c, dict) or "upper" not in c or not c.get("definition"):
                raise RegistryError(
                    f"{origin}: 'circularity' needs 'upper' and 'definition' (the definition of "
                    "the deviation from circularity for which the bound was derived)."
                )
            circularity = CircularityBound(
                upper=_number(c["upper"], f"{origin}: circularity.upper"),
                definition=str(c["definition"]),
                applies_to=str(c.get("applies_to", "")),
                source=c.get("source"),
                location=str(c.get("location", "")),
                note=str(c.get("note", "")),
            )
            if circularity.source is not None and circularity.source not in references:
                raise RegistryError(
                    f"{origin}: circularity cites unknown reference '{circularity.source}'."
                )
        return ShadowDataset(
            **common,
            quantity=str(raw.get("quantity", "r_sh/M")),
            bounds=bounds,
            bounds_location=locations,
            fractional_deviation=deviation,
            assumed_mass_priors=tuple(str(key) for key in assumed),
            mass_priors=mass_priors,
            measured=measured,
            distance=optional("distance"),
            inclination=optional("inclination"),
            circularity=circularity,
        )
    raise RegistryError(
        f"{origin}: unknown data_type '{data_type}'. Supported: {QPO_TYPES + SHADOW_TYPES}."
    )


def load_file(path) -> Dataset:
    """Load a dataset from an arbitrary YAML file (e.g. your own data)."""
    path = Path(path)
    if not path.is_file():
        raise RegistryError(f"Dataset file not found: {path}")
    content = path.read_bytes()
    raw = yaml.safe_load(content)
    return parse_dataset(raw, sha256=hashlib.sha256(content).hexdigest(), origin=str(path))


def available() -> builtins.list[str]:
    """Identifiers of all built-in datasets, sorted."""
    root = registry_path()
    return sorted(p.relative_to(root).with_suffix("").as_posix() for p in root.rglob("*.yaml"))


def load(dataset_id: str) -> Dataset:
    """Load a built-in dataset by identifier.

    Parameters
    ----------
    dataset_id : str
        E.g. ``"qpo/xte_j1550_564"`` or ``"eht/sgra"``.

    Returns
    -------
    QPODataset or ShadowDataset

    Raises
    ------
    RegistryError
        If the identifier is unknown or the file violates the schema.
    """
    path = registry_path() / f"{dataset_id}.yaml"
    if not path.is_file():
        raise RegistryError(
            f"Unknown dataset '{dataset_id}'. Available datasets: {available()}. "
            "To use your own file call blackpy.data.load_file(path)."
        )
    dataset = load_file(path)
    if dataset.id != dataset_id:
        raise RegistryError(f"{path}: the 'id' field ({dataset.id}) does not match the file path.")
    return dataset


def list_datasets(verbose: bool = True) -> builtins.list[Dataset]:
    """List the registry with the verification status of every dataset.

    Parameters
    ----------
    verbose : bool
        Print a table (default). The datasets are returned in any case.

    Returns
    -------
    list of Dataset
    """
    datasets = [load(i) for i in available()]
    if verbose:
        print(f"{'id':<22} {'object':<15} {'type':<13} {'version':<8} status")
        for d in datasets:
            print(f"{d.id:<22} {d.object:<15} {d.data_type:<13} {d.dataset_version:<8} {d.status}")
            for item in d.todo:
                print(f"{'':<22} TODO: {item}")
            for item in d.open_questions:
                print(f"{'':<22} OPEN QUESTION: {item}")
    return datasets
