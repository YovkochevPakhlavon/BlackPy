"""Data registry: schema, loading, integrity flags.

These tests check structure and bookkeeping only. No test depends on the
numerical value of an unverified registry entry.
"""

import copy
import hashlib

import pytest
import yaml

import blackpy as bp
from blackpy.data import MassPrior, Measurement, QPODataset, RegistryError, ShadowDataset

EXPECTED = {
    "eht/m87",
    "eht/sgra",
    "qpo/gro_j1655_40",
    "qpo/xte_j1859_226",
    "qpo/xte_j1550_564",
    "qpo/grs_1915_105",
    "qpo/h1743_322",
    "qpo/m82_x1",
}


def test_registry_contains_initial_targets():
    assert set(bp.data.available()) == EXPECTED


@pytest.mark.parametrize("dataset_id", sorted(EXPECTED))
def test_every_dataset_loads_and_documents_its_sources(dataset_id):
    d = bp.data.load(dataset_id)
    assert d.id == dataset_id and len(d.sha256) == 64
    # Rule: nothing is human-verified until the owner flips the flag by hand.
    assert d.verified_by_human is False
    assert d.references and d.source_access
    for ref in d.references.values():
        assert ref.citation
    if isinstance(d, QPODataset):
        assert d.data_type in ("qpo_pair", "qpo_triplet")
        measurements = [d.upper, d.lower, d.mass_prior] + ([d.low] if d.low else [])
        if d.data_type == "qpo_triplet":
            assert d.low is not None
    else:
        assert isinstance(d, ShadowDataset) and "1sigma" in d.bounds
        measurements = [d.fractional_deviation]
    for m in measurements:
        # Rule: every stored number has a source and an exact location.
        if m.value is not None:
            assert m.source in d.references and m.location, (dataset_id, m)
        # Rule: a missing value must be explained by a TODO entry.
        if not m.is_complete:
            assert d.todo, dataset_id
    if not d.is_complete:
        assert d.todo and "INCOMPLETE" in d.status


def test_typed_objects_and_ordering():
    gro = bp.data.load("qpo/gro_j1655_40")
    assert isinstance(gro, QPODataset) and isinstance(gro.mass_prior, MassPrior)
    assert isinstance(gro.upper, Measurement) and gro.upper.unit == "Hz"
    assert gro.upper.value > gro.lower.value > gro.low.value
    assert gro.mass_prior.unit == "Msun" and gro.mass_prior.independent is True
    for dataset_id in EXPECTED:
        d = bp.data.load(dataset_id)
        if isinstance(d, QPODataset) and d.has_frequencies:
            assert d.upper.value > d.lower.value


def test_circular_mass_prior_is_flagged_in_registry():
    m82 = bp.data.load("qpo/m82_x1")
    assert m82.mass_prior.independent is False
    h1743 = bp.data.load("qpo/h1743_322")
    assert h1743.mass_prior.value is None and not h1743.is_complete and h1743.has_frequencies


def test_shadow_dataset_gaussian_is_consistent_with_bounds():
    sgra = bp.data.load("eht/sgra")
    mean, sigma = sgra.gaussian()
    lo1, hi1 = sgra.bounds["1sigma"]
    lo2, hi2 = sgra.bounds["2sigma"]
    # Internal consistency of the source: the intervals follow from delta and its error.
    assert abs((mean - sigma) - lo1) < 0.006 and abs((mean + sigma) - hi1) < 0.006
    assert abs((mean - 2 * sigma) - lo2) < 0.006 and abs((mean + 2 * sigma) - hi2) < 0.006
    assert lo2 < lo1 < mean < hi1 < hi2


def test_list_prints_verification_status(capsys):
    datasets = bp.data.list()
    out = capsys.readouterr().out
    assert len(datasets) == len(EXPECTED)
    assert "not human-verified" in out and "TODO" in out and "qpo/gro_j1655_40" in out


def test_dataset_hash_does_not_depend_on_line_endings(tmp_path):
    # Git on Windows checks files out with CRLF by default; the recorded hash must not change.
    content = (bp.data.registry_path() / "qpo" / "gro_j1655_40.yaml").read_bytes()
    lf = content.replace(b"\r\n", b"\n")
    expected = hashlib.sha256(lf).hexdigest()
    assert bp.data.load("qpo/gro_j1655_40").sha256 == expected
    for name, data in [
        ("lf", lf),
        ("crlf", lf.replace(b"\n", b"\r\n")),
        ("cr", lf.replace(b"\n", b"\r")),
    ]:
        path = tmp_path / f"{name}.yaml"
        path.write_bytes(data)
        assert bp.data.load_file(path).sha256 == expected, name
    # A change of content still changes the hash.
    path = tmp_path / "changed.yaml"
    path.write_bytes(lf + b"# comment\n")
    assert bp.data.load_file(path).sha256 != expected


def test_unknown_dataset_and_schema_violations(tmp_path):
    with pytest.raises(RegistryError, match="Available datasets"):
        bp.data.load("qpo/does_not_exist")
    raw = yaml.safe_load((bp.data.registry_path() / "qpo" / "gro_j1655_40.yaml").read_text())
    assert isinstance(bp.data.parse_dataset(copy.deepcopy(raw)), QPODataset)

    def broken(mutate):
        bad = copy.deepcopy(raw)
        mutate(bad)
        return bad

    cases = [
        (lambda r: r.pop("verified_by_human"), "verified_by_human"),
        (lambda r: r.update(verified_by_human="yes"), "true or false"),
        (lambda r: r.update(data_type="spectrum"), "unknown data_type"),
        (lambda r: r["frequencies"].pop("lower"), "upper' and 'lower"),
        (lambda r: r["frequencies"]["upper"].pop("err_plus"), "err_plus"),
        (lambda r: r["frequencies"]["upper"].update(value="441"), "number or null"),
        (lambda r: r["frequencies"]["upper"].update(source="nobody"), "unknown reference"),
        (lambda r: r.update(schema_version=99), "not supported"),
        (lambda r: r.pop("mass_prior"), "mass_prior"),
    ]
    for mutate, message in cases:
        with pytest.raises(RegistryError, match=message):
            bp.data.parse_dataset(broken(mutate))
    # User files go through the same validation.
    path = tmp_path / "mine.yaml"
    path.write_text(yaml.safe_dump(raw))
    assert bp.data.load_file(path).object == "GRO J1655-40"
    with pytest.raises(RegistryError, match="not found"):
        bp.data.load_file(tmp_path / "missing.yaml")


def test_measurement_helpers():
    m = Measurement(10.0, 1.0, 3.0, "Hz")
    assert m.sigma == 2.0 and str(m) == "10 +3 -1 Hz"
    assert str(Measurement(None)) == "null"
    with pytest.raises(RegistryError):
        _ = Measurement(None, None, None).sigma


def test_conflicting_published_values_are_stored_with_a_preference(tmp_path):
    """Schema only: a quantity printed twice keeps both values, each with its location."""
    gro = bp.data.load("qpo/gro_j1655_40")
    for m in (gro.spin, gro.published_fit["mass"], gro.published_fit["radius"]):
        assert m.has_conflicting_values and m.preferred_reason
        assert "Table 3" in m.location  # the preferred entry is the table
        for other in m.other_values:
            assert isinstance(other, Measurement) and other.is_complete and other.location
            assert other.unit == m.unit and other.source == m.source  # inherited
            assert not other.other_values
    assert not gro.upper.has_conflicting_values and gro.upper.preferred_reason == ""

    raw = yaml.safe_load((bp.data.registry_path() / "qpo/gro_j1655_40.yaml").read_text())

    def load_modified(change):
        modified = copy.deepcopy(raw)
        change(modified)
        path = tmp_path / "modified.yaml"
        path.write_text(yaml.safe_dump(modified))
        return bp.data.load_file(path)

    with pytest.raises(RegistryError, match="preferred_reason"):
        load_modified(lambda d: d["spin"].pop("preferred_reason"))
    with pytest.raises(RegistryError, match="location"):
        load_modified(lambda d: d["spin"]["other_values"][0].pop("location"))
    with pytest.raises(RegistryError, match="list"):
        load_modified(lambda d: d["spin"].update(other_values="0.29"))
    with pytest.raises(RegistryError, match="nested"):
        load_modified(
            lambda d: d["spin"]["other_values"][0].update(
                other_values=[{"value": 1.0, "err_minus": 0.1, "err_plus": 0.1, "location": "x"}]
            )
        )


def test_open_questions_are_loaded_and_listed(capsys):
    """Schema only: unresolved questions about published values travel with the dataset."""
    xte = bp.data.load("qpo/xte_j1859_226")
    assert len(xte.open_questions) == 1 and "uncertainties" in xte.open_questions[0]
    assert bp.data.load("qpo/gro_j1655_40").open_questions == ()
    bp.data.list()
    assert "OPEN QUESTION:" in capsys.readouterr().out
