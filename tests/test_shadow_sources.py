"""Several shadow sources: registry schema, mass priors, per-source parameters (B3).

Registry entries are tested for structure only; no test depends on a number of an
unverified dataset. Numerical tests use synthetic datasets.
"""

import copy
import math
import warnings

import numpy as np
import pytest
import yaml

import blackpy as bp
from blackpy.data import CircularityBound, MassPrior, Measurement, RegistryError, ShadowDataset
from blackpy.errors import (
    BlackPyError,
    BlackPyWarning,
    CircularInferenceWarning,
    CircularityDefinitionWarning,
)
from blackpy.infer import (
    Fit,
    Gaussian,
    Uniform,
    build_problem,
    inclination_prior,
    simulate_shadow_dataset,
)

RAW = {
    "schema_version": 1,
    "id": "synthetic/ring",
    "object": "Ring",
    "data_type": "shadow_bound",
    "dataset_version": "t",
    "verified_by_human": True,
    "bounds": {"1sigma": {"lower": 4.5, "upper": 5.5}},
    "fractional_deviation": {"value": -0.02, "err_minus": 0.10, "err_plus": 0.10},
    "assumed_mass_priors": ["stars"],
    "mass_priors": {
        "stars": {
            "value": 6.0,
            "err_minus": 0.5,
            "err_plus": 0.5,
            "unit": "1e9 Msun",
            "independent": True,
            "theta_g": {"value": 3.6, "err_minus": 0.3, "err_plus": 0.5, "unit": "muas"},
            "fractional_deviation": {"value": -0.02, "err_minus": 0.10, "err_plus": 0.10},
        },
        "gas": {
            "value": 3.5,
            "err_minus": 0.3,
            "err_plus": 0.8,
            "unit": "1e9 Msun",
            "independent": True,
            "theta_g": {"value": 2.0, "err_minus": 0.2, "err_plus": 0.4, "unit": "muas"},
        },
        "image": {
            "value": 6.4,
            "err_minus": 0.7,
            "err_plus": 0.7,
            "unit": "1e9 Msun",
            "independent": False,
            "theta_g": {"value": 3.8, "err_minus": 0.4, "err_plus": 0.4, "unit": "muas"},
        },
    },
    "measured": {"theta_g": {"value": 3.8, "err_minus": 0.4, "err_plus": 0.4, "unit": "muas"}},
    "distance": {"value": 16.8, "err_minus": 0.8, "err_plus": 0.8, "unit": "Mpc"},
    "inclination": {"value": 17.0, "err_minus": 3.0, "err_plus": 3.0, "unit": "deg"},
    "circularity": {"upper": 0.1, "definition": "fractional_rms", "applies_to": "ring"},
    "references": {"synthetic": {"citation": "synthetic"}},
}


def ring(**changes):
    raw = copy.deepcopy(RAW)
    raw.update(changes)
    return bp.data.parse_dataset(raw)


# ----------------------------------------------------------------------------- registry schema


def test_m87_entry_documents_what_is_measured():
    """Schema and bookkeeping of eht/m87 (no numerical value is tested)."""
    m87 = bp.data.load("eht/m87")
    assert isinstance(m87, ShadowDataset) and m87.verified_by_human is False and m87.is_complete
    # The measured quantity is the ring, not the shadow, and the file says so.
    assert set(m87.measured) >= {"ring_diameter", "theta_g"}
    assert "not of the shadow" in m87.measured["ring_diameter"].note
    for m in m87.measured.values():
        assert m.is_complete and m.source in m87.references and m.location
    # Two discrepant, independent mass measurements and the image-based one.
    assert set(m87.mass_priors) == {"stellar_dynamics", "gas_dynamics", "eht_ring"}
    for key, prior in m87.mass_priors.items():
        assert isinstance(prior, MassPrior) and prior.is_complete and prior.location, key
        assert prior.source in m87.references and isinstance(prior.independent, bool)
        assert prior.theta_g is not None and prior.theta_g.is_complete
    assert m87.mass_priors["stellar_dynamics"].independent is True
    assert m87.mass_priors["gas_dynamics"].independent is True
    assert m87.mass_priors["eht_ring"].independent is False  # derived from the image itself
    for key in ("stellar_dynamics", "gas_dynamics"):  # originals are kept with locations
        original = m87.mass_priors[key].other_values
        assert original and original[0].location and m87.mass_priors[key].preferred_reason
    assert m87.assumed_mass_priors == ("stellar_dynamics",)
    assert m87.mass_prior_is_independent() is True
    assert m87.mass_prior_is_independent("eht_ring") is False
    # Distance, inclination (a jet angle, stored as a measurement) and circularity.
    assert m87.distance.is_complete and m87.distance.unit == "Mpc"
    assert m87.inclination.is_complete and m87.inclination.unit == "deg"
    assert "JET" in m87.inclination.note and m87.inclination.other_values
    assert isinstance(m87.circularity, CircularityBound)
    assert m87.circularity.definition in bp.light.DELTA_C_DEFINITIONS
    assert m87.circularity.source in m87.references and m87.circularity.applies_to
    # The default Gaussian is the published deviation for the assumed mass prior.
    assert m87.gaussian() == m87.gaussian("stellar_dynamics")
    assert m87.gaussian("gas_dynamics") != m87.gaussian()
    assert "1sigma" in m87.bounds and any("2-sigma" in item for item in m87.todo)


def test_sgra_keeps_loading_and_gains_mass_and_distance():
    sgra = bp.data.load("eht/sgra")
    mean, sigma = sgra.gaussian()  # unchanged definition: the published average
    fd = sgra.fractional_deviation
    assert mean == pytest.approx(3 * math.sqrt(3) * (1 + fd.value), rel=1e-14)
    assert sigma == pytest.approx(3 * math.sqrt(3) * fd.sigma, rel=1e-14)
    assert set(sgra.bounds) == {"1sigma", "2sigma"}
    assert set(sgra.mass_priors) == {"keck", "vlti"} and sgra.assumed_mass_priors == (
        "keck",
        "vlti",
    )
    for prior in sgra.mass_priors.values():
        assert prior.independent is True and prior.is_complete and prior.location
        assert prior.theta_g.is_complete and prior.fractional_deviation.is_complete
        assert prior.source in sgra.references and prior.other_values
    assert sgra.mass_prior_is_independent() is True
    assert sgra.distance.is_complete and sgra.distance.unit == "kpc" and sgra.distance.other_values
    assert sgra.inclination is None and sgra.circularity is None
    assert sgra.open_questions  # the Keck numbers differ between papers


def test_shadow_schema_and_deviation_for_a_chosen_mass_prior():
    data = ring()
    scale = 3.0 * math.sqrt(3.0)
    assert data.gaussian() == pytest.approx((scale * 0.98, scale * 0.10))
    assert data.gaussian("stars") == data.gaussian()
    # No published deviation for 'gas': delta = theta_g(image) / theta_g(prior) - 1, with the
    # symmetrised relative uncertainties added in quadrature.
    deviation = data.deviation("gas")
    assert deviation.value == pytest.approx(3.8 / 2.0 - 1.0)
    assert deviation.sigma == pytest.approx(1.9 * math.hypot(0.4 / 3.8, 0.3 / 2.0))
    assert "computed by BlackPy" in deviation.location
    assert data.gaussian("gas") == pytest.approx((scale * 1.9, scale * deviation.sigma))
    assert data.deviation("image").value == pytest.approx(0.0)
    assert data.mass_prior_is_independent("image") is False
    assert data.mass_prior_is_independent("gas") is True
    assert ring(assumed_mass_priors=["stars", "image"]).mass_prior_is_independent() is False
    assert ring(assumed_mass_priors=[]).mass_prior_is_independent() is None
    with pytest.raises(RegistryError, match="no mass prior 'dust'"):
        data.gaussian("dust")
    assert isinstance(data.distance, Measurement) and data.circularity.upper == 0.1

    def broken(change):
        raw = copy.deepcopy(RAW)
        change(raw)
        return bp.data.parse_dataset(raw)

    with pytest.raises(RegistryError, match="independent"):
        broken(lambda r: r["mass_priors"]["gas"].pop("independent"))
    with pytest.raises(RegistryError, match="assumed_mass_priors"):
        broken(lambda r: r.update(assumed_mass_priors=["dust"]))
    with pytest.raises(RegistryError, match="definition"):
        broken(lambda r: r["circularity"].pop("definition"))
    with pytest.raises(RegistryError, match="unknown reference"):
        broken(lambda r: r["distance"].update(source="nowhere"))
    with pytest.raises(RegistryError, match="theta_g"):
        broken(lambda r: r["mass_priors"]["gas"].update(theta_g={"value": 2.0}))
    no_theta = broken(lambda r: r["measured"].pop("theta_g"))
    with pytest.raises(RegistryError, match="no published fractional deviation"):
        no_theta.gaussian("gas")
    # The built-in files round-trip through the parser with the new fields.
    raw = yaml.safe_load((bp.data.registry_path() / "eht/m87.yaml").read_text())
    assert (
        bp.data.parse_dataset(raw).mass_priors.keys() == bp.data.load("eht/m87").mass_priors.keys()
    )


# ----------------------------------------------------------------------------- likelihood


def two_static_sources(rn, q_a=0.3, q_b=0.8, sigma=0.05):
    a = simulate_shadow_dataset(rn, name="a", sigma=sigma, Q=q_a)
    b = simulate_shadow_dataset(rn, name="b", sigma=sigma, Q=q_b)
    return [a, b]


def test_two_shadow_sources_need_an_explicit_choice(rn):
    data = two_static_sources(rn)
    with pytest.raises(ValueError, match=r"shared=\['Q'\].*per_source=\['Q'\]"):
        build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)})
    shared = build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shared=["Q"])
    assert shared.param_names == ["Q"] and shared.description["shared"] == ["Q"]
    separate = build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, per_source=["Q"])
    assert separate.param_names == ["Q_a", "Q_b"] and separate.kinds == ["metric", "metric"]
    assert separate.description["parameter_base"] == {"Q_a": "Q", "Q_b": "Q"}
    assert [row["label"] for row in separate.data_table] == ["r_sh[a]", "r_sh[b]"]
    # Hand computation: each source is compared with the shadow of its own charge.
    centre = [d.gaussian()[0] for d in data]
    for theta in ((0.3, 0.8), (0.5, 0.1)):
        model = [float(bp.light.shadow_radius(rn, Q=q)) for q in theta]
        expected = sum(
            -0.5 * ((m - c) / 0.05) ** 2 - math.log(0.05 * math.sqrt(2 * math.pi))
            for m, c in zip(model, centre, strict=True)
        )
        assert float(separate.log_likelihood(np.array(theta))) == pytest.approx(expected, rel=1e-10)
        np.testing.assert_allclose(separate.predict(np.array(theta)), model, rtol=1e-12)
    both = float(shared.log_likelihood(np.array([0.3])))
    assert both == pytest.approx(float(separate.log_likelihood(np.array([0.3, 0.3]))), rel=1e-12)
    # One copy sampled, the other fixed; a generic prior applies to every copy.
    mixed = build_problem(
        rn, data, priors={"Q_a": Uniform(0.0, 1.0)}, fixed={"Q_b": 0.8}, per_source=["Q"]
    )
    assert mixed.param_names == ["Q_a"] and mixed.fixed == {"Q_b": 0.8}
    assert float(mixed.log_likelihood(np.array([0.3]))) == pytest.approx(
        float(separate.log_likelihood(np.array([0.3, 0.8]))), rel=1e-12
    )
    with pytest.raises(ValueError, match="both as shared and as per_source"):
        build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shared=["Q"], per_source=["Q"])
    with pytest.raises(ValueError, match="unknown parameters"):
        build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shared=["charge"])
    with pytest.raises(ValueError, match="at least two datasets"):
        build_problem(rn, data[0], priors={"Q": Uniform(0.0, 1.0)}, per_source=["Q"])
    # A single shadow source keeps the earlier behaviour: no choice needed.
    assert build_problem(rn, data[0], priors={"Q": Uniform(0.0, 1.0)}).param_names == ["Q"]


def test_rotating_sources_have_their_own_spin_and_inclination(kerr_newman):
    truth = {"a": (0.3, 0.9), "inc": (60.0, 20.0), "Q": 0.4}
    data = [
        simulate_shadow_dataset(
            kerr_newman, name=name, sigma=0.05, inclination=inc, a=a, Q=truth["Q"]
        )
        for name, a, inc in zip(("one", "two"), truth["a"], truth["inc"], strict=True)
    ]
    problem = build_problem(
        kerr_newman,
        data,
        priors={"a": Uniform(0.0, 0.99), "Q": Uniform(0.0, 0.9)},
        shared=["Q"],
    )
    # The spin is per source without being asked; each source has its own inclination.
    assert problem.param_names == ["a_one", "a_two", "Q", "inclination_one", "inclination_two"]
    assert problem.kinds == ["metric", "metric", "metric", "nuisance", "nuisance"]
    assert problem.description["per_source"] == ["a"]
    assert problem.description["inclinations_deg"] == {
        "inclination_one": "sampled",
        "inclination_two": "sampled",
    }
    norm = -math.log(0.05 * math.sqrt(2 * math.pi))
    at_truth = np.array([0.3, 0.9, 0.4, 60.0, 20.0])
    assert float(problem.log_likelihood(at_truth)) == pytest.approx(2 * norm, rel=1e-9)
    swapped = np.array([0.9, 0.3, 0.4, 60.0, 20.0])
    sizes = [
        float(bp.light.shadow_radius(kerr_newman, inclination=math.radians(inc), a=a, Q=0.4))
        for a, inc in ((0.9, 60.0), (0.3, 20.0))
    ]
    centres = [d.gaussian()[0] for d in data]
    expected = 2 * norm - 0.5 * sum(
        ((s - c) / 0.05) ** 2 for s, c in zip(sizes, centres, strict=True)
    )
    assert float(problem.log_likelihood(swapped)) == pytest.approx(expected, rel=1e-8)
    # The black hole domain a^2 + Q^2 <= 1 is applied to every source.
    assert float(problem.log_likelihood(np.array([0.3, 0.95, 0.4, 60.0, 20.0]))) == -np.inf
    # Fixed inclinations per source; a shared spin must be requested explicitly.
    fixed = build_problem(
        kerr_newman,
        data,
        priors={"a": Uniform(0.0, 0.99), "Q": Uniform(0.0, 0.9)},
        shared=["Q", "a"],
        fixed={"inclination_one": 60.0, "inclination_two": 20.0},
    )
    assert fixed.param_names == ["a", "Q"]
    with pytest.raises(ValueError, match="unknown"):
        build_problem(
            kerr_newman, data, priors={"a": Uniform(0.0, 0.9)}, fixed={"inclination_x": 3.0}
        )


def test_mass_prior_choice_and_circular_inference(rn):
    data = ring()
    scale = 3.0 * math.sqrt(3.0)
    with warnings.catch_warnings():
        warnings.simplefilter("error", CircularInferenceWarning)
        default = build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)})
        gas = build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shadow_mass_prior="gas")
    assert default.data_table[0]["value"] == pytest.approx(scale * 0.98)
    assert gas.data_table[0]["value"] == pytest.approx(scale * 1.9)
    assert gas.description["shadow_mass_priors"] == {"ring": "gas"}
    assert default.description["shadow_mass_priors"] == {"ring": ["stars"]}
    # A mass obtained from the image itself is not independent: the fit says so.
    with pytest.warns(CircularInferenceWarning, match="not independent of the image"):
        image = build_problem(
            rn, data, priors={"Q": Uniform(0.0, 1.0)}, shadow_mass_prior={"ring": "image"}
        )
    assert image.description["circular_shadow_mass_priors"] == [["synthetic/ring", "image"]]
    with pytest.warns(CircularInferenceWarning):
        build_problem(rn, ring(assumed_mass_priors=["image"]), priors={"Q": Uniform(0.0, 1.0)})
    with pytest.warns(CircularInferenceWarning):  # the registry flag of the EHT mass of M87*
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", bp.errors.UnverifiedDataWarning)
            build_problem(
                rn, "eht/m87", priors={"Q": Uniform(0.0, 1.0)}, shadow_mass_prior="eht_ring"
            )
    with pytest.raises(RegistryError, match="no mass prior"):
        build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shadow_mass_prior="dust")
    with pytest.raises(ValueError, match="unknown shadow sources"):
        build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shadow_mass_prior={"x": "gas"})
    with pytest.raises(ValueError, match="interval"):
        build_problem(
            rn,
            data,
            priors={"Q": Uniform(0.0, 1.0)},
            shadow_mass_prior="gas",
            shadow_likelihood="interval",
            shadow_level="1sigma",
        )


def test_inclination_is_a_prior_only_on_request(kerr):
    data = ring()
    problem = build_problem(kerr, data, priors={"a": Uniform(0.0, 0.99)})
    index = problem.param_names.index("inclination")
    assert problem.priors[index].to_dict()["type"] == "Isotropic"  # the measurement is not used
    prior = inclination_prior(data)
    assert isinstance(prior, Gaussian) and (prior.mean, prior.sigma) == (17.0, 3.0)
    assert prior.support == (0.0, 90.0)
    explicit = build_problem(kerr, data, priors={"a": Uniform(0.0, 0.99), "inclination": prior})
    assert explicit.priors[explicit.param_names.index("inclination")] == prior
    with pytest.raises(BlackPyError, match="no measured inclination"):
        inclination_prior(bp.data.load("eht/sgra"))


def test_circularity_bound_needs_its_definition(kerr, rn):
    data = ring()  # bound 0.1, derived for 'fractional_rms'
    priors = {"a": Uniform(0.0, 0.999)}
    with warnings.catch_warnings():
        warnings.simplefilter("error", CircularityDefinitionWarning)
        matching = build_problem(kerr, data, priors=priors, circularity="fractional_rms")
    assert [row["kind"] for row in matching.data_table] == ["shadow", "circularity"]
    with pytest.warns(CircularityDefinitionWarning, match="fractional_rms"):
        mismatched = build_problem(kerr, data, priors=priors, circularity="bambi2019")
    # a = 0.99 seen at 80 degrees: fractional rms 0.04 (allowed), four times that excluded.
    theta = np.array([0.99, 80.0])
    assert np.isfinite(float(matching.log_likelihood(theta)))
    assert float(mismatched.log_likelihood(theta)) == -np.inf
    values = np.asarray(mismatched.predict(theta))
    assert values[1] == pytest.approx(
        float(
            bp.light.deviation_from_circularity(
                kerr, math.radians(80.0), a=0.99, definition="bambi2019"
            )
        ),
        rel=1e-6,
    )
    assert values[1] > 0.1 > np.asarray(matching.predict(theta))[1]
    # A bound is not a measurement: it has no pull and adds no degree of freedom.
    fit = Fit(kerr, data, priors=priors, circularity="fractional_rms")
    quality = fit.goodness_of_fit(a=0.5, inclination=40.0)
    assert quality.labels == ["r_sh[ring]"] and "upper bounds" in quality.notes[-1]
    with pytest.raises(ValueError, match="fractional_rms"):
        build_problem(kerr, data, priors=priors, circularity="eht")
    with pytest.raises(ValueError, match="is a circle"):
        build_problem(rn, data, priors={"Q": Uniform(0.0, 1.0)}, circularity="fractional_rms")
    with pytest.raises(ValueError, match="none of the shadow datasets"):
        build_problem(
            kerr,
            simulate_shadow_dataset(kerr, name="k", sigma=0.1, inclination=30.0, a=0.5),
            priors=priors,
            circularity="fractional_rms",
        )


def test_simulated_two_source_dataset_recovers_the_injected_parameters(rn):
    """Joint shadow fit of two sources (mandatory): shared and per-source charge."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)
        # (i) One charge-to-mass ratio per source: each is recovered from its own shadow.
        data = two_static_sources(rn, 0.3, 0.8, sigma=0.03)
        separate = Fit(
            rn, data, priors={"Q": Uniform(0.0, 1.0)}, per_source=["Q"], nsteps=3000
        ).run(seed=1)
        # (ii) The same charge for both: two data, one parameter, one degree of freedom.
        data = two_static_sources(rn, 0.6, 0.6, sigma=0.03)
        shared = Fit(rn, data, priors={"Q": Uniform(0.0, 1.0)}, shared=["Q"], nsteps=3000).run(
            seed=2
        )
    summary = separate.summary()
    for name, injected in (("Q_a", 0.3), ("Q_b", 0.8)):
        assert abs(summary[name]["median"] - injected) < 1.5 * summary[name]["std"], name
        assert summary[name]["std"] < 0.12 and not summary[name]["prior_dominated"]
    assert summary["Q_b"]["std"] < summary["Q_a"]["std"]  # the shadow is steeper at large Q
    assert separate.goodness_of_fit().dof == 0
    summary = shared.summary()["Q"]
    assert abs(summary["median"] - 0.6) < summary["std"] and summary["std"] < 0.03
    quality = shared.goodness_of_fit()
    assert quality.dof == 1 and quality.chi2 < 1e-6
    assert quality.point["Q"] == pytest.approx(0.6, abs=1e-4)
