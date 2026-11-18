"""The public API of BlackPy 1.x (design decision 38).

Public API = the names in ``__all__`` of ``blackpy`` and of its sub-packages and
modules listed below, plus ``blackpy.errors``. Within 1.x no public name may
disappear: a removal needs a deprecation period and a major release (see
``CONTRIBUTING.md``). Adding names is allowed; update the snapshot on purpose.
"""

import importlib
import re
import warnings
from pathlib import Path

import pytest

import blackpy

PUBLIC_API = {
    "blackpy": [
        "StaticSpherical",
        "StationaryAxisymmetric",
        "__version__",
        "constants",
        "data",
        "disk",
        "errors",
        "geometry",
        "infer",
        "lensing",
        "light",
        "metric",
        "orbits",
        "plot",
        "qnm",
        "thermo",
    ],
    "blackpy.constants": [
        "AU",
        "C",
        "G",
        "GM_SUN",
        "MICROARCSECOND",
        "M_SUN",
        "PARSEC",
        "R_SUN_GRAV",
        "T_SUN",
        "gravitational_radius_m",
        "hz_to_omega",
        "omega_to_hz",
    ],
    "blackpy.errors": [
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
    ],
    "blackpy.metric": [
        "Bardeen",
        "CATALOGUE",
        "GMGHS",
        "Hayward",
        "Johannsen",
        "KRZ",
        "Kerr",
        "KerrNewman",
        "KerrSen",
        "Metric",
        "Parameter",
        "ROTATING_CATALOGUE",
        "RZCoefficients",
        "ReissnerNordstrom",
        "RezzollaZhidenko",
        "RotatingBardeen",
        "RotatingHayward",
        "Schwarzschild",
        "SchwarzschildDeSitter",
        "SimpsonVisser",
        "StaticSpherical",
        "StationaryAxisymmetric",
        "SymbolicForm",
        "SymbolicFormAxisymmetric",
        "future",
        "interpolated",
        "krz_deviations",
        "rotating_from_static",
        "rz_coefficients",
    ],
    "blackpy.metric.future": [
        "newman_janis",
    ],
    "blackpy.metric.interpolated": [
        "spline",
    ],
    "blackpy.geometry": [
        "EnergyConditionResult",
        "Horizon",
        "HorizonProfile",
        "HorizonResult",
        "SourceResult",
        "StressEnergy",
        "classify_stress_energy",
        "cosmological_horizon",
        "effective_source",
        "einstein_tensor",
        "einstein_tensor_symbolic",
        "energy_conditions",
        "ergosurface",
        "event_horizon",
        "horizon_profile",
        "horizons",
        "is_vacuum_symbolic",
        "kretschmann",
        "kretschmann_symbolic",
        "report",
        "ricci_scalar",
        "ricci_scalar_symbolic",
        "stress_energy",
        "stress_energy_symbolic",
    ],
    "blackpy.orbits": [
        "Frequencies",
        "MarginallyStableOrbit",
        "QPOModel",
        "QPO_MODELS",
        "RPMSolution",
        "angular_velocity",
        "circular_orbit_exists",
        "circular_orbit_region",
        "frequencies_hz",
        "get_qpo_model",
        "isco",
        "marginally_bound_orbit",
        "marginally_stable_orbits",
        "nodal_precession_frequency",
        "photon_orbit",
        "radial_epicyclic_frequency",
        "radial_epicyclic_frequency_squared",
        "resonance_radius",
        "rpm_kerr_solution",
        "specific_angular_momentum",
        "specific_energy",
        "vertical_epicyclic_frequency",
        "vertical_epicyclic_frequency_squared",
    ],
    "blackpy.light": [
        "CriticalParameters",
        "DELTA_C_DEFINITIONS",
        "HomogeneousPlasma",
        "MetricWithPlasma",
        "PhotonSphere",
        "PlasmaModel",
        "PlasmaProfile",
        "PowerLawPlasma",
        "SeparablePlasma",
        "ShadowContour",
        "ShadowObservables",
        "critical_impact_parameter",
        "critical_impact_parameters",
        "deviation_from_circularity",
        "photon_region",
        "photon_sphere",
        "photon_sphere_in_plasma",
        "photon_spheres",
        "photon_spheres_in_plasma",
        "require_separable",
        "shadow_angular_radius",
        "shadow_angular_radius_in_plasma",
        "shadow_contour",
        "shadow_in_plasma",
        "shadow_observables",
        "shadow_radius",
        "with_plasma",
    ],
    "blackpy.lensing": [
        "StrongDeflection",
        "StrongFieldObservables",
        "closest_approach",
        "deflection_angle",
        "deflection_angle_at",
        "einstein_ring_angle",
        "strong_deflection_angle",
        "strong_deflection_limit",
        "strong_field_observables",
        "weak_deflection_angle",
    ],
    "blackpy.disk": [
        "DiskProfile",
        "inner_edge",
        "page_thorne_flux",
        "radiative_efficiency",
    ],
    "blackpy.data": [
        "CircularityBound",
        "Dataset",
        "MassPrior",
        "Measurement",
        "QPODataset",
        "Reference",
        "RegistryError",
        "ShadowDataset",
        "SpinMeasurement",
        "available",
        "list",
        "load",
        "load_file",
        "parse_dataset",
        "registry_path",
    ],
    "blackpy.infer": [
        "BayesFactor",
        "Fit",
        "FitResult",
        "Forecast",
        "ForecastWarning",
        "Gaussian",
        "GoodnessOfFit",
        "Isotropic",
        "LogUniform",
        "POOR_FIT_P_VALUE",
        "PosteriorPredictive",
        "Prior",
        "Problem",
        "SAMPLERS",
        "Uniform",
        "build_problem",
        "forecast",
        "forecast_from_problem",
        "inclination_prior",
        "prior_from_dict",
        "simulate_qpo_dataset",
        "simulate_shadow_dataset",
    ],
    "blackpy.plot": [
        "QUANTITIES",
        "corner",
        "deflection",
        "disk_flux",
        "forecast",
        "latex_table",
        "profile",
        "residuals",
        "shadow",
    ],
    "blackpy.qnm": [
        "quasinormal_modes",
    ],
    "blackpy.thermo": [
        "entropy",
        "hawking_temperature",
        "heat_capacity",
    ],
}

#: Deprecated names: still importable during 1.x, each with a DeprecationWarning that
#: names the replacement; not part of any ``__all__``.
DEPRECATED = {
    ("blackpy.light", "deflection_angle"): "blackpy.lensing.deflection_angle",
    ("blackpy.light.future", "deflection_angle"): "blackpy.lensing.deflection_angle",
    ("blackpy.light.future", "PlasmaModel"): "blackpy.light.PlasmaModel",
    ("blackpy.light.future", "shadow_in_plasma"): "blackpy.light.shadow_in_plasma",
    ("blackpy.metric.future", "rezzolla_zhidenko"): "blackpy.metric.RezzollaZhidenko",
    ("blackpy.metric.future", "konoplya_rezzolla_zhidenko"): "blackpy.metric.KRZ",
}

SRC = Path(blackpy.__file__).parent


@pytest.mark.parametrize("module", sorted(PUBLIC_API))
def test_no_public_name_disappears(module):
    mod = importlib.import_module(module)
    exported = set(mod.__all__)
    missing = sorted(set(PUBLIC_API[module]) - exported)
    assert not missing, (
        f"{module} no longer exports {missing}. Public names may only be removed after a "
        "deprecation period, in a major release."
    )
    for name in mod.__all__:
        assert hasattr(mod, name), f"{module}.__all__ lists '{name}', which does not exist"


def test_every_subpackage_defines_all():
    for module in PUBLIC_API:
        assert isinstance(importlib.import_module(module).__all__, list)
    # A new public module must be added to the snapshot.
    names = {p.stem for p in SRC.glob("*.py") if not p.stem.startswith("_")}
    names |= {p.name for p in SRC.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    assert {f"blackpy.{n}" for n in names} <= set(PUBLIC_API)


@pytest.mark.parametrize(("module", "name"), sorted(DEPRECATED))
def test_deprecated_aliases_still_work_and_warn(module, name):
    replacement = DEPRECATED[(module, name)]
    target_module, target_name = replacement.rsplit(".", 1)
    target = getattr(importlib.import_module(target_module), target_name)
    mod = importlib.import_module(module)
    with pytest.warns(DeprecationWarning, match=re.escape(replacement)):
        assert getattr(mod, name) is target
    with pytest.warns(DeprecationWarning, match=re.escape(replacement)):
        exec(f"from {module} import {name}", {})
    assert name not in mod.__all__


def test_star_imports_do_not_warn():
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        for module in PUBLIC_API:
            exec(f"from {module} import *", {})


def test_no_source_file_points_to_repository_files():
    # A pip install contains neither the design notes nor the verification reports, so
    # messages and docstrings must point to the public documentation instead.
    pattern = re.compile(r"DESIGN_NOTES|VERIFICATION|CLAUDE\.md|TO_DO\.md|BLACKPY_")
    offenders = [
        f"{path.relative_to(SRC)}:{number}"
        for path in SRC.rglob("*.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if pattern.search(line)
    ]
    assert not offenders, offenders


def test_stubs_point_to_the_roadmap_page():
    from blackpy import qnm, thermo

    for call in (
        lambda: qnm.quasinormal_modes(None),
        lambda: thermo.hawking_temperature(None),
        lambda: blackpy.metric.future.newman_janis(None),
    ):
        with pytest.raises(NotImplementedError, match="roadmap"):
            call()


def test_typing_marker_is_present():
    assert (SRC / "py.typed").exists()


def test_every_public_name_is_in_the_api_reference():
    text = (Path(__file__).parents[1] / "docs" / "api.md").read_text(encoding="utf-8")
    missing = [
        f"{module}.{name}"
        for module, names in PUBLIC_API.items()
        for name in names
        if not re.search(rf"\b{re.escape(name)}\b", text)
    ]
    assert not missing, f"Add these public names to docs/api.md: {missing}"
