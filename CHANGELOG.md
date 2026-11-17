# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [0.3.0] - 2026-11-17

Fit quality, plasma, a second shadow source, parametrised metrics, forecasts,
lensing, thin disks. The v0.2 API is unchanged except where noted under
"Changed".

### Added
- **Fit quality** (`blackpy.infer`): `FitResult.goodness_of_fit()` (pulls,
  chi-square, degrees of freedom = data + Gaussian priors - parameters;
  "undefined" when there are none), `FitResult.posterior_predictive(n, seed)`
  (Gelman, Meng & Stern 1996), `PoorFitWarning`, fit quality in
  `print(result)` and in Bayes factors, `Fit.goodness_of_fit(**theta)`,
  `simulate_shadow_dataset`, `blackpy.plot.residuals`.
- **Plasma** (`blackpy.light`): `HomogeneousPlasma`, `PowerLawPlasma`,
  `PlasmaProfile`, `SeparablePlasma`, `with_plasma`, `shadow_in_plasma`,
  `photon_spheres_in_plasma`; rotating metrics for the separable class of
  Perlick & Tsupko (2017) only; `PlasmaPropagationError`;
  `Fit(..., plasma=...)`.
- **Second shadow source**: registry entry `eht/m87` (ring diameter,
  angular gravitational radius, fractional deviation, two dynamical masses
  and the image-based one, distance, jet inclination, circularity bound with
  its definition); Sgr A\* masses and distances (`eht/sgra` version 2).
  Joint shadow fits with `shared=[...]` / `per_source=[...]`, per-source spin
  and inclination, `shadow_mass_prior`, `circularity="<definition>"`,
  `inclination_prior`.
- **Parametrised metrics**: `RezzollaZhidenko(order, ppn)`,
  `rz_coefficients` (RZ coefficients of any static metric), `KRZ` with
  deviation parameters from Kerr, `krz_deviations`.
- **Forecasts**: `blackpy.infer.forecast` (Fisher matrix by autodiff,
  degenerate directions, `ForecastWarning`).
- **Lensing** (`blackpy.lensing`, static metrics): exact deflection angle,
  Gauss–Bonnet weak deflection, strong deflection limit and observables,
  Einstein ring, deflection in a plasma. Predictions only.
- **Thin disks** (`blackpy.disk`): radiative efficiency, Page–Thorne flux,
  temperature profile, radius of maximum flux.
- **Catalogue**: GMGHS, Simpson–Visser (static); Kerr–Sen, rotating Bardeen,
  rotating Hayward (rotating).
- `blackpy.light.deviation_from_circularity` with three named definitions
  (`fractional_rms`, `jp_asymmetry`, `bambi2019`),
  `CircularityDefinitionWarning`; `blackpy.plot.shadow`.
- Registry schema: `other_values` / `preferred_reason` on any measurement,
  `open_questions` on any dataset, shadow-dataset fields listed above
  (`schema_version` unchanged: no field changed meaning).
- Constants `AU`, `PARSEC`, `MICROARCSECOND`.
- `CONTRIBUTING.md`; examples 09 to 15; `pytest-xdist` in the dev extra.

### Changed
- `qpo/gro_j1655_40` (dataset version 3): the main spin, mass and radius are
  those of Table 3 of Motta et al. (2014); the text values are kept under
  `other_values`. `qpo/xte_j1859_226` (version 3): open question on the
  published uncertainties.
- Joint fits of two or more shadow datasets require an explicit
  `shared` / `per_source` choice for every sampled metric parameter.
- `shadow_contour(...).r_p` of a non-rotating metric holds the photon-sphere
  radius (0.2 returned a placeholder 3.0).
- `blackpy.thermo` and `blackpy.qnm` stay stubs; their messages now say the
  features are not scheduled (no data would use them).
- The reserved names `light.PlasmaModel`, `light.shadow_in_plasma`,
  `light.deflection_angle`, `metric.future.rezzolla_zhidenko` and
  `metric.future.konoplya_rezzolla_zhidenko` now refer to the
  implementations.
- CI: quick subset on Linux, macOS and Windows; full suite on Linux.

### Documented
- The Hawking–Ellis classification was checked against Martín-Moruno &
  Visser (2018); no change was needed.
- Two errors in Konoplya, Rezzolla & Zhidenko (2016) (theory section 21.2).

## [0.2.0] - 2026-11-15

Rotating (stationary, axisymmetric) black holes. The v0.1 API is unchanged.

### Added
- `blackpy.metric.StationaryAxisymmetric`: metrics
  `g_tt, g_tphi, g_phiphi, g_rr, g_thth` of `(r, theta)` from sympy
  expressions or JAX callables; `is_reflection_symmetric`, `is_separable`,
  declared black hole domain. `StaticSpherical.as_axisymmetric()`.
- Rotating catalogue (`ROTATING_CATALOGUE`): Kerr, Kerr-Newman, Johannsen
  (2013). `rotating_from_static` (Azreg-Ainou 2014, no complexification).
- `blackpy.geometry`: horizons of rotating metrics with a theta-independence
  report (`horizon_profile`), `ergosurface`, numerical 4D curvature
  (`kretschmann`, `ricci_scalar`, `einstein_tensor` on an `(r, theta)`
  grid), `effective_source` (vacuum / perfect fluid / anisotropic fluid /
  neither, with energy conditions), full 4D symbolic tensors with a time
  limit, `degenerate_atol` option for near-extremal horizons.
- `blackpy.orbits`: prograde and retrograde equatorial orbits, radial and
  vertical epicyclic frequencies, ISCO, marginally bound and photon orbits
  for any reflection-symmetric rotating metric; nodal precession; all eleven
  QPO models with rotation; `rpm_kerr_solution` (Ingram & Motta 2014).
- `blackpy.light`: `critical_impact_parameters`, `photon_region`,
  `shadow_contour` (observer at infinity or ZAMO at finite distance),
  `shadow_observables` (mean radius, area-equivalent radius, deviation from
  circularity, centroid, extents) for metrics with separable null geodesics.
- `blackpy.infer`: relativistic-precession-model triplet likelihood (nodal
  QPO), spin bounded by the metric's black hole domain, shadow likelihood
  for rotating metrics with the inclination as a nuisance parameter
  (`Isotropic` prior), `SpinIgnoredWarning`, `DegeneracyWarning` with the
  spin-deformation posterior correlation, `FitResult.correlation`.
- Registry: published RPM spin, mass and radius for GRO J1655-40 and
  XTE J1859+226 (`spin`, `published_fit`; dataset version 2).
- Examples 05 (Johannsen fit to the GRO J1655-40 triplet), 06 (published RPM
  results), 07 (rotating shadows, Sgr A*), 08 (tutorial).
- Documentation: `docs/how_it_works.md`, `docs/tutorial.md`, rotating-metric
  theory in LaTeX, `VERIFICATION_v0.2.md`.

### Changed
- Constants are tested against `scipy.constants` (CODATA 2022; `G` is
  unchanged from CODATA 2018). No value changed.
- The Bardeen and Hayward critical values are derived exactly and tested to
  1e-10 (was "within 1%").
- A slow test checks the frequentist coverage of the 68% credible interval
  on 100 noisy datasets.
- `nodal_precession_frequency` and `QPOModel.low_frequency` work for
  rotating metrics; the explicit refusal remains for static metrics.
- `blackpy.metric.future`: the `StationaryAxisymmetric` and `johannsen`
  stubs were replaced by implementations; `newman_janis` still raises and
  points to `rotating_from_static`.

### Known limitations
- No registry value has been verified by a human yet.
- Shadows need separable null geodesics (no ray tracing); the horizon of a
  metric whose `g^rr = 0` surface depends on theta is not computed.

## [0.1.0] - 2026-11-14

First working version (MVP): static, spherically symmetric metrics.

### Added
- `blackpy.metric`: `StaticSpherical` (sympy and callable constructors, named
  parameters with defaults and bounds), spline interpolant for tabulated
  metrics, catalogue (Schwarzschild, Reissner-Nordstrom, Schwarzschild-de
  Sitter, Bardeen, Hayward), stubs for rotating and parametrised metrics.
- `blackpy.geometry`: horizons with classification, Kretschmann and Ricci
  scalars, effective stress-energy (symbolic and numeric), energy conditions,
  one-call report.
- `blackpy.orbits`: circular-orbit quantities, existence regions, ISCO and
  OSCO, marginally bound orbit, epicyclic frequencies, conversion to Hz,
  eleven QPO models, resonance radii.
- `blackpy.light`: photon spheres with stability, critical impact parameter,
  shadow for observers at infinity and at finite radius; plasma and lensing
  stubs.
- `blackpy.data`: registry with seven datasets (six QPO sources, Sgr A*
  shadow bounds), schema validation, typed loader.
- `blackpy.infer`: priors, QPO and shadow likelihoods, emcee and dynesty
  drivers, Bayes factors, integrity warnings, reproducibility bundles.
- `blackpy.plot`: profile plots, corner plots, LaTeX tables.
- Documentation (MkDocs), four examples, GitHub Actions CI, 117 tests.

### Known limitations
- No registry value has been verified by a human yet.
- No rotation; low-frequency QPOs cannot be modelled.
