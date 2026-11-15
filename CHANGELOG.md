# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

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
