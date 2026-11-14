# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

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
