# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## [Unreleased]

### Changed
- `UnverifiedDataWarning` is now also emitted outside fits whenever registry
  values are used: `blackpy.lensing.strong_field_observables(dataset=...)`,
  `blackpy.plot.shadow(band=...)` and `blackpy.infer.inclination_prior`
  (helper `blackpy.data.registry.warn_if_unverified`).
- All eight registry datasets have been checked by the maintainer against their
  papers (journal versions where available, arXiv otherwise) and carry
  `verified_by_human: true`; using them no longer emits `UnverifiedDataWarning`.
  Open points stay in each dataset's `todo` and `open_questions`.

### Fixed
- The tests accept registry files whose `verified_by_human` flag the owner has
  set to `true`; documentation snippets may list `UnverifiedDataWarning` and no
  longer emit it once their datasets are verified. The snippets no longer list
  it.

## [1.0.0rc1] - 2026-11-19

Release candidate of 1.0: a stabilisation release with no new physics. The public
API is frozen and documented, the package builds, installs from a wheel and passes
its tests against the declared minimum dependency versions, and a user guide shows
every calculation and plot with tested outputs.

### Compatibility

From 1.0 on BlackPy follows semantic versioning. The **public API** is the set of
names in `__all__` of `blackpy`, of its sub-packages and public modules, and
`blackpy.errors` (listed in `docs/api.md`). Within 1.x no public name is removed and
no documented behaviour changes without a deprecation period of at least one minor
release, during which the old name emits a `DeprecationWarning` naming its
replacement; removals happen only in 2.0. Result bundles written by 0.x load in 1.x
(format 0) and bundles carry a format number from now on. Numerical results may
change in the last digits between releases (dependency updates, round-off) but not
beyond the tolerances of the test-suite. Registry values are not part of this
promise: they change, with a new `dataset_version`, whenever a human check finds an
error.

The only breaking rename, `inclination` → `inclination_deg`, keeps the old name
working throughout 1.x (see "Deprecated").

### Added
- `scripts/check_windows.ps1`: one command that installs BlackPy and runs the
  linter, the tests, the documentation build and optionally the examples on
  Windows, with a short `SUMMARY.txt` (`-ExamplesQuick` runs the examples in quick
  mode).
- Public-API snapshot test (`tests/test_public_api.py`) and a written deprecation
  policy (`CONTRIBUTING.md`): within 1.x no public name disappears without a
  deprecation period.
- `__all__` for `blackpy.constants`, `blackpy.qnm`, `blackpy.thermo`,
  `blackpy.metric.interpolated` and `blackpy.errors`.
- `src/blackpy/py.typed` (PEP 561 typing marker).
- Result bundles carry `"bundle_format": 1`; `FitResult.load` reads pre-1.0 bundles
  (format 0) and refuses newer formats with the BlackPy version that is needed.
- Public inclination functions in radians (`shadow_contour`, `shadow_observables`,
  `shadow_radius`, `deviation_from_circularity`, `photon_region`, `shadow_in_plasma`,
  `plot.shadow`) refuse values outside [0, π] with a message that suggests
  `math.radians(...)`.
- `docs/roadmap.md`; `docs/api.md` lists every public name.
- `docs/user_guide.md`: every calculation and plot with real outputs, the unit table,
  the integrity guards and errors, an FAQ and an index of the examples; its snippets,
  those of the README, the quickstart and the data page are run by `tests/test_docs.py`.
- Plot helpers `blackpy.plot.deflection` (exact, weak and strong deflection angles),
  `blackpy.plot.disk_flux` (Page–Thorne flux or temperature profiles) and
  `blackpy.plot.forecast` (Fisher-forecast ellipses, alone or on a corner plot);
  examples 13 to 15 use them (example 13 writes `forecast_vs_fit.png`).
- Packaging: wheel and minimum-dependency CI jobs, weekly examples job, GitHub Pages
  and PyPI release workflows (trusted publishing); the full suite moved from `ci.yml`
  to its own workflow `full.yml`, run on manual dispatch (it timed out after 60 min
  as a CI job); `build` and `twine` in the `dev`
  extra; examples honour `BLACKPY_EXAMPLES_QUICK=1`; `check_windows.ps1 -ExamplesQuick`.
- `scripts/registry_checklist.py` lists every registry number with its paper and
  location, for human verification.

### Changed
- **The shadow inclination nuisance parameter is renamed `inclination` →
  `inclination_deg`** (per source: `inclination_<source>` → `inclination_deg_<source>`),
  following the rule "every angle in the public API is in radians; a quantity in
  degrees has `_deg` in its name". Units (degrees) and the default prior
  `Isotropic(0, 90)` are unchanged. `simulate_shadow_dataset(inclination=...)` becomes
  `inclination_deg=...`. The old names keep working during 1.x (see "Deprecated").
- `ForecastWarning` and `RegistryError` are defined in `blackpy.errors` (still importable
  from `blackpy.infer`, `blackpy.infer.forecast` and `blackpy.data`); `RegistryError` is
  now also a `BlackPyError` (it remains a `ValueError`).
- Error messages that did not say what to do next were rewritten; messages no longer
  refer to repository files that a `pip install` does not contain, but to the
  documentation (roadmap page).
- The M87\* registry entry states `unit: deg` on its alternative inclination value
  (no number changed).
- Documentation reorganised for 1.0: `docs/quickstart.md` is "BlackPy in five
  minutes"; the README and the home page start from the user guide; "What version
  0.3 covers" became "What BlackPy covers" and the list of missing features moved to
  `docs/roadmap.md`; the release history of the registry moved from `docs/data.md`
  to this changelog, and the data page has a step-by-step guide for your own data;
  `docs/theory.md` has a topic index (section numbers unchanged; section 25 is now
  "Further catalogue metrics").
- The verification reports moved to `verification/` (`v0.1.md` to `v1.0.md`).
- Trove classifier `Development Status :: 4 - Beta`; Python 3.10 to 3.13 listed.

### Deprecated
- The parameter names `inclination` and `inclination_<source>` (in `priors`, `fixed`,
  `shared`/`per_source`, fiducial points, `Fit.log_likelihood` / `goodness_of_fit`
  keywords, `FitResult[...]`) and the keyword `simulate_shadow_dataset(inclination=...)`:
  use `inclination_deg` / `inclination_deg_<source>`. Giving both spellings is an error.
- `blackpy.light.deflection_angle` and `blackpy.light.future.deflection_angle`: use
  `blackpy.lensing.deflection_angle`.
- `blackpy.light.future.PlasmaModel` and `blackpy.light.future.shadow_in_plasma`: use
  `blackpy.light.PlasmaModel` and `blackpy.light.shadow_in_plasma`.
- `blackpy.metric.future.rezzolla_zhidenko` and
  `blackpy.metric.future.konoplya_rezzolla_zhidenko`: use `blackpy.metric.RezzollaZhidenko`
  and `blackpy.metric.KRZ`.

All deprecated names emit a `DeprecationWarning` naming the replacement and will be
removed in 2.0.

### Fixed
- Dataset hashes (`Dataset.sha256`, recorded in fit results) no longer depend on
  line endings: a registry file checked out with CRLF on Windows has the same hash
  as on Linux and macOS (`blackpy.data.registry.file_hash`). Hashes of files with
  LF line endings are unchanged. `.gitattributes` keeps text files in LF.
- LaTeX labels of per-source parameters escape underscores in object names
  (`M_{\rm src\_a}`), so corner plots of sources such as `src_a` render correctly.
- Three tests no longer depend on details of NumPy >= 2.0 (`np.trapezoid`), SciPy >=
  1.15 (CODATA 2022 table) or on round-off at the exact photon-orbit radius r = 3 M,
  where the sign of the existence condition differs between JAX versions; they now
  pass with the declared minimum versions.

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

### Registry (moved from `docs/data.md` in 1.0)

Read by Claude on 2026-11-15 to 2026-11-17 from the arXiv versions (EHT
M87\* Papers I and VI from the arXiv PDFs). **None has been verified by a
human.**

| Dataset | Change | Source and location |
|---|---|---|
| `qpo/gro_j1655_40` (version 3) | Table 3 values of spin 0.286 ± 0.003, mass 5.307 ± 0.066, radius 5.677 ± 0.035 are the main entries; the text values (0.290, 5.31, 5.68) are kept under `other_values` | Motta et al. (2014), Table 3; abstract, Sec. 4.1 and 5 |
| `qpo/xte_j1859_226` (version 3) | open question: published RPM uncertainties 2.7 times larger than propagation gives | Motta et al. (2022) |
| `eht/sgra` (version 2) | mass priors `keck` (3.964 ± 0.054 × 10⁶ M☉) and `vlti` (4.297 ± 0.042 × 10⁶ M☉) with \(\theta_g\) and \(\delta\) per prior; distance 8.277 ± 0.034 kpc | Do et al. (2019); GRAVITY Collaboration (2022); EHT Sgr A\* Paper VI, Sec. 2.1; Vagnozzi et al. (2023) |
| `eht/m87` (new) | 68% bound 4.31–6.08 on \(r_{\rm sh}/M\); \(\delta = -0.01 \pm 0.17\) (stellar dynamics); ring diameter 42 ± 3 µas; \(\theta_g = 3.8 \pm 0.4\) µas; masses from stellar dynamics (independent), gas dynamics (independent, in tension) and the EHT ring (**not** independent); distance 16.8 ± 0.8 Mpc; jet inclination 17.2 ± 3.3°; circularity < 0.10 (`fractional_rms`) | EHT (2019) Papers I and VI; Kocherlakota et al. (2021), Eq. (14); Gebhardt et al. (2011); Walsh et al. (2013); Mertens et al. (2016) |

The M87\* entry records the chain from the measured ring to a bound on the
shadow size (theory section 20). A fit uses the stellar-dynamics bound by
default; `shadow_mass_prior=` selects another, and the image-based mass
triggers a `CircularInferenceWarning`.

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
  theory in LaTeX, `verification/v0.2.md`.

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

### Registry (moved from `docs/data.md` in 1.0)

All values below were read from the arXiv versions of the papers (ar5iv HTML
rendering) on 2026-11-14; bibliographic details were checked against Crossref.
**None has been verified by a human yet.**

Added in version 0.2 (read on 2026-11-15, dataset version 2), for comparison
with BlackPy's own fits (example 6); these are Kerr QPO fits and **not**
independent measurements:

| Dataset | Published RPM result | Source and location |
|---|---|---|
| `qpo/gro_j1655_40` | \(a = 0.290 \pm 0.003\), \(M = 5.31 \pm 0.07\,M_\odot\), \(R = 5.68 \pm 0.04\,R_g\) | Motta et al. (2014), abstract and Sec. 4.1; their Table 3 gives the spin as 0.286 ± 0.003 |
| `qpo/xte_j1859_226` | \(a = 0.149 \pm 0.005\), \(M = 7.85 \pm 0.46\,M_\odot\), \(R = 6.85 \pm 0.18\,R_g\) | Motta et al. (2022), abstract, Sec. 3 and Table 3 |

| Dataset | Values | Source and location | Mass prior |
|---|---|---|---|
| `qpo/gro_j1655_40` | 441 ± 2, 298 ± 4, 17.3 ± 0.1 Hz | Motta et al. (2014), Table 2, Sample B1 | 5.40 ± 0.30 M☉, Beer & Podsiadlowski (2002), Table 4 (dynamical) |
| `qpo/xte_j1859_226` | 227.5 (+2.1/−2.4), 128.6 (+1.6/−1.8), 3.65 ± 0.01 Hz | Motta et al. (2022), Table 2, triplet | 7.8 ± 1.9 M☉, Yanes-Rizo et al. (2022), abstract (dynamical) |
| `qpo/xte_j1550_564` | 268 ± 3, 188 ± 3 Hz | Miller et al. (2001), Fig. 4 caption | 9.10 ± 0.61 M☉, Orosz et al. (2011), abstract and Sec. IV.1 (dynamical) |
| `qpo/grs_1915_105` | 69.2 ± 0.15, 41.5 ± 0.4 Hz | Strohmayer (2001), Sec. 2 | 12.4 (+2.0/−1.8) M☉, Reid et al. (2014), abstract |
| `qpo/h1743_322` | 241 ± 3, 165 (+9/−5) Hz | Homan et al. (2005), Sec. 3 | **null** (no dynamical mass found) |
| `qpo/m82_x1` | 5.07 ± 0.06, 3.32 ± 0.06 Hz | Pasham et al. (2014), abstract | 415 ± 63 M☉, same paper: **not independent** (RPM fit) |
| `eht/sgra` | 4.55–5.22 (1σ), 4.21–5.56 (2σ) for \(r_{\rm sh}/M\) | Vagnozzi et al. (2023), Eqs. (5)–(8) | – |

#### Open issues

- **XTE J1550-564**: the pair 276 ± 3 / 184 ± 5 Hz used in many QPO papers was
  not found in the original papers accessed (Remillard et al. 2002 give only
  harmonic fits and "near 184 and 276 Hz"). The registry uses the
  independently fitted pair of Miller et al. (2001).
- **GRS 1915+105**: the 113/168 Hz pair is listed under `alternatives` without
  uncertainties (only a review was accessible); the default pair is the
  41/67 Hz detection of Strohmayer (2001).
- **H1743-322**: no mass prior. A fit needs an explicit user prior.
- **M82 X-1**: the only mass estimates come from the QPOs themselves; using
  them is circular and triggers a warning.
- **Sgr A\***: the paper gives lower limits 4.55 and 4.21, whereas 4.54 and
  4.20 are sometimes quoted.

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
