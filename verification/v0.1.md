# Verification report: BlackPy 0.1.0

Date: 2026-11-14. Machine: Apple M2, macOS 27.2 (arm64), Python 3.13.5,
JAX 0.12.0.1, NumPy 2.5.4, SciPy 1.18.1, sympy 1.14.0, emcee 3.1.6,
dynesty 3.2.0.

## 1. Summary

| Check | Result |
|---|---|
| Full test-suite (`pytest`, including slow tests) | **117 passed, 0 failed**, 101 s |
| `ruff check` and `ruff format --check` on `src`, `tests`, `examples` | passed |
| `mkdocs build --strict` | passed |
| Examples 01 to 04 | all ran to completion |
| ISCO performance target (< 0.01 ms per evaluation) | 0.0058 ms for Reissner-Nordstrom; **not met for Bardeen** (0.016 ms) |
| Registry values verified by a human | **none** (all `verified_by_human: false`) |

Everything reported here was run in the project's `.venv` on this machine.
CI (`.github/workflows/ci.yml`) has **not** been run: nothing was pushed, so
Linux and Windows are untested.

## 2. Tests

All 89 test functions (117 cases with parametrisation) passed. Tolerances
are relative unless stated otherwise. "Item n" refers to the validation list
of the specification.

### Constants and units (`tests/test_constants.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_x64_enabled` | JAX runs in float64 | exact | pass |
| `test_constant_values` | c, G, GM_sun as documented; M_sun = 1.98841e30 kg | exact; 1e-5 | pass |
| `test_omega_hz_round_trip` | Hz conversion and its inverse | 1e-15 | pass |
| `test_schwarzschild_isco_frequency_10_msun` (item 6) | nu_phi at ISCO for 10 M_sun equals 6^(-3/2) c^3 / (2 pi GM_sun 10) = 219.86 Hz, computed from the constants | 1e-12 | pass |
| `test_gravitational_radius` | GM_sun/c^2 | 1e-15 | pass |

### Numerical toolbox (`tests/test_numerics.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_derivative_orders` | third autodiff derivative of sin(x) x^3 | 1e-13 | pass |
| `test_refine_root_simple_and_invalid_bracket` | cube root of 2; NaN for an invalid bracket; root at an end point | 1e-15 | pass |
| `test_find_all_roots_of_polynomial` | all five roots of a quintic | 1e-13 | pass |
| `test_close_pair_inside_one_scan_cell` | two roots 1e-3 apart, invisible to a plain scan | 1e-12 | pass |
| `test_pair_closer_than_tolerance_is_reported_as_degenerate` | roots 1e-6 apart merge into one double root | 1e-9 | pass |
| `test_degenerate_root` (2 cases) | double root touching zero from above and below | 1e-12 | pass |
| `test_no_roots_and_truncation_keeps_outermost` | no roots gives NaN; outermost roots are kept | 1e-13 | pass |
| `test_find_roots_is_jit_and_vmap_compatible` | compiled, vectorised solve | 1e-14 | pass |
| `test_gauss_legendre` | integral of sin; exactness for x^5 with 3 nodes | 1e-14 | pass |

### Metric (`tests/test_metric.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_callable_constructor_infers_parameters` | names and defaults from the signature; clear errors | exact | pass |
| `test_parameter_bounds_are_enforced` | out-of-range values raise | – | pass |
| `test_from_sympy_general_f_g_h` | independent f, g, h from sympy; LaTeX output | 1e-8 | pass |
| `test_symbolic_unavailable_for_callable_metric` | explicit error for symbolic requests on numeric metrics | – | pass |
| `test_broadcasting_over_r_and_parameters` | array broadcasting | 1e-15 | pass |
| `test_autodiff_matches_symbolic_derivatives` (item 4; 5 metrics) | f', f'', f''' from autodiff vs sympy (30-digit evaluation) on 40 radii | **1e-12** | pass |
| `test_catalogue_entries_have_reference_and_flat_limit` (5 metrics) | reference present; f reduces to 1 - 2/r | 1e-15 | pass |
| `test_asymptotic_flatness` | flat / de Sitter / deficit-angle cases | – | pass |
| `test_interpolated_metric_matches_closed_form` | metric given only as a table (quintic spline): r_h, r_ph, r_sh, r_ISCO | 1e-9, 1e-8, 1e-8, 1e-6 | pass |
| `test_future_metric_stubs_raise` | rotating metrics raise NotImplementedError | – | pass |

### Geometry (`tests/test_geometry.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_schwarzschild_horizon` (item 1) | r_h = 2 | 1e-14 | pass |
| `test_rn_two_horizons` (items 2, 9; Q = 0.1 to 0.999999) | r_pm = 1 +- sqrt(1 - Q^2), both classified | 1e-13 / sqrt(1 - Q^2) | pass |
| `test_rn_extremal_and_horizonless` (item 9) | double root at Q = 1; Q = 1.1 reported as horizonless | 1e-9 | pass |
| `test_event_horizon_vectorised` | array of charges incl. naked cases (NaN) | 1e-9 | pass |
| `test_sds_event_and_cosmological_horizon` | both horizons; outermost is cosmological | residual < 1e-13 | pass |
| `test_pure_de_sitter_is_horizonless_with_cosmological_horizon` | r_c = sqrt(3/Lambda) | 1e-13 | pass |
| `test_root_of_g_without_root_of_f_is_not_a_horizon` | wormhole-like throat | – | pass |
| `test_regular_black_holes_lose_horizon_above_critical_value` (2) | Bardeen, Hayward critical value 4/(3 sqrt 3) | within 1% | pass |
| `test_closed_form_curvature_matches_full_tensor_computation` | closed forms for G^mu_nu, R, K equal the full tensor computation for arbitrary f, g, h | symbolic identity | pass |
| `test_schwarzschild_is_vacuum` (item 5) | T = 0 symbolically; numerically < 1e-14 / r^2; K = 48/r^6 | exact; 1e-12 | pass |
| `test_rn_stress_energy_is_coulomb_field` (item 5) | rho = -p_r = p_t = Q^2/(8 pi r^4); K of RN | symbolic identity; 1e-10 | pass |
| `test_numeric_stress_energy_for_callable_metric_matches_symbolic` | autodiff stress-energy vs sympy for Hayward | 1e-9 | pass |
| `test_energy_conditions` | Schwarzschild/RN all hold; Lambda > 0 violates only SEC; Hayward DEC violated exactly where r^3 > 4 l^2 | – | pass |
| `test_phantom_like_metric_violates_nec` | negative-energy source detected | – | pass |
| `test_report_runs_for_symbolic_and_numeric` | report text; symbolic checks skipped with a message | – | pass |

### Orbits (`tests/test_orbits.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_schwarzschild_special_orbits` (item 1) | r_ISCO = 6, r_mb = 4, orbits exist from r = 3 | 1e-13 | pass |
| `test_schwarzschild_orbit_quantities` | Omega, E, L closed forms; NaN inside the photon sphere | 1e-14 | pass |
| `test_schwarzschild_epicyclic_relations` (item 1) | nu_theta = nu_phi (exact); nu_r^2 = nu_phi^2 (1 - 6/r) | exact; 1e-12 | pass |
| `test_rn_isco` (item 2) | r_ISCO(Q = 1) = 4; cubic r^3 - 6r^2 + 9Q^2 r - 4Q^4 = 0 | 1e-12; residual 1e-11 | pass |
| `test_rn_epicyclic_frequency_against_independent_formula` | Omega_r^2 vs effective-potential derivation done with sympy | 1e-11 | pass |
| `test_general_f_g_h_metric_against_isotropic_schwarzschild` | f != g, h != r^2: ISCO, r_ph, b_c, r_mb, Omega, Omega_r | 1e-12 (Omega_r: 1e-10) | pass |
| `test_schwarzschild_limit_of_catalogue` (item 3; 4 metrics) | ISCO, r_ph, b_c, r_h reduce to Schwarzschild | **absolute 1e-10** | pass |
| `test_sds_isco_osco_and_static_radius` | ISCO + OSCO; static radius (3/Lambda)^(1/3); no stable orbits above Lambda = 4/5625; r_mb refused | 1e-12 | pass |
| `test_naked_rn_singularity_orbits` | two orbit regions for 1 < Q^2 < 9/8 | 1e-12 | pass |
| `test_frequencies_in_hz` | Hz values and 1/M scaling | 1e-12 | pass |
| `test_qpo_model_definitions` | all 11 models equal Table 1 of Stuchlik & Kolos (2016) | exact | pass |
| `test_resonance_radii_schwarzschild` | 3:2 radii 27/4 (RP) and 54/5 (ER) | 1e-12 | pass |
| `test_nodal_precession_is_refused_explicitly` | explicit error in spherical symmetry | – | pass |
| `test_isco_performance` (item 8, slow) | 10^4 ISCO values, vectorised; budget 0.05 ms per evaluation | measured 0.0058 ms | pass |

### Light (`tests/test_light.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_schwarzschild_photon_sphere_and_shadow` (item 1) | r_ph = 3, b_c = r_sh = 3 sqrt 3 | 1e-14 | pass |
| `test_rn_photon_sphere` (item 2) | r_ph = (3 + sqrt(9 - 8Q^2))/2 and b_c for 21 charges | 1e-13 | pass |
| `test_naked_rn_has_stable_and_unstable_photon_sphere` | two photon spheres with stability; none for Q^2 > 9/8 | 1e-12 | pass |
| `test_static_observer_at_finite_radius` | Synge's formula; pi/2 at the photon sphere; inside it; far limit | 1e-12 (limit: 1e-6) | pass |
| `test_sds_shadow_requires_finite_static_observer` | b_c = 3 sqrt 3 / sqrt(1 - 9 Lambda); refusal at infinity; finite observer; refusal beyond the cosmological horizon | 1e-12 | pass |
| `test_shadow_shrinks_with_charge` | monotonic; extremal RN r_sh = 4 | 1e-12 | pass |
| `test_future_light_stubs_raise` | lensing, plasma, QNM, thermodynamics stubs | – | pass |

### Data (`tests/test_data.py`)

No test depends on the numerical value of a registry entry.

| Test | What is checked | Result |
|---|---|---|
| `test_registry_contains_initial_targets` | the seven datasets exist | pass |
| `test_every_dataset_loads_and_documents_its_sources` (7) | schema; `verified_by_human` is false; every stored number has a source and location; every null has a TODO | pass |
| `test_typed_objects_and_ordering` | typed objects; upper > lower | pass |
| `test_circular_mass_prior_is_flagged_in_registry` | M82 X-1 not independent; H1743-322 null | pass |
| `test_shadow_dataset_gaussian_is_consistent_with_bounds` | intervals follow from delta and its error (0.006) | pass |
| `test_list_prints_verification_status` | listing shows status and TODOs | pass |
| `test_unknown_dataset_and_schema_violations` | nine malformed files rejected with clear messages | pass |
| `test_measurement_helpers` | formatting; error on incomplete values | pass |

### Inference and plots (`tests/test_infer.py`, `tests/test_plot.py`)

| Test | What is checked | Tolerance | Result |
|---|---|---|---|
| `test_prior_consistency` (5 priors) | normalisation, cdf(ppf(u)) = u, support, round trip | 1e-8, 1e-10 | pass |
| `test_invalid_priors_are_rejected` | bad arguments raise | – | pass |
| `test_likelihood_peaks_at_truth_and_matches_hand_computation` | log L at truth and one sigma away, by hand; -inf inside the ISCO | 1e-9 | pass |
| `test_asymmetric_errors_use_split_normal` | split-normal likelihood, by hand | 1e-10 | pass |
| `test_shadow_likelihoods` | Gaussian and interval forms; extremal RN excluded; SdS refused | 1e-10 | pass |
| `test_problem_construction_errors` | seven misuse cases give clear errors | – | pass |
| `test_integrity_guards_on_registry_data` | circular-prior, ignored-QPO and unverified-data warnings; null mass is an error with a remedy | – | pass |
| `test_prior_dominated_parameter_is_flagged` | unconstrained parameter warned and marked in table | – | pass |
| `test_unconverged_chain_warns` | 60-step chain triggers the convergence warning | – | pass |
| `test_samplers_recover_true_charge` (item 7; emcee, dynesty) | all five true values within 2 sigma; sigma(Q) < 0.08; converged | 2 sigma | pass |
| `test_samplers_agree` | emcee vs dynesty means and widths | 0.5 sigma; ratio 0.7 to 1.4 | pass |
| `test_bayes_factor_prefers_true_model_dynesty` (item 7) | ln B_10 > 5 for data with Q = 0.6 (measured: 29.7 +- 0.4) | – | pass |
| `test_bayes_factor_prefers_true_model_savage_dickey` (item 7) | ln B_10 > 3, reported as a lower limit, boundary warning issued | – | pass |
| `test_bayes_factors_prefer_null_when_null_is_true` | data with Q = 0: both methods give ln B_10 < 0.5 and agree | 4 sigma + 0.3 | pass |
| `test_savage_dickey_against_analytic_density` | analytic Gaussian case, interior and boundary | 0.05; 0.15 | pass |
| `test_result_is_reproducible_and_round_trips` | bundle save/load; same seed gives the identical chain | exact | pass |
| `test_registry_dataset_versions_are_recorded` | dataset versions and hashes in the fit record | exact | pass |
| `test_profile_single_metric`, `..._several_metrics_and_quantities`, `test_profile_errors`, `test_corner_and_latex_table` | plot data equal the computed observables; labels; errors; table content | exact | pass |

Notes on the inference tests: the synthetic data are noise-free (the
posterior is centred on the truth), so the 2-sigma test cannot fail by an
unlucky noise draw. Example 3 uses noisy data and recovers
Q = 0.550 (+0.035/-0.039) for a true value of 0.6 (1.4 sigma).

## 3. Formulas and their sources

Full statements and derivations are in `docs/theory.md`.

| Formula | Source | How it was checked |
|---|---|---|
| Horizon classification by the sign of g' | derived | RN, extremal RN, SdS, de Sitter tests |
| Closed forms for G^mu_nu, R, K (E1 to E4) | derived with sympy | symbolic identity against the full tensor computation for arbitrary f, g, h |
| rho = -G^t_t/8pi, p_r = G^r_r/8pi, p_t = G^th_th/8pi | convention (stated) | RN Coulomb field, Schwarzschild vacuum |
| Energy conditions for an anisotropic fluid | Hawking & Ellis (1973); Curiel (2017). **Cited from memory, not re-read** | cases with known outcome (Lambda, Hayward, phantom) |
| Omega_phi^2 = f'/h'; E, L | derived | Schwarzschild; isotropic coordinates |
| Omega_r^2 = g N / (2 f h h') | derived | Schwarzschild; independent sympy derivation for RN; isotropic coordinates |
| Omega_theta = Omega_phi | derived | exact equality in code; follows from spherical symmetry |
| ISCO: N = 0; marginally bound: E = 1 | derived | Schwarzschild 6 and 4; RN cubic; extremal RN 4 |
| SdS stable orbits end at Lambda = 4/5625 | Stuchlik & Hledik (1999). **Value quoted from memory**; DOI checked | reproduced numerically (ISCO and OSCO merge there) |
| nu = Omega c^3 / (2 pi G M) | standard | 219.86 Hz for 10 M_sun |
| c, G (CODATA 2018), GM_sun (IAU 2015 B3) | **values from memory**; references' DOIs checked with Crossref | M_sun = 1.98841e30 kg; GM_sun/c^2 = 1.4766 km |
| QPO models (11) | Stuchlik & Kolos (2016), Table 1. **Read from the arXiv text** | transcription test; 3:2 radii 27/4 and 54/5 |
| Photon sphere f'h - fh' = 0; b_c = sqrt(h/f) | derived | Schwarzschild, RN, SdS |
| sin(alpha_sh) = b_c sqrt(f/h) at the observer | derived; standard (Perlick & Tsupko 2022, **not re-read**) | Synge's Schwarzschild formula |
| Split-normal likelihood | standard | hand computation |
| emcee convergence N > 50 tau; thinning tau/2 | emcee documentation (**from memory**); burn-in 5 tau is my choice | synthetic recovery |
| Savage-Dickey ratio | Dickey (1971); Verdinelli & Wasserman (1995). **Cited from memory** | analytic Gaussian case; agreement with evidence ratio |
| Kass & Raftery scale | Kass & Raftery (1995). **Cited from memory** | – |

Catalogue references (Schwarzschild 1916, Reissner 1916, Nordstrom 1918,
Kottler 1918, Bardeen 1968, Ayon-Beato & Garcia 2000, Hayward 2006) were
written from memory; the DOIs of Reissner, Kottler, Ayon-Beato & Garcia and
Hayward were confirmed with Crossref (title, journal, volume, pages). The
Schwarzschild, Nordstrom and Bardeen references have no DOI and were not
checked. The metric functions themselves are standard and were not re-read
from the original papers.

## 4. Registry values

All values were read by Claude on 2026-11-14 from the arXiv versions of the
papers (ar5iv HTML), not from the journal versions. Bibliographic data were
checked with the arXiv API and Crossref. **Every entry still needs a human
check** and is flagged `verified_by_human: false`.

| Dataset | Quantity | Value | Read from | Status |
|---|---|---|---|---|
| `qpo/gro_j1655_40` | upper HFQPO | 441 +- 2 Hz | Motta et al. (2014), Table 2, Sample B1 | read from original |
| | lower HFQPO | 298 +- 4 Hz | same | read from original |
| | type-C QPO | 17.3 +- 0.1 Hz | same | read from original (not used: nodal precession) |
| | mass | 5.40 +- 0.30 M_sun | Beer & Podsiadlowski (2002), Table 4 | read from original |
| `qpo/xte_j1859_226` | upper HFQPO | 227.5 +2.1 -2.4 Hz | Motta et al. (2022), Table 2, triplet | read from original |
| | lower HFQPO | 128.6 +1.6 -1.8 Hz | same | read from original |
| | type-C QPO | 3.65 +- 0.01 Hz | same | read from original (not used) |
| | mass | 7.8 +- 1.9 M_sun | Yanes-Rizo et al. (2022), abstract | read from the abstract only |
| `qpo/xte_j1550_564` | upper HFQPO | 268 +- 3 Hz | Miller et al. (2001), Fig. 4 caption | read from original |
| | lower HFQPO | 188 +- 3 Hz | same | read from original |
| | mass | 9.10 +- 0.61 M_sun | Orosz et al. (2011), abstract, Sec. IV.1 | read from original |
| | alternative 276 +- 3 / 184 +- 5 Hz | **null** | not found in an accessible original | TODO |
| `qpo/grs_1915_105` | upper HFQPO | 69.2 +- 0.15 Hz | Strohmayer (2001), Sec. 2 | read from original |
| | lower HFQPO | 41.5 +- 0.4 Hz | same | read from original |
| | mass | 12.4 +2.0 -1.8 M_sun | Reid et al. (2014), abstract | read from original |
| | alternative 168 / 113 Hz | values without uncertainties (**null**) | Remillard & McClintock (2006) review | secondary source; TODO |
| `qpo/h1743_322` | upper HFQPO | 241 +- 3 Hz | Homan et al. (2005), Sec. 3 | read from original |
| | lower HFQPO | 165 +9 -5 Hz | same (3 sigma detection, Q fixed) | read from original |
| | mass | **null** | no dynamical mass found | TODO |
| `qpo/m82_x1` | upper QPO | 5.07 +- 0.06 Hz | Pasham et al. (2014), abstract | read from original |
| | lower QPO | 3.32 +- 0.06 Hz | same | read from original |
| | mass | 415 +- 63 M_sun | same; **RPM fit, not independent** | flagged circular |
| `eht/sgra` | r_sh/M, 1 sigma | 4.55 to 5.22 | Vagnozzi et al. (2023), Eq. (7) | read from original |
| | r_sh/M, 2 sigma | 4.21 to 5.56 | Eq. (8) | read from original |
| | delta | -0.060 +- 0.065 | Eq. (5) | read from original |

Published results reproduced from the registry (example 4, not a test):
Vagnozzi et al. (2023) report Q <~ 0.8 M (1 sigma) and Q <~ 0.95 M (2 sigma)
for Reissner-Nordstrom; BlackPy gives 0.798 and 0.939. Their statements that
the Bardeen and Hayward metrics are consistent up to the extremal value are
also reproduced (r_sh = 4.524 and 4.917 at extremality).

## 5. Benchmarks

10^4 parameter values, vectorised, best of 9 runs after compilation,
single process on the Apple M2.

| Metric | Quantity | ms per evaluation | Compile time |
|---|---|---|---|
| Reissner-Nordstrom | event horizon | 0.0026 | 0.4 s |
| Reissner-Nordstrom | ISCO (default, close-root detection on) | **0.0057** | 1.0 s |
| Reissner-Nordstrom | ISCO (`detect_extrema=False`) | 0.0039 | 0.6 s |
| Reissner-Nordstrom | photon sphere / b_c | 0.0087 | 0.8 s |
| Reissner-Nordstrom | orbital and epicyclic frequencies | 0.00002 | 0.06 s |
| Hayward | ISCO | 0.0050 | 1.1 s |
| Schwarzschild-de Sitter | ISCO | 0.0064 | 0.9 s |
| Bardeen | ISCO | 0.0164 | 1.3 s |
| Bardeen | photon sphere / b_c | 0.029 | 1.1 s |

Each ISCO evaluation includes locating the event horizon. For comparison,
the specification quotes 0.0028 ms for a JAX solve from a known starting
point; BlackPy's figure includes a 256-point scan for all roots.

Accuracy: RN horizons agree with the closed form to 5e-15 over 10^4 charges;
the RN ISCO satisfies its cubic to 2e-13.

Inference timings (synthetic two-source RN fit, 5 parameters): emcee,
32 walkers x 5000 steps, 1.6 s; dynesty, 300 live points, 3 s (115 000
likelihood calls); null model for the Bayes factor, 7 s.

## 6. Known limitations

1. **No human verification of the data.** Treat every registry number as
   provisional.
2. **Spherical symmetry only.** The QPO sources spin; fits with non-rotating
   metrics are not physical constraints (see `DESIGN_NOTES.md`, section 4).
   Low-frequency QPOs are ignored with a warning.
3. **Only arXiv versions were read**, through an automatic HTML conversion.
   Table layouts can be garbled there; the GRO J1655-40 row was cross-checked
   against the values quoted by Stuchlik & Kolos (2016), the others were not.
4. **Roots outside `metric.r_range`** (default 1e-3 to 1e4 M) are not found,
   and at most six roots are reported (three in the vectorised fast paths).
5. **Two roots closer than about 1e-5 in relative terms** with |F| < 1e-10
   between them are merged into one degenerate root.
6. **The asymptotic-flatness probe is a heuristic** and can misjudge metrics
   that approach Minkowski slowly; declare the property explicitly then.
7. **emcee starts from a ball around one posterior maximum**; multimodal
   posteriors need nested sampling or a user-supplied start.
8. **The Savage-Dickey estimate** gives only a lower limit when the posterior
   has no samples at the null value, and has no reflection for
   multi-parameter nulls on a boundary.
9. **Kretschmann scalar and stress-energy are NaN exactly at a horizon**
   (0/0 in the closed forms); they are finite arbitrarily close to it.
10. **Windows and Linux are untested**; the CI workflow exists but has not run.
11. **Performance target missed for Bardeen** (0.016 ms per ISCO).

## 7. Points in the specification that I think should change

1. The Sgr A* bounds: the paper gives 4.55 and 4.21 for the lower limits,
   not 4.54 and 4.20.
2. "Event horizon = outermost root" is wrong for de Sitter-like metrics.
3. The "standard" QPO values for XTE J1550-564 and GRS 1915+105 are not in
   the discovery papers that were accessible; a human should decide the
   default pairs.
4. M82 X-1 and H1743-322 have no independent mass and should not be used to
   test metrics unless that changes.
5. Rotation should come before any further data: without it QPO-based
   constraints on extra parameters are not meaningful.

## 8. Commands

```bash
cd /Users/pahlavon/Desktop/BlackPy
source .venv/bin/activate

python -m pytest                       # full suite, about 100 s
python -m pytest -m "not slow"         # quick subset
python -m pytest -s -m slow -k isco    # prints the ISCO timing

python examples/01_new_metric.py
python examples/02_profile_plots.py
python examples/03_rn_fit.py
python examples/04_sgra_shadow_constraints.py

ruff check src tests examples && ruff format --check src tests examples
python -m mkdocs build --strict
```
