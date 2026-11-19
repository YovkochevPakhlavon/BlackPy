# Design notes

Decisions, deviations from the original specification, open questions and
the roadmap. Conventions that every contributor must follow are in
`CLAUDE.md`; formulas are in `docs/theory.md`.

## 1. Architecture

```
src/blackpy/
  constants.py        CODATA / IAU constants, Hz conversion
  _numerics.py        derivatives, root finding, quadrature (the shared toolbox)
  _deprecation.py     deprecation warnings and aliases (v1.0); _links.py: documentation URLs
  errors.py           exceptions and warning classes
  metric/             base.py (Metric, StaticSpherical), axisymmetric.py
                      (StationaryAxisymmetric), rotating.py (Azreg-Ainou generator),
                      catalogue/ (one file per metric), parametrized.py (RZ, KRZ),
                      interpolant, stubs
  geometry/           horizons (both classes), curvature (static closed forms + dispatch),
                      axisymmetric.py (4D autodiff curvature, source classification),
                      symbolic.py (static tensors, 4D tensors with time limit),
                      energy conditions, report
  orbits/             circular.py (static formulas + dispatch + public API),
                      axisymmetric.py (general rotating formulas), frequencies, QPO models,
                      rpm.py (Ingram & Motta analytic solution)
  light/              photon spheres and shadow (static), rotating.py (separable shadows,
                      Delta C definitions), plasma.py (v0.3)
  lensing/            deflection angles, strong deflection limit (v0.3)
  disk.py             thin-disk efficiency and Page-Thorne flux (v0.3)
  data/               schema, loader, registry/*.yaml
  infer/              priors, likelihood, Fit (emcee/dynesty), results,
                      quality.py (fit quality), forecast.py (Fisher) (v0.3)
  plot/               profiles, shadows, corner, residuals, LaTeX tables, predictions
                      (deflection, disk flux, forecast ellipses; v1.0)
  qnm.py, thermo.py   stubs
```

**Core/wrapper pattern.** Each observable is a pure function
`core(metric, p, *scalars, **static_options)` of the parameter array `p`.
Public functions call `metric.evaluate(core, params, ...)`, which resolves
keyword parameters and defaults, checks bounds, broadcasts, and applies
`jit(vmap(core))`, cached per metric instance. Likelihoods call the cores
directly, so the horizon, the ISCO and the shadow are solved *inside* the
compiled likelihood. `evaluate` and the parameter handling live in the
`Metric` base class and are shared by both metric classes.

**Static / rotating dispatch (v0.2).** Core functions check
`metric.is_rotating_class` and forward rotating metrics to the general
formulas. Static metrics keep the v0.1 code unchanged (same numbers, same
speed). `StaticSpherical.as_axisymmetric()` realises "static is the special
case g_tphi = 0" and is used by the tests to check the two paths against
each other.

## 2. Non-obvious choices

1. **Units: M = 1 inside the metric.** Catalogue metrics carry no mass
   parameter; the physical mass enters only in the conversion to hertz. This
   keeps "lengths in units of M" unambiguous and gives each QPO source its own
   mass without touching the metric.
2. **Forward-mode autodiff.** Derivatives use nested `jax.jvp`. For
   scalar-to-scalar functions this gave much smaller graphs than nested
   `jax.grad`: the scan of the derivative of the ISCO function over 256 points
   took 0.0004 ms instead of 0.0025 ms per parameter set.
3. **Root finding with fixed-length output.** Sign-change scan on a
   logarithmic grid (256 points on `metric.r_range`, default 1e-3 to 1e4),
   refinement with safeguarded Newton (`rtsafe`), plus *extremum detection*:
   roots of the derivative are located and both sides of each extremum are
   searched. This finds root pairs closer than the grid spacing (near-extremal
   horizons, ISCO/OSCO close to their merger) and degenerate roots (extremal
   horizons), which a plain scan misses. Outputs are NaN-padded to a fixed
   length so that `vmap` works.
4. **Performance details that mattered.** Padded root slots must terminate
   immediately (under `vmap` a `while_loop` runs until *all* elements finish);
   brackets are selected with `lax.top_k` rather than `jnp.nonzero`; the
   Newton iteration starts from the secant point; the vectorised fast paths
   examine only the three outermost roots. Result: 0.006 ms per ISCO for
   Reissner-Nordstrom, including the horizon search.
5. **Event horizon = outermost black hole horizon, not the cosmological horizon**
   (so not simply "the outermost root"). See deviation D1.
6. **ISCO definition.** "Outermost radius where circular orbits change from
   unstable (inside) to stable (outside)". Unique for ordinary black holes; in
   Schwarzschild-de Sitter it correctly returns the inner edge, not the OSCO.
7. **`asymptotically_flat` is probed numerically when not declared**
   (f, g, h/r^2 within 1e-4 of 1 at r = 1e6 and 1e-6 at r = 1e9), per
   parameter value, so Schwarzschild-de Sitter with Lambda = 0 counts as flat.
   The heuristic is documented; users can declare the property explicitly.
8. **Parameter bounds** describe where the metric functions are defined, not
   where a black hole exists (RN allows Q > 1 so that naked singularities can
   be studied and are reported as horizonless).
9. **Bardeen parameter is called `q`**, not `g`, to avoid a clash with the
   metric function `g`.
10. **emcee initialisation: a small ball around the posterior maximum**
    (best of 4000+ prior draws, polished with Nelder-Mead). Starting from
    scattered prior draws left individual walkers stuck in a minor local
    maximum for the whole run while the autocorrelation check still passed.
    Burn-in is 5 autocorrelation times. Multimodal posteriors need
    `initial_state` or nested sampling.
11. **dynesty default `sample="rwalk"`.** With the default uniform sampling
    the null model of the synthetic test needed 2.4 million likelihood calls
    (240 s); random-walk proposals needed 0.3 million (7 s) and gave the same
    evidence within errors.
12. **Savage-Dickey estimate.** One-dimensional Gaussian KDE with Silverman's
    bandwidth computed from the *unreflected* samples; reflection doubles the
    kernel sum at a boundary. Taking the bandwidth from the reflected sample
    over-smoothed and biased ln B by about 0.6 in the analytic test. If no
    sample is near the null value the result is reported as a lower limit.
13. **Asymmetric uncertainties** use a split normal in the likelihood and are
    symmetrised (mean of the two errors) for default mass priors.
14. **Reproducibility bundle** is a single `.npz` with the samples and a JSON
    record (no pickle): configuration, priors, package versions, sampler
    settings, seed, dataset versions and file hashes.
15. **Results are JAX arrays**, also for scalar inputs (0-d arrays); wrap in
    `float()` when a Python number is needed.

### Added in v0.2

16. **`CATALOGUE` was not extended.** v0.1 tests iterate over `CATALOGUE`
    assuming static metrics, so rotating entries live in `ROTATING_CATALOGUE`.
17. **Orbit directions.** `direction="prograde"` is the `Omega_+` root
    (positive phi direction), `"retrograde"` the `Omega_-` root. Negative spin
    with prograde orbits is the same physics as positive spin with retrograde
    orbits; both are supported. Frequencies are returned as positive numbers;
    the nodal frequency is signed (`nu_phi - nu_theta`), and QPO models use
    its absolute value, following Ingram & Motta (2014).
18. **Pole-free stability functions.** `N_r` and `N_theta` are written so
    that they stay finite at the photon orbit (derivation in
    `docs/theory.md`, section 14); the ISCO scan then needs no special
    handling near it. Kerr ISCO: 0.017 ms per evaluation.
19. **Horizon classification for rotating metrics uses a sign test**, not a
    derivative: `g^rr = 1/g_rr` is evaluated through an infinite `g_rr` at
    an exact root (`Delta = 0` is exactly representable for many spins),
    where its autodiff derivative is undefined.
20. **Source classification in an orthonormal frame.** The first version
    judged "vacuum" from the eigenvalues of `T^mu_nu`; null dust has only
    zero eigenvalues and was misclassified (caught by a test). The size of
    `T` is now measured in an orthonormal frame built from the metric.
21. **Symbolic 4D tensors in `x = cos(theta)`.** Kerr-like metrics become
    rational functions, `sympy.cancel` replaces `simplify`, and the
    Kretschmann sum is reduced block by block: Kerr takes 12 s, Kerr-Newman
    28 s. The time limit is cooperative (checked between components) plus
    `SIGALRM` where available; a subprocess was rejected because `spawn`
    re-imports user scripts that lack a `__main__` guard.
22. **Generic separability instead of metric-specific shadow formulas.** The
    shadow code only needs `W g^tt`, `W g^tphi`, `W g^phiphi` to split into
    radial and polar parts (`W = g_thth`); it never sees `Delta` or `a`. This
    covers Kerr, Kerr-Newman, Johannsen and every Azreg-Ainou metric.
23. **Photon region anchored at the polar orbit.** The ends of the shadow
    contour are the roots of `Theta(i) = 0` nearest to the `xi = 0` orbit
    (`T_r' = 0`). Searching for "the outermost roots" failed at small spin:
    round-off noise in `Q_r'` at large radii produced spurious roots.
24. **Slow rotation.** `xi(r_p)` is 0/0 at `a = 0`. Below a frame-dragging
    measure of 1e-5 (a < 5.6e-7 for Kerr) the shadow is the non-rotating
    circle; below 2e-4 the shape integrals use the displaced circle. The
    neglected terms are documented in `docs/theory.md`, section 15.
25. **Shape integrals by the midpoint rule in the contour parameter**: the
    contour is a smooth closed curve in `t` (`r_p = mid - half cos t`), so
    the rule converges exponentially (64 points: 1e-10) and never evaluates
    the end points, where `d beta/dt` would need `sqrt` at zero.
26. **Finite-distance shadows use a ZAMO** and the screen coordinates
    `R_O (-n_phi, n_theta)` with `R_O = sqrt(g_thth)`, chosen because they
    reduce to v0.1's `sqrt(h) sin(alpha_sh)` for static metrics and to
    Bardeen's coordinates at infinity. Other observers (e.g. the Carter
    observer of Grenzebach et al. 2014) are not implemented.
27. **Spin bounds through a declared domain.** A rotating metric can declare
    `domain(*params) -> bool` (`|a| <= 1`, `a^2 + Q^2 <= 1`, Johannsen's
    regularity bounds). Observables do not enforce it (naked singularities
    remain computable); the likelihood does, together with "an event horizon
    exists", when `require_black_hole=True` (default for rotating metrics).
28. **Inclination in degrees, isotropic by default.** The shadow nuisance
    parameter is named `inclination`; its default prior is `p(i) ~ sin i` on
    [0, 90] degrees (`Isotropic`). *Superseded by decision 39 (1.0): the
    parameter is now `inclination_deg`; units and prior are unchanged.*
29. **`Parameter.role`** (`"spin"`, `"deformation"`, `"charge"`) lets the
    degeneracy guard find the spin without guessing from names.
30. **Coverage test draws the truth from the prior.** Only then is "68%
    interval contains the truth in 68% of cases" an exact property of a
    correct posterior (for a fixed truth it holds only asymptotically).

## 3. Deviations from the specification

| # | Specification | What was done | Why |
|---|---|---|---|
| D1 | "mark the event horizon (outermost)" | Event horizon = outermost root with g' > 0 (or extremal double root); cosmological horizons are classified separately | The outermost root of Schwarzschild-de Sitter is the cosmological horizon |
| D2 | Sgr A* bounds 4.54-5.22 and 4.20-5.56 | Registry stores 4.55-5.22 and 4.21-5.56 | These are the numbers in Eqs. (7)-(8) of the arXiv text of Vagnozzi et al., consistent with its Eq. (6) |
| D3 | XTE J1550-564, GRS 1915+105 from "original discovery papers" | J1550: Miller et al. (2001) pair 268/188 Hz. GRS 1915: Strohmayer (2001) pair 41.5/69.2 Hz | The commonly used pairs (276/184 and 168/113 Hz with uncertainties) could not be found in an original source that was accessible; they are documented as alternatives |
| D4 | Dependencies "numpy, scipy, matplotlib, corner, pyyaml" | plus `emcee`, `dynesty` (required samplers), `jax`, `sympy` (fixed decisions) | Implied by the fixed decisions |
| D5 | `blackpy.data.list()` "prints the registry" | Prints and also returns the datasets | Convenient in notebooks |
| D6 | Dev extras | `.[dev]` = pytest + ruff; MkDocs is in `.[docs]` | Keeps CI test jobs light |
| D7 | Performance test budget | Test asserts < 0.05 ms per ISCO; measured 0.006 ms | A hard 0.01 ms assertion would fail on slow CI runners |
| D8 | Catalogue entries as classes | Factory functions `Schwarzschild()`, ... returning `StaticSpherical` | Simpler than subclasses; the public call syntax is identical |
| D9 | Extra integrity warnings | Added `UnverifiedDataWarning` and `IgnoredDataWarning` | Makes unverified registry data and dropped low-frequency QPOs visible |

| D10 (v0.2) | "deviation from circularity as defined by the EHT M87* papers" | `Delta C` of Bambi et al. (2019), Eqs. (4)-(5) | The EHT M87* papers define no `Delta C`; Paper VI uses the fractional spread of the ring diameter and refers to the asymmetry of Johannsen & Psaltis (2010). See `verification/v0.2.md`, section 8 |
| D11 (v0.2) | "StaticSpherical ideally becomes a special case with g_tphi = 0" | Separate class with `as_axisymmetric()`; dispatch in the core functions | Keeps the v0.1 numbers and speed exactly; the special-case relation is tested instead of structural |
| D12 (v0.2) | Reproduce published RPM values "as validation" (tests) | Example 06, not a test | `CLAUDE.md`: tests must not depend on unverified registry numbers |
| D13 (v0.2) | Rotating metrics in the built-in catalogue | `ROTATING_CATALOGUE`, not `CATALOGUE` | v0.1 tests iterate over `CATALOGUE` |
| D14 (v0.2) | Azreg-Ainou procedure for any static metric | Conformal factor `Psi = h + a^2 cos^2(theta)` by default, replaceable | The paper fixes `Psi` only for `g = f`, `h = r^2` |
| D15 (v0.2) | Shadow "at finite distance" | ZAMO observer | The specification names no observer |
| D16 (v0.2) | v0.1 test `test_future_metric_stubs_raise` | Rewritten | It asserted that `StationaryAxisymmetric` raises `NotImplementedError` |

## 4. Things in the specification that deserve a second look

1. **Spherical symmetry and QPO data.** The sources in the registry are
   believed to spin (the RPM fits give a = 0.29 for GRO J1655-40 and 0.15 for
   XTE J1859+226). Fitting them with non-rotating metrics lets the extra
   metric parameter absorb the effect of spin, so "constraints" on charges
   from QPOs in the MVP are not physical constraints. Rotation should be the
   first post-MVP feature.
2. **Degeneracy.** With a free mass and radius per source, two frequencies
   cannot constrain a metric parameter by themselves: all information comes
   from the independent mass prior. The prior-dominance guard catches the
   extreme case, but users should be told this explicitly (it is in the docs).
3. **Simultaneity.** Several literature "pairs" combine peaks detected in
   different observations. The registry records `simultaneous` and prefers
   simultaneous detections.
4. **The 0.01 ms target** is met for Reissner-Nordstrom (0.006 ms) but not
   for metrics with expensive functions: Bardeen needs 0.016 ms because of
   the fractional power.
5. **Shadow likelihood.** The published interval comes from a Gaussian in
   delta; the Gaussian likelihood is the faithful choice and is the default.
   The top-hat option is kept because many papers use it.

## 5. Open questions for the maintainer

Still open (they wait for papers to be read or authors to answer; see `TO_DO.md`).
The questions the owner answered on 2026-11-19 are decisions 45 to 56 (section 9).

- The factor 4 of the `bambi2019` definition of Delta C is empirical (it
  reproduces Fig. 1 of Bambi et al. 2019; their printed Eq. (5) does not). It
  stays labelled empirical until its origin is found (a later paper or the
  authors); then theory section 17 is updated.

## 6. Roadmap

Moved to the public documentation in 1.0: [`docs/roadmap.md`](docs/roadmap.md)
(error messages of stubs point to its hosted version). Guiding principle,
unchanged since v0.3: BlackPy is "metric in, constraints out"; a feature is
scheduled if it feeds a likelihood, makes a fit more trustworthy, or widens the
set of metrics that can be fitted.

## 7. Status at 1.0 (2026-11-19)

**Done.** The specification `BLACKPY_v1.0_PROMPT.md` (Parts A to F, including the
optional Part D) is implemented and committed, one commit per part (`git log`).
The code is labelled **1.0.0rc1**. Results, the wheel and minimum-version checks,
the error-message audit and the known limitations are in `verification/v1.0.md`.

- Public API frozen and snapshot-tested (decision 38); deprecated aliases warn.
- One rule for angles; `inclination_deg` (decision 39). Versioned result bundles
  (decision 40).
- User guide with tested snippets (decision 41); the docs build strictly with
  anchor validation.
- Packaging: wheel and sdist build and pass `twine check`; the installed wheel
  passes the quick subset; the declared minimum versions pass it on Python 3.10
  (decision 42). CI, docs and release workflows are written but have **never run
  on GitHub** (nothing is pushed by Claude).

**Left, all for the owner** (ordered checklist with commands: `TO_DO.md`):
decisions to confirm (section 5 above and `TO_DO.md`, section 1), human
verification of every registry value (all `verified_by_human: false`), pushing and
watching CI, re-running the Windows check on the final code, GitHub Pages, PyPI
trusted publishing, the 1.0.0 label, tag, release and Zenodo DOI.

**History.** What each version added is in `CHANGELOG.md`; what was tested and
read for each version is in `verification/` (`v0.1.md` to `v1.0.md`). The
status sections of the v0.2 and v0.3 sessions were replaced by this one.

## 8. Decisions added in v0.3

31. **Degrees of freedom** = fitted Gaussian data + informative Gaussian
    priors - free parameters (theory section 18). Interval likelihoods and
    bounds have no pull and are not counted.
32. **Poor-fit threshold 0.05** on the posterior predictive p-value of the
    chi-square discrepancy, computed as the posterior mean of the classical
    p-value (no simulation noise); measured false-alarm rate 1 in 100.
33. **Plasma as a metric wrapper.** `with_plasma(metric, plasma)` returns an
    object with the metric's interface plus one hook; the separable-shadow
    code adds the plasma term to \(W g^{tt}\). No shadow code was duplicated.
34. **Parameter layout for several sources.** `build_problem` keeps a table
    `layout[source][parameter] = free index or fixed value`, so shared and
    per-source parameters, per-source inclinations and the data table use
    one mechanism.
35. **KRZ keeps the full Kerr polar function** \(K_2\) (theory section
    21.2), because the printed lowest-order truncation is Kerr only on the
    equator.
36. **Lensing integrals**: the singular part in closed form, the remainder by
    Gauss-Legendre in \(\sqrt z\), and a Taylor branch for \(z < 10^{-3}\)
    against cancellation (theory section 23.1).
37. **Catalogue names**: the Simpson-Visser length scale is `ell` (the
    paper's `a`) so that `rotating_from_static` can add a spin; the
    power-law plasma index is `k` (Bardeen already has `q`).

## 9. Decisions added in v1.0

38. **Public API and deprecation policy.** The public API is the set of
    names in `__all__` of `blackpy` and of its sub-packages and public
    modules (`constants`, `errors`, `metric`, `metric.future`,
    `metric.interpolated`, `geometry`, `orbits`, `light`, `lensing`, `disk`,
    `data`, `infer`, `plot`, `qnm`, `thermo`). Everything else is private,
    even if importable (`blackpy._numerics`, `blackpy.infer.likelihood.INCLINATION`,
    core functions). From 1.0 on, BlackPy follows semantic versioning: a
    public name is removed, or its behaviour changed, only in a major
    release, and only after at least one minor release in which using it
    emits a `DeprecationWarning` that names the replacement. Deprecated
    aliases are served by a module `__getattr__` (PEP 562), are not in
    `__all__`, and are listed in `tests/test_public_api.py`, whose snapshot
    of `__all__` fails when a public name disappears.
39. **One rule for angles** (supersedes 28): every angle in the public API
    is in radians; a quantity in degrees carries `_deg` in its name. The
    shadow nuisance parameter is therefore `inclination_deg` (per source:
    `inclination_deg_<source>`), while `inclination` stays in radians in
    `blackpy.light` and `blackpy.plot`. Posteriors, corner plots and the
    `Isotropic(0, 90)` prior stay in degrees because the name says so. The
    pre-1.0 names are accepted during 1.x with a `DeprecationWarning`
    (priors, fixed values, `shared`/`per_source`, keyword arguments,
    fiducial points, `FitResult[...]`); a name given in both spellings is an
    error. The registry keeps degrees as printed in the papers, with an
    explicit `unit: deg`; values are converted where they are used. Public
    functions that take an inclination in radians refuse values outside
    [0, π] with a message that suggests `math.radians`. No opposite guard
    on `inclination_deg`: any range test (for example "the whole prior lies
    below 1.6") would have false positives for nearly face-on systems.
40. **Versioned result bundles.** `FitResult.save` writes
    `"bundle_format": 1`. Bundles without the field are format 0 (0.1 to
    0.3) and are upgraded on loading (inclination columns renamed); a newer
    format is refused with the BlackPy version that wrote it. Fixtures of
    both formats (`tests/data/`) guard loading for all of 1.x.
41. **Documentation snippets are tests.** `tests/test_docs.py` executes the
    Python blocks of the README, the quickstart, the data page and the user
    guide in order (one namespace per page) and compares what they print with
    the "Output" block that follows. Exceptions to exact comparison, both
    explicit in the page: "Output (sampler)" blocks are compared with a tolerance
    of 0.05 on numbers (Monte Carlo chains can differ in the last digits between
    platforms), and "Output (registry data)" blocks are run but not compared,
    because tests must not depend on unverified registry values. Warnings are
    part of the contract: a `# warns:` comment lists the BlackPy warnings a block
    emits, and the set must match. Outputs and figures are regenerated from the
    snippets themselves (`scripts/update_doc_outputs.py`, `docs/make_figures.py`).
    The user-guide test is slow (samplers, about two minutes); the others are quick.
42. **Dependency bounds.** The lower bounds of `pyproject.toml` (jax 0.4.30,
    numpy 1.24, scipy 1.10, sympy 1.12, matplotlib 3.7, corner 2.2, pyyaml 6.0,
    emcee 3.1, dynesty 2.1) were installed together on Python 3.10 and pass the
    quick subset; they are pinned in `.github/min-requirements.txt` for the
    `min-deps` CI job. Lower versions were not explored. JAX 0.6.2 is the last
    release for Python 3.10 (JAX >= 0.7.0 requires Python >= 3.11, from the PyPI
    metadata), so 3.10 users run JAX <= 0.6.2. No upper bounds: they would block
    users' security updates, and CI on the latest versions catches breakage.
43. **`build` and `twine` in the `dev` extra** (development tools, not runtime
    dependencies): they build the sdist and wheel and check their metadata,
    locally and in CI. They are not installed by `pip install blackpy-gr`.
44. **Examples in quick mode.** `BLACKPY_EXAMPLES_QUICK=1` divides the sampler
    steps and predictive draws by ten (at least 200) and uses 100 dynesty live
    points, so that a weekly CI job can run every example; unset, every example
    is unchanged. One assertion (example 13, forecast versus sampler) is skipped
    in quick mode because the short chain is too noisy for it.

### Owner decisions of 2026-11-19 (open questions answered)

45. **Unverified data stay a warning in 1.x.** `UnverifiedDataWarning` is not
    turned into an error even after the registry has been checked: that would be
    a breaking change. Revisit for 2.0.
46. **Parameter bounds of observables are not restricted to the black hole
    range.** Observables stay defined for naked singularities and report them as
    horizonless; fits of rotating metrics restrict through
    `require_black_hole=True` and the declared domain.
47. **Repository URL confirmed**: `https://github.com/YovkochevPakhlavon/BlackPy`
    (`pyproject.toml`, `CITATION.cff`, `mkdocs.yml`).
48. **Delta C default stays `fractional_rms`.** It is the only definition fully
    specified by a published formula with its normalisation, and it is the
    quantity of the EHT M87* circularity statement (Paper I). `jp_asymmetry` and
    `bambi2019` remain available by name.
49. **`CATALOGUE` and `ROTATING_CATALOGUE` stay separate in 1.x** (decision 16);
    merging them is a 2.0 question.
50. **M87\* mass prior: stellar dynamics is the default**, and every result says
    which prior it used (`shadow_mass_prior`). The gas-dynamics mass stays
    available; it gives delta = 0.78 ± 0.3, in 2.6 sigma tension with a
    Schwarzschild-like shadow, so studies should report both.
51. **KRZ keeps the full Kerr polar function** (decision 35), not the printed
    truncation, which is not Kerr off the equator; the printed w_01 of the
    rotating dilaton black hole has the wrong sign. Documented in theory
    section 21.2.
52. **RZ coefficients of Reissner–Nordström are accepted as derived** (the RZ
    paper does not treat RN; theory section 21.1 and its test).
53. **NUTS sampler: a roadmap item for a later 1.x**, as an optional extra
    (needs gradients through the root finders by implicit differentiation; see
    `docs/roadmap.md` and `verification/v0.3.md`, section 7); emcee and dynesty
    stay the required samplers.
54. **GRS 1915+105: the default QPO pair stays 41.5 ± 0.4 / 69.2 ± 0.15 Hz**
    (Strohmayer 2001), the only pair detected simultaneously and published with
    uncertainties. Remillard (2004, AIP Conf. Proc. 714, 13) gives 113 and 168 Hz
    only as figure labels, without uncertainties, from power spectra of
    different variability classes; that pair stays under `alternatives`
    (`simultaneous: false`) and is not usable in a likelihood as it is.
55. **GRO J1655-40 spin: 0.286 (Table 3 of Motta et al. 2014) stays the main
    entry.** The journal version (MNRAS 437, 2554) was compared on 2026-11-19 and
    prints the same values as the arXiv version (Table 3: 0.286; text: 0.290);
    0.290 stays under `other_values`.
56. **XTE J1550-564: the default QPO pair stays 268 ± 3 / 188 ± 3 Hz** (Miller et
    al. 2001), the only pair found in an original source with independently fitted
    frequencies and uncertainties. Remillard et al. (2002; journal and arXiv
    versions) give only 'near 184 and 276 Hz' and harmonic-locked fits, and state
    that no individual observation shows both HFQPOs; Remillard (2004) shows them
    only as figure labels. The widely quoted 276 ± 3 / 184 ± 5 Hz was not found in
    any original source and is not stored.
