# Design notes

Decisions, deviations from the original specification, open questions and
the roadmap. Conventions that every contributor must follow are in
`CLAUDE.md`; formulas are in `docs/theory.md`.

## 1. Architecture

```
src/blackpy/
  constants.py        CODATA / IAU constants, Hz conversion
  _numerics.py        derivatives, root finding, quadrature (the shared toolbox)
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
  plot/               profiles, corner, LaTeX tables
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

- Which QPO pair should be the default for XTE J1550-564 and GRS 1915+105?
- Should using a dataset with `verified_by_human: false` be an error unless
  the user opts in, once the registry has been checked?
- Should metric parameter bounds optionally restrict to the black hole range?
  (For rotating metrics the likelihood now does this through the declared
  domain; observables still do not.)
- Repository URL for `pyproject.toml` and `CITATION.cff` (currently a placeholder).
- GRO J1655-40 spin: the registry stores 0.290 (abstract and text of Motta et
  al. 2014) although their Table 3 and the exact solution give 0.286. Keep
  the text value, or store the table value?
- Should `Delta C` follow Bambi et al. (2019) as now, or the Johannsen &
  Psaltis (2010) asymmetry that EHT M87* Paper VI refers to (twice the rms)?
- Should the rotating entries be merged into `CATALOGUE` in a later major
  version?

## 6. Roadmap (interfaces reserved)

**Guiding principle (since v0.3).** BlackPy is "metric in, constraints out".
A feature is scheduled if it feeds a likelihood, makes a fit more
trustworthy, or widens the set of metrics that can be fitted.

| Feature | Reserved interface | Status |
|---|---|---|
| Original Newman-Janis algorithm | `metric.future.newman_janis` (use `rotating_from_static`) | not planned: the Azreg-Ainou version covers it |
| Thermodynamics | `blackpy.thermo` (stubs) | **not scheduled**: there are no thermodynamic data, so nothing would feed a likelihood |
| Quasinormal modes | `blackpy.qnm.quasinormal_modes` (stub) | **not scheduled**: without field equations only test-field modes can be computed, and these cannot be compared with ringdown observations for non-vacuum metrics |
| S2 orbit data | new `data_type` in the registry schema; needs ODE integration | later |
| Shadows of non-separable metrics | not planned inside BlackPy (ray tracing; Gradus.jl) | out of scope |
| Continuum-fitting and iron-line likelihoods | none | out of scope: they need ray tracing of the disk image |
| Event horizon when `g^rr = 0` depends on theta | solve `g^rr + g^thth (dH/dtheta)^2 = 0`; `horizon_profile` reports the case | later |

The v0.2 version of this table also listed plasma, lensing and the RZ/KRZ
parametrisations; all three were implemented in v0.3 (lensing for static
metrics only).

All observable functions take the abstract `Metric` as first argument, so new
metric classes can be added without changing the public API.

## 7. Status at the end of the v0.2 session (2026-11-15)

**Done** (each part committed separately; see `git log`):

- Part A: constants against `scipy.constants`; exact Bardeen/Hayward
  critical values; coverage test with noisy data; event-horizon wording.
- B1: `StationaryAxisymmetric`, Kerr, Kerr-Newman, Johannsen,
  `rotating_from_static`.
- B2: horizons, ergosurface, 4D curvature (numeric and symbolic), effective
  source and energy conditions.
- B3: equatorial orbits, both epicyclic frequencies, ISCO, nodal precession,
  11 QPO models with rotation, analytic RPM solution.
- B4: shadows for separable metrics, observer at infinity and ZAMO at finite
  distance, shape observables.
- B5: triplet likelihood, spin bounds, rotating shadow likelihood with
  inclination, two new warnings, examples 05 to 08.
- Documentation: `how_it_works.md`, `tutorial.md`, LaTeX theory for
  rotation, `verification/v0.2.md`.

**Left for later sessions:**

- Human verification of every registry value (unchanged from v0.1).
- CI has still never run (nothing is pushed): Linux and Windows untested.
  The symbolic time limit has no hard interrupt on Windows.
- A `plot` helper for shadow contours (done in v0.3: `blackpy.plot.shadow`).
- The roadmap of the time listed "v0.3: plasma, lensing, QNMs,
  thermodynamics". QNMs and thermodynamics were later removed from v0.3 on
  purpose (section 6); see section 8 for the v0.3 status.

## 8. Status at the end of the v0.3 work (2026-11-17)

The whole specification `BLACKPY_v0.3_PROMPT.md` (Part A, B1 to B8) is
implemented and committed, one commit per part or smaller (`git log`).
Version 0.3.0. Results, tolerances, papers read and limitations are in
`verification/v0.3.md`.

**Not done, by decision:**

- The optional gradient-based (NUTS) sampler of B5. The log-likelihood is
  differentiable only in forward mode (the safeguarded Newton iteration is a
  `while_loop`, which reverse-mode autodiff cannot traverse). A NUTS sampler
  would need either gradients by `jax.jacfwd` (cost proportional to the
  number of parameters, acceptable for the 3 to 6 parameters of typical
  fits) or implicit differentiation of the root finders (`jax.custom_jvp`
  on `refine_root` with the implicit-function theorem), which would also
  enable reverse mode. The sampler itself (BlackJAX or NumPyro) would be an
  optional extra `blackpy-gr[nuts]`, imported lazily; every integrity guard
  works on samples and would apply unchanged. Not attempted because it adds
  a dependency and the specification says to skip it in case of doubt.
- Lensing by rotating metrics (equatorial plane): "if time allows" in the
  specification; not attempted.

**Left for later sessions:**

- Human verification of every registry value (all `verified_by_human:
  false`), including the new M87\* and Sgr A\* entries.
- CI has never run (nothing is pushed); Linux and Windows are untested.
- Open questions recorded in the registry files and in `verification/v0.3.md`
  (section 7).

## 9. Decisions added in v0.3

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

## 10. Decisions added in v1.0

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
