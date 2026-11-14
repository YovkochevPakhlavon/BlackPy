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
  metric/             StaticSpherical, catalogue (one file per metric), interpolant, stubs
  geometry/           horizons, curvature (numeric closed forms), symbolic tensors, energy conditions
  orbits/             circular orbits, epicyclic frequencies, QPO models
  light/              photon spheres, shadow, plasma/lensing stubs
  data/               schema, loader, registry/*.yaml
  infer/              priors, likelihood, Fit (emcee/dynesty), results
  plot/               profiles, corner, LaTeX tables
  qnm.py, thermo.py   stubs
```

**Core/wrapper pattern.** Each observable is a pure function
`core(metric, p, *scalars, **static_options)` of the parameter array `p`.
Public functions call `metric.evaluate(core, params, ...)`, which resolves
keyword parameters and defaults, checks bounds, broadcasts, and applies
`jit(vmap(core))`, cached per metric instance. Likelihoods call the cores
directly, so the shadow radius is solved *inside* the compiled likelihood.

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
5. **Event horizon is not "the outermost root".** See deviation D1.
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
- Repository URL for `pyproject.toml` and `CITATION.cff` (currently a placeholder).

## 6. Roadmap (interfaces reserved)

| Feature | Reserved interface |
|---|---|
| Rotating metrics, Newman-Janis | `blackpy.metric.StationaryAxisymmetric`, `metric.future.newman_janis` |
| RZ / KRZ / Johannsen parametrisations | `metric.future.rezzolla_zhidenko`, `konoplya_rezzolla_zhidenko`, `johannsen` |
| Nodal precession, low-frequency QPOs | `orbits.nodal_precession_frequency`, `QPOModel.low_frequency` |
| Plasma | `light.PlasmaModel`, `light.shadow_in_plasma` |
| Weak and strong lensing | `light.deflection_angle` (quadrature already in `_numerics`) |
| Quasinormal modes (WKB + Pade) | `blackpy.qnm.quasinormal_modes` |
| Thermodynamics | `blackpy.thermo` |
| S2 orbit data | new `data_type` in the registry schema; needs ODE integration |

All observable functions take the abstract `Metric` as first argument, so new
metric classes can be added without changing the public API.
