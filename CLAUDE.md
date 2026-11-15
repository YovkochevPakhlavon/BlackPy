# BlackPy: conventions for Claude Code sessions

BlackPy takes a black hole metric, computes its standard observables, and
constrains its parameters with observational data (QPOs, EHT shadow size).
Read `DESIGN_NOTES.md` for the reasoning behind the decisions listed here.

## Fixed decisions (do not change without the owner's approval)

| Item | Decision |
|---|---|
| Project / repo name | BlackPy |
| PyPI distribution name | `blackpy-gr` (`blackpy` is taken) |
| Import name | `import blackpy` |
| License | MIT, `Copyright (c) 2026 Pahlavon Yovqochev` |
| Python | 3.10+ |
| Numerical core | JAX with `jax_enable_x64 = True` always (`jit`, `vmap`, autodiff) |
| Symbolic layer | sympy, only for metric input and geometry checks |
| Derivatives | Automatic differentiation only. **No finite differences** in the numerical core |
| Samplers | `sampler="emcee"` or `sampler="dynesty"`; both must work |
| Dependencies | numpy, scipy, matplotlib, corner, pyyaml, emcee, dynesty, jax, sympy. Anything else must be justified in `DESIGN_NOTES.md` |
| Tooling | pyproject.toml (hatchling), pytest, ruff, GitHub Actions CI, MkDocs |
| Language | English everywhere (code, comments, docs, commits) |
| Version | 0.2.0 (single source: `src/blackpy/_version.py`) |

## Design principle: "symbolic in, numeric core"

Two metric classes (both subclasses of `blackpy.metric.Metric`):

- `StaticSpherical`: `ds^2 = -f dt^2 + dr^2/g + h dOmega^2`;
- `StationaryAxisymmetric` (since v0.2):
  `ds^2 = g_tt dt^2 + 2 g_tphi dt dphi + g_phiphi dphi^2 + g_rr dr^2 + g_thth dtheta^2`
  with components functions of `(r, theta)`.

A metric is defined either

1. symbolically (`from_sympy`): sympy runs the geometry checks, then the
   expressions are converted once with `sympy.lambdify(..., modules="jax")`; or
2. numerically (callables): any JAX-traceable callable, including
   interpolants of numerical solutions (`blackpy.metric.interpolated.spline`).
   Symbolic checks are skipped with a clear message; numerical (autodiff)
   versions are used instead.

Every observable is computed numerically by one shared toolbox
(`blackpy/_numerics.py`): forward-mode autodiff derivatives, scan +
safeguarded-Newton root finding, Gauss-Legendre quadrature.

## Code structure rules

- Each observable has a pure, JAX-traceable **core function**
  `xxx_core(metric, p, *scalar_args, **static_options)` where `p` is the 1-D
  parameter array, and a thin **public wrapper** that calls
  `metric.evaluate(core, params, *args, **options)`. `evaluate` handles
  keyword parameters, broadcasting, `vmap`, `jit` and caching. New
  observables must follow this pattern; likelihoods call core functions.
- Root-finding outputs have fixed length (NaN padded) so they stay
  `jit`/`vmap`-compatible. "Not found" is NaN in vectorised functions and an
  explicit result object or message in scalar "list everything" functions.
  A horizonless configuration is a result, not an error.
- **Static and rotating paths.** Core functions dispatch on
  `metric.is_rotating_class`: rotating metrics go to the general formulas in
  `orbits/axisymmetric.py`, `light/rotating.py`, `geometry/axisymmetric.py`;
  static metrics keep their specialised v0.1 formulas, which must not change.
  `StaticSpherical.as_axisymmetric()` sends a static metric through the
  general path; every general formula must be tested against the static one
  this way. Orbit cores take `direction` (+1 prograde, -1 retrograde) as a
  static option; public functions take `direction="prograde" | "retrograde"`.
- A rotating metric must be reflection symmetric for equatorial orbits
  (`require_equatorial_orbits`) and have separable null geodesics for the
  shadow (`require_separable`). Refuse clearly otherwise; never return a
  number that silently assumes the property.
- Generated or deformed rotating metrics (`rotating_from_static`, Johannsen)
  are models: docstrings must say that the source has to be checked with
  `blackpy.geometry.effective_source`, never assumed.
- `CATALOGUE` holds the static metrics (v0.1 API, do not add rotating
  entries to it); rotating entries go to `ROTATING_CATALOGUE`.
- No global state apart from the x64 flag set in `blackpy/__init__.py`.
  Compiled-function caches live on the metric instance.
- Units: G = c = 1, lengths in units of M (Schwarzschild is `f = 1 - 2/r`).
  The physical mass enters only through `blackpy.constants.omega_to_hz`.
- Conventions: signature (-,+,+,+); `G^mu_nu = 8 pi T^mu_nu`;
  `rho = -T^t_t`, `p_r = T^r_r`, `p_t = T^theta_theta`.
- **Event horizon = outermost black hole horizon, not the cosmological
  horizon.** In code: the outermost root of `g` with `g' > 0` (or an extremal
  double root), **not** simply the outermost root: in de Sitter-like metrics
  the outermost root is the cosmological horizon. Use this wording in code,
  docstrings and docs.
- For rotating metrics the horizon functions use `g^rr` and the lapse
  squared; the surface `g^rr = 0` is the event horizon only if it does not
  depend on theta, and the code must report whether it does.
- Out-of-scope features (plasma, weak/strong lensing, QNMs, thermodynamics,
  S2 orbits, non-separable ray tracing, RZ/KRZ parametrisations) are stubs
  that raise `NotImplementedError` with a roadmap message. Do not implement
  them silently or half-way. Nodal precession stays refused for
  `StaticSpherical` metrics (it vanishes identically).

## Coding standards

- Type hints on public functions; NumPy-style docstrings with units and
  literature references.
- Small pure functions in the numerical core; clarity over cleverness.
- Error messages must tell the user what to do next.
- Every formula must have a source or a derivation in `docs/theory.md`, and
  a test in at least one known limit. Never invent formulas or citations; if
  a reference could not be checked, say so in `VERIFICATION.md`.
- Every catalogue metric is one file with its reference and at least one test.
- `ruff check` and `ruff format --check` must pass (line length 100).

## Data-integrity rules (registry under `src/blackpy/data/registry/`)

- **Never write a number from memory.** A value may enter a YAML file only
  if it was read from the original paper text that was actually accessed in
  that session; record the exact location (table/figure/section) and how it
  was accessed in the `location` / `source_access` fields.
- If the original source cannot be accessed, the value stays `null`, a
  `TODO` entry is added to the file, and it is listed in `VERIFICATION.md`.
- A value taken from a secondary source must be labelled as such
  (`source_access`) and is not "verified".
- Every dataset carries `verified_by_human: false` until the project owner
  has checked it against the paper and flipped the flag by hand. Claude must
  never set it to `true`.
- Mass priors must record whether they are independent of QPO modelling
  (`independent: true/false`). Using a non-independent mass prior triggers a
  `CircularInferenceWarning`; do not silence it.
- Tests must not depend on unverified registry numbers (only on schema and
  loading). Reproductions of published constraints go to `examples/`.
- Published spins and fit results (`spin`, `published_fit`) are stored for
  comparison and to mark a source as rotating. They come from QPO fits in the
  Kerr metric (`independent: false`) and must never be used as priors
  automatically.

## Inference integrity guards (warnings, never silenced by default)

- circular mass priors (`CircularInferenceWarning`);
- posterior indistinguishable from prior (`PriorDominatedWarning`): do not
  call such a result a "constraint";
- unconverged chains (`ConvergenceWarning`);
- Savage-Dickey ratio evaluated at a prior boundary (boundary-reflected KDE,
  with a warning);
- a static metric fitted to a source with a published spin or a nodal QPO
  (`SpinIgnoredWarning`);
- spin strongly degenerate with another metric parameter
  (`DegeneracyWarning`; the posterior correlation is always recorded in
  `diagnostics["spin_correlations"]`).

## Commands (macOS / Linux; on Windows use `.venv\Scripts\` instead of `.venv/bin/`)

```bash
python3 -m venv .venv                      # once
.venv/bin/python -m pip install -e ".[dev,docs]"
.venv/bin/python -m pytest                 # full suite (includes slow tests, about 9 min)
.venv/bin/python -m pytest -m "not slow"   # quick suite (about 4.5 min)
.venv/bin/ruff check src tests examples && .venv/bin/ruff format --check src tests examples
.venv/bin/python -m mkdocs build --strict  # documentation
```

## Workflow

- Read `DESIGN_NOTES.md` section 7 first: it records what is done and what
  is left.
- Build bottom-up: constants, metric, geometry, orbits, light, data, infer, plot.
- Run the tests after each module; commit small, meaningful changes.
- Never report a test as passing without having run it.
- Do not push and do not create remotes; the owner does that.
