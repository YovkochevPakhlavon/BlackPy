# BlackPy

**Black hole metric in → observables → joint likelihood → posteriors, Bayes factors, plots.**

BlackPy is an open-source Python package that takes a black hole metric,
computes its standard observables, and constrains its parameters with
observational data (QPOs from X-ray binaries, the EHT shadow size of Sgr A\*).

Status: **0.1.0, first working version (MVP)**. Static, spherically symmetric
metrics only. The data registry has not yet been verified by a human; see
[`VERIFICATION.md`](VERIFICATION.md).

```python
import blackpy as bp
from blackpy.infer import Fit, Uniform

rn = bp.metric.ReissnerNordstrom()

bp.geometry.horizons(rn, Q=0.6).radii            # (0.2, 1.8)
bp.orbits.isco(rn, Q=0.6)                        # 5.42...
bp.light.shadow_radius(rn, Q=0.6)                # 4.86...
bp.orbits.QPO_MODELS["RP"].frequencies(rn, r=6.0, mass=10.0, Q=0.6)   # (nu_U, nu_L) in Hz

fit = Fit(rn, ["qpo/gro_j1655_40", "eht/sgra"], qpo_model="RP",
          priors={"Q": Uniform(0, 1), "r": Uniform(4, 12)}, sampler="emcee")
result = fit.run(seed=1)
print(result, fit.bayes_factor(null={"Q": 0.0}))
```

## Why

Hundreds of papers a year repeat the same pipeline by hand: propose a metric,
compute ISCO, photon sphere, shadow and epicyclic frequencies, then run an
MCMC against QPO tables copied from other papers. Existing tools cover pieces
of it (EinsteinPy: symbolic tensors; PyGRO: geodesic integration; Gradus.jl
and Skylight.jl: ray tracing; GALLIFRAY: VLBI parameter estimation), but not
the pipeline as a whole. BlackPy provides it as one tested package with a
curated, citable data registry.

## Features

- **Metrics**: define `f(r)`, `g(r)`, `h(r)` with sympy *or* as JAX-traceable
  callables (including spline interpolants of numerical solutions). Catalogue:
  Schwarzschild, Reissner–Nordström, Schwarzschild–de Sitter, Bardeen, Hayward.
- **Geometry**: all horizons with classification (event, inner, cosmological,
  extremal; horizonless cases are reported, not errors), Kretschmann scalar,
  effective stress-energy, energy conditions.
- **Orbits**: circular orbits, ISCO (all solutions), marginally bound orbit,
  epicyclic frequencies for general `(f, g, h)`, eleven QPO models.
- **Light**: photon spheres with stability, critical impact parameter, shadow
  for observers at infinity and for static observers at finite radius.
- **Data**: registry of QPO and shadow measurements with exact source
  locations and verification status.
- **Inference**: emcee and dynesty behind one interface, evidence ratios and
  Savage–Dickey Bayes factors, integrity warnings (circular mass priors,
  prior-dominated parameters, unconverged chains), reproducibility bundles.
- **Numerics**: JAX with 64-bit floats, automatic differentiation only,
  `jit` + `vmap`: about 0.006 ms per ISCO evaluation, vectorised over parameters.

## Installation

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"
```

On Windows PowerShell use `.venv\Scripts\python.exe` instead of
`.venv/bin/python`. The PyPI distribution will be called `blackpy-gr`
(import name `blackpy`).

## Usage

```bash
.venv/bin/python -m pytest                      # full test-suite
.venv/bin/python -m pytest -m "not slow"        # quick subset
.venv/bin/python examples/01_new_metric.py
.venv/bin/python examples/02_profile_plots.py
.venv/bin/python examples/03_rn_fit.py
.venv/bin/python examples/04_sgra_shadow_constraints.py
.venv/bin/python -m mkdocs serve                # documentation at http://127.0.0.1:8000
```

Documentation: [`docs/quickstart.md`](docs/quickstart.md),
[`docs/theory.md`](docs/theory.md) (every formula with its source),
[`docs/data.md`](docs/data.md).

## Limitations of version 0.1

- Spherical symmetry only: no rotation, hence no nodal precession and no
  low-frequency QPO modelling. Real black holes spin; constraints obtained
  with non-rotating metrics absorb the spin into the other parameters.
- The registry values were read from arXiv versions by an AI assistant and
  are flagged `verified_by_human: false`.
- See [`DESIGN_NOTES.md`](DESIGN_NOTES.md) for decisions, deviations and the roadmap.

## Citing

See [`CITATION.cff`](CITATION.cff). Please also cite the original data papers
listed by `blackpy.data.load(...).references`.

## License

MIT, Copyright (c) 2026 Pahlavon Yovqochev.
