# BlackPy

**Black hole metric in → observables → joint likelihood → posteriors, Bayes factors, plots.**

BlackPy is an open-source Python package that takes a black hole metric
(static or rotating, given as sympy expressions or as numerical functions),
computes its standard observables, and constrains its parameters with
observational data: quasi-periodic oscillations (QPOs) of X-ray binaries and
the EHT shadow size of Sgr A\*.

Status: **0.2.0**. Static, spherically symmetric metrics and stationary,
axisymmetric (rotating) metrics. The data registry has not yet been verified
by a human; see [`VERIFICATION_v0.2.md`](VERIFICATION_v0.2.md).

## Installation

Requires Python 3.10 or newer.

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"
```

On Windows PowerShell use `.venv\Scripts\python.exe` instead of
`.venv/bin/python`. The PyPI distribution will be called `blackpy-gr`
(import name `blackpy`).

## Quickstart

```python
import math
import blackpy as bp
from blackpy.infer import Fit, Uniform

kerr = bp.metric.Kerr()
bp.geometry.horizons(kerr, a=0.6).radii                       # (0.2, 1.8)
bp.orbits.isco(kerr, a=0.6)                                   # 3.829...
bp.orbits.nodal_precession_frequency(kerr, r=6.0, mass=5.4, a=0.6)   # Hz
bp.light.shadow_observables(kerr, math.radians(60), a=0.6).mean_radius   # 5.12...
fit = Fit(kerr, "qpo/gro_j1655_40", qpo_model="RP", sampler="emcee",
          priors={"a": Uniform(0, 0.998), "r": Uniform(2, 20)})
print(fit.run(seed=1))                                        # a = 0.286 +- 0.003
```

A new metric needs no new code: `bp.StaticSpherical.from_sympy(1 - 2/r + k/r**3)`,
then `bp.metric.rotating_from_static(metric)` for its rotating counterpart.

## Documentation

| Page | Content |
|---|---|
| [`docs/how_it_works.md`](docs/how_it_works.md) | Architecture, numerical methods, how a fit runs, how to add a metric or a dataset |
| [`docs/tutorial.md`](docs/tutorial.md) | Step by step: from a new metric to a corner plot and a Bayes factor |
| [`docs/quickstart.md`](docs/quickstart.md) | Short recipes for each module |
| [`docs/theory.md`](docs/theory.md) | Every formula with its source or derivation |
| [`docs/data.md`](docs/data.md) | The data registry and its integrity rules |
| [`docs/api.md`](docs/api.md) | API reference |
| [`VERIFICATION_v0.2.md`](VERIFICATION_v0.2.md) | What was tested, with tolerances; formula sources; known limitations |
| [`DESIGN_NOTES.md`](DESIGN_NOTES.md), [`CHANGELOG.md`](CHANGELOG.md) | Decisions, roadmap, changes |

Examples are in [`examples/`](examples/) (`01` to `08`); build the docs with
`.venv/bin/python -m mkdocs serve`. Run the tests with
`.venv/bin/python -m pytest` (`-m "not slow"` for the quick subset).

## Citing and license

See [`CITATION.cff`](CITATION.cff); please also cite the original data
papers listed by `blackpy.data.load(...).references`. MIT license,
Copyright (c) 2026 Pahlavon Yovqochev.
