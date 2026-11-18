# BlackPy

**Black hole metric in → observables → joint likelihood → posteriors, Bayes factors, plots.**

**Start here: the [user guide](docs/user_guide.md)**: installation, units, and
how to use every calculation and every plot, with outputs.

BlackPy is an open-source Python package that takes a black hole metric
(static or rotating, given as sympy expressions or as numerical functions),
computes its standard observables, and constrains its parameters with
observational data: quasi-periodic oscillations (QPOs) of X-ray binaries and
the EHT shadow sizes of Sgr A\* and M87\*.

Status: **1.0.0rc1** (release candidate). The public API is frozen: within
1.x no public name disappears and no public behaviour changes without a
deprecation period. The data registry has not yet been verified by a human;
see [`verification/`](verification/README.md).

## Installation

Requires Python 3.10 or newer. Once released on PyPI:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install blackpy-gr
```

From the repository (editable, with test and documentation tools):

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"
```

On Windows PowerShell use `.venv\Scripts\python.exe` instead of
`.venv/bin/python`; `scripts\check_windows.ps1` installs BlackPy and runs every
check in one command. The distribution is called `blackpy-gr`; the import
name is `blackpy`.

## Quickstart

```python
import math

import blackpy as bp
from blackpy.infer import Fit, Uniform

kerr = bp.metric.Kerr()
print([round(x, 6) for x in bp.geometry.horizons(kerr, a=0.6).radii])             # [0.2, 1.8]
print(float(bp.orbits.isco(kerr, a=0.6)))                                       # 3.829...
print(float(bp.orbits.nodal_precession_frequency(kerr, r=6.0, mass=5.4, a=0.6)))  # Hz
print(float(bp.light.shadow_observables(kerr, math.radians(60), a=0.6).mean_radius))  # 5.12...

# warns: UnverifiedDataWarning
fit = Fit(kerr, "qpo/gro_j1655_40", qpo_model="RP", nsteps=2000,
          priors={"a": Uniform(0, 0.998), "M": Uniform(2, 20), "r": Uniform(2, 20)})
result = fit.run(seed=1)
print(result)                     # a = 0.286 +- 0.003, convergence, fit quality
print(result.goodness_of_fit())   # pulls, chi2, degrees of freedom
```

Angles are in radians (`math.radians(60)`); a quantity in degrees has `_deg`
in its name. A new metric needs no new code:
`bp.StaticSpherical.from_sympy(1 - 2/r + k/r**3)`, then
`bp.metric.rotating_from_static(metric)` for its rotating counterpart.

## What BlackPy covers

| Module | Content |
|---|---|
| `blackpy.metric` | static and rotating metrics from sympy or callables; a catalogue of 13 metrics; Rezzolla–Zhidenko and KRZ; the rotating generator `rotating_from_static` |
| `blackpy.geometry` | horizons, ergosurface, curvature, effective source, energy conditions |
| `blackpy.orbits` | circular orbits, ISCO, epicyclic and nodal frequencies, QPO models |
| `blackpy.light` | photon spheres, shadows (also of separable rotating metrics), deviation from circularity, plasma |
| `blackpy.lensing`, `blackpy.disk` | deflection angles and strong lensing, thin disks (predictions only) |
| `blackpy.data` | registry of QPO and shadow data with sources and verification status |
| `blackpy.infer` | likelihoods, emcee and dynesty, Bayes factors, goodness of fit, Fisher forecasts, integrity warnings |
| `blackpy.plot` | profiles, shadow contours, corner and residual plots, LaTeX tables |

What is not included, and why: see the [roadmap](docs/roadmap.md).

## Documentation

| Page | Content |
|---|---|
| [`docs/user_guide.md`](docs/user_guide.md) | **Start here**: every calculation and plot, with outputs; units; warnings and errors; FAQ |
| [`docs/quickstart.md`](docs/quickstart.md) | BlackPy in five minutes |
| [`docs/tutorial.md`](docs/tutorial.md) | Step by step: from a new metric to a corner plot and a Bayes factor |
| [`docs/how_it_works.md`](docs/how_it_works.md) | Architecture and numerical methods |
| [`docs/theory.md`](docs/theory.md) | Every formula with its source or derivation |
| [`docs/data.md`](docs/data.md) | The data registry, its rules, and how to use your own data |
| [`docs/api.md`](docs/api.md) | API reference (the public API) |
| [`docs/roadmap.md`](docs/roadmap.md) | What is planned and what is out of scope |
| [`verification/`](verification/README.md) | What was tested, with tolerances; sources read; known limitations |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Installing, testing, the deprecation policy, adding a metric or a dataset |
| [`CHANGELOG.md`](CHANGELOG.md), [`DESIGN_NOTES.md`](DESIGN_NOTES.md) | Changes; design decisions |

Examples are in [`examples/`](examples/README.md) (`01` to `15`). Build the
documentation with `.venv/bin/python -m mkdocs serve`; run the tests with
`.venv/bin/python -m pytest -n 4 --dist loadfile` (add `-m "not slow"` for
the quick subset).

## Citing and license

See [`CITATION.cff`](CITATION.cff); please also cite the original data
papers listed by `blackpy.data.load(...).references`. MIT license,
Copyright (c) 2026 Pahlavon Yovqochev.
