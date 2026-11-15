# BlackPy

**Metric in → observables → joint likelihood → posteriors, Bayes factors, plots.**

BlackPy takes a black hole metric, computes its standard observables, and
constrains its parameters with observational data: quasi-periodic
oscillations (QPOs) of X-ray binaries and the EHT shadow size of Sgr A\*.

Hundreds of papers a year repeat the same pipeline by hand: propose a metric,
compute the ISCO, photon sphere, shadow and epicyclic frequencies, then run an
MCMC against QPO tables copied from other papers. BlackPy turns this into one
tested package with a curated, citable data registry, so that results become
comparable across papers.

```python
import blackpy as bp

rn = bp.metric.ReissnerNordstrom()
bp.orbits.isco(rn, Q=0.5)                    # 5.607...
bp.light.shadow_radius(rn, Q=0.5)            # 4.968...
bp.geometry.horizons(rn, Q=0.5).radii        # (0.134, 1.866)
```

New here? Read [How BlackPy works](how_it_works.md), then follow the
[tutorial](tutorial.md).

## What version 0.2 covers

Static, spherically symmetric metrics
\( ds^2 = -f(r)\,dt^2 + dr^2/g(r) + h(r)\,d\Omega^2 \) and stationary,
axisymmetric (rotating) metrics
\( ds^2 = g_{tt}dt^2 + 2g_{t\phi}dt\,d\phi + g_{\phi\phi}d\phi^2 + g_{rr}dr^2 + g_{\theta\theta}d\theta^2 \),
in units \(G = c = 1\):

| Module | Content |
|---|---|
| `blackpy.metric` | `StaticSpherical` and `StationaryAxisymmetric` from sympy expressions or JAX callables; catalogue: Schwarzschild, Reissner–Nordström, Schwarzschild–de Sitter, Bardeen, Hayward, Kerr, Kerr–Newman, Johannsen; rotating generator `rotating_from_static` |
| `blackpy.geometry` | all horizons with classification, ergosurface, Kretschmann scalar, effective stress-energy and its type (vacuum, perfect fluid, anisotropic fluid, neither), energy conditions |
| `blackpy.orbits` | circular orbits (prograde and retrograde), ISCO, marginally bound orbit, radial and vertical epicyclic frequencies, nodal precession, QPO models (RP, ER, TD, WD and variants) |
| `blackpy.light` | photon spheres, critical impact parameters, shadows: circles for static metrics, contours and shape observables for rotating metrics with separable null geodesics |
| `blackpy.data` | registry of QPO and shadow data with sources and verification status |
| `blackpy.infer` | priors, likelihoods (QPO pairs and triplets, shadow size), emcee and dynesty, Bayes factors, integrity warnings, reproducibility bundles |
| `blackpy.plot` | profile plots, corner plots, LaTeX tables |

Not in version 0.2 (interfaces reserved, they raise `NotImplementedError`):
ray tracing for non-separable metrics, the RZ and KRZ parametrisations,
plasma, lensing, quasinormal modes, thermodynamics, S2 orbit data.

## Design principle: symbolic in, numeric core

A metric is given either as **sympy expressions** (BlackPy runs symbolic
geometry checks, then converts them once to JAX functions) or as a **plain
JAX-traceable callable**, including interpolants of numerical solutions.
All observables are then computed numerically by one toolbox: automatic
differentiation (never finite differences), scan-plus-Newton root finding,
and quadrature, compiled with `jax.jit` and vectorised with `jax.vmap`.

## Data integrity

Every number in the registry was read from the cited paper, with its exact
location recorded. Values that could not be checked are `null`. All entries
are marked `verified_by_human: false` until the maintainer has confirmed them;
BlackPy warns when such data are used. See [Data registry](data.md).

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"
```

The distribution name is `blackpy-gr`; the import name is `blackpy`.
