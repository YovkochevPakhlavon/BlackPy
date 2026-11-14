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

## What the MVP (version 0.1) covers

Static, spherically symmetric metrics
\( ds^2 = -f(r)\,dt^2 + dr^2/g(r) + h(r)\,d\Omega^2 \) in units \(G = c = 1\):

| Module | Content |
|---|---|
| `blackpy.metric` | `StaticSpherical` from sympy expressions or JAX callables; catalogue: Schwarzschild, Reissner–Nordström, Schwarzschild–de Sitter, Bardeen, Hayward |
| `blackpy.geometry` | all horizons with classification, Kretschmann scalar, effective stress-energy, energy conditions |
| `blackpy.orbits` | circular orbits, ISCO, marginally bound orbit, epicyclic frequencies, QPO models (RP, ER, TD, WD and variants) |
| `blackpy.light` | photon spheres, critical impact parameter, shadow for observers at infinity and at finite radius |
| `blackpy.data` | registry of QPO and shadow data with sources and verification status |
| `blackpy.infer` | priors, likelihoods, emcee and dynesty, Bayes factors, integrity warnings, reproducibility bundles |
| `blackpy.plot` | profile plots, corner plots, LaTeX tables |

Not in the MVP (interfaces reserved, they raise `NotImplementedError`):
rotating metrics, parametrised metrics (RZ, KRZ, Johannsen), plasma, lensing,
quasinormal modes, thermodynamics, S2 orbit data, nodal precession.

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
