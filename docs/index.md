# BlackPy

**Metric in → observables → joint likelihood → posteriors, Bayes factors, plots.**

BlackPy takes a black hole metric, computes its standard observables, and
constrains its parameters with observational data: quasi-periodic
oscillations (QPOs) of X-ray binaries and the EHT shadow sizes of Sgr A\* and
M87\*.

Hundreds of papers a year repeat the same pipeline by hand: propose a metric,
compute the ISCO, photon sphere, shadow and epicyclic frequencies, then run an
MCMC against QPO tables copied from other papers. BlackPy turns this into one
tested package with a curated, citable data registry, so that results become
comparable across papers.

```text
import blackpy as bp

rn = bp.metric.ReissnerNordstrom()
bp.orbits.isco(rn, Q=0.5)                    # 5.607...
bp.light.shadow_radius(rn, Q=0.5)            # 4.968...
bp.geometry.horizons(rn, Q=0.5).radii        # [0.134, 1.866]
```

**New here? Start with the [user guide](user_guide.md)**: installation,
units, and how to use every calculation and every plot. Then
[How BlackPy works](how_it_works.md) explains the architecture and the
[tutorial](tutorial.md) walks through a complete analysis.

## What BlackPy covers

Static, spherically symmetric metrics
\( ds^2 = -f(r)\,dt^2 + dr^2/g(r) + h(r)\,d\Omega^2 \) and stationary,
axisymmetric (rotating) metrics
\( ds^2 = g_{tt}dt^2 + 2g_{t\phi}dt\,d\phi + g_{\phi\phi}d\phi^2 + g_{rr}dr^2 + g_{\theta\theta}d\theta^2 \),
in units \(G = c = 1\):

| Module | Content |
|---|---|
| `blackpy.metric` | `StaticSpherical` and `StationaryAxisymmetric` from sympy expressions or JAX callables; catalogue: Schwarzschild, Reissner–Nordström, Schwarzschild–de Sitter, Bardeen, Hayward, GMGHS, Simpson–Visser, Kerr, Kerr–Newman, Kerr–Sen, Johannsen, rotating Bardeen and Hayward; Rezzolla–Zhidenko and KRZ parametrisations; rotating generator `rotating_from_static` |
| `blackpy.geometry` | all horizons with classification, ergosurface, Kretschmann scalar, effective stress-energy and its type (vacuum, perfect fluid, anisotropic fluid, neither), energy conditions |
| `blackpy.orbits` | circular orbits (prograde and retrograde), ISCO, marginally bound orbit, radial and vertical epicyclic frequencies, nodal precession, QPO models (RP, ER, TD, WD and variants) |
| `blackpy.light` | photon spheres, critical impact parameters, shadows (circles for static metrics, contours and shape observables for separable rotating metrics), deviation from circularity in three named definitions, shadows in a plasma |
| `blackpy.lensing` | exact and weak deflection angles, strong deflection limit and relativistic-image observables (static metrics; predictions only) |
| `blackpy.disk` | radiative efficiency and Page–Thorne flux of thin disks (predictions only) |
| `blackpy.data` | registry of QPO and shadow data (Sgr A\*, M87\*) with sources, mass priors and verification status |
| `blackpy.infer` | priors, likelihoods (QPO pairs and triplets, shadow sizes of several sources, plasma), emcee and dynesty, Bayes factors, goodness of fit and posterior predictive checks, Fisher forecasts, integrity warnings, reproducibility bundles |
| `blackpy.plot` | profile plots, shadow contours, corner plots, residual plots, LaTeX tables |

What is not included, and why (ray tracing, lensing by rotating metrics,
quasinormal modes, thermodynamics, ...): see the [roadmap](roadmap.md).

## Design principle: symbolic in, numeric core

A metric is given either as **sympy expressions** (BlackPy runs symbolic
geometry checks, then converts them once to JAX functions) or as a **plain
JAX-traceable callable**, including interpolants of numerical solutions.
All observables are then computed numerically by one toolbox: automatic
differentiation (never finite differences), scan-plus-Newton root finding,
and quadrature, compiled with `jax.jit` and vectorised with `jax.vmap`.

## Data integrity

Every number in the registry was read from the cited paper, with its exact
location recorded. Values that could not be checked are `null`. An entry is
marked `verified_by_human: false` until the maintainer has confirmed it, and
BlackPy warns when such data are used; every shipped dataset has been
confirmed. See [Data registry](data.md).

## Installation

```text
python3 -m venv .venv
.venv/bin/python -m pip install blackpy-gr          # once released on PyPI
.venv/bin/python -m pip install -e ".[dev,docs]"    # from the repository
```

The distribution name is `blackpy-gr`; the import name is `blackpy`. Details,
including Windows: [user guide, installation](user_guide.md#installation).
