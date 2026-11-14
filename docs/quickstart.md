# Quickstart

All examples use geometric units \(G = c = 1\) with lengths in units of the
black hole mass \(M\); frequencies in hertz need the mass in solar masses.

## 1. Define a metric

From the catalogue:

```python
import blackpy as bp

rn = bp.metric.ReissnerNordstrom()          # parameter: Q (charge in units of M)
```

From sympy expressions (enables symbolic checks):

```python
import sympy as sp

r, beta = sp.symbols("r beta", real=True)
metric = bp.StaticSpherical.from_sympy(1 - 2/r + beta/r**2, parameters={"beta": 0.0})
print(bp.geometry.report(metric, beta=-0.5))
```

From a JAX-traceable callable (parameter names and defaults are read from the
signature). `g` and `h` default to `f` and `r**2`:

```python
metric = bp.StaticSpherical(lambda r, alpha=0.0: 1 - 2/r + alpha/r**2)
```

From a table of numbers (no closed form):

```python
import numpy as np

grid = np.geomspace(1.5, 400.0, 4000)
f_num = bp.metric.interpolated.spline(grid, 1 - 2/grid)        # your numerical solution
metric = bp.StaticSpherical(lambda r: f_num(r), r_range=(1.5, 400.0), asymptotically_flat=True)
```

## 2. Geometry

```python
bp.geometry.horizons(rn, Q=0.6)             # all horizons, classified
bp.geometry.event_horizon(rn, Q=np.linspace(0, 1.2, 50))   # NaN where horizonless
bp.geometry.stress_energy_symbolic(rn)      # {'rho': Q**2/(8*pi*r**4), ...}
print(bp.geometry.energy_conditions(bp.metric.Hayward(), l=0.5).summary())
```

## 3. Orbits and light

Every function accepts arrays for the parameters and is compiled on first use.

```python
Q = np.linspace(0, 1, 10_000)
bp.orbits.isco(rn, Q=Q)                              # ~0.006 ms per value after compilation
bp.orbits.marginally_bound_orbit(rn, Q=0.5)
bp.orbits.frequencies_hz(rn, r=7.0, mass=10.0, Q=0.5)    # (nu_phi, nu_r, nu_theta) in Hz
bp.orbits.QPO_MODELS["RP"].frequencies(rn, 7.0, 10.0, Q=0.5)   # (nu_U, nu_L)

bp.light.photon_spheres(rn, Q=1.05)                  # all, with stability
bp.light.shadow_radius(rn, Q=0.5)                    # observer at infinity
bp.light.shadow_angular_radius(bp.metric.SchwarzschildDeSitter(), r_obs=10.0, Lambda=0.01)
```

!!! note "Low-frequency QPOs"
    In a spherically symmetric spacetime \(\nu_\theta = \nu_\phi\), so the
    nodal precession frequency vanishes. `bp.orbits.nodal_precession_frequency`
    raises an explicit error, and fits ignore the low-frequency QPO of
    triplet datasets with a warning.

## 4. Data

```python
bp.data.list()                                # registry with verification status
gro = bp.data.load("qpo/gro_j1655_40")
gro.upper, gro.lower, gro.mass_prior          # values, uncertainties, sources, locations
sgra = bp.data.load("eht/sgra")
```

## 5. Inference

```python
from blackpy.infer import Fit, Uniform

fit = Fit(
    rn,
    data=["qpo/gro_j1655_40", "qpo/xte_j1550_564"],
    qpo_model="RP",
    priors={"Q": Uniform(0.0, 1.0), "r": Uniform(4.0, 12.0)},
    sampler="emcee",              # or "dynesty"
    nwalkers=32, nsteps=6000,
)
result = fit.run(seed=1)
print(result)
print(fit.bayes_factor(null={"Q": 0.0}))
result.save("my_fit")             # reproducibility bundle (.npz)
```

Each QPO source gets its own mass `M_<source>` (default prior: the independent
mass measurement from the registry) and orbital radius `r_<source>`; metric
parameters are shared. BlackPy warns about circular mass priors, about
parameters whose posterior equals the prior, and about unconverged chains.

## 6. Plots

```python
bp.plot.profile(rn, "isco", Q=np.linspace(0, 1, 200))
bp.plot.profile([(rn, {"Q": q}), (bp.metric.Bardeen(), {"q": q})], "shadow_radius")
bp.plot.corner(result)
print(bp.plot.latex_table(result))
```

Complete scripts are in the `examples/` folder.
