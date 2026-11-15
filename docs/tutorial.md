# Tutorial: from a new metric to a corner plot and a Bayes factor

This page follows one example from start to finish. The complete script is
`examples/08_tutorial_new_metric.py`; every output shown here was produced by
it (BlackPy 0.2.0, about 40 seconds).

Units: \(G = c = 1\), lengths in units of the black hole mass \(M\).

## Step 1. Define a static metric

Take \(f = g = 1 - 2/r + \beta/r^2\) with \(\beta\) of either sign
(\(\beta = Q^2 > 0\) is Reissner–Nordström).

```python
import sympy as sp
import blackpy as bp

r, beta = sp.symbols("r beta", real=True)
static = bp.StaticSpherical.from_sympy(
    1 - 2 / r + beta / r**2,
    parameters=[bp.metric.Parameter("beta", 0.0, latex=r"\beta", role="deformation")],
    name="1/r^2 metric",
    asymptotically_flat=True,
)
print(bp.geometry.report(static, beta=-0.5))
```

```text
1/r^2 metric: event horizon at r = 2.22474487139.
  r = 2.22474487139  (event)
Asymptotically flat: True
Vacuum solution (G^mu_nu = 0): False
  rho = beta/(8*pi*r**4)
  p_r = -beta/(8*pi*r**4)
  p_t = beta/(8*pi*r**4)
NEC: VIOLATED for r in [2.22475, 222.475]
...
```

Because the metric was given symbolically, the report contains the exact
effective stress-energy. For \(\beta < 0\) the energy density is negative
and all energy conditions are violated: this is information you need before
interpreting any fit.

`Parameter(..., role="deformation")` is optional. It tells the inference
guards what the parameter means; a plain `parameters={"beta": 0.0}` works
too.

## Step 2. Make it rotate, and check what you got

The sources in the registry rotate, so a static metric cannot fit their
low-frequency QPO. `rotating_from_static` applies the Azreg-Aïnou (2014)
version of the Newman–Janis algorithm:

```python
metric = bp.metric.rotating_from_static(static, name="rotating 1/r^2 metric")
print(metric.is_reflection_symmetric, metric.is_separable)      # True True
print(bp.geometry.horizons(metric, a=0.5, beta=-0.5))
print(bp.geometry.effective_source(metric, a=0.5, beta=-0.5).summary())
```

```text
rotating 1/r^2 metric: event horizon at r = 2.11803398875.
rotating 1/r^2 metric: the horizon radius r = 2.11803398875 is independent of theta.
Effective source: anisotropic fluid (grid points by type: anisotropic fluid: 168)
NEC: VIOLATED at 168 of 168 grid points (r from 2.22394 to 66.7181)
...
```

The generated metric is a **model**. The algorithm does not guarantee that
it solves any field equations, so BlackPy computes its effective source
numerically (full 4D curvature by automatic differentiation) and classifies
it. Here it is an anisotropic fluid that violates the energy conditions,
as the static seed does for \(\beta < 0\).

Two properties decide what can be computed:

- **reflection symmetry** (\(\theta \to \pi - \theta\)) is required for
  equatorial orbits and epicyclic frequencies;
- **separable null geodesics** are required for the analytic shadow.

If one is missing, the corresponding functions raise an error that says so.

## Step 3. Compute observables

```python
import math

pars = dict(a=0.3, beta=-0.5)
bp.orbits.isco(metric, **pars)                                  # 5.7317
bp.orbits.frequencies_hz(metric, r=6.0, mass=5.4, **pars)       # nu_phi, nu_r, nu_theta in Hz
bp.orbits.nodal_precession_frequency(metric, r=6.0, mass=5.4, **pars)   # 15.74 Hz
bp.light.shadow_observables(metric, math.radians(60.0), **pars) # mean radius 5.5817, Delta C 0.0009
```

```text
ISCO: 5.7317 M (Kerr with the same spin: 4.9786 M)
at r = 6 M, M = 5.4 Msun: nu_phi = 415.0 Hz, nu_r = 85.7 Hz, nu_theta = 399.2 Hz, nodal precession = 15.74 Hz
shadow at 60 deg: mean radius 5.5817 M, Delta C = 0.0009
```

All functions accept arrays for any argument, for example
`bp.orbits.isco(metric, a=np.linspace(0, 0.9, 100), beta=-0.5)`. Use
`direction="retrograde"` for counter-rotating orbits.

## Step 4. Fit data

GRO J1655-40 has three simultaneous QPOs (Motta et al. 2014). In the
relativistic precession model (`"RP"`) they are the orbital, periastron
precession and nodal precession frequencies at one radius.

```python
import warnings
from blackpy.infer import Fit, Uniform

data = bp.data.load("qpo/gro_j1655_40")
priors = {"a": Uniform(0.0, 0.998), "beta": Uniform(-2.0, 0.9), "r": Uniform(2.0, 20.0)}
warnings.simplefilter("always")          # read the integrity warnings; do not silence them
fit = Fit(metric, data, qpo_model="RP", priors=priors, sampler="emcee",
          nsteps=8000, max_steps=64000)
result = fit.run(seed=1)
print(result)
```

What the priors mean:

- `a`, `beta`: metric parameters with a prior are sampled; without one they
  are fixed at their default (or at `fixed={...}`);
- `r`: orbital radius of the QPO source in units of \(M\);
- the mass is not listed, so it takes the registry's *independent*
  (dynamical) mass, \(5.4 \pm 0.3\,M_\odot\), as a Gaussian prior.

```text
FitResult (emcee, 8192 samples)
  a                        = 0.28433 +0.0247 -0.0218
  beta                     = 0.01739 +0.196 -0.235
  M_gro_j1655_40           = 5.3266 +0.304 -0.312   [posterior ~ prior: NOT a constraint]
  r_gro_j1655_40           = 5.656 +0.306 -0.271
  converged: True
  corr(a, beta) = -0.992 (rank -0.991)   [strongly degenerate]
```

BlackPy also prints three warnings. Each one changes how the numbers should
be read:

- `UnverifiedDataWarning`: the registry values have not been checked by a
  human against the paper;
- `DegeneracyWarning`: spin and \(\beta\) are almost perfectly correlated
  (\(-0.99\)). Three frequencies determine three numbers; with four unknowns
  the data fix one *combination* of \(a\) and \(\beta\), and the width along
  it comes from the mass prior. In the Kerr metric the same data give
  \(a = 0.286 \pm 0.003\); here the marginal spin uncertainty is about eight
  times larger;
- the mass line is marked "posterior ~ prior": the QPOs add nothing to the
  mass once \(\beta\) is free.

## Step 5. Compare with Kerr and plot

```python
print(fit.bayes_factor(null={"beta": 0.0}))          # Savage-Dickey ratio from the chain

nested = Fit(metric, data, qpo_model="RP", priors=priors, sampler="dynesty", nlive=400)
nested.run(seed=2)
print(nested.bayes_factor(null={"beta": 0.0}))       # evidence ratio from two nested runs

fig = bp.plot.corner(result, truths={"beta": 0.0})
fig.savefig("tutorial_corner.png", dpi=130)
result.save("tutorial_fit")                          # reproducibility bundle (.npz)
```

```text
ln B_10 = -1.68 +- 0.01 (Savage-Dickey; null: {'beta': 0.0}): positive evidence for the null model
ln B_10 = -1.37 +- 0.25 (evidence ratio; null: {'beta': 0.0}): positive evidence for the null model
```

\(\ln B_{10} < 0\) favours the null model, here Kerr (\(\beta = 0\)). The two
methods are independent (a density ratio at \(\beta = 0\) from the emcee
chain, and the ratio of two nested-sampling evidences) and agree within the
quoted uncertainty. The preference is mild: the extra parameter is not
needed, but it is not excluded either. The Bayes factor depends on the prior
range of \(\beta\).

The corner plot shows the narrow, tilted band in the \((a, \beta)\) panel,
which is the degeneracy the warning reported. Quote the joint constraint,
not the two marginal intervals.

## What to change for your own work

| You want to | Change |
|---|---|
| another metric | Step 1; or pass five components to `StationaryAxisymmetric.from_sympy` |
| a metric known only numerically | `bp.metric.interpolated.spline` and the callable constructor |
| another QPO model | `qpo_model="ER"`, `"TD"`, `"WD"`, ... (`bp.orbits.QPO_MODELS`) |
| more sources | `data=["qpo/gro_j1655_40", "qpo/xte_j1859_226"]`; metric parameters are shared |
| shadow data | add `"eht/sgra"`; a rotating metric then gets an `inclination` nuisance parameter |
| your own data | a YAML file loaded with `bp.data.load_file`, see [Data registry](data.md) |
| retrograde orbits | `direction="retrograde"`, or a prior on `a` that includes negative values |
