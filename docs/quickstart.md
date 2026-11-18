# BlackPy in five minutes

A tour of the main functions, organised by topic. Every function, its
arguments, units, outputs and errors are explained in the
[user guide](user_guide.md); this page only shows what exists. Units:
\(G = c = 1\), lengths in units of \(M\), frequencies in Hz need a mass in
solar masses, **angles in radians** (a quantity in degrees has `_deg` in its
name).

## Metrics

```python
import math

import numpy as np
import sympy as sp

import blackpy as bp

rn = bp.metric.ReissnerNordstrom()  # static catalogue: bp.metric.CATALOGUE
kerr = bp.metric.Kerr()  # rotating catalogue: bp.metric.ROTATING_CATALOGUE

r, k = sp.symbols("r k", real=True)
mine = bp.StaticSpherical.from_sympy(1 - 2 / r + k / r**3, parameters={"k": 0.0})  # symbolic
numeric = bp.StaticSpherical(lambda r, alpha=0.0: 1 - 2 / r + alpha / r**2)  # a callable
rotating = bp.metric.rotating_from_static(bp.metric.Hayward())  # a model: check its source
print(mine.param_names, numeric.param_names, rotating.param_names)
```

```text title="Output"
('k',) ('alpha',) ('a', 'l')
```

## Geometry

```python
print([round(x, 6) for x in bp.geometry.horizons(rn, Q=0.6).radii])  # classified horizons
print(np.asarray(bp.geometry.event_horizon(rn, Q=np.array([0.5, 1.2]))))  # NaN: horizonless
print(bp.geometry.stress_energy_symbolic(rn)["rho"])
print(bp.geometry.energy_conditions(bp.metric.Hayward(), l=0.5).summary().splitlines()[0])
```

```text title="Output"
[0.2, 1.8]
[1.8660254       nan]
Q**2/(8*pi*r**4)
NEC: satisfied on [1.85464, 185.464]
```

## Orbits and QPOs

Every function accepts arrays for the parameters and is compiled on first use.

```python
print(
    round(float(bp.orbits.isco(rn, Q=0.5)), 4),
    np.round(np.asarray(bp.orbits.isco(kerr, a=np.array([0.0, 0.9]))), 4),
)
print(round(float(bp.orbits.isco(kerr, a=0.9, direction="retrograde")), 4))
freqs = bp.orbits.frequencies_hz(kerr, r=6.0, mass=5.4, a=0.3)  # (nu_phi, nu_r, nu_theta) in Hz
print([round(float(f), 1) for f in freqs])
print(round(float(bp.orbits.nodal_precession_frequency(kerr, r=6.0, mass=5.4, a=0.3)), 2))
print([round(float(f), 1) for f in bp.orbits.QPO_MODELS["RP"].frequencies(rn, 7.0, 10.0, Q=0.5)])
print(round(float(bp.orbits.rpm_kerr_solution(17.3, 298.0, 441.0).spin), 3))
```

```text title="Output"
5.6066 [6.     2.3209]
8.7174
[399.0, 157.5, 383.9]
15.08
[171.3, 95.7]
0.286
```

## Light, plasma, lensing, disks

```python
print(round(float(bp.light.shadow_radius(rn, Q=0.5)), 4))
obs = bp.light.shadow_observables(kerr, math.radians(60), a=0.9)  # radians!
print(round(float(obs.mean_radius), 4), round(float(obs.delta_c), 4))
print(
    round(
        float(
            bp.light.deviation_from_circularity(
                kerr, math.radians(60), a=0.9, definition="jp_asymmetry"
            )
        ),
        4,
    )
)
print(
    round(
        float(
            bp.light.shadow_in_plasma(
                bp.metric.Schwarzschild(), bp.light.HomogeneousPlasma(), omega_p=0.3
            )
        ),
        4,
    )
)
print(round(float(bp.lensing.deflection_angle(bp.metric.Schwarzschild(), 20.0)), 4))
print(round(float(bp.lensing.strong_deflection_limit(rn, Q=0.5).a_bar), 4))
print(round(float(bp.disk.radiative_efficiency(kerr, a=0.9)), 4))
```

```text title="Output"
4.9679
4.9995 0.0215
0.0448
5.3639
0.2361
1.0329
0.1558
```

## Data and inference

```python
# warns: UnverifiedDataWarning
from blackpy.infer import Fit, Uniform

print(bp.data.available())
fit = Fit(
    kerr,
    "qpo/gro_j1655_40",
    qpo_model="RP",
    priors={"a": Uniform(0, 0.998), "M": Uniform(2, 20), "r": Uniform(2, 20)},
)
print(fit.param_names)
```

```text title="Output (registry data)"
['eht/m87', 'eht/sgra', 'qpo/gro_j1655_40', 'qpo/grs_1915_105', 'qpo/h1743_322', 'qpo/m82_x1', 'qpo/xte_j1550_564', 'qpo/xte_j1859_226']
['a', 'M_gro_j1655_40', 'r_gro_j1655_40']
```

Then `result = fit.run(seed=1)`, `print(result)`,
`result.goodness_of_fit()`, `result.posterior_predictive(2000, seed=0)`,
`fit.bayes_factor(null={...})`, `result.save("my_fit")`. Shadow data with a
rotating metric add the inclination nuisance parameter `inclination_deg`
(degrees, isotropic prior); joint shadow fits need `shared=[...]` or
`per_source=[...]`; `bp.infer.forecast(...)` gives Fisher forecasts. See
[Inference](user_guide.md#9-inference-blackpyinfer).

## Plots

```python-skip
bp.plot.profile(rn, "isco", Q=np.linspace(0, 1, 200))
bp.plot.shadow(kerr, math.pi / 2, a=[0.0, 0.5, 0.9], annotate_delta_c="fractional_rms")
bp.plot.corner(result)
bp.plot.residuals(result)
print(bp.plot.latex_table(result))
```

Figures and every option: [Plots](user_guide.md#10-plots-blackpyplot).
Complete scripts are in the `examples/` folder; the [tutorial](tutorial.md)
walks through one of them.
