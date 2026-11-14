"""Example 1: a new metric in ten lines.

A Schwarzschild black hole with a "tidal charge"-like term, defined once
symbolically; BlackPy then delivers the geometry checks and all observables.

Run:  python examples/01_new_metric.py
"""

import sympy as sp

import blackpy as bp

r, beta = sp.symbols("r beta", real=True)
metric = bp.StaticSpherical.from_sympy(
    1 - 2 / r + beta / r**2, parameters={"beta": 0.0}, name="tidal charge"
)
print(bp.geometry.report(metric, beta=-0.5))
print("r_ISCO =", float(bp.orbits.isco(metric, beta=-0.5)))
print("r_ph   =", float(bp.light.photon_sphere(metric, beta=-0.5)))
print("r_sh   =", float(bp.light.shadow_radius(metric, beta=-0.5)))
print(
    "QPO (RP model) at r = 8 M, M = 10 Msun [Hz]:",
    [float(v) for v in bp.orbits.QPO_MODELS["RP"].frequencies(metric, 8.0, 10.0, beta=-0.5)],
)

# The same metric as a plain JAX-traceable callable (no sympy): symbolic checks are
# skipped with a clear message, every observable is still available.
numeric = bp.StaticSpherical(
    lambda r, beta=0.0: 1 - 2 / r + beta / r**2, name="tidal charge (numeric)"
)
print("numeric path, r_ISCO =", float(bp.orbits.isco(numeric, beta=-0.5)))
