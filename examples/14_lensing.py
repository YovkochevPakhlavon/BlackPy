"""Example 14: gravitational lensing, from weak deflection to the photon sphere.

1. The exact deflection angle of Schwarzschild against the weak-field series
   and against the strong deflection limit.
2. Strong-deflection coefficients of several metrics.
3. Predicted relativistic images for Sgr A* and M87*, with the masses and
   distances stored in the registry (checked by the maintainer).
4. Deflection in a homogeneous plasma.

These are **predictions**. Relativistic images have not been resolved for any
black hole, so nothing here is a constraint.

Run:  python examples/14_lensing.py        (about 20 seconds)
"""

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import blackpy as bp  # noqa: E402

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
ln, mt = bp.lensing, bp.metric
schwarzschild = mt.Schwarzschild()
b_c = 3.0 * math.sqrt(3.0)

# ----------------------------------------------------------------- 1. one curve, two limits
print("Schwarzschild deflection angle [rad]")
print(f"{'b/M':>12} {'exact':>12} {'4M/b':>12} {'+15pi/4 (M/b)^2':>16} {'strong limit':>13}")
for b in (1.0e4, 100.0, 20.0, 8.0, 6.0, 5.3, b_c * (1 + 1e-3), b_c * (1 + 1e-6)):
    exact = float(ln.deflection_angle(schwarzschild, b))
    second = 4.0 / b + 15.0 * math.pi / (4.0 * b**2)
    strong = float(ln.strong_deflection_angle(schwarzschild, b))
    print(f"{b:12.6f} {exact:12.6f} {4.0 / b:12.6f} {second:16.6f} {strong:13.6f}")
limit = ln.strong_deflection_limit(schwarzschild)
print(
    f"  strong deflection limit: alpha = -a ln(b/b_c - 1) + b with a = {float(limit.a_bar):.6f}, "
    f"b = {float(limit.b_bar):.6f}\n  (Bozza 2002: a = 1, b = ln[216 (7 - 4 sqrt 3)] - pi = "
    f"{math.log(216 * (7 - 4 * math.sqrt(3))) - math.pi:.6f}), b_c = {float(limit.b_c):.6f}\n"
)
fig, ax = plt.subplots(figsize=(6.4, 4.4))
bp.plot.deflection(schwarzschild, ax=ax)  # exact, weak-field and strong-deflection curves
ax.set_ylim(1e-2, 30)
fig.tight_layout()
fig.savefig(out / "deflection_angle.png", dpi=130)

# ----------------------------------------------------------------- 2. other metrics
print("Strong-deflection coefficients")
print(f"{'metric':<32} {'b_c/M':>9} {'a_bar':>9} {'b_bar':>9}")
cases = (
    (schwarzschild, {}),
    (mt.ReissnerNordstrom(), {"Q": 0.8}),
    (mt.ReissnerNordstrom(), {"Q": 1.0}),
    (mt.Bardeen(), {"q": 0.6}),
    (mt.Hayward(), {"l": 0.6}),
    (mt.RezzollaZhidenko(order=1), {"a1": 0.5}),
)
for metric, params in cases:
    c = ln.strong_deflection_limit(metric, **params)
    label = f"{metric.name.split(' (')[0]} {params or ''}"
    print(f"{label:<32} {float(c.b_c):9.5f} {float(c.a_bar):9.5f} {float(c.b_bar):9.5f}")
print("  (extremal Reissner-Nordstrom: Tsukamoto 2017 gives b_c = 4, a = sqrt 2 = 1.41421,")
print(
    f"   b = 2 sqrt 2 ln[4 (2 - sqrt 2)] - pi = {2 * math.sqrt(2) * math.log(4 * (2 - math.sqrt(2))) - math.pi:.5f})\n"  # noqa: E501
)

# ----------------------------------------------------------------- 3. Sgr A* and M87*
print("Predicted relativistic images (Schwarzschild and Reissner-Nordstrom, Q = 0.8)")
print(f"{'lens':<28} {'theta_g':>9} {'theta_inf':>10} {'s':>9} {'r [mag]':>8}   (microarcsec)")
for dataset, key in (
    ("eht/sgra", "vlti"),
    ("eht/sgra", "keck"),
    ("eht/m87", "stellar_dynamics"),
    ("eht/m87", "gas_dynamics"),
):
    for metric, params in ((schwarzschild, {}), (mt.ReissnerNordstrom(), {"Q": 0.8})):
        o = ln.strong_field_observables(metric, dataset=dataset, mass_prior=key, **params)
        label = f"{dataset} [{key}]" + (" RN" if params else "")
        print(
            f"{label:<28} {float(o.theta_g):9.3f} {float(o.theta_infinity):10.3f} "
            f"{float(o.separation):9.4f} {float(o.magnitude_difference):8.2f}"
        )
print(
    "  theta_inf is the shadow radius; the outermost relativistic image lies s outside it and\n"
    "  is r magnitudes brighter than all the others together. Resolving s needs 0.03\n"
    "  microarcseconds. For M87* the prediction depends on which mass is used (a factor 1.8).\n"
)

# ----------------------------------------------------------------- 4. plasma
plasma = bp.light.HomogeneousPlasma()
print("Deflection at b = 1000 M in a homogeneous plasma (omega_p / omega at infinity)")
for omega_p in (0.0, 0.5, 0.9, 0.99):
    exact = float(ln.deflection_angle(schwarzschild, 1.0e3, plasma=plasma, omega_p=omega_p))
    weak = (2.0 / 1.0e3) * (1.0 + 1.0 / (1.0 - omega_p**2))
    print(f"  omega_p/omega = {omega_p:4.2f}: exact {exact:.6e}, weak-field formula {weak:.6e}")
print(
    "  (Bisnovatyi-Kogan & Tsupko 2010, Eq. (38): alpha = (R_S/b) [1 + 1/(1 - omega_p^2/omega^2)])"
)
print(f"\nwrote {out / 'deflection_angle.png'}")
