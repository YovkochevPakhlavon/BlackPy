"""Example 10: black hole shadows in a plasma.

A cold, non-magnetised plasma has refractive index n^2 = 1 - omega_p^2/omega^2,
so radio waves are refracted and the shadow changes size. BlackPy follows
Perlick, Tsupko & Bisnovatyi-Kogan (2015) for static metrics and Perlick &
Tsupko (2017) for rotating ones.

1. Schwarzschild: shadow size against the plasma frequency for a homogeneous
   plasma and for power-law profiles, with the closed-form checks.
2. Kerr: contours for a plasma distribution of the separable class (the only
   class for which the shadow can be computed without ray tracing); a radial
   power law on Kerr is refused.
3. Inference: the Sgr A* shadow size with a Reissner-Nordstrom charge and a
   plasma amplitude as free parameters. Both shrink the shadow, so a single
   shadow size limits only a combination of the two.

The Sgr A* bound is the registry entry ``eht/sgra`` (not verified by a human).
For Sgr A* and M87* at 230 GHz the plasma frequency near the horizon is many
orders of magnitude below the observing frequency, so the effect is negligible
there; it matters at low radio frequencies.

Run:  python examples/10_plasma_shadows.py        (about 40 seconds)
"""

import math
import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.errors import BlackPyWarning, NotSeparableError, PlasmaPropagationError  # noqa: E402
from blackpy.infer import Fit, Uniform  # noqa: E402

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
lt = bp.light
schwarzschild, kerr = bp.metric.Schwarzschild(), bp.metric.Kerr()

# ----------------------------------------------------------------- 1. Schwarzschild
print("Schwarzschild shadow radius r_sh/M at infinity (vacuum: 3 sqrt(3) = 5.1962)")
print(f"{'omega_p/omega_0':>16} {'homogeneous':>12} {'k = 1':>9} {'k = 3/2':>9} {'k = 2':>9}")
models = [lt.HomogeneousPlasma(), *(lt.PowerLawPlasma(k=k) for k in (1.0, 1.5, 2.0))]
for omega_p in (0.0, 0.2, 0.5, 0.8, 1.5, 3.0):
    row = []
    for plasma in models:
        try:
            row.append(f"{float(lt.shadow_in_plasma(schwarzschild, plasma, omega_p=omega_p)):9.4f}")
        except PlasmaPropagationError:
            row.append("   opaque")
    print(f"{omega_p:16.2f} {row[0]:>12} " + " ".join(row[1:]))
print(
    "  power law: omega_p is defined at r = M (beta_0 = omega_p^2 in Perlick et al. 2015);\n"
    "  k = 2 is exact: r_sh = sqrt(27 - omega_p^2), e.g. "
    f"{math.sqrt(27 - 0.8**2):.4f} at 0.8. A homogeneous plasma\n  makes the shadow of a distant "
    "observer larger (the observer is inside the medium);\n  a plasma that thins out makes it "
    "smaller."
)

# ----------------------------------------------------------------- 2. Kerr
# omega_p^2 = (f_r(r) + f_theta(theta)) / (r^2 + a^2 cos^2 theta), Perlick & Tsupko (2017),
# Eq. (26), here with f_r = c sqrt(r) and f_theta = 0 (density ~ r^(-3/2) far away).
separable = lt.SeparablePlasma(lambda r, c=0.0: c * jnp.sqrt(r))
view = lt.with_plasma(kerr, separable)
fig, ax = plt.subplots(figsize=(5.6, 5.6))
bp.plot.shadow(
    view, math.pi / 2, a=0.9, c=[0.0, 2.0, 4.0, 6.0], annotate_delta_c="fractional_rms", ax=ax
)
ax.set_title("Kerr, a = 0.9, edge-on, in a separable plasma")
fig.tight_layout()
fig.savefig(out / "kerr_plasma_shadows.png", dpi=130)
print("\nKerr, a = 0.9, edge-on, omega_p^2 = c sqrt(r) / (r^2 + a^2 cos^2 theta):")
for c in (0.0, 2.0, 4.0, 6.0):
    obs = lt.shadow_observables(view, math.pi / 2, a=0.9, c=c)
    print(
        f"  c = {c:3.1f}: mean radius {float(obs.mean_radius):.4f}, "
        f"Delta C {float(obs.delta_c):.4f}"
    )
try:
    lt.shadow_in_plasma(kerr, lt.PowerLawPlasma(), inclination=1.0, a=0.9, omega_p=0.5)
except NotSeparableError as error:
    print("  a radial power law on Kerr is refused:", str(error).split(". ")[0] + ".")

# ----------------------------------------------------------------- 3. inference
rn = bp.metric.ReissnerNordstrom()
plasma = lt.PowerLawPlasma(k=1.5)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fit = Fit(
        rn,
        "eht/sgra",
        priors={"Q": Uniform(0.0, 1.0), "omega_p": Uniform(0.0, 3.0)},
        plasma=plasma,
        nsteps=4000,
        max_steps=16000,
    )
    result = fit.run(seed=10)
    vacuum = Fit(rn, "eht/sgra", priors={"Q": Uniform(0.0, 1.0)}, nsteps=3000).run(seed=10)
names = sorted({w.category.__name__ for w in caught if issubclass(w.category, BlackPyWarning)})
print("\nSgr A* shadow size, Reissner-Nordstrom charge Q and plasma amplitude omega_p (k = 3/2):")
print(result)
print(f"  warnings: {names}")
q95, q95_vacuum = np.percentile(result["Q"], 95), np.percentile(vacuum["Q"], 95)
print(
    f"  95% upper limit on Q: {q95:.2f} with a free plasma, {q95_vacuum:.2f} in vacuum. A plasma "
    "can only make the\n  shadow smaller, as the charge does, so allowing it tightens the limit "
    "on Q through the prior\n  volume alone; the plasma amplitude itself is limited only by the "
    "same single number.\n  One shadow size cannot separate the two: fix the plasma from "
    "independent information."
)
print(f"\nwrote {out / 'kerr_plasma_shadows.png'}")
