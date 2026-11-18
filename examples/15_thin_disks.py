"""Example 15: thin accretion disks (Novikov-Thorne / Page-Thorne).

1. Radiative efficiency eta = 1 - E(r_ISCO) for Kerr (both directions) and
   for static catalogue metrics.
2. The Page-Thorne flux and the effective-temperature profile for Kerr and
   for a Reissner-Nordstrom charge, with the radius of maximum flux.
3. What BlackPy refuses: a metric without an ISCO, and one with two
   possible inner edges (an explicit inner radius is then required).

No likelihood: comparing disks with spectra needs ray tracing of the disk
image, which BlackPy does not do.

Run:  python examples/15_thin_disks.py        (about 40 seconds)
"""

import math
from pathlib import Path

import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.errors import BlackPyError  # noqa: E402

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
kerr, rn = bp.metric.Kerr(), bp.metric.ReissnerNordstrom()

# ----------------------------------------------------------------- 1. efficiency
print("Radiative efficiency eta = 1 - E(r_ISCO)")
print(f"{'a':>8} {'r_ISCO (pro)':>13} {'eta (pro)':>10} {'r_ISCO (retro)':>15} {'eta (retro)':>12}")
for a in (0.0, 0.5, 0.9, 0.99, 0.999):
    pro = (float(bp.orbits.isco(kerr, a=a)), float(bp.disk.radiative_efficiency(kerr, a=a)))
    retro = (
        float(bp.orbits.isco(kerr, a=a, direction="retrograde")),
        float(bp.disk.radiative_efficiency(kerr, a=a, direction="retrograde")),
    )
    print(f"{a:8.3f} {pro[0]:13.4f} {pro[1]:10.4f} {retro[0]:15.4f} {retro[1]:12.4f}")
print(
    f"  limits: Schwarzschild 1 - sqrt(8/9) = {1 - math.sqrt(8 / 9):.4f}; extremal prograde "
    f"1 - 1/sqrt(3) = {1 - 1 / math.sqrt(3):.4f}"
)
for metric, params in (
    (rn, {"Q": 0.8}),
    (bp.metric.Bardeen(), {"q": 0.6}),
    (bp.metric.Hayward(), {"l": 0.6}),
):
    eta = float(bp.disk.radiative_efficiency(metric, **params))
    print(f"  {metric.name} {params}: eta = {eta:.4f}")

# ----------------------------------------------------------------- 2. flux profiles
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
print("\nPage-Thorne flux (units M^2 / Mdot): peak radius and peak value")
cases = [(kerr, {"a": a}, f"Kerr a = {a}") for a in (0.0, 0.5, 0.9)] + [
    (rn, {"Q": 0.8}, "RN Q = 0.8")
]
for metric, params, label in cases:
    profile = bp.disk.page_thorne_flux(metric, **params)
    print(
        f"  {label:<14} r_in = {profile.r_in:7.4f}  r_max = {profile.r_max:7.4f}  "
        f"F_max = {profile.flux_max:.4e}  eta = {profile.efficiency:.4f}"
    )
bp.plot.disk_flux(cases, newtonian=True, ax=ax1)  # (metric, params, label) triples
bp.plot.disk_flux(cases, quantity="temperature", ax=ax2)
fig.tight_layout()
fig.savefig(out / "thin_disk_flux.png", dpi=130)

# ----------------------------------------------------------------- 3. refusals
print("\nWhat is refused:")
for metric, params in ((rn, {"Q": 1.2}),):
    try:
        bp.disk.radiative_efficiency(metric, **params)
    except BlackPyError as error:
        print("  ", str(error).split(". ")[0] + ".")
bump = bp.StaticSpherical(
    lambda r, A=-0.05: 1 - 2 / r + A * jnp.exp(-(((r - 15.0) / 1.5) ** 2)), name="bump"
)
try:
    bp.disk.page_thorne_flux(bump)
except BlackPyError as error:
    print("  ", str(error).split(". ")[0] + ".")
profile = bp.disk.page_thorne_flux(bump, r_in=19.3)
print(
    f"   with r_in = 19.3 given explicitly: r_max = {profile.r_max:.3f}, "
    f"eta = {profile.efficiency:.4f}"
)
print(f"\nwrote {out / 'thin_disk_flux.png'}")
