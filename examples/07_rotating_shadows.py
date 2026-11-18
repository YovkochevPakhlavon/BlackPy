"""Example 7: shadows of rotating black holes and the Sgr A* shadow size.

1. Kerr shadow contours for several spins, seen edge-on and at 17 degrees
   (drawn with ``blackpy.plot.shadow``), with the size and shape observables
   (mean radius, area-equivalent radius, deviation from circularity in the
   three definitions of ``blackpy.light.deviation_from_circularity``).
2. How much the Kerr shadow size depends on spin and inclination, compared
   with the statement of the EHT Collaboration (2022, Sgr A* Paper VI,
   arXiv:2311.09484): "the radius ranges from ~4.8 GM/c^2 to ~5.2 GM/c^2",
   a "~8% range".
3. The Sgr A* bound on the charge of a Kerr-Newman black hole, with spin and
   inclination as free parameters, compared with the non-rotating
   (Reissner-Nordstrom) bound of example 4.

The Sgr A* bounds are those of Vagnozzi et al. (2023) from the registry
entry ``eht/sgra``; they have not been verified by a human.

Run:  python examples/07_rotating_shadows.py        (about 20 seconds)
"""

import math
import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.errors import BlackPyWarning  # noqa: E402
from blackpy.infer import Fit, Uniform  # noqa: E402

# BLACKPY_EXAMPLES_QUICK=1 (CI, scripts/check_windows.ps1 -ExamplesQuick) reduces the sampler
# settings so that the script finishes in about a minute; the numbers are then less precise.
# Unset, the script runs exactly as documented.
QUICK = os.environ.get("BLACKPY_EXAMPLES_QUICK") == "1"


def steps(n: int) -> int:
    """Number of emcee steps (or posterior predictive draws): n, or n / 10 in quick mode."""
    return max(n // 10, 200) if QUICK else n


def live(n: int) -> int:
    """Number of dynesty live points: n, or 100 in quick mode."""
    return 100 if QUICK else n


out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
kerr = bp.metric.Kerr()

# ----------------------------------------------------------------- 1. contours
fig, axes = plt.subplots(1, 2, figsize=(10.5, 5.2), sharey=True)
print("Kerr shadow observables (lengths in units of M)")
print(
    f"{'i [deg]':>8} {'a':>6} {'mean R':>8} {'area R':>8} {'Delta C':>8} {'centre':>8} {'width':>8}"
)
SPINS = [0.0, 0.5, 0.9, 0.998]
for ax, inclination in zip(axes, (90.0, 17.0), strict=True):
    # The definition of Delta C is always given explicitly: the three definitions in use
    # differ by factors of 2 to 4 (blackpy.light.deviation_from_circularity).
    bp.plot.shadow(
        kerr, math.radians(inclination), a=SPINS, annotate_delta_c="fractional_rms", ax=ax
    )
    for a in SPINS:
        obs = bp.light.shadow_observables(kerr, math.radians(inclination), a=a)
        width = float(obs.alpha_max - obs.alpha_min)
        print(
            f"{inclination:8.0f} {a:6.3f} {float(obs.mean_radius):8.4f} "
            f"{float(obs.area_radius):8.4f} {float(obs.delta_c):8.4f} "
            f"{float(obs.centroid):8.4f} {width:8.4f}"
        )
    ax.set_title(f"Kerr, inclination {inclination:.0f} deg")
axes[1].set_ylabel("")
fig.tight_layout()
fig.savefig(out / "kerr_shadows.png", dpi=130)

print("\nDeviation from circularity of Kerr, a = 0.95, in the three definitions:")
print(f"{'i [deg]':>8} " + " ".join(f"{d:>15}" for d in bp.light.DELTA_C_DEFINITIONS))
for inclination in (17.0, 45.0, 54.0, 90.0):
    row = [
        float(
            bp.light.deviation_from_circularity(
                kerr, math.radians(inclination), a=0.95, definition=d
            )
        )
        for d in bp.light.DELTA_C_DEFINITIONS
    ]
    print(f"{inclination:8.0f} " + " ".join(f"{v:15.4f}" for v in row))
print(
    "  'bambi2019' (4 x the fractional rms) is the quantity whose 0.1 contour reproduces\n"
    "  Fig. 1 of Bambi et al. (2019): it crosses 0.1 at i = 54 deg for a = 0.95."
)

# Recentred contours against the Sgr A* band (the published bound is on a radius).
fig2, ax2 = plt.subplots(figsize=(5.4, 5.4))
bp.plot.shadow(
    kerr,
    math.radians(60.0),
    a=[0.0, 0.9, 0.998],
    band="eht/sgra",
    band_level="1sigma",
    recentre=True,
    reference_circle=True,
    ax=ax2,
)
ax2.set_title("Kerr at 60 deg and the Sgr A* shadow-size band")
fig2.tight_layout()
fig2.savefig(out / "kerr_shadows_sgra_band.png", dpi=130)

# ----------------------------------------------------------------- 2. size vs spin, inclination
spins = np.linspace(0.0, 0.998, 81)[:, None]
inclinations = np.radians(np.linspace(0.5, 90.0, 61))[None, :]
obs = bp.light.shadow_observables(kerr, inclinations, a=spins)
mean = np.asarray(obs.mean_radius)
spread = (mean.max() - mean.min()) / mean.max()
i_min = np.unravel_index(np.argmin(mean), mean.shape)
a_min, inc_min = spins[i_min[0], 0], math.degrees(inclinations[0, i_min[1]])
print(
    f"\nMean radius over 0 <= a <= 0.998 and all inclinations: {mean.min():.4f} to "
    f"{mean.max():.4f} M, a {100 * spread:.1f}% range."
    f"\n  smallest at a = {a_min:.3f}, i = {inc_min:.1f} deg (pole-on); "
    f"edge-on at the same spin: {mean[-1, -1]:.4f} M"
    "\n  EHT (2022): 'the radius ranges from ~4.8 to ~5.2 GM/c^2', '~8% range'."
    f"\n  largest deviation from circularity: {float(np.max(np.asarray(obs.delta_c))):.4f} "
    "(definition 'fractional_rms'; the 'jp_asymmetry' definition is twice as large)"
)
sgra = bp.data.load("eht/sgra")
low1, high1 = sgra.bounds["1sigma"]
inside = (mean >= low1) & (mean <= high1)
print(
    f"  Sgr A* 1-sigma bound {low1} <= r_sh/M <= {high1}: satisfied by {100 * inside.mean():.0f}% "
    "of the (a, i) grid -> the shadow size alone does not constrain the Kerr spin."
)

# ----------------------------------------------------------------- 3. Kerr-Newman charge
kerr_newman = bp.metric.KerrNewman()
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BlackPyWarning)  # unverified data; prior-dominated spin
    fit = Fit(
        kerr_newman,
        sgra,
        priors={"a": Uniform(0.0, 0.998), "Q": Uniform(0.0, 1.0)},  # inclination: isotropic
        sampler="emcee",
        nsteps=steps(4000),
        max_steps=steps(16000),
    )
    result = fit.run(seed=4)
    static = Fit(
        bp.metric.ReissnerNordstrom(), sgra, priors={"Q": Uniform(0.0, 1.0)}, nsteps=steps(3000)
    )
    static_result = static.run(seed=0)
q68, q95 = np.percentile(result["Q"], [68.27, 95.45])
s68, s95 = np.percentile(static_result["Q"], [68.27, 95.45])
print(
    "\nSgr A* shadow size (Gaussian likelihood on the mean radius), flat priors on a and Q "
    "inside a^2 + Q^2 <= 1,\nisotropic inclination:"
    f"\n  Kerr-Newman:        Q < {q68:.2f} (68%), Q < {q95:.2f} (95%);  parameters "
    f"{result.param_names}"
    f"\n  Reissner-Nordstrom: Q < {s68:.2f} (68%), Q < {s95:.2f} (95%)   (example 4)"
    f"\n  posterior correlation of a and Q: {result.correlation('a', 'Q'):+.2f}; "
    f"spin posterior prior-dominated: {result.diagnostics['prior_dominated']['a']}"
    "\n  (The Kerr-Newman limit is tighter mostly because the prior volume a^2 + Q^2 <= 1 "
    "disfavours large Q.)"
)
print(f"\nwrote {out / 'kerr_shadows.png'}")
