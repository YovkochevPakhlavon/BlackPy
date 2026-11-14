"""Example 2: profile plots of observables against the metric parameter.

Produces ``examples/output/profiles.png`` with four panels comparing the
Reissner-Nordstrom, Bardeen and Hayward black holes.

Run:  python examples/02_profile_plots.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

import blackpy as bp  # noqa: E402

rn, bardeen, hayward = bp.metric.ReissnerNordstrom(), bp.metric.Bardeen(), bp.metric.Hayward()
# Each parameter runs up to (just below) the value at which the horizon disappears.
models = [
    (rn, {"Q": np.linspace(0.0, 1.0, 200)}),
    (bardeen, {"q": np.linspace(0.0, 0.769, 200)}),
    (hayward, {"l": np.linspace(0.0, 0.769, 200)}),
]

fig, axes = plt.subplots(2, 2, figsize=(10, 7.5))
bp.plot.profile(models, "event_horizon", ax=axes[0, 0])
bp.plot.profile(models, "isco", ax=axes[0, 1])
bp.plot.profile(models, "shadow_radius", ax=axes[1, 0])
bp.plot.profile(models, "nu_phi_isco", mass=10.0, ax=axes[1, 1])

# Sgr A* shadow bounds from the registry (Vagnozzi et al. 2023), for orientation.
sgra = bp.data.load("eht/sgra")
for level, alpha in (("1sigma", 0.25), ("2sigma", 0.12)):
    low, high = sgra.bounds[level]
    axes[1, 0].axhspan(low, high, color="grey", alpha=alpha, lw=0)
axes[1, 0].set_title("grey: Sgr A* 1 and 2 sigma bounds (registry, not human-verified)", fontsize=8)
axes[1, 1].set_title(r"$M = 10\,M_\odot$", fontsize=9)
for ax in axes.flat:
    ax.set_xlabel("charge / core parameter (units of $M$)")

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
fig.tight_layout()
fig.savefig(out / "profiles.png", dpi=150)
print("wrote", out / "profiles.png")
