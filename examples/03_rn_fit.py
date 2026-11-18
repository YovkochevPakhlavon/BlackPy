"""Example 3: fit the Reissner-Nordstrom charge with emcee and dynesty.

Synthetic twin-peak QPO data for two sources are generated from a known
charge (Q = 0.6), then fitted with both samplers. The script prints the
posterior summaries, the Bayes factors against Schwarzschild (Q = 0), a
LaTeX table, and saves a corner plot and the reproducibility bundles to
``examples/output/``.

Run:  python examples/03_rn_fit.py        (about half a minute)
"""

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.infer import Fit, FitResult, Uniform, simulate_qpo_dataset  # noqa: E402

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

rn = bp.metric.ReissnerNordstrom()
truth = {"Q": 0.6, "M_src_a": 6.0, "r_src_a": 5.6, "M_src_b": 10.0, "r_src_b": 6.2}
rng = np.random.default_rng(2026)  # noisy data; drop rng=... for noise-free data
data = [
    simulate_qpo_dataset(
        rn, "RP", name="src_a", mass=6.0, r=5.6, sigma_nu=2.0, mass_sigma=0.1, rng=rng, Q=0.6
    ),
    simulate_qpo_dataset(
        rn, "RP", name="src_b", mass=10.0, r=6.2, sigma_nu=1.5, mass_sigma=0.15, rng=rng, Q=0.6
    ),
]
for d in data:
    print(f"{d.object}: nu_U = {d.upper}, nu_L = {d.lower}, mass prior = {d.mass_prior}")

# The masses use the (synthetic) independent mass measurements as Gaussian priors.
priors = {"Q": Uniform(0.0, 1.0), "r": Uniform(4.0, 12.0)}

emcee_fit = Fit(
    rn, data, qpo_model="RP", priors=priors, sampler="emcee", nwalkers=32, nsteps=steps(6000)
)
emcee_result = emcee_fit.run(seed=1)
print("\n", emcee_result, sep="")
print(
    "emcee diagnostics:",
    {k: emcee_result.diagnostics[k] for k in ("converged", "steps_over_tau", "burn_in", "thin")},
)
print(emcee_fit.bayes_factor(null={"Q": 0.0}))

dynesty_fit = Fit(rn, data, qpo_model="RP", priors=priors, sampler="dynesty", nlive=live(400))
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UserWarning)  # dynesty's own efficiency hints
    dynesty_result = dynesty_fit.run(seed=2)
    print("\n", dynesty_result, sep="")
    print(dynesty_fit.bayes_factor(null={"Q": 0.0}))

print("\n" + bp.plot.latex_table({"emcee": emcee_result, "dynesty": dynesty_result}))

fig = bp.plot.corner(dynesty_result, truths=truth)
fig.savefig(out / "rn_fit_corner.png", dpi=130)
path = emcee_result.save(out / "rn_fit_emcee")
dynesty_result.save(out / "rn_fit_dynesty")
reloaded = FitResult.load(path)
print(
    f"\nwrote {out / 'rn_fit_corner.png'} and the bundles; reloaded {reloaded.samples.shape[0]} "
    f"samples made with blackpy {reloaded.metadata['versions']['blackpy']}, seed "
    f"{reloaded.metadata['seed']}"
)
