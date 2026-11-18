"""Example 9: does the best model fit the data at all?

A Bayes factor ranks models; it does not say that the better one describes
the data. This example shows the two tools added in BlackPy 0.3:

* ``result.goodness_of_fit()``: pulls, chi-square and degrees of freedom at
  the best-fit point;
* ``result.posterior_predictive()``: replicated data and posterior
  predictive p-values (Gelman, Meng & Stern 1996).

Three fits of the GRO J1655-40 QPOs (registry data, not verified by a human):

1. Kerr, relativistic precession model, three QPOs, dynamical mass prior:
   3 data + 1 Gaussian prior - 3 parameters = 1 degree of freedom.
2. Schwarzschild, same model, the two high-frequency QPOs and the same mass
   prior: 2 + 1 - 2 = 1 degree of freedom. A non-rotating metric cannot
   reproduce the frequencies with the dynamical mass: PoorFitWarning.
3. Reissner-Nordstrom with a free charge: 2 + 1 - 3 = 0 degrees of freedom.
   The charge absorbs the misfit, the fit is exact, and the goodness of fit
   is undefined: nothing can be said about whether the model is adequate.

Run:  python examples/09_fit_quality.py        (about 30 seconds)
"""

import os
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.errors import PoorFitWarning  # noqa: E402
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
DATA = "qpo/gro_j1655_40"

cases = {
    "Kerr (triplet)": (bp.metric.Kerr(), {"a": Uniform(0.0, 0.998), "r": Uniform(2.0, 20.0)}),
    "Schwarzschild (pair)": (bp.metric.Schwarzschild(), {"r": Uniform(6.0, 20.0)}),
    "Reissner-Nordstrom (pair)": (
        bp.metric.ReissnerNordstrom(),
        {"Q": Uniform(0.0, 1.0), "r": Uniform(4.0, 20.0)},
    ),
}
fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, (title, (metric, priors)) in zip(axes, cases.items(), strict=True):
    print(f"=== {title} " + "=" * (60 - len(title)))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = Fit(
            metric, DATA, qpo_model="RP", priors=priors, nsteps=steps(4000), max_steps=steps(16000)
        )
        result = fit.run(seed=9)
    poor = [w for w in caught if issubclass(w.category, PoorFitWarning)]
    other = sorted({w.category.__name__ for w in caught if w not in poor})
    print(result)
    print(result.goodness_of_fit())
    print(result.posterior_predictive(steps(4000), seed=0, warn=False))
    print(f"PoorFitWarning raised: {bool(poor)}; other warnings: {other}\n")
    bp.plot.residuals(result, ax=ax)
    ax.set_title(f"{title}\n{ax.get_title()}", fontsize="medium")
fig.tight_layout()
fig.savefig(out / "fit_quality.png", dpi=130)
print(
    "Reading: the Kerr fit passes. The Schwarzschild fit fails (the mass it needs is 3.8 sigma\n"
    "below the dynamical one), so a Bayes\n"
    "factor that preferred it over some other failing model would mean nothing. The\n"
    "Reissner-Nordstrom fit is exact because it has as many parameters as data: its charge\n"
    "is not a measurement (the source rotates: see the SpinIgnoredWarning), and its fit\n"
    "quality is undefined, not good.\n"
    f"wrote {out / 'fit_quality.png'}"
)
