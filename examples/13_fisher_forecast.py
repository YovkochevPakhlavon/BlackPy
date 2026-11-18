"""Example 13: Fisher-matrix forecasts.

A forecast answers "what would this dataset constrain?" in about a second,
without running a sampler. The derivatives of the model with respect to the
parameters come from automatic differentiation of the same functions the
likelihood uses.

1. A Kerr QPO triplet: forecast against the emcee posterior.
2. Design study: how well must the three frequencies be measured to detect a
   Johannsen deformation alpha13? The forecast shows that the answer is
   "it does not help": spin and deformation are degenerate, and it names the
   degenerate direction. An independent mass measurement breaks it.
3. A forecast that must not be trusted: the Kerr spin from a shadow size.

Run:  python examples/13_fisher_forecast.py        (about 40 seconds)
"""

import os
import time
import warnings

import numpy as np

import blackpy as bp
from blackpy.errors import BlackPyWarning
from blackpy.infer import (
    Fit,
    Gaussian,
    Uniform,
    forecast,
    simulate_qpo_dataset,
    simulate_shadow_dataset,
)

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


kerr, johannsen = bp.metric.Kerr(), bp.metric.Johannsen()
TRUTH = {"mass": 6.0, "r": 5.7, "a": 0.3}


def triplet(sigma_nu, sigma_low, mass_sigma=0.3):
    return simulate_qpo_dataset(
        kerr, "RP", name="src", mass=TRUTH["mass"], r=TRUTH["r"], sigma_nu=sigma_nu,
        sigma_low=sigma_low, mass_sigma=mass_sigma, a=TRUTH["a"],
    )  # fmt: skip


# ----------------------------------------------------------------- 1. forecast vs sampler
data = triplet(2.0, 0.1)
fiducial = {"a": 0.3, "M_src": 6.0, "r_src": 5.7}
priors = {"a": Uniform(0.0, 0.998), "M": Uniform(2.0, 20.0), "r": Uniform(2.0, 20.0)}
start = time.perf_counter()
predicted = forecast(kerr, data, fiducial, "RP", priors)
elapsed_first = time.perf_counter() - start
start = time.perf_counter()
forecast(kerr, data, fiducial, "RP", priors)
elapsed_again = time.perf_counter() - start
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BlackPyWarning)
    result = Fit(kerr, data, "RP", priors, nsteps=steps(4000), max_steps=steps(16000)).run(seed=1)
print("Kerr RPM triplet (2 Hz on the twin peaks, 0.1 Hz on the type-C QPO), flat priors")
print(predicted)
print(f"  {'parameter':<10} {'forecast sigma':>15} {'emcee sigma':>13}")
for name, summary in result.summary().items():
    print(f"  {name:<10} {predicted.sigma[name]:15.4g} {summary['std']:13.4g}")
print(
    f"  forecast: {elapsed_first:.1f} s including compilation, {1e3 * elapsed_again:.0f} ms when "
    f"repeated; emcee: {result.metadata['runtime_seconds']:.1f} s\n"
)

# ----------------------------------------------------------------- 2. design study
print("Johannsen metric, deformation alpha13 free: forecast sigma for different data quality")
print(
    f"  {'sigma(HF) [Hz]':>15} {'sigma(LF) [Hz]':>15} {'mass prior':>12} {'sigma(a)':>10} {'sigma(alpha13)':>15}"  # noqa: E501
)
fiducial_j = {"a": 0.3, "alpha13": 0.0, "M_src": 6.0, "r_src": 5.7}
priors_j = {"a": Uniform(0.0, 0.998), "alpha13": Uniform(-3.0, 3.0), "r": Uniform(2.0, 20.0)}
last = None
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BlackPyWarning)
    for sigma_nu, sigma_low, mass_sigma in (
        (2.0, 0.1, None),
        (0.2, 0.01, None),
        (2.0, 0.1, 0.3),
        (2.0, 0.1, 0.03),
        (0.2, 0.01, 0.03),
    ):
        data = triplet(sigma_nu, sigma_low, mass_sigma or 0.3)
        mass_prior = (
            Uniform(2.0, 20.0) if mass_sigma is None else Gaussian(6.0, mass_sigma, low=0.0)
        )
        last = forecast(johannsen, data, fiducial_j, "RP", {**priors_j, "M": mass_prior})
        label = "flat" if mass_sigma is None else f"{mass_sigma} Msun"
        print(
            f"  {sigma_nu:15.2f} {sigma_low:15.2f} {label:>12} {last.sigma['a']:10.3g} "
            f"{last.sigma['alpha13']:15.3g}"
        )
        if mass_sigma is None and sigma_nu == 2.0:
            flat = last
print("\n  with a flat mass prior:")
print("  " + str(flat).replace("\n", "\n  "))
print(
    "  Three frequencies fix three parameters (a, M, r); a fourth one (alpha13) leaves one\n"
    "  direction unconstrained, whatever the precision of the frequencies. Only independent\n"
    "  information, here a mass measurement, makes sigma(alpha13) finite, and then it scales\n"
    "  with the mass uncertainty, not with the frequency uncertainties.\n"
)

# ----------------------------------------------------------------- 3. a forecast not to trust
shadow = simulate_shadow_dataset(kerr, name="img", sigma=0.34, inclination_deg=30.0, a=0.5)
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    spin = forecast(
        kerr, shadow, {"a": 0.5}, priors={"a": Uniform(0.0, 0.998)}, fixed={"inclination_deg": 30.0}
    )
print("Kerr spin from one shadow size (6.5% precision, inclination fixed at 30 deg):")
print(spin)
print(f"  warnings: {sorted({w.category.__name__ for w in caught})}")
print(
    f"  A forecast sigma of {spin.sigma['a']:.1f} for a parameter confined to [0, 0.998] is not an\n"  # noqa: E501
    "  expected constraint: the shadow size does not measure the spin, and the Gaussian\n"
    "  approximation has no meaning here. The warnings say so."
)
if not QUICK:  # the short quick-mode chain is too noisy for this check
    np.testing.assert_allclose(predicted.sigma["a"], result.summary()["a"]["std"], rtol=0.15)
