"""Example 4: reproduce published Sgr A* shadow constraints.

Vagnozzi et al. (2023), Class. Quantum Grav. 40, 165007 (arXiv:2205.07787),
Section III.A-C, report from the EHT shadow size of Sgr A*:

* Reissner-Nordstrom: Q <~ 0.8 M (1 sigma) and Q <~ 0.95 M (2 sigma);
* Bardeen: consistent within 2 sigma for every allowed charge, including
  the extremal value;
* Hayward: consistent for every allowed value of the parameter, including
  the extremal value.

This script recomputes these statements from the registry entry
``eht/sgra`` (bounds of Eqs. 7-8 of the same paper). The registry values
have been checked by the maintainer (``verified_by_human: true``); a
reproduction of a published constraint is an example, not a test.

Run:  python examples/04_sgra_shadow_constraints.py
"""

import os
import warnings

import numpy as np
from scipy.optimize import brentq

import blackpy as bp
from blackpy.errors import UnverifiedDataWarning
from blackpy.infer import Fit, Uniform

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


sgra = bp.data.load("eht/sgra")
print(f"Sgr A* r_sh/M bounds: 1 sigma {sgra.bounds['1sigma']}, 2 sigma {sgra.bounds['2sigma']}")

rn = bp.metric.ReissnerNordstrom()
shadow = lambda q: float(bp.light.shadow_radius(rn, Q=q))
print(
    "\nReissner-Nordstrom (the shadow shrinks with charge, so the lower bounds give upper limits)"
)
for level, published in (("1sigma", 0.8), ("2sigma", 0.95)):
    lower = sgra.bounds[level][0]
    limit = brentq(lambda q, target=lower: shadow(q) - target, 0.0, 1.0)
    print(f"  {level}: Q < {limit:.3f} M   (published: Q <~ {published} M)")
print(f"  extremal RN (Q = 1): r_sh = {shadow(1.0):.3f} M, below the 2 sigma lower bound: excluded")

extremal = 4.0 / (3.0 * np.sqrt(3.0))
for metric, name in ((bp.metric.Bardeen(), "q"), (bp.metric.Hayward(), "l")):
    value = float(bp.light.shadow_radius(metric, **{name: extremal * (1 - 1e-9)}))
    low2, low1 = sgra.bounds["2sigma"][0], sgra.bounds["1sigma"][0]
    verdict = "inside 1 sigma" if value > low1 else "inside 2 sigma" if value > low2 else "excluded"
    print(
        f"\n{metric.name}: r_sh at the extremal value {name} = {extremal:.4f} is {value:.3f} M "
        f"-> {verdict} (published: consistent up to the extremal value)"
    )

# The same constraint as a posterior: Gaussian likelihood on r_sh/M, flat prior on Q.
with warnings.catch_warnings():
    warnings.simplefilter("ignore", UnverifiedDataWarning)  # only for datasets with the flag false
    fit = Fit(
        rn, sgra, priors={"Q": Uniform(0.0, 1.0)}, sampler="emcee", nwalkers=32, nsteps=steps(3000)
    )
result = fit.run(seed=0)
q68, q95 = np.percentile(result["Q"], [68.27, 95.45])
print(f"\nPosterior with a flat prior on Q in [0, 1]: Q < {q68:.2f} (68%), Q < {q95:.2f} (95%)")
print("(Bayesian upper limits depend on the prior and need not coincide with the bound crossings.)")
