"""Example 11: two shadow sources, M87* and Sgr A*.

1. What the M87* entry of the registry contains: the EHT measures the
   diameter of the emission ring; the bound on the shadow radius in units of
   M follows through a chain of steps and depends entirely on the mass prior.
2. Reproduction of a published M87* constraint: the Reissner-Nordstrom charge
   bound of Kocherlakota et al. (2021, Phys. Rev. D 103, 104047), who exclude
   charges above q ~ 0.90 at 68% confidence.
3. Joint fits of both sources. Metric parameters are in units of each
   object's own mass, so the user must say whether a parameter is common to
   both black holes (``shared``) or not (``per_source``):
   a static metric parameter (the Reissner-Nordstrom charge), and a rotating
   metric (Kerr-Newman) with one spin and one inclination per source.
4. The role of the mass prior, and the circular-inference guard.

All registry values are unverified by a human (``verified_by_human: false``).

Run:  python examples/11_m87_and_sgra_shadows.py        (about 1.5 minutes)
"""

import math
import warnings

import numpy as np
from scipy.optimize import brentq

import blackpy as bp
from blackpy.errors import BlackPyWarning, CircularInferenceWarning
from blackpy.infer import Fit, Uniform, inclination_prior

m87, sgra = bp.data.load("eht/m87"), bp.data.load("eht/sgra")
rn, kerr_newman = bp.metric.ReissnerNordstrom(), bp.metric.KerrNewman()

# ----------------------------------------------------------------- 1. the M87* chain
print("M87*: from the image to a bound on r_sh / M")
print(f"  measured ring diameter        d       = {m87.measured['ring_diameter']}")
print(f"  image-based grav. radius      theta_g = {m87.measured['theta_g']}  (Kerr-calibrated)")
print(f"  distance                      D       = {m87.distance}")
print(f"  jet inclination (not used unless asked): {m87.inclination}")
print(f"  circularity bound: < {m87.circularity.upper} ({m87.circularity.definition})")
print(
    f"  {'mass prior':<18} {'M':>26} {'theta_dyn':>22} {'indep.':>7} {'delta':>14} {'r_sh/M':>14}"
)
for key, prior in m87.mass_priors.items():
    deviation = m87.deviation(key)
    mean, sigma = m87.gaussian(key)
    origin = "" if prior.fractional_deviation is not None else " (computed)"
    print(
        f"  {key:<18} {str(prior):>26} {str(prior.theta_g):>22} {str(prior.independent):>7} "
        f"{deviation.value:+6.2f} +- {deviation.sigma:4.2f} {mean:6.2f} +- {sigma:4.2f}{origin}"
    )
print(f"  default bound uses: {m87.assumed_mass_priors}; 68% interval {m87.bounds['1sigma']}\n")

# ----------------------------------------------------------------- 2. published charge bound
low, high = m87.bounds["1sigma"]
q_max = brentq(lambda q: float(bp.light.shadow_radius(rn, Q=q)) - low, 0.0, 1.0)
print("Reissner-Nordstrom charge allowed by the M87* shadow size at 68% confidence:")
print(f"  shadow radius falls below {low} M at Q/M = {q_max:.3f}")
print("  Kocherlakota et al. (2021), Sec. V and conclusions: 0 < q <~ 0.90")
print(
    f"  (the extremal value Q = 1 has r_sh = {float(bp.light.shadow_radius(rn, Q=1.0)):.2f} M, "
    f"outside the interval)\n"
)

# ----------------------------------------------------------------- 3. joint fits
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BlackPyWarning)  # unverified data, stated above
    single = {
        name: Fit(rn, data, priors={"Q": Uniform(0.0, 1.0)}, nsteps=3000).run(seed=1)
        for name, data in (("M87*", m87), ("Sgr A*", sgra))
    }
    shared = Fit(rn, [sgra, m87], priors={"Q": Uniform(0.0, 1.0)}, shared=["Q"], nsteps=3000).run(
        seed=2
    )
    separate = Fit(
        rn, [sgra, m87], priors={"Q": Uniform(0.0, 1.0)}, per_source=["Q"], nsteps=3000
    ).run(seed=3)
try:
    Fit(rn, [sgra, m87], priors={"Q": Uniform(0.0, 1.0)})
except ValueError as error:
    print("Without a choice the joint fit is refused:\n  " + str(error)[:160] + " ...\n")

limit = lambda samples: np.percentile(samples, [68.27, 95.45])
print("Reissner-Nordstrom charge Q/M, upper limits (68%, 95%), flat prior on [0, 1]:")
for name, result in single.items():
    print(f"  {name:<28} Q < {limit(result['Q'])[0]:.2f}, {limit(result['Q'])[1]:.2f}")
print(
    f"  {'both, shared Q':<28} Q < {limit(shared['Q'])[0]:.2f}, {limit(shared['Q'])[1]:.2f}"
    f"   ({shared.goodness_of_fit().summary_line()})"
)
for name in separate.param_names:
    print(
        f"  {'both, per source: ' + name:<28} Q < {limit(separate[name])[0]:.2f}, "
        f"{limit(separate[name])[1]:.2f}"
    )
print(
    "  A shared charge-to-mass ratio is an assumption about two different objects; with one\n"
    "  value per source the joint fit is just the two single fits side by side.\n"
)

with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    rotating = Fit(
        kerr_newman,
        [sgra, m87],
        priors={
            "Q": Uniform(0.0, 1.0),
            "a": Uniform(0.0, 0.998),  # applies to a_sgra and a_m87
            "inclination_m87": inclination_prior(m87),  # the jet angle, on request only
        },
        shared=["Q"],
        nsteps=4000,
        max_steps=16000,
    )
    result = rotating.run(seed=4)
print("Kerr-Newman: shared Q, one spin and one inclination per source")
print(f"  parameters: {result.param_names}")
print(result)
print(f"  warnings: {sorted({w.category.__name__ for w in caught})}")
print(
    f"  Q < {limit(result['Q'])[0]:.2f} (68%), {limit(result['Q'])[1]:.2f} (95%). The spins are "
    "not constrained by the shadow sizes:\n  the flags above say so, and their intervals must "
    "not be quoted as measurements.\n"
)

# ----------------------------------------------------------------- 4. the mass prior
print("M87* with other mass priors (Reissner-Nordstrom, Q in [0, 1]):")
with warnings.catch_warnings():
    warnings.simplefilter("ignore", BlackPyWarning)
    for key in ("stellar_dynamics", "gas_dynamics"):
        mean, sigma = m87.gaussian(key)
        best = (float(bp.light.shadow_radius(rn, Q=0.0)) - mean) / sigma
        print(
            f"  {key:<17}: r_sh/M = {mean:.2f} +- {sigma:.2f}; the largest RN shadow (Q = 0, "
            f"3 sqrt(3) = {3 * math.sqrt(3):.2f}) is {best:+.1f} sigma away"
        )
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    Fit(rn, m87, priors={"Q": Uniform(0.0, 1.0)}, shadow_mass_prior="eht_ring")
circular = [w for w in caught if issubclass(w.category, CircularInferenceWarning)]
print(f"  eht_ring         : CircularInferenceWarning raised: {bool(circular)}")
print(
    "  With the gas-dynamical mass no metric of this family fits (the shadow would have to be\n"
    "  78% larger than 3 sqrt(3) M); the EHT mass comes from the image itself and cannot be\n"
    "  used to test the metric with that image."
)
