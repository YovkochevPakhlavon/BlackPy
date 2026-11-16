"""Example 12: model-agnostic constraints with the Rezzolla-Zhidenko parametrisation.

Instead of testing one theory at a time, the data constrain the coefficients
of a general expansion of the metric around Schwarzschild (Rezzolla &
Zhidenko 2014; rotating version: Konoplya, Rezzolla & Zhidenko 2016).

1. Any static catalogue metric can be mapped onto the parametrisation:
   coefficients of Reissner-Nordstrom, Bardeen and Hayward.
2. Reproduction of a published bound: Psaltis et al. (2020, Phys. Rev. Lett.
   125, 141104) find -1.2 < a_1 < 1.3 from the M87* shadow size.
3. A fit of the leading coefficients (epsilon, a_1) to the Sgr A* and M87*
   shadows: one combination is constrained, the coefficients are not.
4. Adding QPOs: a static RZ metric fitted to GRO J1655-40. BlackPy warns
   that the source rotates; the example shows what the warnings mean.

Registry data are unverified by a human.

Run:  python examples/12_parametrised_metrics.py        (about 45 seconds)
"""

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from scipy.optimize import brentq  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.errors import BlackPyWarning  # noqa: E402
from blackpy.infer import Fit, Uniform  # noqa: E402

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)
mt = bp.metric

# ----------------------------------------------------------------- 1. mapping metrics to RZ
print("Rezzolla-Zhidenko coefficients of catalogue metrics (r_0 = 2 / (1 + epsilon) M)")
print(
    f"{'metric':<28} {'epsilon':>9} {'a0':>9} {'b0':>9} {'a1':>9} {'a2':>9} "
    f"{'beta':>8} {'gamma':>7}"
)
for metric, params in (
    (mt.ReissnerNordstrom(), {"Q": 0.5}),
    (mt.Bardeen(), {"q": 0.5}),
    (mt.Hayward(), {"l": 0.5}),
):
    c = mt.rz_coefficients(metric, order=3, **params)
    label = f"{metric.name} {params}"
    print(
        f"{label:<28} {c.epsilon:9.5f} {c.a0:9.5f} {c.b0:9.5f} {c.a[0]:9.5f} {c.a[1]:9.5f} "
        f"{c.beta:8.5f} {c.gamma:7.4f}"
    )
    for order in (1, 3):
        rz = mt.RezzollaZhidenko(order=order, ppn="free")
        error = float(bp.light.shadow_radius(rz, **c.as_params(order))) - float(
            bp.light.shadow_radius(metric, **params)
        )
        print(f"{'':<28}   shadow radius of the order-{order} RZ metric: error {error:+.1e} M")
print(
    "  Reissner-Nordstrom is exact at lowest order (a0 = epsilon = Q^2/r_0^2, a1 = 0); it needs\n"
    "  a0 != 0, i.e. it is excluded by the default ppn='gr', which assumes beta = gamma = 1.\n"
)

# ----------------------------------------------------------------- 2. published bound
rz = mt.RezzollaZhidenko(order=1)
schwarzschild = 3.0 * math.sqrt(3.0)
shadow = lambda a1: float(bp.light.shadow_radius(rz, a1=a1))
lower = brentq(lambda a1: shadow(a1) - 1.17 * schwarzschild, -3.0, 0.0)
upper = brentq(lambda a1: shadow(a1) - 0.83 * schwarzschild, 0.0, 3.0)
print("M87* shadow size within 17% of 3 sqrt(3) M (epsilon = a0 = 0, only a_1):")
print(f"  BlackPy:               {lower:.2f} < a_1 < {upper:.2f}")
print("  Psaltis et al. (2020): -1.2 < a_1 < 1.3")
print(
    f"  (to first order r_sh = 3 sqrt(3) (1 - 4 a_1/9) would give |a_1| < {0.17 * 9 / 4:.2f}; "
    "the 1/r^4 term\n   of the RZ metric cancels two thirds of the effect at the photon sphere)\n"
)

# ----------------------------------------------------------------- 3. shadows of two sources
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    fit = Fit(
        rz,
        ["eht/sgra", "eht/m87"],
        priors={"epsilon": Uniform(-0.4, 0.45), "a1": Uniform(-2.0, 2.0)},
        shared=["epsilon", "a1"],  # an assumption: the same deviations for both objects
        nsteps=5000,
        max_steps=20000,
    )
    result = fit.run(seed=12)
print("RZ (order 1) fitted to the Sgr A* and M87* shadow sizes, shared coefficients:")
print(result)
print(f"  warnings: {sorted({w.category.__name__ for w in caught})}")
correlation = result.correlation("epsilon", "a1", method="spearman")
samples = result.samples
size = np.asarray(bp.light.shadow_radius(rz, epsilon=samples[:2000, 0], a1=samples[:2000, 1]))
median, low, high = np.median(size), np.percentile(size, 16), np.percentile(size, 84)
print(
    f"  rank correlation of epsilon and a_1: {correlation:+.2f}. What the data constrain is the "
    f"shadow size itself:\n  r_sh/M = {median:.2f} +{high - median:.2f} -{median - low:.2f}; "
    "the marginal intervals of epsilon and a_1 above are set by the\n  prior ranges and must "
    "not be quoted as measurements.\n"
)
figure = bp.plot.corner(result)
figure.savefig(out / "rz_shadow_corner.png", dpi=110)

# ----------------------------------------------------------------- 4. adding QPOs
with warnings.catch_warnings(record=True) as caught:
    warnings.simplefilter("always")
    joint = Fit(
        rz,
        ["qpo/gro_j1655_40", "eht/sgra"],
        qpo_model="RP",
        priors={"epsilon": Uniform(-0.4, 0.45), "a1": Uniform(-2.0, 2.0), "r": Uniform(3.0, 15.0)},
        sampler="dynesty",  # the degeneracy is curved and narrow: emcee does not converge
        nlive=400,
    )
    joint_result = joint.run(seed=13)
print("RZ (order 1) fitted to the GRO J1655-40 twin QPOs and the Sgr A* shadow size:")
print(joint_result)
names = sorted({w.category.__name__ for w in caught if issubclass(w.category, BlackPyWarning)})
print(f"  warnings: {names}")
print(
    "  The QPOs pull the coefficients away from zero, but GRO J1655-40 rotates (a = 0.29 in the\n"
    "  Kerr fit) and a static metric cannot model its nodal-precession QPO: the deviation\n"
    "  parameters absorb the spin (SpinIgnoredWarning). A rotating parametrised metric is\n"
    "  blackpy.metric.KRZ(); its shadow needs ray tracing and is refused, its QPOs work.\n"
    f"wrote {out / 'rz_shadow_corner.png'}"
)
