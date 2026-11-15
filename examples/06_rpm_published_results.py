"""Example 6: reproduce published relativistic-precession-model results.

For the QPO triplets of GRO J1655-40 (Motta et al. 2014) and XTE J1859+226
(Motta et al. 2022) this script computes mass, spin and radius in the Kerr
metric in three ways and compares them with the published values stored in
the registry:

1. the analytic solution of Ingram & Motta (2014) for the central
   frequencies;
2. Monte Carlo propagation of the frequency uncertainties through that
   solution (the method of the original papers);
3. a Bayesian fit with BlackPy's general rotating-metric machinery (no
   Kerr-specific formula), with flat priors on mass, spin and radius.

It also compares with Stuchlik & Kolos (2016), who report
M = 5.3 +- 0.1 Msun and a = 0.286 +- 0.004 for GRO J1655-40.

This is an example and not a test, because the registry values have not
been verified by a human (project rule: tests must not depend on them).

Run:  python examples/06_rpm_published_results.py        (about 20 seconds)
"""

import math
import warnings

import numpy as np

import blackpy as bp
from blackpy.errors import BlackPyWarning
from blackpy.infer import Fit, Uniform

N_DRAWS = 200_000
BETA = 1.0 / (2.0 * math.pi * bp.constants.T_SUN)  # c^3 / (2 pi G M_sun) in Hz


def draw(measurement, rng, size):
    """Draws from a split normal (asymmetric 1-sigma errors)."""
    z = rng.standard_normal(size)
    return measurement.value + np.where(z >= 0, measurement.err_plus, measurement.err_minus) * z


def report(label, mass, spin, radius):
    print(f"  {label:<37} M = {mass:<16} a = {spin:<18} r = {radius}")


def pm(samples, digits):
    return f"{np.mean(samples):.{digits}f} +- {np.std(samples):.{digits}f}"


print(f"c^3 / (2 pi G M_sun) = {BETA:.2f} Hz (BlackPy constants); Ingram & Motta quote 3.237e4 Hz.")
print(f"Ratio: {32370.0 / BETA:.5f}. A mass computed with 3.237e4 Hz is larger by this factor.\n")

kerr = bp.metric.Kerr()
rng = np.random.default_rng(2026)
for dataset_id in ("qpo/gro_j1655_40", "qpo/xte_j1859_226"):
    data = bp.data.load(dataset_id)
    print(f"{data.object}: nu_low = {data.low}, nu_L = {data.lower}, nu_U = {data.upper}")

    exact = bp.orbits.rpm_kerr_solution(data.low.value, data.lower.value, data.upper.value)
    report(
        "1. analytic, central values",
        f"{float(exact.mass):.3f}",
        f"{float(exact.spin):.4f}",
        f"{float(exact.radius):.3f}",
    )
    report(
        "   same, with beta = 3.237e4 Hz",
        f"{float(exact.mass) * 32370.0 / BETA:.3f}",
        f"{float(exact.spin):.4f}",
        f"{float(exact.radius):.3f}",
    )

    draws = bp.orbits.rpm_kerr_solution(
        draw(data.low, rng, N_DRAWS), draw(data.lower, rng, N_DRAWS), draw(data.upper, rng, N_DRAWS)
    )
    mass, spin, radius = (np.asarray(v) for v in (draws.mass, draws.spin, draws.radius))
    ok = np.isfinite(mass)
    report("2. Monte Carlo propagation", pm(mass[ok], 3), pm(spin[ok], 4), pm(radius[ok], 3))

    slug = dataset_id.split("/")[-1]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlackPyWarning)  # unverified data: stated above
        fit = Fit(
            kerr,
            data,
            qpo_model="RP",
            priors={"a": Uniform(0.0, 0.998), "M": Uniform(2.0, 20.0), "r": Uniform(2.0, 20.0)},
            sampler="emcee",
            nsteps=4000,
            max_steps=16000,
        )
        result = fit.run(seed=7)
    report(
        "3. BlackPy fit (general metric code)",
        pm(result[f"M_{slug}"], 3),
        pm(result["a"], 4),
        pm(result[f"r_{slug}"], 3),
    )
    published = data.published_fit
    report(
        "published (" + data.references[data.spin.source].citation.split(",")[0] + ")",
        str(published["mass"]).replace(" Msun", ""),
        str(data.spin),
        str(published["radius"]).replace(" G M / c^2", ""),
    )
    if dataset_id == "qpo/gro_j1655_40":
        report("Motta et al. (2014), Table 3", "5.307 +- 0.066", "0.286 +- 0.003", "5.677 +- 0.035")
        report("Ingram & Motta (2014), Table 1", "5.31 +- 0.07", "0.285 +- 0.003", "5.68 +- 0.04")
        report("Stuchlik & Kolos (2016), RP model", "5.3 +- 0.1", "0.286 +- 0.004", "-")
    r_isco = float(bp.orbits.isco(kerr, a=float(exact.spin)))
    print(f"  ISCO for the analytic spin: {r_isco:.3f}; QPO radius: {float(exact.radius):.3f}\n")

print(
    "Reading of the comparison (details in VERIFICATION_v0.2.md):\n"
    "- GRO J1655-40: all determinations agree on a = 0.286 except the value 0.290 quoted in the\n"
    "  abstract and text of Motta et al. (2014), which differs from their own Table 3 (0.286).\n"
    "  Stuchlik & Kolos use the same frequencies; their 0.286 is the exact solution, and their\n"
    "  larger errors are maximum errors from the edges of the frequency intervals.\n"
    "- The published masses are 0.18% above BlackPy's because of the constant beta.\n"
    "- XTE J1859+226: the central values agree; the published uncertainties are larger than\n"
    "  the propagated frequency uncertainties."
)
