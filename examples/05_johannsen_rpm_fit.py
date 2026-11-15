"""Example 5: Johannsen-metric fit to the QPO triplet of GRO J1655-40.

The relativistic precession model (RP) identifies the three simultaneous
QPOs of GRO J1655-40 (Motta et al. 2014) with the orbital, periastron
precession and nodal precession frequencies at one radius. In the Kerr
metric the three frequencies fix mass, spin and radius. Here the Johannsen
(2013) deformation parameter alpha13 is left free as well: three frequencies
then cannot separate the spin from the deformation, and the degeneracy is
broken only by the independent (dynamical) mass measurement.

The script fits the Kerr metric and the Johannsen metric with emcee and
dynesty, prints the posterior correlation between spin and deformation,
the Bayes factor of Johannsen against Kerr, and saves a corner plot to
``examples/output/johannsen_rpm_corner.png``.

The registry values have not been verified by a human, and the mass prior
is the dynamical mass of Beer & Podsiadlowski (2002). BlackPy says so in its
warnings; they are printed, not silenced.

Run:  python examples/05_johannsen_rpm_fit.py        (about half a minute)
"""

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import blackpy as bp  # noqa: E402
from blackpy.errors import DegeneracyWarning, UnverifiedDataWarning  # noqa: E402
from blackpy.infer import Fit, Uniform  # noqa: E402

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)

DATA = "qpo/gro_j1655_40"
data = bp.data.load(DATA)
print(f"{data.object}: nu_U = {data.upper}, nu_L = {data.lower}, nu_low = {data.low}")
print(f"mass prior (independent: {data.mass_prior.independent}): {data.mass_prior}")
print(f"published RPM spin: {data.spin}  [{data.spin.method}]\n")

kerr = bp.metric.Kerr()
johannsen = bp.metric.Johannsen()
print(johannsen.reference)
print("The Johannsen metric is not a solution of GR; its effective source at alpha13 = 0.5:")
print(bp.geometry.effective_source(johannsen, a=0.29, alpha13=0.5).summary(), "\n")

# The mass uses the dynamical measurement as a Gaussian prior (default).
kerr_priors = {"a": Uniform(0.0, 0.998), "r": Uniform(2.0, 20.0)}
priors = {**kerr_priors, "alpha13": Uniform(-4.0, 4.0)}

warnings.simplefilter("always", DegeneracyWarning)
warnings.simplefilter("once", UnverifiedDataWarning)

kerr_fit = Fit(kerr, DATA, qpo_model="RP", priors=kerr_priors, sampler="emcee", nsteps=6000)
kerr_result = kerr_fit.run(seed=1)
print("Kerr, emcee\n", kerr_result, "\n", sep="")

emcee_fit = Fit(
    johannsen,
    DATA,
    qpo_model="RP",
    priors=priors,
    sampler="emcee",
    nsteps=8000,
    max_steps=64000,
)
emcee_result = emcee_fit.run(seed=2)
print("Johannsen (alpha13 free), emcee\n", emcee_result, sep="")
print(emcee_fit.bayes_factor(null={"alpha13": 0.0}), "\n")

dynesty_fit = Fit(johannsen, DATA, qpo_model="RP", priors=priors, sampler="dynesty", nlive=400)
with warnings.catch_warnings():
    warnings.simplefilter("ignore", RuntimeWarning)  # dynesty's own efficiency hints
    dynesty_result = dynesty_fit.run(seed=3)
    print("Johannsen (alpha13 free), dynesty\n", dynesty_result, sep="")
    print(dynesty_fit.bayes_factor(null={"alpha13": 0.0}), "\n")

correlation = emcee_result.diagnostics["spin_correlations"]["alpha13"]
kerr_sigma = kerr_result.summary()["a"]["std"]
johannsen_sigma = emcee_result.summary()["a"]["std"]
print(
    f"Posterior correlation of a and alpha13: {correlation['pearson']:+.3f} "
    f"(rank {correlation['spearman']:+.3f}).\n"
    f"Spin uncertainty: {kerr_sigma:.4f} in Kerr, {johannsen_sigma:.4f} with alpha13 free "
    f"({johannsen_sigma / kerr_sigma:.0f} times larger).\n"
    "The triplet constrains one combination of spin and deformation; the width along the\n"
    "degeneracy is set by the mass prior, not by the QPOs."
)

print("\n" + bp.plot.latex_table({"emcee": emcee_result, "dynesty": dynesty_result}))

fig = bp.plot.corner(
    emcee_result,
    truths={"alpha13": 0.0, "a": kerr_result.summary()["a"]["median"]},
)
fig.suptitle("GRO J1655-40, RP model, Johannsen metric (lines: Kerr fit)", fontsize=11)
fig.savefig(out / "johannsen_rpm_corner.png", dpi=130)
emcee_result.save(out / "johannsen_rpm_emcee")
dynesty_result.save(out / "johannsen_rpm_dynesty")
print(f"\nwrote {out / 'johannsen_rpm_corner.png'} and the reproducibility bundles")
