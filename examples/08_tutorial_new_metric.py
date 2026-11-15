"""Example 8: the tutorial of docs/tutorial.md as one script.

From a new metric to a corner plot and a Bayes factor:

1. define a static metric with sympy and inspect its geometry;
2. generate its rotating counterpart and check what its source is;
3. compute the observables (ISCO, frequencies, shadow);
4. fit it to the QPO triplet of GRO J1655-40 with emcee;
5. compare it with Kerr (Bayes factor, two methods) and make a corner plot.

Run:  python examples/08_tutorial_new_metric.py        (about one minute)
"""

import math
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import sympy as sp  # noqa: E402

import blackpy as bp  # noqa: E402
from blackpy.infer import Fit, Uniform  # noqa: E402

out = Path(__file__).parent / "output"
out.mkdir(exist_ok=True)

# --- Step 1: a new static metric, f = g = 1 - 2/r + beta/r^2 (beta of either sign) ----------
r, beta = sp.symbols("r beta", real=True)
static = bp.StaticSpherical.from_sympy(
    1 - 2 / r + beta / r**2,
    parameters=[bp.metric.Parameter("beta", 0.0, latex=r"\beta", role="deformation")],
    name="1/r^2 metric",
    asymptotically_flat=True,
)
print(bp.geometry.report(static, beta=-0.5), "\n")

# --- Step 2: its rotating counterpart (a model, not a solution: check the source) -----------
metric = bp.metric.rotating_from_static(static, name="rotating 1/r^2 metric")
print(metric)
print("reflection symmetric:", metric.is_reflection_symmetric, "| separable:", metric.is_separable)
print(bp.geometry.horizons(metric, a=0.5, beta=-0.5))
print(bp.geometry.effective_source(metric, a=0.5, beta=-0.5).summary(), "\n")

# --- Step 3: observables ---------------------------------------------------------------------
pars = dict(a=0.3, beta=-0.5)
r_isco = float(bp.orbits.isco(metric, **pars))
freqs = bp.orbits.frequencies_hz(metric, r=6.0, mass=5.4, **pars)
nodal = float(bp.orbits.nodal_precession_frequency(metric, r=6.0, mass=5.4, **pars))
shadow = bp.light.shadow_observables(metric, math.radians(60.0), **pars)
kerr_isco = float(bp.orbits.isco(bp.metric.Kerr(), a=0.3))
print(f"ISCO: {r_isco:.4f} M (Kerr with the same spin: {kerr_isco:.4f} M)")
print(
    f"at r = 6 M, M = 5.4 Msun: nu_phi = {float(freqs.nu_phi):.1f} Hz, "
    f"nu_r = {float(freqs.nu_r):.1f} Hz, nu_theta = {float(freqs.nu_theta):.1f} Hz, "
    f"nodal precession = {nodal:.2f} Hz"
)
print(
    f"shadow at 60 deg: mean radius {float(shadow.mean_radius):.4f} M, "
    f"Delta C = {float(shadow.delta_c):.4f}\n"
)

# --- Step 4: fit the QPO triplet of GRO J1655-40 (RP model) ---------------------------------
data = bp.data.load("qpo/gro_j1655_40")
priors = {"a": Uniform(0.0, 0.998), "beta": Uniform(-2.0, 0.9), "r": Uniform(2.0, 20.0)}
warnings.simplefilter("always")  # show BlackPy's integrity warnings; do not silence them
fit = Fit(
    metric, data, qpo_model="RP", priors=priors, sampler="emcee", nsteps=8000, max_steps=64000
)
result = fit.run(seed=1)
print(result, "\n")

# --- Step 5: model comparison and plots ------------------------------------------------------
print(fit.bayes_factor(null={"beta": 0.0}))  # Savage-Dickey density ratio (emcee)
nested = Fit(metric, data, qpo_model="RP", priors=priors, sampler="dynesty", nlive=400)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    nested.run(seed=2)
    print(nested.bayes_factor(null={"beta": 0.0}))  # evidence ratio (two nested-sampling runs)

fig = bp.plot.corner(result, truths={"beta": 0.0})
fig.savefig(out / "tutorial_corner.png", dpi=130)
result.save(out / "tutorial_fit")
print(f"\nwrote {out / 'tutorial_corner.png'} and the reproducibility bundle")
