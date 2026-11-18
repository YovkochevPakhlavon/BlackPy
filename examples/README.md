# Examples

Each script runs on its own from the repository root, for example
`.venv/bin/python examples/03_rn_fit.py`. Figures and result bundles go to
`examples/output/` (gitignored). Every function used here is explained in the
[user guide](../docs/user_guide.md).

**Quick mode.** With the environment variable `BLACKPY_EXAMPLES_QUICK=1` the
sampler settings are reduced (fewer steps, live points and predictive draws),
so that each script finishes in about a minute; the numbers are then less
precise. Without it the scripts run exactly as documented. The weekly CI job
`examples` and `scripts\check_windows.ps1 -ExamplesQuick` use quick mode.

Each script runs on its own: `python examples/NN_name.py`. Figures go to
`examples/output/`. Times are on an Apple M2 laptop (normal mode); with
`BLACKPY_EXAMPLES_QUICK=1` every example takes under a minute.

| Script | What it shows | Time | Output files |
|---|---|---|---|
| `01_new_metric.py` | a new metric from sympy: horizons, ISCO, photon sphere, shadow, energy conditions | ~10 s | – |
| `02_profile_plots.py` | `plot.profile` for several metrics | ~15 s | `profiles.png` |
| `03_rn_fit.py` | synthetic Reissner–Nordström QPO fit with emcee and dynesty, Bayes factor, reproducibility bundle | ~1 min | `rn_fit_corner.png`, `rn_fit.npz` |
| `04_sgra_shadow_constraints.py` | Sgr A\* shadow bound on the RN charge | ~20 s | – |
| `05_johannsen_rpm_fit.py` | Kerr and Johannsen RPM triplet fits, degeneracy | ~1.5 min | `johannsen_rpm_corner.png` |
| `06_rpm_published_results.py` | the analytic RPM solution and fits compared with published values | ~30 s | – |
| `07_rotating_shadows.py` | Kerr shadows, \(\Delta C\) definitions, Kerr–Newman shadow fit | ~1.5 min | `kerr_shadows.png`, `kerr_shadows_sgra_band.png` |
| `08_tutorial_new_metric.py` | the [tutorial](../docs/tutorial.md) as a script | ~2 min | `tutorial_corner.png` |
| `09_fit_quality.py` | goodness of fit and posterior predictive checks | ~25 s | `fit_quality.png` |
| `10_plasma_shadows.py` | shadows in a plasma, plasma–charge degeneracy | ~1 min | `kerr_plasma_shadows.png` |
| `11_m87_and_sgra_shadows.py` | two shadow sources, shared and per-source parameters, mass priors | ~2.5 min | – |
| `12_parametrised_metrics.py` | RZ coefficients, RZ shadow fit, KRZ | ~1.5 min | `rz_shadow_corner.png` |
| `13_fisher_forecast.py` | Fisher forecasts compared with a fit (`plot.forecast` on a corner plot); a forecast not to trust | ~30 s | `forecast_vs_fit.png` |
| `14_lensing.py` | deflection angles (`plot.deflection`), strong deflection limit, Sgr A\* observables | ~30 s | `deflection_angle.png` |
| `15_thin_disks.py` | radiative efficiency and Page–Thorne flux (`plot.disk_flux`) | ~1 min | `thin_disk_flux.png` |
