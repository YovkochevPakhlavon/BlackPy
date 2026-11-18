# API reference

The **public API** of BlackPy is the set of names listed on this page: the names in
`__all__` of `blackpy` and of its sub-packages and modules, plus `blackpy.errors`.
Within 1.x no public name is removed and no public behaviour changes without a
deprecation period (see `CONTRIBUTING.md`). Everything else is private, even if it
can be imported; the private numerical toolbox is documented at the end of this
page for contributors only.

Units: G = c = 1, lengths in units of M; every angle is in radians unless its name
ends in `_deg`. The table of units of every argument is in the
[user guide](user_guide.md#units).

Top level: `blackpy.StaticSpherical`, `blackpy.StationaryAxisymmetric` (the same
classes as in `blackpy.metric`), `blackpy.__version__`, and the sub-packages below.

## Metric (`blackpy.metric`)

Static and rotating metric classes, the built-in catalogue, parametrised metrics and the rotating generator.

| Name | Meaning |
|---|---|
| `CATALOGUE` | dict: name -> factory of the static catalogue metrics. |
| `ROTATING_CATALOGUE` | dict: name -> factory of the rotating catalogue metrics. |

`blackpy.metric.interpolated.spline` builds metric functions from tables; `blackpy.metric.future` holds the reserved `newman_janis` stub.

::: blackpy.metric.interpolated.spline

::: blackpy.metric.catalogue.simpson_visser.SimpsonVisser
::: blackpy.metric.catalogue.rotating_regular.RotatingHayward
::: blackpy.metric.catalogue.rotating_regular.RotatingBardeen
::: blackpy.metric.catalogue.kerr_sen.KerrSen
::: blackpy.metric.catalogue.gmghs.GMGHS
::: blackpy.metric.parametrized.RZCoefficients
::: blackpy.metric.catalogue.bardeen.Bardeen
::: blackpy.metric.catalogue.hayward.Hayward
::: blackpy.metric.catalogue.johannsen.Johannsen
::: blackpy.metric.parametrized.KRZ
::: blackpy.metric.catalogue.kerr.Kerr
::: blackpy.metric.catalogue.kerr_newman.KerrNewman
::: blackpy.metric.base.Metric
::: blackpy.metric.base.Parameter
::: blackpy.metric.catalogue.reissner_nordstrom.ReissnerNordstrom
::: blackpy.metric.parametrized.RezzollaZhidenko
::: blackpy.metric.catalogue.schwarzschild.Schwarzschild
::: blackpy.metric.catalogue.schwarzschild_de_sitter.SchwarzschildDeSitter
::: blackpy.metric.base.StaticSpherical
::: blackpy.metric.axisymmetric.StationaryAxisymmetric
::: blackpy.metric.base.SymbolicForm
::: blackpy.metric.axisymmetric.SymbolicFormAxisymmetric
::: blackpy.metric.parametrized.krz_deviations
::: blackpy.metric.rotating.rotating_from_static
::: blackpy.metric.parametrized.rz_coefficients

## Geometry (`blackpy.geometry`)

Horizons, curvature, effective source and energy conditions.

::: blackpy.geometry.energy_conditions.EnergyConditionResult
::: blackpy.geometry.horizons.Horizon
::: blackpy.geometry.horizons.HorizonProfile
::: blackpy.geometry.horizons.HorizonResult
::: blackpy.geometry.axisymmetric.SourceResult
::: blackpy.geometry.curvature.StressEnergy
::: blackpy.geometry.axisymmetric.classify_stress_energy
::: blackpy.geometry.horizons.cosmological_horizon
::: blackpy.geometry.axisymmetric.effective_source
::: blackpy.geometry.curvature.einstein_tensor
::: blackpy.geometry.curvature.einstein_tensor_symbolic
::: blackpy.geometry.energy_conditions.energy_conditions
::: blackpy.geometry.horizons.ergosurface
::: blackpy.geometry.horizons.event_horizon
::: blackpy.geometry.horizons.horizon_profile
::: blackpy.geometry.horizons.horizons
::: blackpy.geometry.curvature.is_vacuum_symbolic
::: blackpy.geometry.curvature.kretschmann
::: blackpy.geometry.curvature.kretschmann_symbolic
::: blackpy.geometry.report.report
::: blackpy.geometry.curvature.ricci_scalar
::: blackpy.geometry.curvature.ricci_scalar_symbolic
::: blackpy.geometry.curvature.stress_energy
::: blackpy.geometry.curvature.stress_energy_symbolic

## Orbits (`blackpy.orbits`)

Circular orbits, epicyclic frequencies, QPO models.

| Name | Meaning |
|---|---|
| `QPO_MODELS` | dict: name -> `QPOModel` (RP, ER, TD, WD and variants). |

::: blackpy.orbits.frequencies.Frequencies
::: blackpy.orbits.circular.MarginallyStableOrbit
::: blackpy.orbits.qpo.QPOModel
::: blackpy.orbits.rpm.RPMSolution
::: blackpy.orbits.circular.angular_velocity
::: blackpy.orbits.circular.circular_orbit_exists
::: blackpy.orbits.circular.circular_orbit_region
::: blackpy.orbits.frequencies.frequencies_hz
::: blackpy.orbits.qpo.get_qpo_model
::: blackpy.orbits.circular.isco
::: blackpy.orbits.circular.marginally_bound_orbit
::: blackpy.orbits.circular.marginally_stable_orbits
::: blackpy.orbits.frequencies.nodal_precession_frequency
::: blackpy.orbits.circular.photon_orbit
::: blackpy.orbits.circular.radial_epicyclic_frequency
::: blackpy.orbits.circular.radial_epicyclic_frequency_squared
::: blackpy.orbits.qpo.resonance_radius
::: blackpy.orbits.rpm.rpm_kerr_solution
::: blackpy.orbits.circular.specific_angular_momentum
::: blackpy.orbits.circular.specific_energy
::: blackpy.orbits.circular.vertical_epicyclic_frequency
::: blackpy.orbits.circular.vertical_epicyclic_frequency_squared

## Light (`blackpy.light`)

Photon spheres, shadows, deviation from circularity, plasma.

| Name | Meaning |
|---|---|
| `DELTA_C_DEFINITIONS` | tuple of the names of the deviation-from-circularity definitions (`fractional_rms`, `jp_asymmetry`, `bambi2019`). |

::: blackpy.light.rotating.CriticalParameters
::: blackpy.light.plasma.HomogeneousPlasma
::: blackpy.light.plasma.MetricWithPlasma
::: blackpy.light.photon_sphere.PhotonSphere
::: blackpy.light.plasma.PlasmaModel
::: blackpy.light.plasma.PlasmaProfile
::: blackpy.light.plasma.PowerLawPlasma
::: blackpy.light.plasma.SeparablePlasma
::: blackpy.light.rotating.ShadowContour
::: blackpy.light.rotating.ShadowObservables
::: blackpy.light.photon_sphere.critical_impact_parameter
::: blackpy.light.rotating.critical_impact_parameters
::: blackpy.light.rotating.deviation_from_circularity
::: blackpy.light.rotating.photon_region
::: blackpy.light.photon_sphere.photon_sphere
::: blackpy.light.plasma.photon_sphere_in_plasma
::: blackpy.light.photon_sphere.photon_spheres
::: blackpy.light.plasma.photon_spheres_in_plasma
::: blackpy.light.rotating.require_separable
::: blackpy.light.shadow.shadow_angular_radius
::: blackpy.light.plasma.shadow_angular_radius_in_plasma
::: blackpy.light.rotating.shadow_contour
::: blackpy.light.plasma.shadow_in_plasma
::: blackpy.light.rotating.shadow_observables
::: blackpy.light.shadow.shadow_radius
::: blackpy.light.plasma.with_plasma

## Lensing (`blackpy.lensing`)

Deflection angles and the strong deflection limit (static metrics; predictions only).

::: blackpy.lensing.strong.StrongDeflection
::: blackpy.lensing.strong.StrongFieldObservables
::: blackpy.lensing.deflection.closest_approach
::: blackpy.lensing.deflection.deflection_angle
::: blackpy.lensing.deflection.deflection_angle_at
::: blackpy.lensing.strong.einstein_ring_angle
::: blackpy.lensing.strong.strong_deflection_angle
::: blackpy.lensing.strong.strong_deflection_limit
::: blackpy.lensing.strong.strong_field_observables
::: blackpy.lensing.deflection.weak_deflection_angle

## Thin disks (`blackpy.disk`)

Radiative efficiency and Page-Thorne flux (predictions only).

::: blackpy.disk.DiskProfile
::: blackpy.disk.inner_edge
::: blackpy.disk.page_thorne_flux
::: blackpy.disk.radiative_efficiency

## Data (`blackpy.data`)

The data registry.

::: blackpy.data.schema.CircularityBound
::: blackpy.data.schema.Dataset
::: blackpy.data.schema.MassPrior
::: blackpy.data.schema.Measurement
::: blackpy.data.schema.QPODataset
::: blackpy.data.schema.Reference
`RegistryError` (also importable from `blackpy.errors`): see below.

::: blackpy.data.schema.ShadowDataset
::: blackpy.data.schema.SpinMeasurement
::: blackpy.data.registry.available
`blackpy.data.list` is `list_datasets`:

::: blackpy.data.registry.list_datasets
::: blackpy.data.registry.load
::: blackpy.data.registry.load_file
::: blackpy.data.registry.parse_dataset
::: blackpy.data.registry.registry_path

## Inference (`blackpy.infer`)

Priors, likelihoods, fits, model comparison, fit quality, forecasts.

| Name | Meaning |
|---|---|
| `POOR_FIT_P_VALUE` | float: posterior predictive p-value below which `PoorFitWarning` is raised (0.05). |
| `SAMPLERS` | tuple of the supported samplers (`emcee`, `dynesty`). |

::: blackpy.infer.result.BayesFactor
::: blackpy.infer.fit.Fit
::: blackpy.infer.forecast.Forecast
`ForecastWarning` (defined in `blackpy.errors`): see below.

::: blackpy.infer.result.FitResult
::: blackpy.infer.priors.Gaussian
::: blackpy.infer.quality.GoodnessOfFit
::: blackpy.infer.priors.Isotropic
::: blackpy.infer.priors.LogUniform
::: blackpy.infer.quality.PosteriorPredictive
::: blackpy.infer.priors.Prior
::: blackpy.infer.likelihood.Problem
::: blackpy.infer.priors.Uniform
::: blackpy.infer.likelihood.build_problem
::: blackpy.infer.forecast.forecast
::: blackpy.infer.forecast.forecast_from_problem
::: blackpy.infer.likelihood.inclination_prior
::: blackpy.infer.priors.prior_from_dict
::: blackpy.infer.likelihood.simulate_qpo_dataset
::: blackpy.infer.likelihood.simulate_shadow_dataset

## Plotting (`blackpy.plot`)

Profile plots, shadow contours, corner and residual plots, LaTeX tables.

| Name | Meaning |
|---|---|
| `QUANTITIES` | dict: name -> description of the quantities `plot.profile` can draw. |

::: blackpy.plot.posterior.corner
::: blackpy.plot.posterior.latex_table
::: blackpy.plot.profiles.profile
::: blackpy.plot.posterior.residuals
::: blackpy.plot.shadows.shadow

## Constants (`blackpy.constants`)

Physical constants (SI) and unit conversions.

| Name | Meaning |
|---|---|
| `AU` | astronomical unit [m] |
| `C` | speed of light [m/s] |
| `G` | Newton's constant [m^3 kg^-1 s^-2] |
| `GM_SUN` | solar mass parameter G M_sun [m^3/s] |
| `MICROARCSECOND` | one microarcsecond [rad] |
| `M_SUN` | solar mass [kg] |
| `PARSEC` | parsec [m] |
| `R_SUN_GRAV` | G M_sun / c^2 [m] |
| `T_SUN` | G M_sun / c^3 [s] |

::: blackpy.constants.gravitational_radius_m
::: blackpy.constants.hz_to_omega
::: blackpy.constants.omega_to_hz

## Errors and warnings (`blackpy.errors`)

Every exception and warning class. What triggers each one and what to do about it is explained in the [user guide](user_guide.md#11-integrity-guards-and-errors).

::: blackpy.errors.BlackPyError
::: blackpy.errors.BlackPyWarning
::: blackpy.errors.CircularInferenceWarning
::: blackpy.errors.CircularityDefinitionWarning
::: blackpy.errors.ConvergenceWarning
::: blackpy.errors.DegeneracyWarning
::: blackpy.errors.ForecastWarning
::: blackpy.errors.IgnoredDataWarning
::: blackpy.errors.NodalPrecessionUnavailableError
::: blackpy.errors.NotAsymptoticallyFlatError
::: blackpy.errors.NotSeparableError
::: blackpy.errors.PlasmaPropagationError
::: blackpy.errors.PoorFitWarning
::: blackpy.errors.PriorDominatedWarning
::: blackpy.errors.ReflectionSymmetryError
::: blackpy.errors.RegistryError
::: blackpy.errors.SpinIgnoredWarning
::: blackpy.errors.SymbolicTimeoutError
::: blackpy.errors.SymbolicUnavailableError
::: blackpy.errors.UnverifiedDataWarning

## Reserved interfaces (stubs)

These names exist so that code written against them fails with a clear message; they raise `NotImplementedError`. See the [roadmap](roadmap.md).

::: blackpy.metric.future.newman_janis
::: blackpy.qnm.quasinormal_modes
::: blackpy.thermo.hawking_temperature
::: blackpy.thermo.entropy
::: blackpy.thermo.heat_capacity

## Internals (not part of the public API)

The shared numerical toolbox. These functions may change in any release; they are
documented for contributors.

::: blackpy._numerics.find_roots
::: blackpy._numerics.refine_root
::: blackpy._numerics.derivative
