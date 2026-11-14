# API reference

## Metric

::: blackpy.metric.base.StaticSpherical
::: blackpy.metric.base.Parameter
::: blackpy.metric.catalogue.schwarzschild.Schwarzschild
::: blackpy.metric.catalogue.reissner_nordstrom.ReissnerNordstrom
::: blackpy.metric.catalogue.schwarzschild_de_sitter.SchwarzschildDeSitter
::: blackpy.metric.catalogue.bardeen.Bardeen
::: blackpy.metric.catalogue.hayward.Hayward
::: blackpy.metric.interpolated.spline

## Geometry

::: blackpy.geometry.horizons.horizons
::: blackpy.geometry.horizons.event_horizon
::: blackpy.geometry.horizons.cosmological_horizon
::: blackpy.geometry.curvature.kretschmann
::: blackpy.geometry.curvature.stress_energy
::: blackpy.geometry.curvature.stress_energy_symbolic
::: blackpy.geometry.curvature.kretschmann_symbolic
::: blackpy.geometry.energy_conditions.energy_conditions
::: blackpy.geometry.report.report

## Orbits

::: blackpy.orbits.circular.angular_velocity
::: blackpy.orbits.circular.specific_energy
::: blackpy.orbits.circular.specific_angular_momentum
::: blackpy.orbits.circular.circular_orbit_region
::: blackpy.orbits.circular.isco
::: blackpy.orbits.circular.marginally_stable_orbits
::: blackpy.orbits.circular.marginally_bound_orbit
::: blackpy.orbits.circular.radial_epicyclic_frequency
::: blackpy.orbits.circular.vertical_epicyclic_frequency
::: blackpy.orbits.frequencies.frequencies_hz
::: blackpy.orbits.frequencies.nodal_precession_frequency
::: blackpy.orbits.qpo.QPOModel
::: blackpy.orbits.qpo.resonance_radius

## Light

::: blackpy.light.photon_sphere.photon_sphere
::: blackpy.light.photon_sphere.photon_spheres
::: blackpy.light.photon_sphere.critical_impact_parameter
::: blackpy.light.shadow.shadow_radius
::: blackpy.light.shadow.shadow_angular_radius

## Data

::: blackpy.data.registry.load
::: blackpy.data.registry.list_datasets
::: blackpy.data.registry.load_file
::: blackpy.data.schema.QPODataset
::: blackpy.data.schema.ShadowDataset
::: blackpy.data.schema.Measurement
::: blackpy.data.schema.MassPrior

## Inference

::: blackpy.infer.fit.Fit
::: blackpy.infer.result.FitResult
::: blackpy.infer.result.BayesFactor
::: blackpy.infer.likelihood.build_problem
::: blackpy.infer.likelihood.simulate_qpo_dataset
::: blackpy.infer.priors.Uniform
::: blackpy.infer.priors.Gaussian
::: blackpy.infer.priors.LogUniform

## Plotting

::: blackpy.plot.profiles.profile
::: blackpy.plot.posterior.corner
::: blackpy.plot.posterior.latex_table

## Constants and numerics

::: blackpy.constants
::: blackpy._numerics.find_roots
::: blackpy._numerics.refine_root
::: blackpy._numerics.derivative
