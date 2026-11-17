# API reference

## Metric

::: blackpy.metric.base.StaticSpherical
::: blackpy.metric.axisymmetric.StationaryAxisymmetric
::: blackpy.metric.base.Parameter
::: blackpy.metric.rotating.rotating_from_static
::: blackpy.metric.catalogue.kerr.Kerr
::: blackpy.metric.catalogue.kerr_newman.KerrNewman
::: blackpy.metric.catalogue.johannsen.Johannsen
::: blackpy.metric.catalogue.schwarzschild.Schwarzschild
::: blackpy.metric.catalogue.reissner_nordstrom.ReissnerNordstrom
::: blackpy.metric.catalogue.schwarzschild_de_sitter.SchwarzschildDeSitter
::: blackpy.metric.catalogue.bardeen.Bardeen
::: blackpy.metric.catalogue.hayward.Hayward
::: blackpy.metric.catalogue.gmghs.GMGHS
::: blackpy.metric.catalogue.simpson_visser.SimpsonVisser
::: blackpy.metric.catalogue.kerr_sen.KerrSen
::: blackpy.metric.catalogue.rotating_regular.RotatingBardeen
::: blackpy.metric.catalogue.rotating_regular.RotatingHayward
::: blackpy.metric.parametrized.RezzollaZhidenko
::: blackpy.metric.parametrized.rz_coefficients
::: blackpy.metric.parametrized.RZCoefficients
::: blackpy.metric.parametrized.KRZ
::: blackpy.metric.parametrized.krz_deviations
::: blackpy.metric.interpolated.spline

## Geometry

::: blackpy.geometry.horizons.horizons
::: blackpy.geometry.horizons.event_horizon
::: blackpy.geometry.horizons.cosmological_horizon
::: blackpy.geometry.horizons.horizon_profile
::: blackpy.geometry.horizons.ergosurface
::: blackpy.geometry.curvature.kretschmann
::: blackpy.geometry.curvature.ricci_scalar
::: blackpy.geometry.curvature.einstein_tensor
::: blackpy.geometry.curvature.einstein_tensor_symbolic
::: blackpy.geometry.axisymmetric.effective_source
::: blackpy.geometry.axisymmetric.SourceResult
::: blackpy.geometry.axisymmetric.classify_stress_energy
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
::: blackpy.orbits.circular.photon_orbit
::: blackpy.orbits.circular.radial_epicyclic_frequency
::: blackpy.orbits.circular.vertical_epicyclic_frequency
::: blackpy.orbits.frequencies.frequencies_hz
::: blackpy.orbits.frequencies.nodal_precession_frequency
::: blackpy.orbits.qpo.QPOModel
::: blackpy.orbits.qpo.resonance_radius
::: blackpy.orbits.rpm.rpm_kerr_solution

## Light

::: blackpy.light.photon_sphere.photon_sphere
::: blackpy.light.photon_sphere.photon_spheres
::: blackpy.light.photon_sphere.critical_impact_parameter
::: blackpy.light.shadow.shadow_radius
::: blackpy.light.shadow.shadow_angular_radius
::: blackpy.light.rotating.shadow_contour
::: blackpy.light.rotating.shadow_observables
::: blackpy.light.rotating.critical_impact_parameters
::: blackpy.light.rotating.photon_region
::: blackpy.light.rotating.deviation_from_circularity
::: blackpy.light.plasma.shadow_in_plasma
::: blackpy.light.plasma.shadow_angular_radius_in_plasma
::: blackpy.light.plasma.photon_spheres_in_plasma
::: blackpy.light.plasma.photon_sphere_in_plasma
::: blackpy.light.plasma.with_plasma
::: blackpy.light.plasma.PlasmaModel
::: blackpy.light.plasma.HomogeneousPlasma
::: blackpy.light.plasma.PowerLawPlasma
::: blackpy.light.plasma.PlasmaProfile
::: blackpy.light.plasma.SeparablePlasma

## Lensing

::: blackpy.lensing.deflection.deflection_angle
::: blackpy.lensing.deflection.deflection_angle_at
::: blackpy.lensing.deflection.closest_approach
::: blackpy.lensing.deflection.weak_deflection_angle
::: blackpy.lensing.strong.strong_deflection_limit
::: blackpy.lensing.strong.strong_deflection_angle
::: blackpy.lensing.strong.strong_field_observables
::: blackpy.lensing.strong.einstein_ring_angle

## Thin disks

::: blackpy.disk.radiative_efficiency
::: blackpy.disk.page_thorne_flux
::: blackpy.disk.inner_edge

## Data

::: blackpy.data.registry.load
::: blackpy.data.registry.list_datasets
::: blackpy.data.registry.load_file
::: blackpy.data.schema.QPODataset
::: blackpy.data.schema.ShadowDataset
::: blackpy.data.schema.Measurement
::: blackpy.data.schema.MassPrior
::: blackpy.data.schema.CircularityBound

## Inference

::: blackpy.infer.fit.Fit
::: blackpy.infer.result.FitResult
::: blackpy.infer.result.BayesFactor
::: blackpy.infer.likelihood.build_problem
::: blackpy.infer.likelihood.simulate_qpo_dataset
::: blackpy.infer.likelihood.simulate_shadow_dataset
::: blackpy.infer.likelihood.inclination_prior
::: blackpy.infer.quality.GoodnessOfFit
::: blackpy.infer.quality.PosteriorPredictive
::: blackpy.infer.forecast.forecast
::: blackpy.infer.forecast.Forecast
::: blackpy.infer.priors.Uniform
::: blackpy.infer.priors.Gaussian
::: blackpy.infer.priors.LogUniform
::: blackpy.infer.priors.Isotropic

## Plotting

::: blackpy.plot.profiles.profile
::: blackpy.plot.posterior.corner
::: blackpy.plot.posterior.latex_table
::: blackpy.plot.posterior.residuals
::: blackpy.plot.shadows.shadow

## Constants and numerics

::: blackpy.constants
::: blackpy._numerics.find_roots
::: blackpy._numerics.refine_root
::: blackpy._numerics.derivative
