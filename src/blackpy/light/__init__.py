"""Null geodesics: photon spheres, critical impact parameters and shadows."""

from blackpy.light.future import deflection_angle
from blackpy.light.photon_sphere import (
    PhotonSphere,
    critical_impact_parameter,
    photon_sphere,
    photon_spheres,
)
from blackpy.light.plasma import (
    HomogeneousPlasma,
    MetricWithPlasma,
    PlasmaModel,
    PlasmaProfile,
    PowerLawPlasma,
    SeparablePlasma,
    photon_sphere_in_plasma,
    photon_spheres_in_plasma,
    shadow_angular_radius_in_plasma,
    shadow_in_plasma,
    with_plasma,
)
from blackpy.light.rotating import (
    DELTA_C_DEFINITIONS,
    CriticalParameters,
    ShadowContour,
    ShadowObservables,
    critical_impact_parameters,
    deviation_from_circularity,
    photon_region,
    require_separable,
    shadow_contour,
    shadow_observables,
)
from blackpy.light.shadow import shadow_angular_radius, shadow_radius

__all__ = [
    "DELTA_C_DEFINITIONS",
    "CriticalParameters",
    "HomogeneousPlasma",
    "MetricWithPlasma",
    "PhotonSphere",
    "PlasmaModel",
    "PlasmaProfile",
    "PowerLawPlasma",
    "SeparablePlasma",
    "ShadowContour",
    "ShadowObservables",
    "critical_impact_parameter",
    "critical_impact_parameters",
    "deflection_angle",
    "deviation_from_circularity",
    "photon_region",
    "photon_sphere",
    "photon_sphere_in_plasma",
    "photon_spheres",
    "photon_spheres_in_plasma",
    "require_separable",
    "shadow_angular_radius",
    "shadow_angular_radius_in_plasma",
    "shadow_contour",
    "shadow_in_plasma",
    "shadow_observables",
    "shadow_radius",
    "with_plasma",
]
