"""Null geodesics: photon spheres, critical impact parameters and shadows."""

from blackpy.light.future import PlasmaModel, deflection_angle, shadow_in_plasma
from blackpy.light.photon_sphere import (
    PhotonSphere,
    critical_impact_parameter,
    photon_sphere,
    photon_spheres,
)
from blackpy.light.rotating import (
    CriticalParameters,
    ShadowContour,
    ShadowObservables,
    critical_impact_parameters,
    photon_region,
    require_separable,
    shadow_contour,
    shadow_observables,
)
from blackpy.light.shadow import shadow_angular_radius, shadow_radius

__all__ = [
    "CriticalParameters",
    "PhotonSphere",
    "PlasmaModel",
    "ShadowContour",
    "ShadowObservables",
    "critical_impact_parameter",
    "critical_impact_parameters",
    "deflection_angle",
    "photon_region",
    "photon_sphere",
    "photon_spheres",
    "require_separable",
    "shadow_angular_radius",
    "shadow_contour",
    "shadow_in_plasma",
    "shadow_observables",
    "shadow_radius",
]
