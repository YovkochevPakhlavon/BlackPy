"""Null geodesics: photon spheres, critical impact parameters and shadows."""

from blackpy._deprecation import alias_getattr
from blackpy.light import future  # noqa: F401  (deprecated aliases, kept importable)
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

# ``blackpy.light.deflection_angle`` was an alias of ``blackpy.lensing.deflection_angle``
# (v0.3); it is deprecated since 1.0 and not part of ``__all__``.
__getattr__ = alias_getattr(
    __name__,
    {
        "deflection_angle": (
            "blackpy.lensing.deflection",
            "deflection_angle",
            "blackpy.lensing.deflection_angle",
        ),
    },
)

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
