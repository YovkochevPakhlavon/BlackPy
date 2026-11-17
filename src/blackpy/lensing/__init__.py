"""Gravitational lensing by static, spherically symmetric metrics.

Exact deflection angles, the weak-deflection (Gauss-Bonnet) estimate, the
strong deflection limit and its observables, with and without a plasma.

There are no observational data for the strong-deflection observables yet
(relativistic images have not been resolved), so this sub-package produces
**predictions, not constraints**: nothing here feeds a likelihood.
"""

from blackpy.lensing.deflection import (
    closest_approach,
    deflection_angle,
    deflection_angle_at,
    weak_deflection_angle,
)
from blackpy.lensing.strong import (
    StrongDeflection,
    StrongFieldObservables,
    einstein_ring_angle,
    strong_deflection_angle,
    strong_deflection_limit,
    strong_field_observables,
)

__all__ = [
    "StrongDeflection",
    "StrongFieldObservables",
    "closest_approach",
    "deflection_angle",
    "deflection_angle_at",
    "einstein_ring_angle",
    "strong_deflection_angle",
    "strong_deflection_limit",
    "strong_field_observables",
    "weak_deflection_angle",
]
