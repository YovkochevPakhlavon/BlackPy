"""Geometry of the spacetime: horizons, curvature, stress-energy, energy conditions."""

from blackpy.geometry.axisymmetric import SourceResult, classify_stress_energy, effective_source
from blackpy.geometry.curvature import (
    StressEnergy,
    einstein_tensor,
    einstein_tensor_symbolic,
    is_vacuum_symbolic,
    kretschmann,
    kretschmann_symbolic,
    ricci_scalar,
    ricci_scalar_symbolic,
    stress_energy,
    stress_energy_symbolic,
)
from blackpy.geometry.energy_conditions import EnergyConditionResult, energy_conditions
from blackpy.geometry.horizons import (
    Horizon,
    HorizonProfile,
    HorizonResult,
    cosmological_horizon,
    ergosurface,
    event_horizon,
    horizon_profile,
    horizons,
)
from blackpy.geometry.report import report

__all__ = [
    "EnergyConditionResult",
    "Horizon",
    "HorizonProfile",
    "HorizonResult",
    "SourceResult",
    "StressEnergy",
    "classify_stress_energy",
    "cosmological_horizon",
    "effective_source",
    "einstein_tensor",
    "einstein_tensor_symbolic",
    "energy_conditions",
    "ergosurface",
    "event_horizon",
    "horizon_profile",
    "horizons",
    "is_vacuum_symbolic",
    "kretschmann",
    "kretschmann_symbolic",
    "report",
    "ricci_scalar",
    "ricci_scalar_symbolic",
    "stress_energy",
    "stress_energy_symbolic",
]
