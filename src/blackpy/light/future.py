"""Names kept for backward compatibility.

Plasma (:mod:`blackpy.light.plasma`) and lensing (:mod:`blackpy.lensing`) are implemented
since v0.3. Their names reserved in earlier versions still import from here.
"""

from __future__ import annotations

from blackpy.lensing.deflection import deflection_angle
from blackpy.light.plasma import PlasmaModel, shadow_in_plasma

__all__ = ["PlasmaModel", "deflection_angle", "shadow_in_plasma"]
