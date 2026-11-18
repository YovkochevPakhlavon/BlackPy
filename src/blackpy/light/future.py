"""Deprecated names kept for backward compatibility.

Plasma (:mod:`blackpy.light.plasma`) and lensing (:mod:`blackpy.lensing`) are
implemented since v0.3. The names reserved for them in earlier versions still
import from here, but emit a :class:`DeprecationWarning` (since 1.0) and will be
removed in 2.0:

* ``blackpy.light.future.PlasmaModel`` -> :class:`blackpy.light.PlasmaModel`
* ``blackpy.light.future.shadow_in_plasma`` -> :func:`blackpy.light.shadow_in_plasma`
* ``blackpy.light.future.deflection_angle`` -> :func:`blackpy.lensing.deflection_angle`
"""

from __future__ import annotations

from blackpy._deprecation import alias_getattr

_ALIASES = {
    "PlasmaModel": ("blackpy.light.plasma", "PlasmaModel", "blackpy.light.PlasmaModel"),
    "shadow_in_plasma": (
        "blackpy.light.plasma",
        "shadow_in_plasma",
        "blackpy.light.shadow_in_plasma",
    ),
    "deflection_angle": (
        "blackpy.lensing.deflection",
        "deflection_angle",
        "blackpy.lensing.deflection_angle",
    ),
}

#: No public names: everything here is deprecated.
__all__: list[str] = []

__getattr__ = alias_getattr(__name__, _ALIASES)
