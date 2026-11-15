"""Interfaces reserved for light-propagation features that are not implemented yet.

Implemented since v0.3 (no longer here): plasma (:mod:`blackpy.light.plasma`). The names
``PlasmaModel`` and ``shadow_in_plasma`` are re-exported for backward compatibility.
"""

from __future__ import annotations

from blackpy.light.plasma import PlasmaModel, shadow_in_plasma

__all__ = ["PlasmaModel", "deflection_angle", "shadow_in_plasma"]

_ROADMAP = (
    "{what} is not implemented in this version of BlackPy. The interface is reserved; see "
    "DESIGN_NOTES.md (roadmap)."
)


def deflection_angle(metric, impact_parameter, **params):
    """Light deflection angle (weak and strong lensing). **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="Gravitational lensing"))
