"""Black hole thermodynamics. **Not implemented (planned for v0.3); interface only.**

Planned: Hawking temperature ``T = sqrt(f'(r_h) g'(r_h)) / (4 pi)``, entropy,
heat capacity and phase-transition diagnostics.
"""

from __future__ import annotations

_MESSAGE = (
    "Thermodynamics is not part of the BlackPy MVP or of v0.2 (planned for v0.3). The interface "
    "is reserved; see DESIGN_NOTES.md (roadmap)."
)


def hawking_temperature(metric, **params):
    """Hawking temperature of the event horizon. **Not implemented.**"""
    raise NotImplementedError(_MESSAGE)


def entropy(metric, **params):
    """Bekenstein-Hawking entropy. **Not implemented.**"""
    raise NotImplementedError(_MESSAGE)


def heat_capacity(metric, **params):
    """Heat capacity at fixed parameters. **Not implemented.**"""
    raise NotImplementedError(_MESSAGE)
