"""Quasinormal modes. **Not implemented (planned for v0.3); interface only.**

Planned: WKB approximation with Pade resummation for static, spherically
symmetric metrics (Konoplya, Zhidenko & Zinhailo 2019, Class. Quantum Grav.
36, 155002), with the effective potential built by automatic
differentiation.
"""

from __future__ import annotations


def quasinormal_modes(metric, *, spin: int = 2, ell: int = 2, overtone: int = 0, **params):
    """Quasinormal frequency ``omega`` in units of 1/M. **Not implemented.**

    Parameters
    ----------
    metric : StaticSpherical
    spin : int
        Spin weight of the perturbing field (0 scalar, 1 electromagnetic, 2 gravitational).
    ell : int
        Multipole number.
    overtone : int
        Overtone number ``n``.
    """
    raise NotImplementedError(
        "Quasinormal modes (WKB + Pade) are not part of the BlackPy MVP or of v0.2 (planned "
        "for v0.3). The interface is reserved; see DESIGN_NOTES.md (roadmap)."
    )
