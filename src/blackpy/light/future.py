"""Interfaces reserved for post-MVP light-propagation features (not implemented)."""

from __future__ import annotations

from abc import ABC, abstractmethod

_ROADMAP = (
    "{what} is not part of the BlackPy MVP. The interface is reserved; see DESIGN_NOTES.md "
    "(roadmap)."
)


class PlasmaModel(ABC):
    """Refractive medium for light propagation. **Interface only.**

    A plasma enters the photon Hamiltonian through its plasma frequency
    ``omega_p(r)``; the photon sphere condition and the shadow formula then
    involve ``h(r) (1/f(r) - omega_p(r)^2 / omega_0^2)`` instead of
    ``h(r) / f(r)`` (Perlick, Tsupko & Bisnovatyi-Kogan 2015, Phys. Rev. D
    92, 104031). Subclasses will implement :meth:`plasma_frequency_squared`.
    """

    @abstractmethod
    def plasma_frequency_squared(self, r, *params):
        """Squared plasma frequency at radius ``r`` (JAX-traceable)."""


def deflection_angle(metric, impact_parameter, **params):
    """Light deflection angle (weak and strong lensing). **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="Gravitational lensing"))


def shadow_in_plasma(metric, plasma: PlasmaModel, r_obs=None, **params):
    """Shadow radius in a plasma. **Not implemented.**"""
    raise NotImplementedError(_ROADMAP.format(what="Light propagation in plasma"))
