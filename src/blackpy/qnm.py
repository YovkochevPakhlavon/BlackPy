"""Quasinormal modes. **Not implemented and not scheduled; interface only.**

Quasinormal modes were removed from the roadmap of version 0.3 on purpose.
Within BlackPy's scope (a metric without field equations) only test-field
modes could be computed, and those cannot be compared with ringdown
observations for non-vacuum metrics, so the feature would not feed a
likelihood. The function below reserves the name and raises
``NotImplementedError``. See the 'Roadmap' page of the documentation.
"""

from __future__ import annotations

from blackpy._links import ROADMAP

__all__ = ["quasinormal_modes"]


def quasinormal_modes(metric, *, spin: int = 2, ell: int = 2, overtone: int = 0, **params):
    """Quasinormal frequency ``omega`` in units of 1/M. **Not implemented, not scheduled.**

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
        "Quasinormal modes are not implemented in BlackPy and are not scheduled for a release: "
        "without field equations only test-field modes could be computed, and these cannot be "
        "compared with ringdown observations for non-vacuum metrics. The interface is "
        "reserved; see " + ROADMAP + "."
    )
