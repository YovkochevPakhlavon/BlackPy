"""Schwarzschild-de Sitter (Kottler) metric.

Reference
---------
F. Kottler, "Ueber die physikalischen Grundlagen der Einsteinschen
Gravitationstheorie", Ann. Phys. 361 (1918) 401-462.
"""

from __future__ import annotations

import math

import sympy as sp

from blackpy.metric.base import Parameter, StaticSpherical

REFERENCE = "Kottler (1918), Ann. Phys. 361, 401"


def SchwarzschildDeSitter() -> StaticSpherical:
    r"""Schwarzschild-de Sitter black hole, :math:`f = g = 1 - 2/r - \Lambda r^2/3`.

    Units: G = c = M = 1, so ``Lambda`` is the dimensionless product
    :math:`\Lambda M^2`.

    Parameters of the metric
    ------------------------
    Lambda : float, default 0
        Cosmological constant in units of 1/M^2, ``Lambda >= 0``. An event
        horizon and a cosmological horizon both exist for
        ``0 < Lambda < 1/9``.

    Notes
    -----
    The spacetime is **not asymptotically flat** for ``Lambda > 0``:

    * the static region is bounded by the cosmological horizon, so there is
      no observer "at infinity". The shadow must be computed for a static
      observer at a finite radius ``r_obs``;
    * the specific energy is not normalised to one at infinity, so the
      marginally bound orbit is not defined;
    * stable circular orbits exist only between an inner (ISCO) and an outer
      (OSCO) marginally stable orbit, and only for
      ``Lambda < 4/5625`` (Stuchlik & Hledik 1999, Phys. Rev. D 60, 044006).

    The asymptotic behaviour is probed numerically per parameter value, so
    ``Lambda = 0`` is correctly treated as flat.

    Returns
    -------
    StaticSpherical
    """
    r = sp.Symbol("r", positive=True)
    lam = sp.Symbol("Lambda", real=True)
    return StaticSpherical.from_sympy(
        1 - 2 / r - lam * r**2 / 3,
        r=r,
        parameters=[
            Parameter(
                "Lambda",
                0.0,
                (0.0, math.inf),
                latex=r"\Lambda",
                description="cosmological constant in units of 1/M^2",
            )
        ],
        name="Schwarzschild-de Sitter",
        asymptotically_flat=None,
        reference=REFERENCE,
    )
