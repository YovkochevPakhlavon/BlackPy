r"""Geodesic models of twin high-frequency QPOs.

Each model maps the three fundamental frequencies of quasi-circular geodesic
motion, :math:`(\nu_\phi, \nu_r, \nu_\theta)`, to the upper and lower
frequency of the observed twin peak, :math:`(\nu_U, \nu_L)`.

Source of the definitions
-------------------------
All definitions below are transcribed from **Table 1 of Stuchlik & Kolos
(2016), A&A 586, A130, arXiv:1603.07366** ("Models of quasi-periodic
oscillations related to mass and spin of the GRO J1655-40 black hole"):

===== ============================== ==============================
model :math:`\nu_U`                  :math:`\nu_L`
===== ============================== ==============================
RP    :math:`\nu_\phi`               :math:`\nu_\phi - \nu_r`
RP1   :math:`\nu_\theta`             :math:`\nu_\phi - \nu_r`
RP2   :math:`\nu_\phi`               :math:`\nu_\theta - \nu_r`
ER    :math:`\nu_\theta`             :math:`\nu_r`
ER1   :math:`\nu_\theta`             :math:`\nu_\theta - \nu_r`
ER2   :math:`\nu_\theta - \nu_r`     :math:`\nu_r`
ER3   :math:`\nu_\theta + \nu_r`     :math:`\nu_\theta`
ER4   :math:`\nu_\theta + \nu_r`     :math:`\nu_\theta - \nu_r`
ER5   :math:`\nu_r`                  :math:`\nu_\theta - \nu_r`
TD    :math:`\nu_\phi + \nu_r`       :math:`\nu_\phi`
WD    :math:`2\nu_\phi - \nu_r`      :math:`2(\nu_\phi - \nu_r)`
===== ============================== ==============================

In all of them the same paper assigns the low-frequency QPO to the nodal
precession, :math:`\nu_{\rm low} = \nu_\phi - \nu_\theta`.

Original proposals, as cited by Stuchlik & Kolos (2016): RP - Stella &
Vietri (1999), Stella, Vietri & Morsink (1999); RP1 - Bursa (2005); RP2 and
the frequency-ratio method - Stuchlik, Kotrlova & Torok (2013); ER -
Abramowicz & Kluzniak (2001); TD - Cadez, Calvani & Kostic (2008), Kostic et
al. (2009); WD - Kato (2004, 2008).

Spherical symmetry
------------------
In a spherically symmetric spacetime :math:`\nu_\theta = \nu_\phi`, so
several models coincide (RP = RP1 = RP2 = ER1, ER3 = TD) and the nodal
precession frequency vanishes: low-frequency QPOs cannot be modelled with a
static metric (:func:`blackpy.orbits.nodal_precession_frequency` raises an
explicit error).

Rotating metrics
----------------
All eleven models work with any reflection-symmetric
:class:`blackpy.metric.StationaryAxisymmetric` metric: there
:math:`\nu_\theta \ne \nu_\phi`, the models are distinct, and
:meth:`QPOModel.low_frequency` returns the nodal precession frequency.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import jax.numpy as jnp

from blackpy._numerics import find_roots, nanmax_or_nan
from blackpy.geometry.horizons import static_domain_core
from blackpy.orbits.circular import _options, isco_core
from blackpy.orbits.frequencies import frequencies_hz_core, nodal_precession_hz_core

_SOURCE = "Stuchlik & Kolos (2016), A&A 586, A130, Table 1 (arXiv:1603.07366)"


@dataclass(frozen=True)
class QPOModel:
    """A twin-peak QPO model.

    Attributes
    ----------
    name : str
        Short name, e.g. ``"RP"``.
    description : str
        Full name of the model.
    upper, lower : callable
        ``(nu_phi, nu_r, nu_theta) -> nu``; the upper and lower twin-peak
        frequency.
    formula : str
        Human-readable definition.
    reference : str
        Where the definition was taken from.
    """

    name: str
    description: str
    upper: Callable
    lower: Callable
    formula: str
    reference: str = _SOURCE

    def frequencies_core(self, metric, p, r, mass_msun, direction=1):
        """``(nu_U, nu_L)`` in Hz (jit/vmap-compatible)."""
        nu_phi, nu_r, nu_theta = frequencies_hz_core(metric, p, r, mass_msun, direction)
        return self.upper(nu_phi, nu_r, nu_theta), self.lower(nu_phi, nu_r, nu_theta)

    def low_frequency_core(self, metric, p, r, mass_msun, direction=1):
        """``|nu_phi - nu_theta|`` in Hz: the low-frequency QPO of the model."""
        return jnp.abs(nodal_precession_hz_core(metric, p, r, mass_msun, direction))

    def frequencies(self, metric, r, mass, *, direction="prograde", **params):
        """Upper and lower QPO frequency in Hz.

        Parameters
        ----------
        metric : StaticSpherical or StationaryAxisymmetric
        r : float or array
            Orbit radius in units of M.
        mass : float or array
            Black hole mass in solar masses.
        direction : {"prograde", "retrograde"}
            Rotating metrics only.
        **params
            Metric parameters.

        Returns
        -------
        (jax.Array, jax.Array)
            ``(nu_U, nu_L)`` in Hz; NaN where the model is undefined (inside
            the ISCO).
        """
        return metric.evaluate(
            self.frequencies_core, params, r, mass, **_options(metric, direction)
        )

    def low_frequency(self, metric, r, mass, *, direction="prograde", **params):
        r"""Low-frequency QPO of the model, :math:`|\nu_\phi - \nu_\theta|` in Hz.

        Stuchlik & Kolos (2016) assign the low-frequency QPO to the nodal
        precession in all eleven models. Needs a rotating metric; for a
        static metric :class:`blackpy.errors.NodalPrecessionUnavailableError`
        is raised.
        """
        from blackpy.orbits.frequencies import nodal_precession_frequency

        return jnp.abs(nodal_precession_frequency(metric, r, mass, direction=direction, **params))


def _model(name, description, upper, lower, formula) -> QPOModel:
    return QPOModel(name=name, description=description, upper=upper, lower=lower, formula=formula)


#: QPO models by name. Definitions from Stuchlik & Kolos (2016), Table 1.
QPO_MODELS: dict[str, QPOModel] = {
    m.name: m
    for m in (
        _model(
            "RP",
            "relativistic precession",
            lambda k, r, v: k,
            lambda k, r, v: k - r,
            "nu_U = nu_phi, nu_L = nu_phi - nu_r",
        ),
        _model(
            "RP1",
            "relativistic precession, variant 1 (Bursa 2005)",
            lambda k, r, v: v,
            lambda k, r, v: k - r,
            "nu_U = nu_theta, nu_L = nu_phi - nu_r",
        ),
        _model(
            "RP2",
            "total precession (Stuchlik, Kotrlova & Torok 2013)",
            lambda k, r, v: k,
            lambda k, r, v: v - r,
            "nu_U = nu_phi, nu_L = nu_theta - nu_r",
        ),
        _model(
            "ER",
            "epicyclic resonance",
            lambda k, r, v: v,
            lambda k, r, v: r,
            "nu_U = nu_theta, nu_L = nu_r",
        ),
        _model(
            "ER1",
            "epicyclic resonance, variant 1",
            lambda k, r, v: v,
            lambda k, r, v: v - r,
            "nu_U = nu_theta, nu_L = nu_theta - nu_r",
        ),
        _model(
            "ER2",
            "epicyclic resonance, variant 2",
            lambda k, r, v: v - r,
            lambda k, r, v: r,
            "nu_U = nu_theta - nu_r, nu_L = nu_r",
        ),
        _model(
            "ER3",
            "epicyclic resonance, variant 3",
            lambda k, r, v: v + r,
            lambda k, r, v: v,
            "nu_U = nu_theta + nu_r, nu_L = nu_theta",
        ),
        _model(
            "ER4",
            "epicyclic resonance, variant 4",
            lambda k, r, v: v + r,
            lambda k, r, v: v - r,
            "nu_U = nu_theta + nu_r, nu_L = nu_theta - nu_r",
        ),
        _model(
            "ER5",
            "epicyclic resonance, variant 5",
            lambda k, r, v: r,
            lambda k, r, v: v - r,
            "nu_U = nu_r, nu_L = nu_theta - nu_r",
        ),
        _model(
            "TD",
            "tidal disruption",
            lambda k, r, v: k + r,
            lambda k, r, v: k,
            "nu_U = nu_phi + nu_r, nu_L = nu_phi",
        ),
        _model(
            "WD",
            "warped disc",
            lambda k, r, v: 2.0 * k - r,
            lambda k, r, v: 2.0 * (k - r),
            "nu_U = 2 nu_phi - nu_r, nu_L = 2 (nu_phi - nu_r)",
        ),
    )
}

_ALIASES = {"RPM": "RP", "RELATIVISTIC_PRECESSION": "RP", "WARPED_DISC": "WD", "TIDAL": "TD"}


def get_qpo_model(model) -> QPOModel:
    """Look up a QPO model by name (case-insensitive) or pass a :class:`QPOModel` through."""
    if isinstance(model, QPOModel):
        return model
    key = str(model).upper().replace(" ", "_").replace("-", "_")
    key = _ALIASES.get(key, key)
    if key not in QPO_MODELS:
        raise ValueError(f"Unknown QPO model '{model}'. Available models: {sorted(QPO_MODELS)}.")
    return QPO_MODELS[key]


def _resonance_radius_core(metric, p, model=None, ratio=1.5, n_scan=256, direction=1):
    """Outermost radius outside the ISCO where ``nu_U / nu_L = ratio``."""
    _, _, _, r_out = static_domain_core(metric, p, n_scan=n_scan)
    r_isco = isco_core(metric, p, n_scan=n_scan, direction=direction)

    def mismatch(r):
        upper, lower = model.frequencies_core(metric, p, r, 1.0, direction)
        return upper - ratio * lower

    lo = jnp.where(jnp.isfinite(r_isco), r_isco * (1.0 + 1e-9), 1.0)
    hi = jnp.minimum(r_out * (1.0 - 1e-9), 1e3 * lo)
    roots, _ = find_roots(mismatch, lo, hi, n_scan=n_scan, detect_extrema=False)
    best = nanmax_or_nan(roots, jnp.isfinite(roots))
    return jnp.where(jnp.isfinite(r_isco), best, jnp.nan)


def resonance_radius(
    metric,
    model="RP",
    *,
    ratio: float = 1.5,
    n_scan: int = 256,
    direction="prograde",
    **params,
):
    r"""Radius at which the model frequencies are in a given ratio.

    Solves :math:`\nu_U(r) / \nu_L(r) = \mathrm{ratio}` outside the ISCO
    (default 3:2, the ratio commonly observed in microquasars). The ratio is
    independent of the black hole mass.

    Parameters
    ----------
    metric : StaticSpherical
    model : str or QPOModel
        QPO model.
    ratio : float
        Frequency ratio ``nu_U / nu_L``.
    **params
        Metric parameters (scalars or arrays).

    Returns
    -------
    jax.Array
        Outermost solution within 1000 ISCO radii, in units of M; NaN if
        there is none.
    """
    model = get_qpo_model(model)
    return metric.evaluate(
        _resonance_radius_core,
        params,
        model=model,
        ratio=float(ratio),
        n_scan=n_scan,
        **_options(metric, direction),
    )
