r"""Light propagation and shadows in a cold, non-magnetised, pressureless plasma.

The plasma has refractive index (Perlick & Tsupko 2022, Phys. Rep. 947, 1,
Eq. (81))

.. math:: n^2 = 1 - \frac{\omega_p(x)^2}{\omega(x)^2},

where :math:`\omega_p` is the electron plasma frequency and :math:`\omega`
the photon frequency measured by a static observer. Light rays are the
solutions of Hamilton's equations for
:math:`\mathcal{H} = \tfrac12(g^{\mu\nu}p_\mu p_\nu + \omega_p^2)` with
:math:`\mathcal{H} = 0` (their Eq. (80)). The constant of motion
:math:`\omega_0 = -p_t` is the photon frequency at infinity;
:math:`\omega(r) = \omega_0/\sqrt{-g_{tt}}`.

Throughout BlackPy a plasma profile is the **dimensionless** function

.. math:: X(r, \theta) = \frac{\omega_p(r, \theta)^2}{\omega_0^2} .

Static metrics
--------------
Perlick, Tsupko & Bisnovatyi-Kogan (2015), Phys. Rev. D 92, 104031
(arXiv:1507.04217). In BlackPy's notation (their :math:`A = f`,
:math:`D = h`), their Eq. (17) is

.. math:: h_{\rm pl}(r)^2 = \frac{h}{f}\,\bigl(1 - f X\bigr).

* Light with constant of motion :math:`\omega_0` propagates only where
  :math:`f X < 1` (their Eq. (14): the local photon frequency exceeds the
  plasma frequency).
* Photon spheres: :math:`\tfrac{d}{dr}h_{\rm pl}^2 = 0` (Eq. (32)); a minimum
  of :math:`h_{\rm pl}^2` is an unstable photon sphere.
* Shadow of a static observer at :math:`r_O` outside the photon sphere:
  :math:`\sin^2\alpha_{\rm sh} = h_{\rm pl}(r_{\rm ph})^2/h_{\rm pl}(r_O)^2`
  (Eq. (43)).

Rotating metrics
----------------
Perlick & Tsupko (2017), Phys. Rev. D 95, 104003 (arXiv:1702.08768). The
Hamilton-Jacobi equation separates on Kerr if and only if (their Eq. (26))

.. math:: \omega_p(r,\theta)^2 = \frac{f_r(r) + f_\theta(\theta)}{r^2 + a^2\cos^2\theta}.

In BlackPy's separability framework (:mod:`blackpy.light.rotating`, with
:math:`W = g_{\theta\theta}`) the plasma adds :math:`W\omega_p^2` to
:math:`W g^{tt}`, so the condition is that :math:`g_{\theta\theta}\,X` is a
sum of a function of ``r`` and a function of ``theta``. For Kerr,
:math:`g_{\theta\theta} = r^2 + a^2\cos^2\theta`, which is exactly their
condition. Profiles that violate it are refused. In particular a
non-constant profile that depends only on ``r`` is not separable on a
rotating metric.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from blackpy._numerics import derivative, find_roots, scan_grid, value_and_derivative
from blackpy.errors import (
    BlackPyError,
    NotAsymptoticallyFlatError,
    NotSeparableError,
    PlasmaPropagationError,
)
from blackpy.geometry.horizons import static_domain_core
from blackpy.metric.base import Metric, Parameter, _as_parameters, _parameters_from_signature

HALF_PI = 0.5 * math.pi

#: Radius (units of M) at which a profile is evaluated to obtain its value "at infinity".
R_INFINITY = 1e12

_EDGE = 1e-9

#: Number of radii on which the propagation condition is tested along the path.
N_PATH = 64


# ----------------------------------------------------------------------------- plasma models


class PlasmaModel(ABC):
    r"""A plasma distribution, :math:`X = \omega_p^2/\omega_0^2`.

    :math:`\omega_0` is the photon frequency at infinity (the constant of
    motion :math:`-p_t`). The profile may have parameters (for example its
    amplitude), which behave like metric parameters: they can be passed as
    keywords to the light functions and sampled in a fit.

    Subclasses implement :meth:`plasma_frequency_squared`. Built-in models:
    :class:`HomogeneousPlasma`, :class:`PowerLawPlasma`,
    :class:`PlasmaProfile` (any radial callable) and
    :class:`SeparablePlasma` (the Perlick-Tsupko class for rotating metrics).

    Attributes
    ----------
    name : str
    parameters : tuple of Parameter
        Plasma parameters, with defaults and bounds.
    """

    name: str = "plasma"
    parameters: tuple[Parameter, ...] = ()

    @abstractmethod
    def plasma_frequency_squared(self, r, *params):
        r"""Dimensionless :math:`\omega_p(r)^2/\omega_0^2` (JAX-traceable).

        Called with the radius and the plasma parameters in the order of
        :attr:`parameters`.
        """

    def frequency_squared_array(self, metric, p, q, r, theta):
        r""":math:`X(r, \theta)` for metric parameters ``p`` and plasma parameters ``q``.

        The default is a spherically symmetric profile; models that depend
        on ``theta`` or on the metric override this method.
        """
        return self.plasma_frequency_squared(r, *[q[i] for i in range(len(self.parameters))])

    @property
    def param_names(self) -> tuple[str, ...]:
        return tuple(par.name for par in self.parameters)

    def __repr__(self) -> str:
        pars = ", ".join(f"{par.name}={par.default}" for par in self.parameters)
        return f"{type(self).__name__}({pars})"


def _amplitude(default: float, description: str) -> Parameter:
    return Parameter(
        "omega_p",
        float(default),
        (0.0, math.inf),
        latex=r"\omega_p/\omega_0",
        description=description,
    )


class HomogeneousPlasma(PlasmaModel):
    r"""Homogeneous plasma, :math:`\omega_p = {\rm const}`.

    Parameters of the model
    -----------------------
    omega_p : float, default 0
        :math:`\omega_p/\omega_0`, the plasma frequency in units of the
        photon frequency at infinity, :math:`0 \le \omega_p/\omega_0 < 1`.

    On a Kerr-like metric (:math:`g_{\theta\theta} = r^2 + a^2\cos^2\theta`)
    a homogeneous plasma satisfies the separability condition (Perlick &
    Tsupko 2017, Eq. (36)). The observer at infinity is inside the plasma,
    where the refractive index is :math:`n_\infty = \sqrt{1 - \omega_p^2/
    \omega_0^2}`; impact parameters are divided by :math:`n_\infty`.
    """

    name = "homogeneous plasma"

    def __init__(self, omega_p: float = 0.0):
        self.parameters = (_amplitude(omega_p, "plasma frequency / photon frequency at infinity"),)

    def plasma_frequency_squared(self, r, omega_p):
        return omega_p**2 + 0.0 * r


class PowerLawPlasma(PlasmaModel):
    r"""Power-law plasma.

    .. math::

        \frac{\omega_p(r)^2}{\omega_0^2}
            = \omega_{p,\rm ref}^2\left(\frac{r_{\rm ref}}{r}\right)^{k}

    With ``r_ref = 1`` (the default, i.e. :math:`r_{\rm ref} = M`) this is
    Eq. (57) of Perlick, Tsupko & Bisnovatyi-Kogan (2015) with
    :math:`\beta_0 = \omega_{p,\rm ref}^2`.

    Parameters
    ----------
    omega_p, k : float
        Default values of the two parameters of the model (below).
    r_ref : float
        Reference radius in units of M at which ``omega_p`` is defined.

    Parameters of the model
    -----------------------
    omega_p : float, default 0
        :math:`\omega_p(r_{\rm ref})/\omega_0 \ge 0`.
    k : float, default 1.5
        Power-law index of :math:`\omega_p^2`, i.e. the electron density
        falls as :math:`r^{-k}`; ``k >= 0``. The default 3/2 is the density
        profile of spherical dust accretion. (It is called ``k`` as in the
        paper, not ``q``, which is the name of the Bardeen parameter.)
    """

    name = "power-law plasma"

    def __init__(self, omega_p: float = 0.0, k: float = 1.5, *, r_ref: float = 1.0):
        if not r_ref > 0:
            raise ValueError("r_ref must be positive.")
        self.r_ref = float(r_ref)
        self.parameters = (
            _amplitude(
                omega_p, f"plasma frequency at r = {r_ref:g} M / photon frequency at infinity"
            ),
            Parameter("k", float(k), (0.0, math.inf), latex="k", description="power-law index"),
        )

    def plasma_frequency_squared(self, r, omega_p, k):
        return omega_p**2 * (self.r_ref / r) ** k


class PlasmaProfile(PlasmaModel):
    r"""User-supplied radial profile.

    Parameters
    ----------
    fn : callable
        ``fn(r, *params)`` returning :math:`\omega_p(r)^2/\omega_0^2`,
        JAX-traceable and smooth (the photon sphere needs two derivatives).
    parameters : sequence or mapping, optional
        Parameter definitions (:class:`blackpy.metric.Parameter` objects,
        names, or a mapping ``name -> default``). Default: read from the
        signature of ``fn``.
    name : str

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> import blackpy as bp
    >>> plasma = bp.light.PlasmaProfile(lambda r, x0=0.2, scale=5.0: x0 * jnp.exp(-r / scale))
    >>> schwarzschild = bp.metric.Schwarzschild()
    >>> float(bp.light.shadow_in_plasma(schwarzschild, plasma, x0=0.0))   # doctest: +SKIP
    5.196152422706632
    """

    def __init__(
        self,
        fn: Callable,
        *,
        parameters: Sequence | Mapping | None = None,
        name: str = "plasma profile",
    ):
        if not callable(fn):
            raise TypeError("fn must be a callable fn(r, *params).")
        self._fn = fn
        self.name = name
        self.parameters = (
            _parameters_from_signature(fn) if parameters is None else _as_parameters(parameters)
        )

    def plasma_frequency_squared(self, r, *params):
        return jnp.asarray(self._fn(r, *params), dtype=jnp.float64) + 0.0 * r


class SeparablePlasma(PlasmaModel):
    r"""The plasma class for which light rays separate on a rotating metric.

    .. math::

        \frac{\omega_p(r,\theta)^2}{\omega_0^2}
            = \frac{f_r(r) + f_\theta(\theta)}{g_{\theta\theta}(r,\theta)} .

    For the Kerr metric :math:`g_{\theta\theta} = r^2 + a^2\cos^2\theta` and
    this is Eq. (26) of Perlick & Tsupko (2017), the necessary and
    sufficient condition for a generalised Carter constant. For a static
    metric :math:`g_{\theta\theta} = h(r)` (their Eq. (66) for
    Schwarzschild); the shadow is then a circle whose size depends on the
    polar angle of the observer.

    Parameters
    ----------
    f_r : callable
        ``f_r(r, *params)``, in units of :math:`M^2\omega_0^2`.
    f_theta : callable, optional
        ``f_theta(theta, *params)`` with the same parameters. Default: zero.
    parameters : sequence or mapping, optional
        Default: read from the signature of ``f_r``.
    name : str
    """

    def __init__(
        self,
        f_r: Callable,
        f_theta: Callable | None = None,
        *,
        parameters: Sequence | Mapping | None = None,
        name: str = "separable plasma",
    ):
        if not callable(f_r) or (f_theta is not None and not callable(f_theta)):
            raise TypeError("f_r and f_theta must be callables f(r, *params), f(theta, *params).")
        self._f_r = f_r
        self._f_theta = f_theta
        self.name = name
        self.parameters = (
            _parameters_from_signature(f_r) if parameters is None else _as_parameters(parameters)
        )

    def plasma_frequency_squared(self, r, *params):
        raise BlackPyError(
            "A SeparablePlasma is defined relative to a metric (it divides by g_thetatheta); "
            "use it through blackpy.light.shadow_in_plasma or blackpy.light.with_plasma."
        )

    def frequency_squared_array(self, metric, p, q, r, theta):
        values = [q[i] for i in range(len(self.parameters))]
        total = jnp.asarray(self._f_r(r, *values), dtype=jnp.float64) + 0.0 * r
        if self._f_theta is not None:
            total = total + jnp.asarray(self._f_theta(theta, *values), dtype=jnp.float64)
        if metric.is_rotating_class:
            return total / metric.gthth_array(r, theta, p)
        return total / metric.h_array(r, p)


# ----------------------------------------------------------------------------- metric + plasma


class MetricWithPlasma(Metric):
    """A metric together with a plasma: the object the light functions work on.

    Its parameters are the metric parameters followed by the plasma
    parameters. Every ``*_array`` method of the underlying metric is
    available and ignores the plasma parameters, so orbits, horizons and
    the separable-shadow code of :mod:`blackpy.light.rotating` accept it
    unchanged; the shadow code additionally picks up the plasma through
    :meth:`plasma_term_array`.

    Create it with :func:`with_plasma`.
    """

    def __init__(self, metric: Metric, plasma: PlasmaModel):
        if isinstance(metric, MetricWithPlasma):
            raise TypeError(
                "This metric already carries a plasma: pass the bare metric (metric.metric) "
                "together with the new plasma."
            )
        if not isinstance(plasma, PlasmaModel):
            raise TypeError(
                "plasma must be a blackpy.light.PlasmaModel, for example HomogeneousPlasma() or "
                "PowerLawPlasma()."
            )
        clash = set(metric.param_names) & set(plasma.param_names)
        if clash:
            raise ValueError(
                f"Plasma parameter(s) {sorted(clash)} have the same name as parameters of metric "
                f"'{metric.name}'. Rename them (PlasmaProfile / SeparablePlasma take "
                "parameters=[...])."
            )
        self.metric = metric
        self.plasma = plasma
        self.n_metric = metric.n_params
        self._init_common(
            f"{metric.name} + {plasma.name}",
            metric.reference,
            tuple(metric.parameters) + tuple(plasma.parameters),
            metric.r_range,
            reserved=(),
        )
        self._axisymmetric = None
        self._separable = None

    def __getattr__(self, name):
        # Only reached when normal attribute lookup fails: forward to the metric.
        if name.startswith("__") or name in ("metric", "plasma", "n_metric"):
            raise AttributeError(name)
        attr = getattr(self.metric, name)
        if name.endswith("_array") and callable(attr):
            n = self.n_metric
            return lambda *args: attr(*args[:-1], args[-1][..., :n])
        return attr

    # ------------------------------------------------------------------ basic properties

    @property
    def is_spherically_symmetric(self) -> bool:
        return self.metric.is_spherically_symmetric

    @property
    def is_rotating_class(self) -> bool:
        return self.metric.is_rotating_class

    @property
    def is_symbolic(self) -> bool:
        return False

    def split(self, p):
        """Split a parameter array into ``(metric parameters, plasma parameters)``."""
        return p[..., : self.n_metric], p[..., self.n_metric :]

    def metric_params(self, params: Mapping) -> dict:
        """The entries of ``params`` that belong to the metric."""
        return {k: v for k, v in params.items() if k in self.metric.param_names}

    def is_asymptotically_flat(self, **params) -> bool:
        return self.metric.is_asymptotically_flat(**self.metric_params(params))

    def as_axisymmetric(self):
        """The same plasma on the axisymmetric form of a static metric."""
        if self.metric.is_rotating_class:
            return self
        if self._axisymmetric is None:
            self._axisymmetric = MetricWithPlasma(self.metric.as_axisymmetric(), self.plasma)
        return self._axisymmetric

    def __repr__(self) -> str:
        return f"MetricWithPlasma({self.metric!r}, {self.plasma!r})"

    # ------------------------------------------------------------------ plasma functions

    def plasma_local_array(self, r, theta, p):
        r"""Local :math:`X = \omega_p^2/\omega_0^2` at ``(r, theta)`` (JAX-traceable)."""
        pm, q = self.split(p)
        return self.plasma.frequency_squared_array(self.metric, pm, q, r, theta)

    def plasma_term_array(self, r, theta, p):
        r""":math:`g_{\theta\theta}X`, the term added to :math:`W g^{tt}` (rotating metrics)."""
        pm, _ = self.split(p)
        return self.metric.gthth_array(r, theta, pm) * self.plasma_local_array(r, theta, p)

    def plasma_infinity_array(self, theta, p):
        r""":math:`X` at spatial infinity in the direction ``theta``."""
        return self.plasma_local_array(jnp.asarray(R_INFINITY), theta, p)

    # ------------------------------------------------------------------ separability

    def _representative(self, params: Mapping) -> np.ndarray:
        if params:
            return np.atleast_2d(np.asarray(self.pack(**params), dtype=float))
        metric_sets = self.metric._parameter_sets({})
        plasma = []
        for par in self.plasma.parameters:
            value = par.default if par.default not in (None, 0.0) else 0.3
            plasma.append(float(np.clip(value, *par.bounds)))
        return np.array([[*row, *plasma] for row in metric_sets])

    def check_separability(self, **params) -> bool:
        r"""Numerical test that light rays in this plasma separate.

        The metric must pass its own test, and
        :math:`\partial_r\partial_\theta(g_{\theta\theta}X) = 0` on a grid of
        ``(r, theta)`` points, for the given parameters or for
        representative values (the condition of Perlick & Tsupko 2017,
        Eq. (26), for Kerr).
        """
        if not self.metric.is_rotating_class:
            return True
        if not self.metric.check_separability(**self.metric_params(params)):
            return False
        r, theta = self.metric._check_points()
        r, theta = jnp.asarray(r.ravel()), jnp.asarray(theta.ravel())

        def residual(p, x, y):
            def d_theta(xx):
                return value_and_derivative(lambda yy: self.plasma_term_array(xx, yy, p), y)[1]

            mixed = value_and_derivative(d_theta, x)[1]
            scale = jnp.abs(self.plasma_term_array(x, y, p)) + 1e-30
            return jnp.abs(mixed) * x / scale

        for p in self._representative(params):
            p = jnp.asarray(p)
            values = np.asarray(jax.vmap(lambda x, y, p=p: residual(p, x, y))(r, theta))
            if not np.all(values[np.isfinite(values)] <= 1e-8):
                return False
        return True

    @property
    def is_separable(self) -> bool:
        if self._separable is None:
            self._separable = bool(self.check_separability())
        return self._separable


def with_plasma(metric: Metric, plasma: PlasmaModel) -> MetricWithPlasma:
    """Combine a metric and a plasma for the light functions.

    The result can be passed to :func:`blackpy.light.shadow_contour`,
    :func:`blackpy.light.shadow_observables`, :func:`blackpy.plot.shadow`
    and the other separable-shadow functions; plasma parameters are given
    as keywords next to the metric parameters. Repeated calls with the same
    two objects return the same instance (so compiled functions are reused).

    Raises
    ------
    NotSeparableError
        For a rotating metric if the plasma violates the separability
        condition of Perlick & Tsupko (2017).
    """
    if not isinstance(plasma, PlasmaModel):
        raise TypeError(
            "plasma must be a blackpy.light.PlasmaModel (for example HomogeneousPlasma(), "
            "PowerLawPlasma(), PlasmaProfile(fn) or SeparablePlasma(f_r, f_theta))."
        )
    if isinstance(metric, MetricWithPlasma):
        raise TypeError(
            f"'{metric.name}' already carries a plasma: pass the bare metric together with the "
            "plasma you want."
        )
    cache = plasma.__dict__.setdefault("_views", {})
    key = id(metric)
    if key not in cache or cache[key][0] is not metric:
        cache[key] = (metric, MetricWithPlasma(metric, plasma))
    view = cache[key][1]
    if metric.is_rotating_class and not view.is_separable:
        if not metric.is_separable:
            from blackpy.light.rotating import require_separable

            require_separable(metric)
        raise NotSeparableError(
            f"Light rays in the plasma '{plasma.name}' do not separate on metric "
            f"'{metric.name}'. Separability requires g_thetatheta * omega_p^2 to be a sum of a "
            "function of r and a function of theta (Perlick & Tsupko 2017, Eq. (26): on Kerr, "
            "omega_p^2 = (f_r(r) + f_theta(theta)) / (r^2 + a^2 cos^2 theta)). A non-constant "
            "profile that depends only on r is not of this form on a rotating metric. Use "
            "blackpy.light.SeparablePlasma(f_r, f_theta) or HomogeneousPlasma; other "
            "distributions need numerical ray tracing, which is outside the scope of BlackPy."
        )
    return view


# ----------------------------------------------------------------------------- static cores


def h_plasma_squared(view: MetricWithPlasma, p, r):
    r""":math:`h_{\rm pl}^2 = (h/f)(1 - fX)` of Perlick et al. (2015), Eq. (17)."""
    f = view.f_array(r, p)
    h = view.h_array(r, p)
    return h / f - h * view.plasma_local_array(r, HALF_PI, p)


def photon_spheres_core(view: MetricWithPlasma, p, n_scan=256, max_roots=6):
    """All photon spheres in the plasma: ``(radii, unstable)`` (NaN padded)."""
    pm, _ = view.split(p)
    _, _, r_in, r_out = static_domain_core(view.metric, pm, n_scan=n_scan)
    h2 = lambda x: h_plasma_squared(view, p, x)
    slope = derivative(h2)
    roots, degenerate = find_roots(
        slope, r_in * (1.0 + _EDGE), r_out * (1.0 - _EDGE), n_scan=n_scan, max_roots=max_roots
    )
    safe = jnp.where(jnp.isfinite(roots), roots, 1.0)
    curvature = jax.vmap(derivative(slope))(safe)
    values = jax.vmap(h2)(safe)
    static = jax.vmap(lambda x: (view.f_array(x, p) > 0.0) & (view.g_array(x, p) > 0.0))(safe)
    valid = jnp.isfinite(roots) & static & (~degenerate) & (values > 0.0)
    return jnp.where(valid, roots, jnp.nan), (curvature > 0.0) & valid


class StaticShadow(NamedTuple):
    """Result of :func:`static_shadow_core`."""

    radius: jax.Array
    alpha: jax.Array
    r_ph: jax.Array
    impact: jax.Array
    propagates: jax.Array


def static_shadow_core(view: MetricWithPlasma, p, r_obs, n_scan=256, finite=False) -> StaticShadow:
    r"""Shadow of a static metric in a plasma (Perlick et al. 2015, Eq. (43)).

    Returns the shadow radius (``b_c / n_infinity`` at infinity,
    :math:`\sqrt{h(r_O)}\sin\alpha_{\rm sh}` at finite distance), the
    angular radius, the photon sphere that forms the shadow, the critical
    impact parameter :math:`b_c = h_{\rm pl}(r_{\rm ph})`, and whether light
    can propagate from the photon sphere to the observer.
    """
    pm, _ = view.split(p)
    _, r_cosmo, r_in, r_out = static_domain_core(view.metric, pm, n_scan=n_scan)
    h2 = lambda x: h_plasma_squared(view, p, x)
    roots, unstable = photon_spheres_core(view, p, n_scan=n_scan, max_roots=3)
    outer = r_obs if finite else jnp.asarray(jnp.inf)
    candidates = unstable & (roots < outer)
    safe = jnp.where(candidates, roots, 1.0)
    values = jnp.where(candidates, jax.vmap(h2)(safe), jnp.inf)
    # A ray is captured if its impact parameter is below h_pl everywhere outside the
    # horizon, so the edge of the shadow is the smallest minimum of h_pl^2.
    index = jnp.argmin(values)
    impact2 = values[index]
    found = jnp.isfinite(impact2)
    r_ph = jnp.where(found, roots[index], jnp.nan)
    safe_ph = jnp.where(found, r_ph, 3.0)
    safe_b2 = jnp.where(found, impact2, 1.0)

    # Propagation along the path, from the photon sphere to the observer (or outwards).
    # If no photon sphere was found, the whole static region is tested instead, so that
    # an opaque plasma is reported as such and not as "no photon sphere".
    edge = r_out * (1.0 - _EDGE)
    far = jnp.minimum(r_obs, edge) if finite else edge
    start = jnp.where(found, safe_ph, r_in * (1.0 + 1e-6))
    grid = scan_grid(start, jnp.maximum(far, start * (1.0 + 1e-9)), N_PATH)
    local = jax.vmap(lambda x: view.f_array(x, p) * view.plasma_local_array(x, HALF_PI, p))(grid)
    path = jax.vmap(h2)(grid)
    propagates = jnp.all(local < 1.0) & (jnp.all(path >= safe_b2 * (1.0 - 1e-10)) | (~found))
    if finite:
        inside_cosmo = jnp.where(jnp.isfinite(r_cosmo), r_obs < r_cosmo, True)
        h2_obs = h2(r_obs)
        ok = found & (r_obs > safe_ph) & inside_cosmo & (h2_obs > 0.0)
        sin_alpha = jnp.sqrt(jnp.clip(safe_b2 / jnp.where(ok, h2_obs, 1.0), 0.0, 1.0))
        alpha = jnp.arcsin(sin_alpha)
        radius = jnp.sqrt(view.h_array(r_obs, p)) * sin_alpha
    else:
        n2 = 1.0 - view.plasma_infinity_array(jnp.asarray(HALF_PI), p)
        ok = found & (n2 > 0.0)
        propagates = propagates & (n2 > 0.0)
        radius = jnp.sqrt(safe_b2 / jnp.where(n2 > 0.0, n2, 1.0))
        alpha = jnp.zeros_like(radius)
    nan = jnp.nan
    return StaticShadow(
        jnp.where(ok & propagates, radius, nan),
        jnp.where(ok & propagates, alpha, nan),
        r_ph,
        jnp.where(found, jnp.sqrt(safe_b2), nan),
        propagates,
    )


def static_shadow_radius_core(view, p, n_scan=256):
    """Shadow radius at infinity of a static metric in a plasma (used by the likelihood)."""
    return static_shadow_core(view, p, jnp.asarray(jnp.inf), n_scan=n_scan, finite=False).radius


# ----------------------------------------------------------------------------- rotating: path


def rotating_propagation_core(view, p, inclination, r_obs, n_scan=256, finite=False):
    r"""Whether the rays that form the shadow rim can reach the observer.

    For a handful of points of the contour the radial potential
    :math:`\mathcal{R}(r) = -T_r + 2P_r\xi - Q_r\xi^2 - \eta` (with the
    plasma term in :math:`T_r`) must be non-negative between the spherical
    photon orbit and the observer: this is the condition that the photon
    frequency exceeds the plasma frequency along the ray, in separated form.
    """
    from blackpy.light.rotating import _radial_parts, _setup, critical_core

    setup = _setup(view, p, inclination, n_scan)
    pm, _ = view.split(p)
    _, _, _, r_out = static_domain_core(view.metric, pm, n_scan=n_scan)
    mid, half = 0.5 * (setup.r_hi + setup.r_lo), 0.5 * (setup.r_hi - setup.r_lo)
    t = jnp.linspace(0.05, math.pi - 0.05, 7)
    r_p = jnp.where(setup.rotating, mid - half * jnp.cos(t), setup.r_sphere)
    far = r_obs if finite else jnp.minimum(r_out * (1.0 - _EDGE), 1e4)

    def ray_ok(radius):
        xi, eta = critical_core(view, p, radius)
        xi = jnp.where(setup.rotating, xi, 0.0)
        eta = jnp.where(setup.rotating, eta, -_radial_parts(view, p, radius)[0])
        grid = scan_grid(radius * (1.0 + 1e-6), jnp.maximum(far, radius * (1.0 + 2e-6)), N_PATH)

        def potential(x):
            t_r, p_r, q_r = _radial_parts(view, p, x)
            return (-t_r + 2.0 * p_r * xi - q_r * xi**2 - eta) / (jnp.abs(t_r) + 1.0)

        return jnp.all(jax.vmap(potential)(grid) >= -1e-9)

    ok = jnp.all(jax.vmap(ray_ok)(jnp.where(jnp.isfinite(r_p), r_p, 3.0)))
    # Without a photon region the path test is void; the condition at the observer is not.
    ok = ok | (~jnp.isfinite(setup.r_sphere))
    if finite:
        local = view.plasma_local_array(r_obs, setup.inclination, p)
        g_tt = view.gtt_array(r_obs, setup.inclination, p)
        return ok & (local * (-g_tt) < 1.0)
    return ok & (view.plasma_infinity_array(setup.inclination, p) < 1.0)


# ----------------------------------------------------------------------------- public API


def _refuse_propagation(metric, plasma, what):
    raise PlasmaPropagationError(
        f"{what}: light of this frequency cannot propagate from the photon sphere (or photon "
        f"region) of '{metric.name}' to the observer through the plasma '{plasma.name}'. "
        "Somewhere along the path the plasma frequency reaches the local photon frequency "
        "(f * omega_p^2 / omega_0^2 >= 1 for a static metric; in general the radial potential "
        "becomes negative), so the plasma is opaque there and no shadow is observed at this "
        "frequency. Lower the plasma density (omega_p) or use a higher photon frequency."
    )


def photon_spheres_in_plasma(metric, plasma: PlasmaModel, *, n_scan: int = 256, **params):
    r"""All photon spheres of a static metric in a plasma (scalar parameters).

    Roots of :math:`\tfrac{d}{dr}h_{\rm pl}^2 = 0` with
    :math:`h_{\rm pl}^2 = (h/f)(1 - f\omega_p^2/\omega_0^2)` (Perlick, Tsupko
    & Bisnovatyi-Kogan 2015, Eqs. (17), (32)). For Schwarzschild and a
    homogeneous plasma there is one, at
    :math:`r_{\rm ph} = [3 - 4x + \sqrt{9 - 8x}]/[2(1 - x)]`,
    :math:`x = \omega_p^2/\omega_0^2` (from their Eq. (33)).

    Parameters
    ----------
    metric : StaticSpherical
    plasma : PlasmaModel
    **params
        Scalar metric and plasma parameters.

    Returns
    -------
    list of blackpy.light.PhotonSphere
        In ascending radius; ``impact_parameter`` is :math:`h_{\rm pl}` at
        the sphere.
    """
    from blackpy.light.photon_sphere import PhotonSphere

    if metric.is_rotating_class:
        raise TypeError(
            "photon_spheres_in_plasma is for static metrics; a rotating metric has a photon "
            "region: use blackpy.light.photon_region(blackpy.light.with_plasma(metric, plasma))."
        )
    view = with_plasma(metric, plasma)
    roots, unstable = (
        np.asarray(v) for v in view.evaluate(photon_spheres_core, params, n_scan=n_scan)
    )
    if roots.ndim != 1:
        raise ValueError(
            "photon_spheres_in_plasma() lists every photon sphere of one configuration, so it "
            "needs scalar parameters; use photon_sphere_in_plasma() for arrays."
        )
    out = []
    for radius, is_unstable in zip(roots, unstable, strict=True):
        if np.isfinite(radius):
            impact = float(np.sqrt(view.evaluate(h_plasma_squared, params, radius)))
            out.append(PhotonSphere(float(radius), not bool(is_unstable), impact))
    return sorted(out, key=lambda sphere: sphere.radius)


def photon_sphere_in_plasma(metric, plasma: PlasmaModel, *, n_scan: int = 256, **params):
    """Radius of the photon sphere that forms the shadow in a plasma (vectorised).

    The unstable photon sphere with the smallest critical impact parameter
    (for a black hole with a monotonic plasma profile there is only one).

    Returns
    -------
    jax.Array
        NaN if there is none.
    """
    if metric.is_rotating_class:
        raise TypeError(
            "photon_sphere_in_plasma is for static metrics; a rotating metric has a photon "
            "region: use blackpy.light.photon_region(blackpy.light.with_plasma(metric, plasma))."
        )
    view = with_plasma(metric, plasma)
    return view.evaluate(static_shadow_core, params, np.inf, n_scan=n_scan, finite=False).r_ph


def shadow_in_plasma(
    metric,
    plasma: PlasmaModel,
    r_obs=None,
    *,
    inclination=None,
    kind: str = "mean",
    n_scan: int = 256,
    **params,
):
    r"""Shadow radius (units of M) of a black hole surrounded by a plasma.

    **Static metrics** (Perlick, Tsupko & Bisnovatyi-Kogan 2015): with
    :math:`h_{\rm pl}^2 = (h/f)(1 - f\omega_p^2/\omega_0^2)` and
    :math:`r_{\rm ph}` the photon sphere in the plasma,

    * ``r_obs=None`` (observer at infinity, asymptotically flat metrics):
      :math:`r_{\rm sh} = h_{\rm pl}(r_{\rm ph})/n_\infty`, where
      :math:`n_\infty^2 = 1 - \omega_p(\infty)^2/\omega_0^2` (one unless the
      plasma extends to infinity);
    * finite ``r_obs`` (static observer outside the photon sphere):
      :math:`\sqrt{h(r_O)}\,\sin\alpha_{\rm sh}` with
      :math:`\sin\alpha_{\rm sh} = h_{\rm pl}(r_{\rm ph})/h_{\rm pl}(r_O)`.

    **Rotating metrics** (Perlick & Tsupko 2017): only plasma distributions
    for which the Hamilton-Jacobi equation separates are accepted;
    ``inclination`` (radians) is required and the result is the mean
    (``kind="mean"``) or area-equivalent (``kind="area"``) radius of the
    contour, as in :func:`blackpy.light.shadow_radius`. For contours and
    shape observables call :func:`blackpy.light.shadow_contour` or
    :func:`blackpy.light.shadow_observables` on
    ``with_plasma(metric, plasma)``.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    plasma : PlasmaModel
    r_obs : float or array, optional
    inclination : float or array, optional
        Rotating metrics: polar angle of the observer in radians. Static
        metrics: used only by :class:`SeparablePlasma` profiles that depend
        on ``theta``.
    kind : {"mean", "area"}
    **params
        Metric and plasma parameters (scalars or arrays), e.g. ``Q=0.3,
        omega_p=0.2``.

    Returns
    -------
    jax.Array

    Raises
    ------
    PlasmaPropagationError
        If light of this frequency cannot propagate from the photon sphere
        to the observer (the plasma frequency exceeds the photon frequency
        somewhere on the path).
    NotSeparableError
        Rotating metrics with a plasma outside the separable class.
    NotAsymptoticallyFlatError
        If ``r_obs`` is None and the metric is not asymptotically flat.
    """
    from blackpy.light.rotating import check_inclination_radians

    check_inclination_radians(inclination, "shadow_in_plasma()")
    view = with_plasma(metric, plasma)
    finite = r_obs is not None
    if not finite and not view.is_asymptotically_flat(**params):
        raise NotAsymptoticallyFlatError(
            f"Metric '{metric.name}' is not asymptotically flat for these parameters: pass "
            "r_obs=<radius of the observer>."
        )
    theta_dependent = isinstance(plasma, SeparablePlasma) and plasma._f_theta is not None
    if metric.is_rotating_class or (theta_dependent and inclination is not None):
        from blackpy.light.shadow import shadow_radius

        if inclination is None:
            raise TypeError(
                f"The shadow of the rotating metric '{metric.name}' depends on the observer's "
                "inclination: pass inclination=<polar angle in radians>."
            )
        rotating = view.as_axisymmetric()
        ok = rotating.evaluate(
            rotating_propagation_core,
            params,
            inclination,
            r_obs if finite else np.inf,
            n_scan=n_scan,
            finite=finite,
        )
        if not np.all(np.asarray(ok)):
            _refuse_propagation(metric, plasma, "shadow_in_plasma")
        return shadow_radius(
            rotating, r_obs, inclination=inclination, kind=kind, n_scan=n_scan, **params
        )
    out = view.evaluate(
        static_shadow_core, params, r_obs if finite else np.inf, n_scan=n_scan, finite=finite
    )
    if not np.all(np.asarray(out.propagates)):
        _refuse_propagation(metric, plasma, "shadow_in_plasma")
    if np.all(np.isnan(np.asarray(out.radius))):
        raise ValueError(
            f"shadow_in_plasma: no shadow could be computed for '{metric.name}' in "
            f"'{plasma.name}'. Either there is no unstable photon sphere for these parameters "
            "(see blackpy.light.photon_spheres_in_plasma), or the observer is not outside it "
            "in the static region."
        )
    return out.radius


def shadow_angular_radius_in_plasma(
    metric, plasma: PlasmaModel, r_obs, *, n_scan: int = 256, **params
):
    r"""Angular radius :math:`\alpha_{\rm sh}` (radians) of the shadow in a plasma.

    Static metrics, static observer at ``r_obs`` outside the photon sphere:
    :math:`\sin^2\alpha_{\rm sh} = h_{\rm pl}(r_{\rm ph})^2/h_{\rm pl}(r_O)^2`
    (Perlick, Tsupko & Bisnovatyi-Kogan 2015, Eq. (43)).
    """
    if metric.is_rotating_class:
        raise TypeError(
            "shadow_angular_radius_in_plasma is for static metrics; use "
            "blackpy.light.shadow_contour(with_plasma(metric, plasma), inclination, r_obs)."
        )
    view = with_plasma(metric, plasma)
    out = view.evaluate(static_shadow_core, params, r_obs, n_scan=n_scan, finite=True)
    if not np.all(np.asarray(out.propagates)):
        _refuse_propagation(metric, plasma, "shadow_angular_radius_in_plasma")
    return out.alpha
