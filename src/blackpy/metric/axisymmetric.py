r"""Stationary, axisymmetric metrics in Boyer-Lindquist-like coordinates.

.. math::

    ds^2 = g_{tt}\,dt^2 + 2 g_{t\phi}\,dt\,d\phi + g_{\phi\phi}\,d\phi^2
           + g_{rr}\,dr^2 + g_{\theta\theta}\,d\theta^2 ,

with all components functions of :math:`(r, \theta)` and of the metric
parameters. Units: G = c = 1 and lengths in units of the mass M, signature
(-,+,+,+).

The design principle is the same as for :class:`blackpy.metric.StaticSpherical`
("symbolic in, numeric core"): the components are given either as sympy
expressions (:meth:`StationaryAxisymmetric.from_sympy`) or as JAX-traceable
callables ``fn(r, theta, *params)``. The numerical core only sees JAX
functions.

Notation used throughout the rotating code
------------------------------------------
On the equatorial plane ``theta = pi/2``:

``A = -g_tt``, ``B = g_tphi``, ``C = g_phiphi``, and
``rho2 = B^2 + A C = g_tphi^2 - g_tt g_phiphi`` (minus the determinant of the
(t, phi) block; it vanishes on a Killing horizon). The squared lapse of
zero-angular-momentum observers is ``N^2 = rho2 / g_phiphi``; for a static
metric it equals ``f``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from blackpy._numerics import value_and_derivative
from blackpy.errors import SymbolicUnavailableError
from blackpy.metric.base import (
    DEFAULT_R_RANGE,
    Metric,
    Parameter,
    _as_parameters,
    _parameters_from_signature,
)

HALF_PI = 0.5 * math.pi

#: Relative tolerance of the numerical symmetry and separability checks.
CHECK_RTOL = 1e-8

_COMPONENTS = ("g_tt", "g_tphi", "g_phiphi", "g_rr", "g_thth")


@dataclass(frozen=True)
class SymbolicFormAxisymmetric:
    """Symbolic (sympy) form of a stationary, axisymmetric metric.

    Attributes
    ----------
    r, theta : sympy.Symbol
        Coordinates.
    g_tt, g_tphi, g_phiphi, g_rr, g_thth : sympy.Expr
        Covariant metric components.
    params : tuple of sympy.Symbol
        Parameter symbols, in the same order as the metric's parameters.
    """

    r: Any
    theta: Any
    g_tt: Any
    g_tphi: Any
    g_phiphi: Any
    g_rr: Any
    g_thth: Any
    params: tuple

    @property
    def components(self) -> tuple:
        """``(g_tt, g_tphi, g_phiphi, g_rr, g_thth)``."""
        return (self.g_tt, self.g_tphi, self.g_phiphi, self.g_rr, self.g_thth)


def _wrap(fn: Callable) -> Callable:
    """Wrap ``fn(r, theta, *params)`` as ``fn(r, theta, p)`` with ``p`` a 1-D array."""

    def wrapped(r, theta, p):
        out = fn(r, theta, *[p[i] for i in range(p.shape[-1])])
        shape = jnp.zeros_like(
            jnp.asarray(r, dtype=jnp.float64) + jnp.asarray(theta, dtype=jnp.float64)
        )
        return jnp.asarray(out, dtype=jnp.float64) + shape

    return wrapped


class StationaryAxisymmetric(Metric):
    r"""Stationary, axisymmetric metric in Boyer-Lindquist-like coordinates.

    Parameters
    ----------
    g_tt, g_tphi, g_phiphi, g_rr, g_thth : callable
        Covariant components as JAX-traceable callables
        ``fn(r, theta, *params)`` (use ``jax.numpy`` functions). They must be
        smooth: the ISCO needs three radial derivatives.
    parameters : sequence or mapping, optional
        Parameter definitions (:class:`Parameter` objects, names, or a
        mapping ``name -> default``). If omitted they are read from the
        signature of ``g_tt``.
    name : str
        Display name.
    asymptotically_flat : bool or None
        Declare the asymptotic behaviour; ``None`` means "probe numerically".
    r_range : tuple of float
        Radial range (units of M) scanned for horizons and special orbits.
    reference : str
        Literature reference.
    domain : callable, optional
        ``domain(*params) -> bool`` (JAX-traceable): the part of parameter
        space in which the metric describes a regular black hole exterior
        (for example ``|a| <= 1`` for Kerr). It is *not* enforced when
        observables are computed; the inference module uses it to bound the
        spin and deformation parameters.

    Notes
    -----
    Equatorial circular orbits and epicyclic frequencies require symmetry
    under :math:`\theta \to \pi - \theta` (:attr:`is_reflection_symmetric`).
    The analytic shadow requires separable null geodesics
    (:attr:`is_separable`).

    Examples
    --------
    >>> import blackpy as bp
    >>> kerr = bp.metric.Kerr()
    >>> round(float(bp.orbits.isco(kerr, a=0.0)), 12)
    6.0
    """

    def __init__(
        self,
        g_tt: Callable,
        g_tphi: Callable,
        g_phiphi: Callable,
        g_rr: Callable,
        g_thth: Callable,
        *,
        parameters: Sequence | Mapping | None = None,
        name: str = "custom",
        asymptotically_flat: bool | None = None,
        r_range: tuple[float, float] = DEFAULT_R_RANGE,
        reference: str = "",
        domain: Callable | None = None,
    ):
        user = (g_tt, g_tphi, g_phiphi, g_rr, g_thth)
        for label, fn in zip(_COMPONENTS, user, strict=True):
            if not callable(fn):
                raise TypeError(
                    f"{label} must be a callable {label}(r, theta, *params). For sympy "
                    "expressions use StationaryAxisymmetric.from_sympy(...)."
                )
        self._init_common(
            name,
            reference,
            _parameters_from_signature(g_tt, n_coordinates=2)
            if parameters is None
            else _as_parameters(parameters),
            r_range,
            reserved=("r", "theta"),
        )
        self.asymptotically_flat = asymptotically_flat
        self._user = user
        self._fns = tuple(_wrap(fn) for fn in user)
        self._domain_user = domain
        self._symbolic: SymbolicFormAxisymmetric | None = None
        self._checks: dict = {}
        #: Set by constructors that know the answer (catalogue, generators).
        self._declared: dict = {}

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_sympy(
        cls,
        g_tt,
        g_tphi,
        g_phiphi,
        g_rr,
        g_thth,
        *,
        r=None,
        theta=None,
        parameters: Sequence | Mapping | None = None,
        name: str = "custom",
        asymptotically_flat: bool | None = None,
        r_range: tuple[float, float] = DEFAULT_R_RANGE,
        reference: str = "",
        domain: Callable | None = None,
    ) -> StationaryAxisymmetric:
        """Build a metric from sympy expressions.

        Parameters
        ----------
        g_tt, g_tphi, g_phiphi, g_rr, g_thth : sympy.Expr
            Covariant components as functions of ``r`` and ``theta``.
        r, theta : sympy.Symbol, optional
            Coordinates. Default: the free symbols named ``r`` and ``theta``.
        parameters : sequence or mapping, optional
            Parameter definitions; names must match the free symbols. If
            omitted, all other free symbols become parameters without
            defaults, sorted by name.
        name, asymptotically_flat, r_range, reference, domain
            See :class:`StationaryAxisymmetric`.

        Returns
        -------
        StationaryAxisymmetric
        """
        import sympy as sp

        exprs = tuple(sp.sympify(e) for e in (g_tt, g_tphi, g_phiphi, g_rr, g_thth))
        free = set().union(*[e.free_symbols for e in exprs])

        def coordinate(symbol, label):
            if symbol is not None:
                return symbol
            candidates = [s for s in free if s.name == label]
            if len(candidates) != 1:
                raise ValueError(
                    f"Could not identify the coordinate '{label}': pass {label}=<sympy symbol> "
                    "explicitly."
                )
            return candidates[0]

        r = coordinate(r, "r")
        theta = coordinate(theta, "theta")
        other = sorted(free - {r, theta}, key=lambda s: s.name)
        if parameters is None:
            params = tuple(Parameter(s.name) for s in other)
        else:
            params = _as_parameters(parameters)
            missing = {s.name for s in other} - {p.name for p in params}
            if missing:
                raise ValueError(
                    f"The expressions contain symbols {sorted(missing)} that are not listed in "
                    "parameters. Add them, e.g. parameters={'name': default_value}."
                )
        by_name = {s.name: s for s in other}
        symbols = tuple(by_name.get(p.name, sp.Symbol(p.name, real=True)) for p in params)
        fns = [sp.lambdify([r, theta, *symbols], e, modules="jax") for e in exprs]
        metric = cls(
            *fns,
            parameters=params,
            name=name,
            asymptotically_flat=asymptotically_flat,
            r_range=r_range,
            reference=reference,
            domain=domain,
        )
        metric._symbolic = SymbolicFormAxisymmetric(r, theta, *exprs, params=symbols)
        return metric

    # ------------------------------------------------------------------ basic properties

    @property
    def is_spherically_symmetric(self) -> bool:
        """Always False: spherical symmetry is not assumed for this class."""
        return False

    @property
    def is_rotating_class(self) -> bool:
        return True

    @property
    def is_symbolic(self) -> bool:
        """True if the metric was built from sympy expressions."""
        return self._symbolic is not None

    @property
    def symbolic(self) -> SymbolicFormAxisymmetric:
        """Symbolic form of the metric.

        Raises
        ------
        SymbolicUnavailableError
            If the metric was defined with plain callables.
        """
        if self._symbolic is None:
            raise SymbolicUnavailableError(
                f"Metric '{self.name}' was defined with numerical callables, so symbolic results "
                "are not available. Use the numerical functions instead, or define the metric "
                "with StationaryAxisymmetric.from_sympy(...)."
            )
        return self._symbolic

    def latex(self) -> str:
        """LaTeX string of the line element (symbolic metrics only)."""
        import sympy as sp

        s = self.symbolic
        terms = [
            (s.g_tt, "dt^2"),
            (2 * s.g_tphi, r"dt\,d\phi"),
            (s.g_phiphi, r"d\phi^2"),
            (s.g_rr, "dr^2"),
            (s.g_thth, r"d\theta^2"),
        ]
        return "ds^2 = " + " + ".join(
            r"\left(" + sp.latex(e) + r"\right) " + d for e, d in terms if e != 0
        )

    def __repr__(self) -> str:
        pars = ", ".join(
            f"{p.name}={p.default}" if p.default is not None else p.name for p in self.parameters
        )
        kind = "symbolic" if self.is_symbolic else "numeric"
        return f"StationaryAxisymmetric(name='{self.name}', parameters=[{pars}], {kind})"

    @property
    def spin_parameter(self) -> str | None:
        """Name of the parameter with ``role == "spin"`` (None if there is none)."""
        for par in self.parameters:
            if par.role == "spin":
                return par.name
        return None

    # ------------------------------------------------------------------ metric functions

    def gtt_array(self, r, theta, p):
        """``g_tt(r, theta)`` with parameters as a 1-D array ``p`` (JAX-traceable)."""
        return self._fns[0](r, theta, p)

    def gtphi_array(self, r, theta, p):
        """``g_tphi(r, theta)`` with parameters as a 1-D array ``p`` (JAX-traceable)."""
        return self._fns[1](r, theta, p)

    def gphiphi_array(self, r, theta, p):
        """``g_phiphi(r, theta)`` with parameters as a 1-D array ``p`` (JAX-traceable)."""
        return self._fns[2](r, theta, p)

    def grr_array(self, r, theta, p):
        """``g_rr(r, theta)`` with parameters as a 1-D array ``p`` (JAX-traceable)."""
        return self._fns[3](r, theta, p)

    def gthth_array(self, r, theta, p):
        """``g_thetatheta(r, theta)`` with parameters as a 1-D array ``p`` (JAX-traceable)."""
        return self._fns[4](r, theta, p)

    def components_array(self, r, theta, p):
        """``(g_tt, g_tphi, g_phiphi, g_rr, g_thth)`` at ``(r, theta)`` (JAX-traceable)."""
        return tuple(fn(r, theta, p) for fn in self._fns)

    def rho2_array(self, r, theta, p):
        """``rho2 = g_tphi^2 - g_tt g_phiphi`` (zero on a Killing horizon)."""
        g_tt, g_tphi, g_phiphi, _, _ = self.components_array(r, theta, p)
        return g_tphi**2 - g_tt * g_phiphi

    def lapse2_array(self, r, theta, p):
        """Squared lapse ``N^2 = rho2 / g_phiphi`` of zero-angular-momentum observers."""
        g_tt, g_tphi, g_phiphi, _, _ = self.components_array(r, theta, p)
        return -g_tt + g_tphi**2 / g_phiphi

    def grr_inverse_array(self, r, theta, p):
        """``g^rr = 1 / g_rr`` (its roots are the horizon candidates)."""
        return 1.0 / self.grr_array(r, theta, p)

    def matrix_array(self, r, theta, p):
        """Covariant metric as a 4x4 array, coordinates ``(t, r, theta, phi)``."""
        g_tt, g_tphi, g_phiphi, g_rr, g_thth = self.components_array(r, theta, p)
        zero = jnp.zeros_like(g_tt)
        return jnp.array(
            [
                [g_tt, zero, zero, g_tphi],
                [zero, g_rr, zero, zero],
                [zero, zero, g_thth, zero],
                [g_tphi, zero, zero, g_phiphi],
            ]
        )

    def in_domain_array(self, p):
        """Whether ``p`` lies in the declared black hole domain (True if none was declared)."""
        if self._domain_user is None:
            return jnp.asarray(True)
        return jnp.asarray(self._domain_user(*[p[i] for i in range(p.shape[-1])]), dtype=bool)

    def in_domain(self, **params):
        """Whether the parameters lie in the declared black hole domain (vectorised)."""
        return self.evaluate(_in_domain_core, params)

    def g_tt(self, r, theta, **params):
        """Component ``g_tt``; broadcasts over ``r``, ``theta`` and parameters."""
        return self.evaluate(_component_core, params, r, theta, index=0)

    def g_tphi(self, r, theta, **params):
        """Component ``g_tphi``; broadcasts over ``r``, ``theta`` and parameters."""
        return self.evaluate(_component_core, params, r, theta, index=1)

    def g_phiphi(self, r, theta, **params):
        """Component ``g_phiphi``; broadcasts over ``r``, ``theta`` and parameters."""
        return self.evaluate(_component_core, params, r, theta, index=2)

    def g_rr(self, r, theta, **params):
        """Component ``g_rr``; broadcasts over ``r``, ``theta`` and parameters."""
        return self.evaluate(_component_core, params, r, theta, index=3)

    def g_thth(self, r, theta, **params):
        """Component ``g_thetatheta``; broadcasts over ``r``, ``theta`` and parameters."""
        return self.evaluate(_component_core, params, r, theta, index=4)

    # ------------------------------------------------------------------ symmetry checks

    def _sample_parameters(self, n: int = 3, seed: int = 0) -> np.ndarray:
        """Representative parameter sets for the numerical checks, shape ``(n, n_params)``.

        Finite bounds: uniform in the central 80% of the range. Otherwise the
        default value (or the finite bound, or zero) plus an offset between
        0.2 and 0.9. This is a heuristic; use ``check_...(**params)`` to test
        specific values.
        """
        rng = np.random.default_rng(seed)
        columns = []
        for par in self.parameters:
            low, high = par.bounds
            u = rng.uniform(0.2, 0.9, size=n)
            if math.isfinite(low) and math.isfinite(high):
                col = low + (0.1 + 0.8 * rng.uniform(size=n)) * (high - low)
            elif math.isfinite(low):
                col = max(low, par.default if par.default is not None else low) + u
            elif math.isfinite(high):
                col = min(high, par.default if par.default is not None else high) - u
            else:
                col = (par.default if par.default is not None else 0.0) + u
            columns.append(col)
        if not columns:
            return np.zeros((n, 0))
        return np.column_stack(columns)

    def _check_points(self):
        r = np.array([2.9, 4.3, 7.7, 19.0])
        theta = np.array([0.35, 0.8, 1.2, 1.45])
        return np.meshgrid(r, theta, indexing="ij")

    def _parameter_sets(self, params: Mapping[str, Any]) -> np.ndarray:
        if params or not self.n_params:
            return np.atleast_2d(np.asarray(self.pack(**params), dtype=float))
        return self._sample_parameters()

    def check_reflection_symmetry(self, **params) -> bool:
        r"""Numerical test of the symmetry :math:`\theta \to \pi - \theta`.

        All five components are compared at mirrored points of an
        ``(r, theta)`` grid, for the given parameters or, if none are given,
        for representative values.
        """
        r, theta = self._check_points()
        for p in self._parameter_sets(params):
            p = jnp.asarray(p)
            upper = np.asarray(jnp.stack(self.components_array(r, theta, p)))
            lower = np.asarray(jnp.stack(self.components_array(r, math.pi - theta, p)))
            scale = np.maximum(np.abs(upper), np.abs(lower)) + 1e-300
            if not np.all(np.abs(upper - lower) <= CHECK_RTOL * scale, where=np.isfinite(scale)):
                return False
        return True

    @property
    def is_reflection_symmetric(self) -> bool:
        r"""Whether the metric is symmetric under :math:`\theta \to \pi - \theta`.

        Checked symbolically when the metric was built from sympy
        expressions (falling back to the numerical test if sympy cannot
        decide), numerically otherwise. The result is cached.
        """
        if "reflection" in self._declared:
            return self._declared["reflection"]
        if "reflection" not in self._checks:
            decided = None
            if self.is_symbolic:
                decided = self._reflection_symmetric_symbolic()
            self._checks["reflection"] = (
                self.check_reflection_symmetry() if decided is None else decided
            )
        return self._checks["reflection"]

    def _reflection_symmetric_symbolic(self) -> bool | None:
        """True/False if sympy can decide, None otherwise."""
        import sympy as sp

        s = self.symbolic
        for expr in s.components:
            diff = expr.subs(s.theta, sp.pi - s.theta) - expr
            if diff == 0:
                continue
            if sp.simplify(diff) != 0:
                return None  # undecided: let the numerical test answer
        return True

    def check_separability(self, **params) -> bool:
        r"""Numerical test of the separability of null geodesics.

        The null Hamilton-Jacobi equation separates in these coordinates if,
        with :math:`W = g_{\theta\theta}`, the functions
        :math:`W g^{tt}`, :math:`W g^{t\phi}`, :math:`W g^{\phi\phi}` are
        each a sum of a function of ``r`` and a function of ``theta``, and
        :math:`W g^{rr}` depends on ``r`` only (see ``docs/theory.md``). The
        mixed derivatives are evaluated by automatic differentiation on an
        ``(r, theta)`` grid, for the given parameters or for representative
        values. This is a sufficient condition; it covers Kerr,
        Kerr-Newman, the Johannsen metric and Azreg-Ainou-generated metrics.
        """
        r, theta = self._check_points()
        r, theta = jnp.asarray(r.ravel()), jnp.asarray(theta.ravel())
        for p in self._parameter_sets(params):
            p_array = jnp.asarray(p)
            residual_at = jax.vmap(lambda x, y, q=p_array: _separability_residual(self, q, x, y))
            residual = np.asarray(residual_at(r, theta))
            if not np.all(residual[np.isfinite(residual)] <= CHECK_RTOL):
                return False
        return True

    @property
    def is_separable(self) -> bool:
        """Whether null geodesics separate (numerical test, cached); see
        :meth:`check_separability`."""
        if "separable" in self._declared:
            return self._declared["separable"]
        if "separable" not in self._checks:
            self._checks["separable"] = self.check_separability()
        return self._checks["separable"]

    # ------------------------------------------------------------------ asymptotics

    def is_asymptotically_flat(self, **params) -> bool:
        """Whether the metric is asymptotically flat for the given parameters.

        If declared at construction, that value is returned. Otherwise the
        components are probed at ``r = 1e6`` and ``r = 1e9`` (tolerances 1e-4
        and 1e-6) at two polar angles: ``-g_tt``, ``g_rr``,
        ``g_thth / r^2`` and ``g_phiphi / (r sin theta)^2`` must approach
        one and ``g_tphi / r`` zero.
        """
        if self.asymptotically_flat is not None:
            return bool(self.asymptotically_flat)
        return bool(np.all(np.asarray(self.evaluate(_flatness_core, params))))


def _component_core(metric, p, r, theta, index=0):
    return metric._fns[index](r, theta, p)


def _in_domain_core(metric, p):
    return metric.in_domain_array(p)


def _flatness_core(metric, p):
    ok = jnp.asarray(True)
    for radius, tol in ((1e6, 1e-4), (1e9, 1e-6)):
        r = jnp.asarray(radius)
        for angle in (HALF_PI, 1.0):
            th = jnp.asarray(angle)
            g_tt, g_tphi, g_phiphi, g_rr, g_thth = metric.components_array(r, th, p)
            for value in (-g_tt, g_rr, g_thth / r**2, g_phiphi / (r * jnp.sin(th)) ** 2):
                ok = ok & (jnp.abs(value - 1.0) < tol)
            ok = ok & (jnp.abs(g_tphi / r) < tol)
    return ok


def separable_functions(metric, p, r, theta):
    r"""The four functions that must split for null geodesics to separate.

    Returns ``(W g^tt, W g^tphi, W g^phiphi, W g^rr)`` with
    ``W = g_thetatheta``, at the point ``(r, theta)``.
    """
    g_tt, g_tphi, g_phiphi, g_rr, g_thth = metric.components_array(r, theta, p)
    rho2 = g_tphi**2 - g_tt * g_phiphi
    return (
        -g_thth * g_phiphi / rho2,
        g_thth * g_tphi / rho2,
        -g_thth * g_tt / rho2,
        g_thth / g_rr,
    )


def _separability_residual(metric, p, r, theta):
    """Largest scaled violation of the separability conditions at ``(r, theta)``."""

    def stacked(x, y):
        return jnp.stack(separable_functions(metric, p, x, y))

    values = stacked(r, theta)

    def d_theta(x):
        return value_and_derivative(lambda y: stacked(x, y), theta)[1]

    mixed = value_and_derivative(d_theta, r)[1]
    scale = jnp.abs(values) + 1.0
    sums = jnp.abs(mixed[:3]) * r / scale[:3]
    radial = jnp.abs(d_theta(r)[3]) / scale[3]
    return jnp.maximum(jnp.max(sums), radial)
