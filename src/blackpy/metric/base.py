"""Metric base class and static, spherically symmetric spacetimes

(rotating metrics: :mod:`blackpy.metric.axisymmetric`). This module implements

    ds^2 = -f(r) dt^2 + dr^2 / g(r) + h(r) (dtheta^2 + sin^2(theta) dphi^2)

in geometric units G = c = 1, with lengths in units of the black hole mass M
(so the Schwarzschild metric is ``f = 1 - 2/r``). Defaults: ``g = f`` and
``h = r^2``.

Design principle: *symbolic in, numeric core*. A metric can be given as sympy
expressions (:meth:`StaticSpherical.from_sympy`) or as plain JAX-traceable
callables. Either way the numerical core only ever sees JAX functions.
"""

from __future__ import annotations

import inspect
import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from blackpy.errors import SymbolicUnavailableError

#: Default radial search range (in units of M) for horizons, orbits and photon spheres.
DEFAULT_R_RANGE: tuple[float, float] = (1e-3, 1e4)


@dataclass(frozen=True)
class Parameter:
    """A named metric parameter.

    Parameters
    ----------
    name : str
        Name used as keyword argument throughout the package.
    default : float or None
        Default value. ``None`` means the value must always be supplied.
    bounds : tuple of float
        Allowed range ``(low, high)``, inclusive. This is the range in which
        the metric functions are well defined, not the (smaller) range in
        which a black hole exists.
    latex : str or None
        LaTeX symbol used in plots and tables.
    description : str
        Human-readable description, including units.
    role : str
        Physical role, used by the inference guards: ``"spin"`` for the
        rotation parameter, ``"deformation"`` for a parameter that deforms
        the metric away from a reference solution, ``"charge"``, or ``""``.
    """

    name: str
    default: float | None = None
    bounds: tuple[float, float] = (-math.inf, math.inf)
    latex: str | None = None
    description: str = ""
    role: str = ""

    def __post_init__(self):
        low, high = self.bounds
        if not low < high:
            raise ValueError(
                f"Parameter '{self.name}': bounds must satisfy low < high, got {self.bounds}."
            )
        if self.default is not None and not (low <= self.default <= high):
            raise ValueError(
                f"Parameter '{self.name}': default {self.default} is outside bounds {self.bounds}."
            )

    @property
    def symbol(self) -> str:
        """LaTeX symbol, falling back to the plain name."""
        return self.latex if self.latex is not None else self.name


@dataclass(frozen=True)
class SymbolicForm:
    """Symbolic (sympy) form of a static, spherically symmetric metric.

    Attributes
    ----------
    r : sympy.Symbol
        Radial coordinate.
    f, g, h : sympy.Expr
        Metric functions.
    params : tuple of sympy.Symbol
        Parameter symbols, in the same order as the metric's parameters.
    """

    r: Any
    f: Any
    g: Any
    h: Any
    params: tuple


class Metric(ABC):
    """Abstract base class of every spacetime in BlackPy.

    Concrete subclasses: :class:`StaticSpherical` and
    :class:`blackpy.metric.StationaryAxisymmetric`. Parameter handling and the
    evaluation engine (:meth:`evaluate`) are shared.
    """

    name: str
    parameters: tuple[Parameter, ...]

    @property
    def param_names(self) -> tuple[str, ...]:
        """Parameter names, in positional order."""
        return tuple(p.name for p in self.parameters)

    @property
    def n_params(self) -> int:
        """Number of metric parameters."""
        return len(self.parameters)

    @property
    @abstractmethod
    def is_spherically_symmetric(self) -> bool:
        """Whether the spacetime is spherically symmetric."""

    @property
    def is_rotating_class(self) -> bool:
        """True for :class:`StationaryAxisymmetric` metrics (general code path)."""
        return False

    def _init_common(self, name, reference, parameters, r_range, reserved=("r",)):
        """Shared part of the constructors: names, parameters, scan range, caches."""
        self.name = name
        self.reference = reference
        self.parameters = parameters
        if len(set(self.param_names)) != len(self.param_names):
            raise ValueError(f"Duplicate parameter names: {self.param_names}.")
        for word in reserved:
            if word in self.param_names:
                raise ValueError(
                    f"'{word}' is reserved for a coordinate and cannot be a parameter name."
                )
        r_min, r_max = float(r_range[0]), float(r_range[1])
        if not 0.0 < r_min < r_max:
            raise ValueError(f"r_range must satisfy 0 < r_min < r_max, got {r_range}.")
        self.r_range = (r_min, r_max)
        self._compiled: dict = {}
        self._symbolic_cache: dict = {}

    # ------------------------------------------------------------------ parameters

    def resolve(self, params: Mapping[str, Any], *, check_bounds: bool = True) -> list:
        """Return parameter values in positional order, filling in defaults.

        Parameters
        ----------
        params : mapping
            ``name -> value`` (scalars or arrays).
        check_bounds : bool
            Validate concrete values against the parameter bounds.

        Returns
        -------
        list
            One value per metric parameter.
        """
        unknown = set(params) - set(self.param_names)
        if unknown:
            raise TypeError(
                f"Metric '{self.name}' has no parameter(s) {sorted(unknown)}. "
                f"Available parameters: {list(self.param_names) or 'none'}."
            )
        values = []
        for par in self.parameters:
            value = params.get(par.name, par.default)
            if value is None:
                raise TypeError(
                    f"Parameter '{par.name}' of metric '{self.name}' has no default value; "
                    f"pass {par.name}=<value>."
                )
            if check_bounds and not isinstance(value, jax.core.Tracer):
                arr = np.asarray(value, dtype=float)
                low, high = par.bounds
                if arr.size and (np.nanmin(arr) < low or np.nanmax(arr) > high):
                    raise ValueError(
                        f"Parameter '{par.name}' of metric '{self.name}' must lie in "
                        f"[{low}, {high}]; got values in [{np.nanmin(arr)}, {np.nanmax(arr)}]."
                    )
            values.append(value)
        return values

    def pack(self, **params) -> jnp.ndarray:
        """Pack keyword parameters into an array of shape ``(..., n_params)``."""
        values = [jnp.asarray(v, dtype=jnp.float64) for v in self.resolve(params)]
        if not values:
            return jnp.zeros((0,), dtype=jnp.float64)
        return jnp.stack(jnp.broadcast_arrays(*values), axis=-1)

    # ------------------------------------------------------------------ evaluation engine

    def evaluate(self, core: Callable, params: Mapping[str, Any], *args, **options):
        """Evaluate a scalar core function with broadcasting, ``vmap`` and ``jit``.

        This is the single entry point through which every public observable
        is computed. ``core(metric, p, *args, **options)`` must be a pure,
        JAX-traceable function of a 1-D parameter array ``p`` and scalar
        ``args``. The compiled function is cached on the metric instance.

        Parameters
        ----------
        core : callable
            Scalar core function.
        params : mapping
            Metric parameters by name (scalars or arrays).
        *args
            Extra array arguments (e.g. radii) broadcast against the parameters.
        **options
            Hashable static options, baked into the compiled function.

        Returns
        -------
        Output of ``core`` with the broadcast shape of the inputs prepended.
        """
        values = self.resolve(params)
        arrays = [jnp.asarray(a, dtype=jnp.float64) for a in (*args, *values)]
        arrays = jnp.broadcast_arrays(*arrays) if arrays else []
        shape = arrays[0].shape if arrays else ()
        size = int(np.prod(shape)) if shape else 1
        flat_args = [a.reshape(size) for a in arrays[: len(args)]]
        if self.n_params:
            p = jnp.stack([a.reshape(size) for a in arrays[len(args) :]], axis=-1)
        else:
            p = jnp.zeros((size, 0), dtype=jnp.float64)

        key = (core, len(args), tuple(sorted(options.items())))
        fn = self._compiled.get(key)
        if fn is None:
            fn = jax.jit(jax.vmap(lambda p_, *a_: core(self, p_, *a_, **options)))
            self._compiled[key] = fn
        out = fn(p, *flat_args)
        return jax.tree_util.tree_map(lambda x: x.reshape(shape + x.shape[1:]), out)


def _as_parameters(parameters) -> tuple[Parameter, ...]:
    """Normalise the ``parameters`` argument to a tuple of :class:`Parameter`."""
    if parameters is None:
        return ()
    if isinstance(parameters, Mapping):
        return tuple(
            v if isinstance(v, Parameter) else Parameter(str(k), None if v is None else float(v))
            for k, v in parameters.items()
        )
    out = []
    for item in parameters:
        if isinstance(item, Parameter):
            out.append(item)
        elif isinstance(item, str):
            out.append(Parameter(item))
        else:
            raise TypeError(
                "parameters must be a sequence of Parameter objects or names, or a mapping "
                f"name -> default; got element {item!r}."
            )
    return tuple(out)


def _parameters_from_signature(fn: Callable, n_coordinates: int = 1) -> tuple[Parameter, ...]:
    """Infer parameters from the signature ``fn(r, *params)`` or ``fn(r, theta, *params)``."""
    sig = inspect.signature(fn)
    names = list(sig.parameters.values())[n_coordinates:]
    out = []
    for p in names:
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            raise TypeError(
                "Cannot infer metric parameters from a callable with *args/**kwargs; "
                "pass parameters=[...] explicitly."
            )
        default = None if p.default is inspect.Parameter.empty else float(p.default)
        out.append(Parameter(p.name, default))
    return tuple(out)


def _wrap(fn: Callable) -> Callable:
    """Wrap ``fn(r, *params)`` as ``fn(r, p)`` with ``p`` a 1-D parameter array."""

    def wrapped(r, p):
        out = fn(r, *[p[i] for i in range(p.shape[-1])])
        # Constant expressions come back as Python scalars; give them r's shape.
        return jnp.asarray(out, dtype=jnp.float64) + jnp.zeros_like(
            jnp.asarray(r, dtype=jnp.float64)
        )

    return wrapped


class StaticSpherical(Metric):
    r"""Static, spherically symmetric metric.

    .. math::

        ds^2 = -f(r)\,dt^2 + \frac{dr^2}{g(r)} + h(r)\,d\Omega^2

    Parameters
    ----------
    f : callable
        ``f(r, *params)``, JAX-traceable (use ``jax.numpy`` functions). Must
        be smooth: the ISCO needs three derivatives of ``f``.
    g : callable, optional
        ``g(r, *params)`` with the same signature. Default: ``g = f``.
    h : callable, optional
        ``h(r, *params)`` with the same signature. Default: ``h = r^2``.
    parameters : sequence or mapping, optional
        Parameter definitions: :class:`Parameter` objects, plain names, or a
        mapping ``name -> default``. If omitted they are read from the
        signature of ``f`` (argument names and default values).
    name : str
        Display name.
    asymptotically_flat : bool or None
        Declare the asymptotic behaviour. ``None`` (default) means "probe
        numerically" at large radius for the parameter values in use; see
        :meth:`is_asymptotically_flat`.
    r_range : tuple of float
        Radial range (units of M) scanned for horizons, photon spheres and
        special orbits. Roots outside this range are not found.
    reference : str
        Literature reference for the metric.

    Notes
    -----
    Units: G = c = 1 and lengths in units of M, unless the metric functions
    carry their own mass parameter.

    Examples
    --------
    >>> import blackpy as bp
    >>> rn = bp.metric.StaticSpherical(lambda r, Q=0.0: 1 - 2 / r + Q**2 / r**2)
    >>> float(bp.orbits.isco(rn, Q=0.0))
    6.0
    """

    def __init__(
        self,
        f: Callable,
        g: Callable | None = None,
        h: Callable | None = None,
        *,
        parameters: Sequence | Mapping | None = None,
        name: str = "custom",
        asymptotically_flat: bool | None = None,
        r_range: tuple[float, float] = DEFAULT_R_RANGE,
        reference: str = "",
    ):
        if not callable(f):
            raise TypeError(
                "f must be a callable f(r, *params). For sympy expressions use "
                "StaticSpherical.from_sympy(...)."
            )
        self._init_common(
            name,
            reference,
            _parameters_from_signature(f) if parameters is None else _as_parameters(parameters),
            r_range,
        )
        self.asymptotically_flat = asymptotically_flat
        self.g_equals_f = g is None
        self.h_is_r_squared = h is None

        self._f_user = f
        self._g_user = f if g is None else g
        self._h_user = (lambda r, *p: r**2) if h is None else h
        self._f = _wrap(self._f_user)
        self._g = self._f if g is None else _wrap(self._g_user)
        self._h = _wrap(self._h_user)
        self._symbolic: SymbolicForm | None = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def from_sympy(
        cls,
        f,
        g=None,
        h=None,
        *,
        r=None,
        parameters: Sequence | Mapping | None = None,
        name: str = "custom",
        asymptotically_flat: bool | None = None,
        r_range: tuple[float, float] = DEFAULT_R_RANGE,
        reference: str = "",
    ) -> StaticSpherical:
        """Build a metric from sympy expressions.

        The expressions are converted once to JAX functions with
        ``sympy.lambdify(..., modules="jax")``; the symbolic form is kept for
        geometry checks and LaTeX output.

        Parameters
        ----------
        f, g, h : sympy.Expr
            Metric functions. Defaults: ``g = f``, ``h = r**2``.
        r : sympy.Symbol, optional
            Radial coordinate. Default: the free symbol named ``r``.
        parameters : sequence or mapping, optional
            Parameter definitions (see :class:`StaticSpherical`); names must
            match the free symbols. If omitted, all free symbols other than
            ``r`` become parameters without defaults, sorted by name.
        name, asymptotically_flat, r_range, reference
            See :class:`StaticSpherical`.

        Returns
        -------
        StaticSpherical
        """
        import sympy as sp

        f = sp.sympify(f)
        g_expr = f if g is None else sp.sympify(g)
        free = set(f.free_symbols) | set(g_expr.free_symbols)
        if h is not None:
            free |= set(sp.sympify(h).free_symbols)
        if r is None:
            candidates = [s for s in free if s.name == "r"]
            if len(candidates) != 1:
                raise ValueError(
                    "Could not identify the radial coordinate: pass r=<sympy symbol> explicitly."
                )
            r = candidates[0]
        h_expr = r**2 if h is None else sp.sympify(h)

        other = sorted(free - {r}, key=lambda s: s.name)
        if parameters is None:
            params = tuple(Parameter(s.name) for s in other)
        else:
            params = _as_parameters(parameters)
            missing = {s.name for s in other} - {p.name for p in params}
            if missing:
                raise ValueError(
                    f"The expressions contain symbols {sorted(missing)} that are not listed in "
                    "parameters."
                )
        by_name = {s.name: s for s in other}
        symbols = tuple(by_name.get(p.name, sp.Symbol(p.name, real=True)) for p in params)

        def lam(expr):
            return sp.lambdify([r, *symbols], expr, modules="jax")

        metric = cls(
            lam(f),
            None if g is None else lam(g_expr),
            None if h is None else lam(h_expr),
            parameters=params,
            name=name,
            asymptotically_flat=asymptotically_flat,
            r_range=r_range,
            reference=reference,
        )
        metric._symbolic = SymbolicForm(r=r, f=f, g=g_expr, h=h_expr, params=symbols)
        return metric

    # ------------------------------------------------------------------ basic properties

    @property
    def is_spherically_symmetric(self) -> bool:
        return True

    @property
    def is_symbolic(self) -> bool:
        """True if the metric was built from sympy expressions."""
        return self._symbolic is not None

    @property
    def symbolic(self) -> SymbolicForm:
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
                "with StaticSpherical.from_sympy(...)."
            )
        return self._symbolic

    def latex(self) -> str:
        """LaTeX string of the line element (symbolic metrics only)."""
        import sympy as sp

        s = self.symbolic
        return (
            r"ds^2 = -\left("
            + sp.latex(s.f)
            + r"\right) dt^2 + \left("
            + sp.latex(s.g)
            + r"\right)^{-1} dr^2 + \left("
            + sp.latex(s.h)
            + r"\right) d\Omega^2"
        )

    def __repr__(self) -> str:
        pars = ", ".join(
            f"{p.name}={p.default}" if p.default is not None else p.name for p in self.parameters
        )
        kind = "symbolic" if self.is_symbolic else "numeric"
        return f"StaticSpherical(name='{self.name}', parameters=[{pars}], {kind})"

    # ------------------------------------------------------------------ metric functions

    def f_array(self, r, p):
        """``f(r)`` with parameters given as a 1-D array ``p`` (JAX-traceable)."""
        return self._f(r, p)

    def g_array(self, r, p):
        """``g(r)`` with parameters given as a 1-D array ``p`` (JAX-traceable)."""
        return self._g(r, p)

    def h_array(self, r, p):
        """``h(r)`` with parameters given as a 1-D array ``p`` (JAX-traceable)."""
        return self._h(r, p)

    def f(self, r, **params):
        """Metric function ``f(r) = -g_tt``; broadcasts over ``r`` and parameters."""
        return self.evaluate(_f_core, params, r)

    def g(self, r, **params):
        """Metric function ``g(r) = 1 / g_rr``; broadcasts over ``r`` and parameters."""
        return self.evaluate(_g_core, params, r)

    def h(self, r, **params):
        """Metric function ``h(r) = g_theta_theta``; broadcasts over ``r`` and parameters."""
        return self.evaluate(_h_core, params, r)

    # ------------------------------------------------------------------ conversion

    def as_axisymmetric(self):
        r"""The same spacetime as a :class:`blackpy.metric.StationaryAxisymmetric`.

        The components are :math:`g_{tt} = -f`, :math:`g_{t\phi} = 0`,
        :math:`g_{rr} = 1/g`, :math:`g_{\theta\theta} = h`,
        :math:`g_{\phi\phi} = h\sin^2\theta`. The static metric is thus the
        special case :math:`g_{t\phi} = 0` of the rotating class. The result
        goes through the general (rotating) code path, which makes it an
        independent check of the specialised static formulas.

        Returns
        -------
        StationaryAxisymmetric
            With the same parameters, name, scan range and reference.
        """
        from blackpy.metric.axisymmetric import StationaryAxisymmetric

        options = {
            "parameters": self.parameters,
            "name": self.name,
            "asymptotically_flat": self.asymptotically_flat,
            "r_range": self.r_range,
            "reference": self.reference,
        }
        if self.is_symbolic:
            import sympy as sp

            s = self.symbolic
            theta = sp.Symbol("theta", real=True)
            metric = StationaryAxisymmetric.from_sympy(
                -s.f,
                sp.S.Zero,
                s.h * sp.sin(theta) ** 2,
                1 / s.g,
                s.h,
                r=s.r,
                theta=theta,
                **options,
            )
        else:
            f, g, h = self._f_user, self._g_user, self._h_user
            metric = StationaryAxisymmetric(
                lambda r, theta, *p: -f(r, *p),
                lambda r, theta, *p: 0.0 * r,
                lambda r, theta, *p: h(r, *p) * jnp.sin(theta) ** 2,
                lambda r, theta, *p: 1.0 / g(r, *p),
                lambda r, theta, *p: h(r, *p),
                **options,
            )
        metric._declared.update(reflection=True, separable=True)
        return metric

    # ------------------------------------------------------------------ asymptotics

    def is_asymptotically_flat(self, **params) -> bool:
        """Whether the metric is asymptotically flat for the given parameters.

        If ``asymptotically_flat`` was declared at construction, that value is
        returned. Otherwise the metric is probed numerically: it is accepted
        as flat if ``f``, ``g`` and ``h / r^2`` are within 1e-4 of one at
        ``r = 1e6`` and within 1e-6 at ``r = 1e9``. This heuristic rejects
        (anti-)de Sitter asymptotics and constant deficits; it also rejects
        flat metrics that approach Minkowski more slowly than ``~ 1/r``, in
        which case declare ``asymptotically_flat=True`` explicitly.
        """
        if self.asymptotically_flat is not None:
            return bool(self.asymptotically_flat)
        return bool(np.all(np.asarray(self.evaluate(_flatness_core, params))))


def _f_core(metric, p, r):
    return metric.f_array(r, p)


def _g_core(metric, p, r):
    return metric.g_array(r, p)


def _h_core(metric, p, r):
    return metric.h_array(r, p)


def _flatness_core(metric, p):
    ok = jnp.asarray(True)
    for radius, tol in ((1e6, 1e-4), (1e9, 1e-6)):
        r = jnp.asarray(radius)
        for value in (metric.f_array(r, p), metric.g_array(r, p), metric.h_array(r, p) / r**2):
            ok = ok & (jnp.abs(value - 1.0) < tol)
    return ok
