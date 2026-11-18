r"""Parametrised black hole metrics: Rezzolla-Zhidenko (static) and KRZ (rotating).

These are not solutions of any field equations. They are families of metrics
whose parameters measure deviations from Schwarzschild and Kerr, so that a
fit constrains deviation parameters instead of one theory at a time.

References (both read for BlackPy 0.3)
--------------------------------------
L. Rezzolla & A. Zhidenko, "New parametrization for spherically symmetric
black holes in metric theories of gravity", Phys. Rev. D 90 (2014) 084009,
arXiv:1407.3086 ("RZ").

R. Konoplya, L. Rezzolla & A. Zhidenko, "General parametrization of
axisymmetric black holes in metric theories of gravity", Phys. Rev. D 93
(2016) 064015, arXiv:1602.02378 ("KRZ").
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
import sympy as sp

from blackpy._numerics import value_and_derivative
from blackpy.errors import BlackPyError
from blackpy.metric.axisymmetric import StationaryAxisymmetric
from blackpy.metric.base import Parameter, StaticSpherical

RZ_REFERENCE = "Rezzolla & Zhidenko (2014), Phys. Rev. D 90, 084009"
KRZ_REFERENCE = "Konoplya, Rezzolla & Zhidenko (2016), Phys. Rev. D 93, 064015"


def _continued_fraction(coefficients, x):
    r""":math:`c_1/(1 + c_2x/(1 + c_3x/(1 + \dots)))` truncated after the last coefficient."""
    if not coefficients:
        return sp.S.Zero
    tail = sp.S.One
    for c in reversed(coefficients[1:]):
        tail = 1 + c * x / tail
    return coefficients[0] / tail


# ----------------------------------------------------------------------------- Rezzolla-Zhidenko


def RezzollaZhidenko(order: int = 2, ppn: str = "gr") -> StaticSpherical:
    r"""Rezzolla-Zhidenko parametrised static, spherically symmetric black hole.

    RZ Eqs. (1)-(7), (19), with :math:`f = N^2`, :math:`g = N^2/B^2`,
    :math:`h = r^2` in BlackPy's notation:

    .. math::

        x = 1 - \frac{r_0}{r}, \qquad N^2 = x\,A(x),

        A = 1 - \epsilon(1-x) + (a_0 - \epsilon)(1-x)^2 + \tilde A(x)(1-x)^3, \qquad
        B = 1 + b_0(1-x) + \tilde B(x)(1-x)^2,

        \tilde A = \cfrac{a_1}{1 + \cfrac{a_2x}{1 + \cfrac{a_3x}{1 + \dots}}}, \qquad
        \tilde B = \cfrac{b_1}{1 + \cfrac{b_2x}{1 + \cfrac{b_3x}{1 + \dots}}} .

    **Unit of length.** BlackPy measures lengths in units of the ADM mass
    :math:`M`. In the RZ expansion the horizon radius :math:`r_0` is a
    parameter and :math:`\epsilon = 2M/r_0 - 1` (RZ Eq. (14)), so here

    .. math:: r_0 = \frac{2}{1 + \epsilon}\ \ (\text{in units of } M) .

    :math:`\epsilon > 0` is a horizon smaller than Schwarzschild's. This
    holds as long as :math:`A(x) > 0` for :math:`0 \le x \le 1` (RZ Eq. (5));
    with only :math:`\epsilon` non-zero that means :math:`\epsilon < 1/2`.
    If :math:`A` has a zero outside :math:`r_0`, that zero is the event
    horizon, and :func:`blackpy.geometry.event_horizon` returns it.

    Parameters
    ----------
    order : int
        Truncation order :math:`n \ge 0` of the continued fractions: the
        coefficients :math:`a_1 \dots a_n` and :math:`b_1 \dots b_n` are kept
        and :math:`a_{n+1} = b_{n+1} = 0` (RZ, end of section II). ``order=0``
        keeps only :math:`\epsilon` (and :math:`a_0, b_0`).
    ppn : {"gr", "free"}
        Treatment of the lowest coefficients, which RZ relate to the PPN
        parameters, :math:`a_0 = (\beta-\gamma)(1+\epsilon)^2/2` and
        :math:`b_0 = (\gamma-1)(1+\epsilon)/2` (RZ Eqs. (15)-(16)).

        * ``"gr"`` (default): :math:`a_0 = b_0 = 0`, i.e. :math:`\beta =
          \gamma = 1`. This **assumes** that the solar-system bounds
          :math:`|\beta - 1| \lesssim 2.3\times10^{-4}`, :math:`|\gamma - 1|
          \lesssim 2.3\times10^{-5}` (RZ Eq. (10)) also hold for the black
          hole, i.e. that the object has no charge-like hair that changes
          the :math:`1/r^2` term of the metric.
        * ``"free"``: :math:`a_0` and :math:`b_0` are parameters. Needed to
          represent, for example, Reissner-Nordstrom (:math:`a_0 =
          Q^2/r_0^2`) or the dilaton black hole of RZ section IV.

    Parameters of the metric
    ------------------------
    epsilon : float, default 0
        :math:`\epsilon > -1`; fixes the horizon radius.
    a0, b0 : float, default 0
        Only for ``ppn="free"``.
    a1 ... a<order>, b1 ... b<order> : float, default 0
        Near-horizon coefficients. With all parameters zero the metric is
        Schwarzschild exactly. ``b_i`` enter only :math:`g_{rr}`, so photon
        orbits, the shadow, the ISCO and the frequencies depend on
        :math:`\epsilon` and the ``a_i`` only (RZ, section V).

    Returns
    -------
    StaticSpherical
        Built from sympy expressions, so the symbolic geometry checks work.

    Notes
    -----
    This is a parametrised model, not a solution: check what source it
    would need with :func:`blackpy.geometry.energy_conditions`. Coefficients
    are expected to be strongly degenerate in a fit to few data. The
    continued fractions have poles for some coefficient values (a
    denominator vanishing outside the horizon); such points have no
    regular black hole and are reported as such.

    See Also
    --------
    rz_coefficients : the RZ coefficients of any static metric.
    """
    if not isinstance(order, int) or order < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}.")
    if ppn not in ("gr", "free"):
        raise ValueError("ppn must be 'gr' (a0 = b0 = 0) or 'free' (a0, b0 are parameters).")
    r = sp.Symbol("r", positive=True)
    epsilon = sp.Symbol("epsilon", real=True)
    a = sp.symbols(f"a1:{order + 1}", real=True) if order else ()
    b = sp.symbols(f"b1:{order + 1}", real=True) if order else ()
    a0, b0 = (sp.Symbol("a0", real=True), sp.Symbol("b0", real=True)) if ppn == "free" else (0, 0)
    r0 = 2 / (1 + epsilon)
    x = 1 - r0 / r
    big_a = (
        1
        - epsilon * (1 - x)
        + (a0 - epsilon) * (1 - x) ** 2
        + _continued_fraction(list(a), x) * (1 - x) ** 3
    )
    big_b = 1 + b0 * (1 - x) + _continued_fraction(list(b), x) * (1 - x) ** 2
    f = x * big_a
    parameters = [
        Parameter(
            "epsilon",
            0.0,
            (-1.0 + 1e-9, math.inf),
            latex=r"\epsilon",
            description="2M/r_0 - 1: deviation of the horizon radius from 2M",
            role="deformation",
        )
    ]
    if ppn == "free":
        parameters += [
            Parameter("a0", 0.0, latex="a_0", description="PPN coefficient", role="deformation"),
            Parameter("b0", 0.0, latex="b_0", description="PPN coefficient", role="deformation"),
        ]
    for k in range(1, order + 1):
        parameters.append(
            Parameter(
                f"a{k}", 0.0, latex=f"a_{k}", description="g_tt coefficient", role="deformation"
            )
        )
    for k in range(1, order + 1):
        parameters.append(
            Parameter(
                f"b{k}", 0.0, latex=f"b_{k}", description="g_rr coefficient", role="deformation"
            )
        )
    metric = StaticSpherical.from_sympy(
        f,
        f / big_b**2,
        r=r,
        parameters=parameters,
        name=f"Rezzolla-Zhidenko (order {order}, ppn={ppn})",
        asymptotically_flat=True,
        reference=RZ_REFERENCE,
    )
    metric.rz_order = order
    metric.rz_ppn = ppn
    return metric


@dataclass(frozen=True)
class RZCoefficients:
    """Rezzolla-Zhidenko coefficients of a static metric (see :func:`rz_coefficients`).

    Attributes
    ----------
    epsilon, a0, b0 : float
        Asymptotic coefficients.
    a, b : tuple of float
        ``(a1, a2, ...)`` and ``(b1, b2, ...)``. A vanishing coefficient
        ends the continued fraction: the following ones are reported as 0.
    r0 : float
        Areal radius of the event horizon, in the length unit of the input
        metric.
    mass : float
        ADM mass read from the asymptotic behaviour, in the same unit
        (1 for a metric that follows BlackPy's conventions).
    """

    epsilon: float
    a0: float
    b0: float
    a: tuple
    b: tuple
    r0: float
    mass: float

    @property
    def beta(self) -> float:
        """PPN parameter beta (RZ Eq. (17))."""
        e = 1.0 + self.epsilon
        return 1.0 + 2.0 * (self.a0 + self.b0 * e) / e**2

    @property
    def gamma(self) -> float:
        """PPN parameter gamma (RZ Eq. (18))."""
        return 1.0 + 2.0 * self.b0 / (1.0 + self.epsilon)

    def as_params(self, order: int | None = None, ppn: str = "free") -> dict:
        """Keyword parameters for ``RezzollaZhidenko(order, ppn)``."""
        order = len(self.a) if order is None else order
        out = {"epsilon": self.epsilon}
        if ppn == "free":
            out.update(a0=self.a0, b0=self.b0)
        for k in range(order):
            out[f"a{k + 1}"] = self.a[k] if k < len(self.a) else 0.0
        for k in range(order):
            out[f"b{k + 1}"] = self.b[k] if k < len(self.b) else 0.0
        return out


def _series_reciprocal(c: np.ndarray) -> np.ndarray:
    out = np.zeros_like(c)
    out[0] = 1.0 / c[0]
    for k in range(1, len(c)):
        out[k] = -np.dot(c[1 : k + 1], out[k - 1 :: -1][:k]) / c[0]
    return out


def _series_product(c: np.ndarray, d: np.ndarray) -> np.ndarray:
    n = min(len(c), len(d))
    return np.array([np.dot(c[: k + 1], d[k::-1]) for k in range(n)])


def _series_sqrt(c: np.ndarray) -> np.ndarray:
    out = np.zeros_like(c)
    out[0] = math.sqrt(c[0])
    for k in range(1, len(c)):
        cross = np.dot(out[1:k], out[k - 1 : 0 : -1]) if k > 1 else 0.0
        out[k] = (c[k] - cross) / (2.0 * out[0])
    return out


def series_to_continued_fraction(series, tol: float = 1e-9) -> list[float]:
    r"""Coefficients of :math:`c_1/(1 + c_2x/(1 + c_3x/\dots))` from a Taylor series in ``x``.

    A coefficient smaller than ``tol`` (relative to the largest series
    coefficient) ends the fraction; the remaining ones are zero.
    """
    current = np.asarray(series, dtype=float)
    scale = max(float(np.max(np.abs(current))), 1e-300)
    out = [0.0] * len(current)
    for k in range(len(out)):
        lead = float(current[0])
        if abs(lead) <= tol * max(scale, 1.0):
            break
        out[k] = lead
        if len(current) == 1:
            break
        # current = lead / (1 + x * next)  =>  next = (lead / current - 1) / x
        current = (lead * _series_reciprocal(current))[1:]
    return out


def _rz_taylor_core(metric, p, order=4, n_scan=256):
    """Taylor data in ``x = 1 - r0/r`` at the horizon and limits at infinity."""
    from blackpy.geometry.horizons import static_domain_core

    rho_h = static_domain_core(metric, p, n_scan=n_scan)[0]
    f = lambda rho: metric.f_array(rho, p)
    g = lambda rho: metric.g_array(rho, p)
    h = lambda rho: metric.h_array(rho, p)
    r0 = jnp.sqrt(h(rho_h))
    x = lambda rho: 1.0 - r0 / jnp.sqrt(h(rho))

    def d_dx(fn):
        return lambda rho: value_and_derivative(fn, rho)[1] / value_and_derivative(x, rho)[1]

    def taylor(fn, count, rho):
        values, factorial = [], 1.0
        for k in range(count):
            values.append(fn(rho) / factorial)
            fn = d_dx(fn)
            factorial *= k + 1.0
        return jnp.stack(values)

    def areal(rho):  # 4 h / h'^2: relates d rho to d(areal radius)
        return 4.0 * h(rho) / value_and_derivative(h, rho)[1] ** 2

    at_horizon = (
        taylor(f, order + 2, rho_h),
        taylor(g, order + 2, rho_h),
        taylor(areal, order + 1, rho_h),
    )

    # Asymptotic coefficients: derivatives with respect to x at four large radii; the
    # limit x -> 1 is taken afterwards by polynomial extrapolation in (1 - x).
    big_a = lambda rho: f(rho) / x(rho)
    big_b = lambda rho: jnp.sqrt(areal(rho) * f(rho) / g(rho))

    def limits(rho):
        first, second = d_dx(big_a), d_dx(d_dx(big_a))
        return jnp.stack(
            [first(rho), second(rho), d_dx(big_b)(rho), rho * (1.0 - f(rho)) / 2.0, 1.0 - x(rho)]
        )

    far = jnp.stack([limits(rho_h * factor) for factor in _FAR_RADII])
    return (*at_horizon, far, r0, rho_h)


#: Radii (in units of the horizon radius) used to extrapolate to spatial infinity. Larger
#: radii lose accuracy to cancellation in the second derivative; with these four the
#: asymptotic coefficients are accurate to about 1e-10.
_FAR_RADII = (50.0, 100.0, 200.0, 400.0)


def _extrapolate_to_zero(u: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Value at ``u = 0`` of the polynomial through ``(u_i, values_i)`` (Neville)."""
    table = [np.asarray(v, dtype=float) for v in values]
    for level in range(1, len(u)):
        table = [
            (u[i] * table[i + 1] - u[i + level] * table[i]) / (u[i] - u[i + level])
            for i in range(len(u) - level)
        ]
    return table[0]


def rz_coefficients(metric, order: int = 4, **params) -> RZCoefficients:
    r"""Rezzolla-Zhidenko coefficients of a static, spherically symmetric metric.

    Maps any ``StaticSpherical`` metric (also one with :math:`g \ne f` or
    :math:`h \ne r^2`, which is first expressed in the areal radius) onto
    the RZ parametrisation. The coefficients are obtained by automatic
    differentiation:

    * at the event horizon, the Taylor coefficients in
      :math:`x = 1 - r_0/r` of the metric functions, which fix
      :math:`a_1, a_2, \dots` and :math:`b_1, b_2, \dots` through the
      continued fractions (RZ Eq. (19));
    * towards spatial infinity, the derivatives :math:`dA/dx = \epsilon`,
      :math:`d^2A/dx^2 = 2(a_0 - \epsilon)` and :math:`dB/dx = -b_0`. They
      are evaluated by automatic differentiation at 50, 100, 200 and 400
      horizon radii and the limit :math:`x \to 1` is the value at
      :math:`1 - x = 0` of the cubic polynomial through the four points.

    The asymptotic coefficients are accurate to about :math:`10^{-10}`. The
    higher coefficients inherit this error divided by the lower ones: if
    :math:`a_1` is small, :math:`a_2` is less accurate.

    Parameters
    ----------
    metric : StaticSpherical
        Must be asymptotically flat and have an event horizon.
    order : int
        Number of coefficients :math:`a_i`, :math:`b_i` to compute (1 to 6).
    **params
        Scalar metric parameters.

    Returns
    -------
    RZCoefficients
        ``result.as_params(n)`` gives the keyword parameters of
        ``RezzollaZhidenko(order=n, ppn="free")``, in units of the ADM mass.

    Raises
    ------
    BlackPyError
        If the metric has no event horizon or is not asymptotically flat.
    """
    if metric.is_rotating_class:
        raise TypeError(
            "rz_coefficients needs a StaticSpherical metric. For a rotating metric close to "
            "Kerr use blackpy.metric.krz_deviations."
        )
    if not 1 <= order <= 6:
        raise ValueError("order must be between 1 and 6.")
    f_k, g_k, m_k, far, r0, rho_h = (
        np.asarray(v, dtype=float) for v in metric.evaluate(_rz_taylor_core, params, order=order)
    )
    if not np.isfinite(rho_h):
        raise BlackPyError(
            f"Metric '{metric.name}' has no event horizon for these parameters, so the "
            "Rezzolla-Zhidenko expansion (in x = 1 - r0/r) is not defined. Choose parameters "
            "with an event horizon (see blackpy.geometry.horizons)."
        )
    limit = _extrapolate_to_zero(far[:, 4], far[:, :4])
    epsilon, a0 = float(limit[0]), float(limit[0] + 0.5 * limit[1])
    b0, mass = float(-limit[2]), float(limit[3])
    if not np.all(np.isfinite(limit)) or abs(far[-1, 3] - mass) > 1e-2 * max(abs(mass), 1e-12):
        raise BlackPyError(
            f"Metric '{metric.name}' does not look asymptotically flat (f does not approach "
            "1 - 2M/r), so its Rezzolla-Zhidenko coefficients are not defined. The "
            "parametrisation applies to asymptotically flat metrics in units of their "
            "asymptotic mass only."
        )
    n = order
    # N^2 = x A(x): the Taylor coefficients of A are those of f shifted by one.
    series_a = f_k[1 : n + 1].copy()
    series_a[0] += -1.0 + epsilon - (a0 - epsilon)
    if n > 1:
        series_a[1] += -epsilon + 2.0 * (a0 - epsilon)
    if n > 2:
        series_a[2] += -(a0 - epsilon)
    binomial = np.array([(k + 1) * (k + 2) / 2.0 for k in range(n)])  # (1 - x)^(-3)
    tilde_a = _series_product(series_a, binomial)
    # B^2 = (f / g) * 4 h / h'^2; f and g both vanish at the horizon.
    ratio = _series_product(f_k[1 : n + 1], _series_reciprocal(g_k[1 : n + 1]))
    series_b = _series_sqrt(_series_product(ratio, m_k[:n]))
    series_b[0] += -1.0 - b0
    if n > 1:
        series_b[1] += b0
    tilde_b = _series_product(series_b, np.arange(1.0, n + 1.0))  # (1 - x)^(-2)
    return RZCoefficients(
        epsilon=epsilon,
        a0=a0,
        b0=b0,
        a=tuple(series_to_continued_fraction(tilde_a)),
        b=tuple(series_to_continued_fraction(tilde_b)),
        r0=float(r0),
        mass=mass,
    )


# ----------------------------------------------------------------------------- KRZ


def _kerr_horizon(a):
    return 1.0 + jnp.sqrt(jnp.clip(1.0 - a**2, 0.0, None))


def KRZ() -> StationaryAxisymmetric:
    r"""Konoplya-Rezzolla-Zhidenko parametrised rotating black hole.

    The line element is KRZ Eq. (7),

    .. math::

        ds^2 = -\frac{N^2 - W^2\sin^2\theta}{K^2}dt^2 - 2Wr\sin^2\theta\,dt\,d\phi
               + K^2r^2\sin^2\theta\,d\phi^2
               + \Sigma\left(\frac{B^2}{N^2}dr^2 + r^2d\theta^2\right),
        \qquad \Sigma = 1 + \frac{a^2\cos^2\theta}{r^2},

    with the functions of the lowest-order form recommended in KRZ
    Appendix A (their Eqs. (86)-(89) and Table 3): second order in the polar
    direction (:math:`\cos^2\theta`, reflection symmetric), first order of
    the continued fractions in the radial direction. In units of the mass,
    with :math:`r_0` the equatorial horizon radius:

    .. math::

        N^2 &= \left(1 - \frac{r_0}{r}\right)\left[1 - \frac{\epsilon_0r_0}{r}
               + (a_{00} - \epsilon_0 + k_{00})\frac{r_0^2}{r^2}
               + \frac{a_{01}r_0^3}{r^3}\right]
               + \left[(\delta k_{21} + \delta a_{20})\frac{r_0^3}{r^3}
               + \delta a_{21}\frac{r_0^4}{r^4}\right]\cos^2\theta, \\
        B &= 1 + \frac{b_{00}r_0}{r} + \frac{b_{01}r_0^2}{r^2}
             + \frac{b_{21}r_0^2}{r^2}\cos^2\theta, \\
        W &= \frac{1}{\Sigma}\left[\frac{w_{00}r_0^2}{r^2} + \frac{w_{01}r_0^3}{r^3}
             + \frac{w_{21}r_0^3}{r^3}\cos^2\theta\right], \\
        K^2 &= 1 + \frac{aW}{r} + \frac{1}{\Sigma}\left[\frac{k_{00}r_0^2}{r^2}
               + \left(\frac{a^4}{r^4} - \frac{2a^2}{r^3}
               + \delta k_{21}\frac{r_0^3}{r^3}\right)\cos^2\theta\right],

    with :math:`\epsilon_0 = 2/r_0 - 1`, :math:`k_{00} = a^2/r_0^2`,
    :math:`w_{00} = 2a/r_0^2` fixed by the mass and the angular momentum, and
    :math:`\epsilon_2 = b_{20} = w_{20} = k_{20} = 0` ("astrophysically
    realistic black holes", KRZ Table 3).

    **Deviation parameters.** BlackPy's parameters are deviations from the
    Kerr values of the KRZ coefficients, so that all of them zero is Kerr:

    ===============  ==========================================================
    ``d_eps0``       :math:`\epsilon_0 - \epsilon_0^{\rm Kerr}`; moves the
                     equatorial horizon, :math:`r_0 = 2/(1 + \epsilon_0)`
    ``a00``, ``b00`` PPN coefficients (KRZ Eqs. (30b)-(30c)); zero in GR
    ``a01``          deformation of :math:`g_{tt}` (zero for Kerr)
    ``b01``          deformation of :math:`g_{rr}` (zero for Kerr)
    ``w01``          rotational deformation (zero for Kerr)
    ``d_a20``        :math:`a_{20} - 2a^2/r_0^3`: deviation of the quadrupole
                     moment, :math:`Q = -(a_{20}r_0^3 + a^2)/3`
    ``d_a21``        :math:`a_{21} + a^4/r_0^4`: horizon deformation
    ``d_k21``        :math:`k_{21} - (a^4/r_0^4 - 2a^2/r_0^3)`: horizon deformation
    ``w21``, ``b21`` polar deformations of :math:`W` and :math:`B` (zero for Kerr)
    ===============  ==========================================================

    :func:`krz_deviations` converts the coefficients as printed in the
    paper into these parameters.

    **One departure from the printed lowest-order form.** With the Kerr
    coefficients of KRZ Appendix A.1, their Eqs. (86)-(89) reproduce the
    Kerr metric only in the equatorial plane; off it the components differ
    (by about :math:`10^{-3}` at :math:`r = 3.5`, :math:`a = 0.6`). The
    reason is that the Kerr polar function is :math:`K_2 = a^4/r^4 -
    2a^2/r^3`, whose :math:`r^{-4}` term needs the next two coefficients of
    its continued fraction. BlackPy keeps the full Kerr :math:`K_2` and adds
    :math:`\delta k_{21}r_0^3/r^3`; this is a member of the KRZ family (the
    coefficients :math:`k_{22}, k_{23}` are tied to :math:`k_{21}`) and makes
    the zero-deviation metric Kerr to round-off everywhere.

    Properties
    ----------
    * Reflection symmetric: equatorial orbits, both epicyclic frequencies and
      the QPO models are available.
    * **Null geodesics do not separate** for generic deviations, so the
      analytic shadow is refused (:class:`blackpy.errors.NotSeparableError`);
      it needs ray tracing.
    * For generic deviations the surface :math:`N^2 = 0` depends on
      :math:`\theta` and BlackPy reports that it is then not the event
      horizon. It is a sphere if :math:`\delta a_{20} + \delta a_{21} +
      \delta k_{21} = 0` (KRZ, Appendix A).
    * Declared domain: :math:`|a| \le 1`.

    This is a parametrised model, not a solution of any field equations:
    the source it would need has to be checked with
    :func:`blackpy.geometry.effective_source`, never assumed.

    Returns
    -------
    StationaryAxisymmetric
    """
    r = sp.Symbol("r", positive=True)
    theta = sp.Symbol("theta", real=True)
    names = "a d_eps0 a00 b00 a01 b01 w01 d_a20 d_a21 d_k21 w21 b21"
    a, d_eps0, a00, b00, a01, b01, w01, d_a20, d_a21, d_k21, w21, b21 = sp.symbols(names, real=True)
    r0_kerr = 1 + sp.sqrt(1 - a**2)
    eps0 = (2 - r0_kerr) / r0_kerr + d_eps0
    r0 = 2 / (1 + eps0)
    cos2, sin2 = sp.cos(theta) ** 2, sp.sin(theta) ** 2
    k00, w00 = a**2 / r0**2, 2 * a / r0**2
    sigma = 1 + a**2 * cos2 / r**2
    n2 = (1 - r0 / r) * (
        1 - eps0 * r0 / r + (a00 - eps0 + k00) * r0**2 / r**2 + a01 * r0**3 / r**3
    ) + ((d_k21 + d_a20) * r0**3 / r**3 + d_a21 * r0**4 / r**4) * cos2
    big_b = 1 + b00 * r0 / r + b01 * r0**2 / r**2 + b21 * r0**2 / r**2 * cos2
    w = (w00 * r0**2 / r**2 + w01 * r0**3 / r**3 + w21 * r0**3 / r**3 * cos2) / sigma
    k2 = (
        1
        + a * w / r
        + (k00 * r0**2 / r**2 + (a**4 / r**4 - 2 * a**2 / r**3 + d_k21 * r0**3 / r**3) * cos2)
        / sigma
    )
    deformation = lambda name, latex, text: Parameter(
        name, 0.0, latex=latex, description=text, role="deformation"
    )
    parameters = [
        Parameter("a", 0.0, (-1.0, 1.0), latex="a", description="spin a/M = J/M^2", role="spin"),
        deformation("d_eps0", r"\delta\epsilon_0", "epsilon_0 minus its Kerr value"),
        deformation("a00", "a_{00}", "PPN coefficient (beta - gamma)"),
        deformation("b00", "b_{00}", "PPN coefficient (gamma - 1)"),
        deformation("a01", "a_{01}", "deformation of g_tt"),
        deformation("b01", "b_{01}", "deformation of g_rr"),
        deformation("w01", "w_{01}", "rotational deformation"),
        deformation("d_a20", r"\delta a_{20}", "a_20 minus its Kerr value (quadrupole)"),
        deformation("d_a21", r"\delta a_{21}", "a_21 minus its Kerr value"),
        deformation("d_k21", r"\delta k_{21}", "k_21 minus its Kerr value"),
        deformation("w21", "w_{21}", "polar rotational deformation"),
        deformation("b21", "b_{21}", "polar deformation of g_rr"),
    ]
    return StationaryAxisymmetric.from_sympy(
        -(n2 - w**2 * sin2) / k2,
        -w * r * sin2,
        k2 * r**2 * sin2,
        sigma * big_b**2 / n2,
        sigma * r**2,
        r=r,
        theta=theta,
        parameters=parameters,
        name="KRZ (lowest order)",
        asymptotically_flat=True,
        reference=KRZ_REFERENCE,
        domain=lambda a, *deviations: jnp.abs(a) <= 1.0,
    )


def krz_deviations(
    a: float,
    *,
    r0: float | None = None,
    a00: float = 0.0,
    b00: float = 0.0,
    a01: float = 0.0,
    b01: float = 0.0,
    w01: float = 0.0,
    a20: float | None = None,
    a21: float | None = None,
    k21: float | None = None,
    w21: float = 0.0,
    b21: float = 0.0,
) -> dict:
    r"""Convert KRZ coefficients as printed in the paper into the parameters of :func:`KRZ`.

    All quantities in units of the mass :math:`M`. ``r0`` is the equatorial
    horizon radius; the default is the Kerr value :math:`1 + \sqrt{1 - a^2}`.
    ``a20``, ``a21`` and ``k21`` default to their Kerr expressions
    :math:`2a^2/r_0^3`, :math:`-a^4/r_0^4` and :math:`a^4/r_0^4 - 2a^2/r_0^3`
    (KRZ Appendix A.1) evaluated with the given ``r0``.

    Returns
    -------
    dict
        Keyword parameters for the metric returned by :func:`KRZ`.
    """
    if not abs(a) <= 1.0:
        raise ValueError(
            f"The KRZ deviation parameters are defined for |a| <= 1, got a = {a}; choose a spin "
            "in that range."
        )
    r0_kerr = 1.0 + math.sqrt(1.0 - a**2)
    r0 = r0_kerr if r0 is None else float(r0)
    kerr_a20, kerr_a21 = 2.0 * a**2 / r0**3, -(a**4) / r0**4
    kerr_k21 = a**4 / r0**4 - 2.0 * a**2 / r0**3
    return {
        "a": float(a),
        "d_eps0": (2.0 / r0 - 1.0) - (2.0 / r0_kerr - 1.0),
        "a00": float(a00),
        "b00": float(b00),
        "a01": float(a01),
        "b01": float(b01),
        "w01": float(w01),
        "d_a20": 0.0 if a20 is None else float(a20) - kerr_a20,
        "d_a21": 0.0 if a21 is None else float(a21) - kerr_a21,
        "d_k21": 0.0 if k21 is None else float(k21) - kerr_k21,
        "w21": float(w21),
        "b21": float(b21),
    }
