r"""Curvature and effective source of stationary, axisymmetric metrics (numerical).

Everything here is computed from the metric tensor
:math:`g_{\mu\nu}(r, \theta)` with the textbook definitions, using forward-mode
automatic differentiation for the derivatives with respect to ``r`` and
``theta`` (the metric does not depend on ``t`` and ``phi``):

.. math::

    \Gamma^a{}_{bc} &= \tfrac12 g^{ad}(\partial_b g_{dc} + \partial_c g_{db}
                       - \partial_d g_{bc}), \\
    R^a{}_{bcd} &= \partial_c\Gamma^a{}_{bd} - \partial_d\Gamma^a{}_{bc}
                   + \Gamma^a{}_{ce}\Gamma^e{}_{bd} - \Gamma^a{}_{de}\Gamma^e{}_{bc}, \\
    R_{bd} &= R^a{}_{bad}, \qquad G^a{}_b = R^a{}_b - \tfrac12\delta^a_b R, \qquad
    K = R_{abcd}R^{abcd} .

Conventions as in :mod:`blackpy.geometry.symbolic` (Misner, Thorne & Wheeler
1973); coordinates ``(t, r, theta, phi)``; ``G^a_b = 8 pi T^a_b``.

Classification of the source
----------------------------
The effective stress-energy tensor :math:`T^a{}_b = G^a{}_b / 8\pi` is
classified at every grid point by its eigenvalue structure (Hawking & Ellis
1973, section 4.3; cited from memory, see ``docs/theory.md``):

* all eigenvalues real, a complete set of eigenvectors, exactly one of them
  timelike (type I): :math:`T^a{}_b` is diagonal in an orthonormal frame,
  with energy density :math:`\rho = -\lambda_{\rm timelike}` and principal
  pressures :math:`p_i` equal to the other three eigenvalues. Three equal
  pressures: **perfect fluid**; otherwise **anisotropic fluid**;
* complex eigenvalues, a defective matrix, or no timelike eigenvector
  (types II-IV): **neither**;
* :math:`T^a{}_b = 0` within round-off: **vacuum**.

Degenerate eigenvalues (as for an electromagnetic field, where
:math:`\rho = -p_1`) are handled by looking at the whole eigenspace: it
contains a timelike direction if the metric restricted to it has a negative
eigenvalue.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

HALF_PI = 0.5 * math.pi
_EIGHT_PI = 8.0 * math.pi

CONDITIONS = ("NEC", "WEC", "SEC", "DEC")
KINDS = ("vacuum", "perfect fluid", "anisotropic fluid", "neither")


def _inverse(g):
    """Inverse of the block-structured metric (t-phi block plus diagonal r, theta)."""
    g_tt, g_tphi, g_phiphi, g_rr, g_thth = g[0, 0], g[0, 3], g[3, 3], g[1, 1], g[2, 2]
    det = g_tt * g_phiphi - g_tphi**2
    zero = jnp.zeros_like(g_tt)
    return jnp.array(
        [
            [g_phiphi / det, zero, zero, -g_tphi / det],
            [zero, 1.0 / g_rr, zero, zero],
            [zero, zero, 1.0 / g_thth, zero],
            [-g_tphi / det, zero, zero, g_tt / det],
        ]
    )


def _pad(derivative_r_theta):
    """Insert zero derivatives along t and phi: (..., 2) -> (..., 4)."""
    zero = jnp.zeros_like(derivative_r_theta[..., 0])
    return jnp.stack([zero, derivative_r_theta[..., 0], derivative_r_theta[..., 1], zero], axis=-1)


def christoffel_core(metric, p, x):
    """Christoffel symbols ``Gamma[a, b, c]`` at ``x = (r, theta)``."""
    g_fn = lambda y: metric.matrix_array(y[0], y[1], p)
    g_inv = _inverse(g_fn(x))
    dg = _pad(jax.jacfwd(g_fn)(x))  # dg[a, b, c] = d_c g_ab
    lower = 0.5 * (jnp.transpose(dg, (0, 2, 1)) + dg - jnp.transpose(dg, (2, 0, 1)))
    # lower[d, b, c] = (d_b g_dc + d_c g_db - d_d g_bc) / 2
    return jnp.einsum("ad,dbc->abc", g_inv, lower)


def curvature_core(metric, p, r, theta):
    """Kretschmann scalar, Ricci scalar and mixed Einstein tensor at ``(r, theta)``.

    Returns
    -------
    kretschmann, ricci_scalar : scalars
    einstein_mixed : array (4, 4)
        ``G^a_b`` with coordinates ``(t, r, theta, phi)``.
    """
    x = jnp.stack([jnp.asarray(r, dtype=jnp.float64), jnp.asarray(theta, dtype=jnp.float64)])
    gamma_fn = lambda y: christoffel_core(metric, p, y)
    gamma = gamma_fn(x)
    dgamma = _pad(jax.jacfwd(gamma_fn)(x))  # dgamma[a, b, c, d] = d_d Gamma^a_bc
    riemann = (
        jnp.transpose(dgamma, (0, 1, 3, 2))
        - dgamma
        + jnp.einsum("ace,ebd->abcd", gamma, gamma)
        - jnp.einsum("ade,ebc->abcd", gamma, gamma)
    )  # R^a_bcd
    g = metric.matrix_array(x[0], x[1], p)
    g_inv = _inverse(g)
    ricci = jnp.einsum("abad->bd", riemann)
    ricci_scalar = jnp.einsum("bd,bd->", g_inv, ricci)
    einstein_mixed = jnp.einsum("ac,cb->ab", g_inv, ricci) - 0.5 * ricci_scalar * jnp.eye(4)
    riemann_low = jnp.einsum("ae,ebcd->abcd", g, riemann)
    riemann_up = jnp.einsum("abcd,bf,cg,dh->afgh", riemann, g_inv, g_inv, g_inv)
    kretschmann = jnp.einsum("abcd,abcd->", riemann_low, riemann_up)
    return kretschmann, ricci_scalar, einstein_mixed


def kretschmann_core(metric, p, r, theta):
    """Kretschmann scalar ``K = R_abcd R^abcd`` at ``(r, theta)``."""
    return curvature_core(metric, p, r, theta)[0]


def ricci_scalar_core(metric, p, r, theta):
    """Ricci scalar at ``(r, theta)``."""
    return curvature_core(metric, p, r, theta)[1]


def einstein_core(metric, p, r, theta):
    """Mixed Einstein tensor ``G^a_b`` (4x4) at ``(r, theta)``."""
    return curvature_core(metric, p, r, theta)[2]


def _source_inputs_core(metric, p, r, theta):
    kretschmann, _, einstein = curvature_core(metric, p, r, theta)
    return einstein / _EIGHT_PI, metric.matrix_array(r, theta, p), kretschmann


# ----------------------------------------------------------------------------- classification


def classify_stress_energy(t_mixed, g_cov, curvature_scale, rtol: float = 1e-7):
    """Classify one stress-energy tensor by its eigenvalue structure.

    Parameters
    ----------
    t_mixed : array (4, 4)
        ``T^a_b``.
    g_cov : array (4, 4)
        Covariant metric at the same point.
    curvature_scale : float
        ``sqrt(K)``; sets the absolute size below which ``T`` counts as zero.
    rtol : float
        Relative tolerance for "zero", "equal eigenvalues" and "real".

    Returns
    -------
    kind : str
        One of ``"vacuum"``, ``"perfect fluid"``, ``"anisotropic fluid"``,
        ``"neither"``.
    rho : float
        Energy density (NaN unless type I; 0 for vacuum).
    pressures : numpy.ndarray (3,)
        Principal pressures in ascending order (NaN unless type I).
    """
    nan3 = np.full(3, np.nan)
    if not (np.all(np.isfinite(t_mixed)) and np.all(np.isfinite(g_cov))):
        return "neither", np.nan, nan3
    # Work in an orthonormal frame E (E^T g E = diag(-1, 1, 1, 1) up to ordering), where
    # the components of T have a common scale. The size of T is judged from these
    # components, not from its eigenvalues: null dust has only zero eigenvalues.
    metric_values, metric_vectors = np.linalg.eigh(g_cov)
    frame = metric_vectors / np.sqrt(np.abs(metric_values))
    eta = np.diag(np.sign(metric_values))
    t_mixed = np.linalg.solve(frame, t_mixed @ frame)
    g_cov = eta
    size = float(np.max(np.abs(t_mixed)))
    if _EIGHT_PI * size <= rtol * max(curvature_scale, 1e-300):
        return "vacuum", 0.0, np.zeros(3)
    values = np.linalg.eigvals(t_mixed)
    tol = 100.0 * rtol * size
    if np.max(np.abs(values.imag)) > tol:
        return "neither", np.nan, nan3
    values = np.sort(values.real)
    clusters: list[list[float]] = [[values[0]]]
    for value in values[1:]:
        if value - clusters[-1][-1] <= tol:
            clusters[-1].append(value)
        else:
            clusters.append([value])
    rho, pressures, n_timelike = np.nan, [], 0
    for cluster in clusters:
        eigenvalue = float(np.mean(cluster))
        # Eigenspace = null space of (T - lambda); it must have the full multiplicity.
        _, singular, vh = np.linalg.svd(t_mixed - eigenvalue * np.eye(4))
        basis = vh[singular <= 10.0 * tol].T
        if basis.shape[1] != len(cluster):
            return "neither", np.nan, nan3  # defective: no complete set of eigenvectors
        gram = np.linalg.eigvalsh(basis.T @ g_cov @ basis)
        negative = int(np.sum(gram < 0.0))
        n_timelike += negative
        if negative == 1:
            rho = -eigenvalue
        pressures += [eigenvalue] * (len(cluster) - negative)
    if n_timelike != 1 or len(pressures) != 3:
        return "neither", np.nan, nan3
    pressures = np.sort(np.asarray(pressures))
    isotropic = pressures[2] - pressures[0] <= tol
    return ("perfect fluid" if isotropic else "anisotropic fluid"), rho, pressures


@dataclass(frozen=True)
class SourceResult:
    """Effective source of a metric on an ``(r, theta)`` grid.

    Attributes
    ----------
    r, theta : numpy.ndarray
        Grid coordinates, both of shape ``(n_r, n_theta)``.
    stress_energy : numpy.ndarray
        ``T^a_b = G^a_b / (8 pi)``, shape ``(n_r, n_theta, 4, 4)``, in units
        of 1/M^2, coordinates ``(t, r, theta, phi)``.
    kinds : numpy.ndarray of str
        Point-wise classification.
    rho : numpy.ndarray
        Energy density (NaN where the source is not of type I).
    pressures : numpy.ndarray
        Principal pressures, shape ``(n_r, n_theta, 3)``, ascending.
    satisfied : dict of str -> numpy.ndarray of bool
        NEC, WEC, SEC, DEC point-wise (False where the source is not of
        type I, because the conditions below do not apply there).
    kretschmann : numpy.ndarray
        Kretschmann scalar on the grid.
    rtol : float
        Tolerance used for the classification and the energy conditions.
    """

    r: np.ndarray
    theta: np.ndarray
    stress_energy: np.ndarray
    kinds: np.ndarray
    rho: np.ndarray
    pressures: np.ndarray
    satisfied: dict
    kretschmann: np.ndarray
    rtol: float

    @property
    def kind(self) -> str:
        """Overall classification on the grid.

        ``"vacuum"`` if every point is vacuum; ``"perfect fluid"`` if every
        non-vacuum point is a perfect fluid; ``"anisotropic fluid"`` if every
        point is of type I; otherwise ``"neither"``.
        """
        found = set(self.kinds.ravel().tolist())
        if found <= {"vacuum"}:
            return "vacuum"
        if found <= {"vacuum", "perfect fluid"}:
            return "perfect fluid"
        if "neither" not in found:
            return "anisotropic fluid"
        return "neither"

    def holds(self, condition: str) -> bool:
        """True if ``condition`` (NEC, WEC, SEC or DEC) holds on the whole grid."""
        return bool(np.all(self.satisfied[condition.upper()]))

    def fraction(self, kind: str) -> float:
        """Fraction of grid points classified as ``kind``."""
        return float(np.mean(self.kinds == kind))

    def summary(self) -> str:
        """Human-readable summary of the source type and the energy conditions."""
        counts = ", ".join(
            f"{k}: {int(np.sum(self.kinds == k))}" for k in KINDS if np.any(self.kinds == k)
        )
        lines = [
            f"Effective source: {self.kind} (grid points by type: {counts})",
            f"  grid: r in [{self.r.min():.6g}, {self.r.max():.6g}], "
            f"theta in [{self.theta.min():.4g}, {self.theta.max():.4g}]",
        ]
        for name in CONDITIONS:
            ok = self.satisfied[name]
            if ok.all():
                lines.append(f"{name}: satisfied on the grid")
            else:
                bad_r = self.r[~ok]
                lines.append(
                    f"{name}: VIOLATED at {int((~ok).sum())} of {ok.size} grid points "
                    f"(r from {bad_r.min():.6g} to {bad_r.max():.6g})"
                )
        return "\n".join(lines)


def effective_source(
    metric,
    r=None,
    theta=None,
    *,
    r_max: float | None = None,
    n_r: int = 24,
    n_theta: int = 7,
    rtol: float = 1e-7,
    **params,
) -> SourceResult:
    r"""Effective stress-energy, its type and the energy conditions on an ``(r, theta)`` grid.

    The Einstein tensor is computed with automatic differentiation and
    :math:`T^a{}_b = G^a{}_b/8\pi` is classified by its eigenvalue structure
    (see the module docstring). For type I sources the point-wise energy
    conditions are, with :math:`p_i` the three principal pressures:

    * NEC: :math:`\rho + p_i \ge 0`;
    * WEC: NEC and :math:`\rho \ge 0`;
    * SEC: NEC and :math:`\rho + \sum p_i \ge 0`;
    * DEC: :math:`\rho \ge |p_i|`.

    Parameters
    ----------
    metric : StationaryAxisymmetric or StaticSpherical
        A static metric is converted with ``as_axisymmetric()``.
    r : array, optional
        Radii. Default: ``n_r`` logarithmically spaced points from 1.05
        times the equatorial event horizon (or the lower end of the scan
        range if horizonless) to ``r_max`` (default: 30 times the inner
        edge).
    theta : array, optional
        Polar angles. Default: ``n_theta`` points from 0.15 to pi - 0.15.
    rtol : float
        Relative tolerance; values this small relative to the local
        curvature scale :math:`\sqrt{K}` count as zero.
    **params
        Scalar metric parameters.

    Returns
    -------
    SourceResult
    """
    from blackpy.geometry.horizons import static_domain_core

    if not metric.is_rotating_class:
        metric = metric.as_axisymmetric()
    if r is None:
        _, r_cosmo, r_in, r_out = (float(v) for v in metric.evaluate(static_domain_core, params))
        lo = 1.05 * r_in
        hi = 30.0 * lo if r_max is None else float(r_max)
        if np.isfinite(r_cosmo):
            hi = min(hi, 0.95 * r_out)
        r = np.geomspace(lo, hi, n_r)
    if theta is None:
        theta = np.linspace(0.15, math.pi - 0.15, n_theta)
    r_grid, theta_grid = np.meshgrid(
        np.atleast_1d(np.asarray(r, dtype=float)),
        np.atleast_1d(np.asarray(theta, dtype=float)),
        indexing="ij",
    )
    t_mixed, g_cov, kretschmann = (
        np.asarray(v) for v in metric.evaluate(_source_inputs_core, params, r_grid, theta_grid)
    )
    shape = r_grid.shape
    kinds = np.empty(shape, dtype=object)
    rho = np.full(shape, np.nan)
    pressures = np.full((*shape, 3), np.nan)
    for index in np.ndindex(shape):
        scale = math.sqrt(abs(float(kretschmann[index])))
        kinds[index], rho[index], pressures[index] = classify_stress_energy(
            t_mixed[index], g_cov[index], scale, rtol=rtol
        )
    tol = rtol * np.sqrt(np.abs(kretschmann)) / _EIGHT_PI
    type_one = np.isfinite(rho)
    with np.errstate(invalid="ignore"):
        nec = type_one & np.all(rho[..., None] + pressures >= -tol[..., None], axis=-1)
        wec = nec & (rho >= -tol)
        sec = nec & (rho + pressures.sum(axis=-1) >= -tol)
        dec = type_one & np.all(rho[..., None] - np.abs(pressures) >= -tol[..., None], axis=-1)
    return SourceResult(
        r=r_grid,
        theta=theta_grid,
        stress_energy=t_mixed,
        kinds=kinds,
        rho=rho,
        pressures=pressures,
        satisfied={"NEC": nec, "WEC": wec, "SEC": sec, "DEC": dec},
        kretschmann=kretschmann,
        rtol=rtol,
    )
