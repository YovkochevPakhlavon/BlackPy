r"""Fisher-matrix forecasts.

The model prediction for every fitted datum is a JAX function of the
parameters (root finding for horizons, photon regions and shadows
included), so its derivatives are available by automatic differentiation;
no finite differences are used.

For independent Gaussian data :math:`y_k` with uncertainties
:math:`\sigma_k` and model values :math:`m_k(\theta)` the Fisher matrix is

.. math::

    F_{ij} = \sum_k \frac{1}{\sigma_k^2}
             \frac{\partial m_k}{\partial\theta_i}\frac{\partial m_k}{\partial\theta_j}
             \;-\; \delta_{ij}\,\frac{\partial^2\ln\pi_i}{\partial\theta_i^2},

where the second term is the contribution of the (independent) priors
:math:`\pi_i`. The first term equals minus the Hessian of the log-likelihood
when the data equal the model prediction; this was verified for QPO pairs,
QPO triplets and shadow data, for static and rotating metrics (the whole
log-likelihood is twice differentiable by forward-mode autodiff), and is
tested. The forecast covariance is :math:`F^{-1}`.

Limits
------
* **Gaussian approximation**: the forecast is the curvature of the
  log-posterior at the fiducial point. It is accurate when the posterior is
  close to Gaussian, and wrong for curved (banana-shaped) or multimodal
  posteriors.
* **Prior boundaries**: a hard boundary truncates the posterior; the Fisher
  matrix does not see it. A warning is raised when the fiducial value is
  within one forecast sigma of a boundary.
* **Prior-dominated parameters**: if the data alone leave a parameter
  unconstrained on the scale of its prior, the forecast sigma reflects the
  prior, or is infinite for a flat one. This is reported and warned about.
* Asymmetric uncertainties are symmetrised; interval (top-hat) likelihoods
  and bounds carry no curvature and are ignored, with a note.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from blackpy.errors import BlackPyError, ForecastWarning, PriorDominatedWarning
from blackpy.infer.likelihood import Problem, build_problem, rename_legacy_inclination
from blackpy.infer.priors import Gaussian, Isotropic, LogUniform, Prior, Uniform
from blackpy.infer.quality import gaussian_rows

#: A direction in (scaled) parameter space is called degenerate if its Fisher eigenvalue
#: is below this fraction of the largest one.
DEGENERATE_EIGENVALUE_RATIO = 1e-8


def _prior_curvature(prior: Prior, value: float) -> float:
    r""":math:`-\partial^2\ln\pi/\partial\theta^2` at ``value`` (0 for a flat prior)."""
    if isinstance(prior, Gaussian):
        return 1.0 / prior.sigma**2
    if isinstance(prior, Uniform):
        return 0.0
    if isinstance(prior, LogUniform):
        return -1.0 / value**2
    if isinstance(prior, Isotropic):
        # ln pi = ln sin(i) + const with i in degrees.
        radians = math.radians(value)
        return (math.pi / 180.0) ** 2 / max(math.sin(radians) ** 2, 1e-12)
    return 0.0


def _prior_scale(prior: Prior) -> float:
    """Half-width of the central 68% interval of a prior."""
    return 0.5 * float(prior.ppf(0.8413) - prior.ppf(0.1587))


@dataclass
class Forecast:
    r"""Fisher-matrix forecast (see :func:`forecast`).

    Attributes
    ----------
    param_names : list of str
    fiducial : dict of str -> float
    fisher : numpy.ndarray
        Fisher matrix including the priors.
    fisher_data : numpy.ndarray
        Contribution of the data alone.
    covariance : numpy.ndarray
        Inverse of ``fisher``. Entries that involve a degenerate direction
        are infinite (variances) or NaN (covariances).
    sigma : dict of str -> float
        Forecast 1-sigma uncertainties (``inf`` along degenerate directions).
    sigma_data : dict of str -> float
        The same from the data alone (no priors).
    correlation : numpy.ndarray
        Correlation matrix (NaN where a variance is infinite).
    condition_number : float
        Ratio of the largest to the smallest eigenvalue of the Fisher matrix
        in units of the prior widths (``inf`` if singular).
    degenerate_directions : list of dict
        One entry per direction that the data and priors do not constrain:
        ``{"eigenvalue_ratio", "direction"}``, where ``direction`` maps
        parameter names to the components of the unit eigenvector in units
        of the prior widths (so that components are comparable).
    weakest_direction : dict
        The least constrained direction, in the same form, whether or not it
        counts as degenerate.
    prior_dominated : dict of str -> bool
        True if the data alone constrain the parameter less tightly than its
        prior does (or not at all).
    near_boundary : dict of str -> bool
        True if the fiducial value is within one forecast sigma of a prior
        boundary.
    notes : list of str
    """

    param_names: list
    fiducial: dict
    fisher: np.ndarray
    fisher_data: np.ndarray
    covariance: np.ndarray
    sigma: dict
    sigma_data: dict
    correlation: np.ndarray
    condition_number: float
    degenerate_directions: list
    weakest_direction: dict
    prior_dominated: dict
    near_boundary: dict
    notes: list = field(default_factory=list)

    @property
    def is_singular(self) -> bool:
        """True if at least one direction is degenerate."""
        return bool(self.degenerate_directions)

    def __str__(self) -> str:
        lines = ["Fisher forecast (Gaussian approximation at the fiducial point)"]
        for name in self.param_names:
            flags = []
            if self.prior_dominated[name]:
                flags.append("prior dominated: NOT a constraint")
            if self.near_boundary[name]:
                flags.append("within 1 sigma of a prior boundary")
            flag = f"   [{'; '.join(flags)}]" if flags else ""
            lines.append(
                f"  {name:<24} = {self.fiducial[name]:.5g} +- {self.sigma[name]:.3g} "
                f"(data alone: {self.sigma_data[name]:.3g}){flag}"
            )
        lines.append(f"  condition number (prior-scaled): {self.condition_number:.3g}")
        for item in self.degenerate_directions:
            parts = ", ".join(
                f"{value:+.3f} {name}"
                for name, value in item["direction"].items()
                if abs(value) > 1e-3
            )
            lines.append(f"  degenerate direction (in units of the prior widths): {parts}")
        if not self.degenerate_directions:
            parts = ", ".join(
                f"{value:+.3f} {name}"
                for name, value in self.weakest_direction["direction"].items()
                if abs(value) > 1e-3
            )
            lines.append(f"  least constrained direction (prior-scaled): {parts}")
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)


def _sigmas(matrix: np.ndarray, scales: np.ndarray):
    """Covariance and 1-sigma values from a (possibly singular) Fisher matrix.

    The matrix is diagonalised in units of ``scales``; directions whose
    eigenvalue is below ``DEGENERATE_EIGENVALUE_RATIO`` times the largest one
    (or not positive) are treated as unconstrained.
    """
    scaled = matrix * np.outer(scales, scales)
    scaled = 0.5 * (scaled + scaled.T)
    values, vectors = np.linalg.eigh(scaled)
    largest = max(float(values[-1]), 0.0)
    good = values > DEGENERATE_EIGENVALUE_RATIO * largest if largest > 0 else values > np.inf
    n = len(scales)
    variance = np.zeros(n)
    covariance = np.zeros((n, n))
    for k in range(n):
        if good[k]:
            covariance += np.outer(vectors[:, k], vectors[:, k]) / values[k]
    infinite = np.zeros(n, dtype=bool)
    for k in range(n):
        if not good[k]:
            infinite |= np.abs(vectors[:, k]) > 1e-9
    covariance = covariance * np.outer(scales, scales)
    variance = np.where(infinite, np.inf, np.diag(covariance))
    mask = np.logical_or.outer(infinite, infinite)
    covariance = np.where(mask, np.nan, covariance)
    covariance[np.diag_indices(n)] = variance
    return covariance, np.sqrt(variance), values, vectors, good


def forecast_from_problem(problem: Problem, fiducial, *, warn: bool = True) -> Forecast:
    """Fisher forecast for an assembled problem at a parameter vector.

    Parameters
    ----------
    problem : Problem
        From :func:`blackpy.infer.build_problem` (or built by hand: it needs
        ``predict``, ``data_table``, ``param_names`` and ``priors``).
    fiducial : array or mapping
        Fiducial values of all sampled parameters.
    warn : bool
        Emit the boundary and prior-dominance warnings.

    Returns
    -------
    Forecast
    """
    names = list(problem.param_names)
    if isinstance(fiducial, Mapping):
        fiducial = rename_legacy_inclination(fiducial, "the fiducial point")
        missing = set(names) - set(fiducial)
        if missing:
            raise TypeError(
                f"The fiducial point lacks values for {sorted(missing)}. Sampled parameters of "
                f"this problem: {names}."
            )
        fiducial = [float(fiducial[n]) for n in names]
    theta = np.asarray(fiducial, dtype=float)
    if theta.shape != (len(names),):
        raise ValueError(f"The fiducial point must have {len(names)} values, for {names}.")
    if not np.isfinite(problem.log_prior(theta)[0]):
        raise BlackPyError(
            "The fiducial point lies outside the support of the priors; a forecast there has "
            "no meaning. Choose fiducial values inside the prior ranges (see problem.priors)."
        )
    rows = gaussian_rows(problem.data_table)
    notes = []
    skipped = [row["label"] for k, row in enumerate(problem.data_table) if k not in rows]
    if skipped:
        notes.append(
            f"{skipped} are interval likelihoods or bounds: they have no curvature and add "
            "nothing to the Fisher matrix"
        )
    values = np.asarray(problem.predict(jnp.asarray(theta)))
    jacobian = np.asarray(jax.jacfwd(problem.predict)(jnp.asarray(theta)))
    if not (np.all(np.isfinite(values[rows])) and np.all(np.isfinite(jacobian[rows]))):
        raise BlackPyError(
            "The model prediction or its derivative is not finite at the fiducial point (for "
            "example an orbit inside the ISCO, or no shadow). Choose a fiducial point where "
            "the model is defined."
        )
    sigma = np.array(
        [
            0.5 * (problem.data_table[k]["err_minus"] + problem.data_table[k]["err_plus"])
            for k in rows
        ]
    )
    weighted = jacobian[rows] / sigma[:, None]
    fisher_data = weighted.T @ weighted
    prior_term = np.array(
        [_prior_curvature(prior, value) for prior, value in zip(problem.priors, theta, strict=True)]
    )
    fisher = fisher_data + np.diag(prior_term)
    scales = np.array([_prior_scale(prior) for prior in problem.priors])
    covariance, sigma_total, eigenvalues, vectors, good = _sigmas(fisher, scales)
    _, sigma_data, _, _, _ = _sigmas(fisher_data, scales)

    def direction(k):
        vector = vectors[:, k] * np.sign(vectors[np.argmax(np.abs(vectors[:, k])), k])
        largest = max(float(eigenvalues[-1]), 1e-300)
        return {
            "eigenvalue_ratio": float(max(eigenvalues[k], 0.0) / largest),
            "direction": dict(zip(names, vector.tolist(), strict=True)),
        }

    degenerate = [direction(k) for k in range(len(names)) if not good[k]]
    smallest = float(eigenvalues[0])
    condition = float(eigenvalues[-1] / smallest) if smallest > 0 and not degenerate else math.inf
    with np.errstate(invalid="ignore", divide="ignore"):
        correlation = covariance / np.outer(sigma_total, sigma_total)
    prior_dominated, near_boundary = {}, {}
    for i, (name, prior) in enumerate(zip(names, problem.priors, strict=True)):
        prior_dominated[name] = bool(not sigma_data[i] < scales[i])
        low, high = prior.support
        distance = min(abs(theta[i] - low), abs(high - theta[i]))
        # An infinite sigma is a degenerate direction, reported as such; the boundary
        # warning is for finite forecasts that a boundary would truncate.
        near_boundary[name] = bool(
            math.isfinite(distance) and math.isfinite(sigma_total[i]) and distance < sigma_total[i]
        )
    result = Forecast(
        param_names=names,
        fiducial=dict(zip(names, theta.tolist(), strict=True)),
        fisher=fisher,
        fisher_data=fisher_data,
        covariance=covariance,
        sigma=dict(zip(names, sigma_total.tolist(), strict=True)),
        sigma_data=dict(zip(names, sigma_data.tolist(), strict=True)),
        correlation=correlation,
        condition_number=condition,
        degenerate_directions=degenerate,
        weakest_direction=direction(0),
        prior_dominated=prior_dominated,
        near_boundary=near_boundary,
        notes=notes,
    )
    if warn:
        kinds = dict(zip(names, problem.kinds, strict=True)) if problem.kinds else {}
        dominated = [
            n for n in names if prior_dominated[n] and kinds.get(n, "metric") != "nuisance"
        ]
        if dominated:
            warnings.warn(
                f"Forecast: the data alone do not constrain {dominated} on the scale of their "
                "priors (the forecast uncertainty from the data is larger than the prior "
                "width, or infinite). The quoted sigma then reflects the prior, and the "
                "Gaussian approximation is not valid for a bounded prior: do not report it as "
                "an expected constraint.",
                PriorDominatedWarning,
                stacklevel=3,
            )
        close = [n for n in names if near_boundary[n]]
        if close:
            warnings.warn(
                f"Forecast: the fiducial value of {close} is within one forecast sigma of a "
                "prior boundary. The posterior will be truncated there and the Fisher "
                "forecast (a Gaussian centred on the fiducial point) is not valid for these "
                "parameters; run a sampler instead.",
                ForecastWarning,
                stacklevel=3,
            )
    return result


def forecast(
    metric,
    data,
    fiducial: Mapping[str, float],
    qpo_model=None,
    priors: Mapping[str, Prior] | None = None,
    *,
    warn: bool = True,
    **options,
) -> Forecast:
    r"""Fisher-matrix forecast of the constraints a dataset would give.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
    data : dataset, registry id, or sequence of them
        Only the uncertainties of the data matter, not their central values:
        the forecast assumes that the data equal the model prediction at the
        fiducial point. Use a registry dataset, or a simulated one
        (:func:`simulate_qpo_dataset`, :func:`simulate_shadow_dataset`) with
        the uncertainties you want to explore.
    fiducial : mapping of str -> float
        Fiducial values of every sampled parameter (the names are those of
        :func:`build_problem`, e.g. ``{"a": 0.3, "M_src": 6, "r_src": 5.7}``;
        a shadow inclination is ``"inclination_deg"``, in degrees).
    qpo_model, priors, **options
        As in :class:`blackpy.infer.Fit` / :func:`build_problem`; the priors
        decide which parameters are free and enter the Fisher matrix
        through their curvature (Gaussian: ``1/sigma^2``; uniform: none).
    warn : bool
        Emit the warnings of this function (boundary, prior dominance) and
        of :func:`build_problem`.

    Returns
    -------
    Forecast
        ``print(result)`` shows the uncertainties and the degenerate or
        least constrained direction. A singular or badly conditioned Fisher
        matrix is a result, not an error: the degenerate direction is
        reported as an eigenvector in parameter names, and the affected
        uncertainties are infinite. This complements the correlation-based
        degeneracy guard of :class:`Fit`, which needs samples and misses
        non-monotonic degeneracies.

    Notes
    -----
    **Limits.** This is a Gaussian approximation around the fiducial point.
    It is invalid at prior boundaries (a warning is raised when the
    fiducial value is within one forecast sigma of one) and for
    prior-dominated parameters (also warned about), and it cannot represent
    curved or multimodal posteriors. Use it to explore designs quickly, and
    a sampler for results.
    """
    problem = build_problem(metric, data, qpo_model, priors, warn=warn, **options)
    return forecast_from_problem(problem, fiducial, warn=warn)
