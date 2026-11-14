"""Likelihood construction.

The joint log-likelihood is the sum over the chosen datasets:

* **QPO datasets** (one per source): Gaussian likelihood on the upper and
  lower high-frequency QPO. Each source has its own mass ``M_<source>`` (in
  solar masses) and orbital radius ``r_<source>`` (in units of M); the metric
  parameters are shared by all sources. Asymmetric uncertainties are treated
  with a split normal: ``sigma_plus`` is used when the model lies above the
  measured value and ``sigma_minus`` when it lies below, with normalisation
  ``sqrt(2/pi) / (sigma_minus + sigma_plus)``.
* **Shadow datasets**: Gaussian or interval (top-hat) likelihood on
  ``r_sh / M`` for an observer at infinity.

The log-likelihood is a pure JAX function of the parameter vector and is
compiled once; root finding for the shadow happens inside it.
"""

from __future__ import annotations

import math
import warnings
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from blackpy.data import QPODataset, ShadowDataset, load
from blackpy.errors import (
    BlackPyError,
    CircularInferenceWarning,
    IgnoredDataWarning,
    NotAsymptoticallyFlatError,
    UnverifiedDataWarning,
)
from blackpy.infer.priors import Gaussian, Prior, Uniform
from blackpy.light.shadow import shadow_radius_core
from blackpy.orbits.qpo import QPOModel, get_qpo_model

#: Default prior on the orbital radius of a QPO source, in units of M.
DEFAULT_RADIUS_PRIOR = (1.0, 50.0)

_HALF_LOG_2PI = 0.5 * math.log(2.0 * math.pi)


def _slug(dataset_id: str) -> str:
    return dataset_id.split("/")[-1]


def _split_normal_logpdf(model, obs, sigma_minus, sigma_plus):
    sigma = jnp.where(model >= obs, sigma_plus, sigma_minus)
    norm = math.log(math.sqrt(2.0 / math.pi) / (sigma_minus + sigma_plus))
    return norm - 0.5 * ((model - obs) / sigma) ** 2


@dataclass
class Problem:
    """A fully specified inference problem.

    Attributes
    ----------
    param_names : list of str
        Names of the sampled parameters, in order.
    param_labels : list of str
        LaTeX labels for plots and tables.
    priors : list of Prior
        One prior per sampled parameter.
    kinds : list of str
        ``"metric"``, ``"mass"`` or ``"radius"`` for each parameter.
    fixed : dict
        Metric parameters held fixed, with their values.
    log_likelihood : callable
        Jit-compiled ``theta -> log L`` for a single parameter vector.
    log_likelihood_batch : callable
        Jit-compiled, vectorised version for arrays of shape ``(n, ndim)``.
    description : dict
        JSON-serialisable summary, stored in the fit result.
    """

    param_names: list[str]
    param_labels: list[str]
    priors: list[Prior]
    kinds: list[str]
    fixed: dict[str, float]
    log_likelihood: callable
    log_likelihood_batch: callable
    description: dict = field(default_factory=dict)

    @property
    def ndim(self) -> int:
        return len(self.param_names)

    def log_prior(self, theta) -> np.ndarray:
        """Log prior density for ``theta`` of shape ``(ndim,)`` or ``(n, ndim)``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        total = np.zeros(theta.shape[0])
        for i, prior in enumerate(self.priors):
            total = total + prior.logpdf(theta[:, i])
        return total

    def prior_transform(self, u) -> np.ndarray:
        """Map the unit cube to parameter space (for nested sampling)."""
        u = np.asarray(u, dtype=float)
        return np.array([prior.ppf(u[i]) for i, prior in enumerate(self.priors)])

    def log_posterior_batch(self, theta) -> np.ndarray:
        """Unnormalised log posterior for an array of shape ``(n, ndim)``."""
        theta = np.atleast_2d(np.asarray(theta, dtype=float))
        log_prior = self.log_prior(theta)
        out = np.full(theta.shape[0], -np.inf)
        ok = np.isfinite(log_prior)
        if ok.any():
            # Evaluate the whole batch (keeps the compiled shape fixed) and mask.
            log_like = np.asarray(self.log_likelihood_batch(jnp.asarray(theta)))
            out[ok] = log_prior[ok] + log_like[ok]
        return np.where(np.isnan(out), -np.inf, out)

    def sample_prior(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw ``n`` parameter vectors from the prior."""
        return np.column_stack([prior.sample(rng, n) for prior in self.priors])


def _resolve_datasets(data) -> list:
    if isinstance(data, str | QPODataset | ShadowDataset):
        data = [data]
    datasets = [load(d) if isinstance(d, str) else d for d in data]
    if not datasets:
        raise ValueError("At least one dataset is required.")
    slugs = [_slug(d.id) for d in datasets]
    if len(set(slugs)) != len(slugs):
        raise ValueError(f"Dataset names must be unique, got {slugs}.")
    return datasets


def build_problem(
    metric,
    data,
    qpo_model: str | QPOModel | None = None,
    priors: Mapping[str, Prior] | None = None,
    *,
    fixed: Mapping[str, float] | None = None,
    shadow_likelihood: str = "gaussian",
    shadow_level: str = "2sigma",
    warn: bool = True,
) -> Problem:
    """Assemble parameters, priors and the joint log-likelihood.

    Parameters
    ----------
    metric : StaticSpherical
        The spacetime model.
    data : dataset, id, or sequence of them
        QPO and/or shadow datasets (objects or registry identifiers).
    qpo_model : str or QPOModel
        QPO model; required if any QPO dataset is present.
    priors : mapping of str -> Prior
        Priors by parameter name. Metric parameters with a prior are sampled;
        the others are fixed (see ``fixed``). Source parameters are named
        ``M_<source>`` and ``r_<source>``; the generic keys ``"M"`` and
        ``"r"`` apply to every source without a specific entry. Defaults: the
        registry mass measurement (Gaussian, truncated at zero) for the mass
        and ``Uniform(1, 50)`` for the radius.
    fixed : mapping of str -> float
        Values of metric parameters that are not sampled (default: the
        parameter's default value).
    shadow_likelihood : {"gaussian", "interval"}
        Form of the shadow likelihood. ``"interval"`` is flat inside the
        bounds of ``shadow_level`` and zero outside.
    shadow_level : str
        Which bound of the dataset the interval likelihood uses.
    warn : bool
        Emit the integrity warnings (circular mass prior, ignored
        low-frequency QPO, data not verified by a human).

    Returns
    -------
    Problem
    """
    priors = dict(priors or {})
    fixed = dict(fixed or {})
    datasets = _resolve_datasets(data)
    qpo_sets = [d for d in datasets if isinstance(d, QPODataset)]
    shadow_sets = [d for d in datasets if isinstance(d, ShadowDataset)]
    if len(qpo_sets) + len(shadow_sets) != len(datasets):
        raise TypeError("Datasets must be QPODataset or ShadowDataset objects (or registry ids).")
    model = None
    if qpo_sets:
        if qpo_model is None:
            raise ValueError("QPO datasets were given: choose a QPO model, e.g. qpo_model='RP'.")
        model = get_qpo_model(qpo_model)
    if shadow_likelihood not in ("gaussian", "interval"):
        raise ValueError("shadow_likelihood must be 'gaussian' or 'interval'.")

    names, labels, prior_list, kinds = [], [], [], []
    overlap = set(fixed) & set(priors) & set(metric.param_names)
    if overlap:
        raise ValueError(f"Parameters {sorted(overlap)} have both a prior and a fixed value.")
    unknown_fixed = set(fixed) - set(metric.param_names)
    if unknown_fixed:
        raise ValueError(f"'fixed' contains unknown metric parameters {sorted(unknown_fixed)}.")
    fixed_values, free_index = {}, {}
    for par in metric.parameters:
        if par.name in priors:
            free_index[par.name] = len(names)
            names.append(par.name)
            labels.append(par.symbol)
            prior_list.append(priors.pop(par.name))
            kinds.append("metric")
        else:
            value = fixed.get(par.name, par.default)
            if value is None:
                raise ValueError(
                    f"Metric parameter '{par.name}' has no prior, no fixed value and no default. "
                    f"Add priors={{'{par.name}': ...}} or fixed={{'{par.name}': value}}."
                )
            fixed_values[par.name] = float(value)

    generic_mass = priors.pop("M", None)
    generic_radius = priors.pop("r", None)
    source_index, circular, ignored_low = [], [], []
    for d in qpo_sets:
        slug = _slug(d.id)
        if not d.has_frequencies:
            raise BlackPyError(
                f"Dataset '{d.id}' has null QPO frequencies (not yet verified from the source), "
                "so it cannot be used in a fit. See its 'todo' field."
            )
        mass_prior = priors.pop(f"M_{slug}", generic_mass)
        if mass_prior is None:
            mp = d.mass_prior
            if not mp.is_complete:
                raise BlackPyError(
                    f"Dataset '{d.id}' has no usable mass measurement (null in the registry). "
                    f"Pass an explicit prior, e.g. priors={{'M_{slug}': Uniform(3, 20)}}."
                )
            if mp.independent is not True:
                circular.append(d.id)
            mass_prior = Gaussian(mp.value, mp.sigma, low=0.0)
        radius_prior = priors.pop(f"r_{slug}", generic_radius)
        if radius_prior is None:
            radius_prior = Uniform(*DEFAULT_RADIUS_PRIOR)
        if d.low is not None:
            ignored_low.append(d.id)
        source_index.append((len(names), len(names) + 1))
        names += [f"M_{slug}", f"r_{slug}"]
        labels += [rf"M_{{\rm {d.object}}}/M_\odot", rf"r_{{\rm {d.object}}}/M"]
        prior_list += [mass_prior, radius_prior]
        kinds += ["mass", "radius"]
    if priors:
        raise ValueError(
            f"Priors were given for unknown parameters {sorted(priors)}. "
            f"Sampled parameters of this problem: {names}; metric parameters: "
            f"{list(metric.param_names)}."
        )
    if not names:
        raise ValueError("Nothing to sample: give a prior for at least one parameter.")

    if shadow_sets:
        reference = {
            **fixed_values,
            **{n: float(prior_list[i].ppf(0.5)) for n, i in free_index.items()},
        }
        if not metric.is_asymptotically_flat(**reference):
            raise NotAsymptoticallyFlatError(
                f"The shadow likelihood compares r_sh/M seen from infinity, but metric "
                f"'{metric.name}' is not asymptotically flat."
            )
    shadow_terms = []
    for d in shadow_sets:
        if shadow_likelihood == "gaussian":
            shadow_terms.append(("gaussian", *d.gaussian()))
        else:
            if shadow_level not in d.bounds or None in d.bounds[shadow_level]:
                raise BlackPyError(
                    f"Dataset '{d.id}' has no complete '{shadow_level}' bound. "
                    f"Available: {list(d.bounds)}."
                )
            shadow_terms.append(("interval", *d.bounds[shadow_level]))

    if warn:
        if circular:
            warnings.warn(
                f"Mass prior of {circular} is not an independent measurement (it was derived "
                "from a QPO fit in the Kerr/Schwarzschild metric, or its origin is unknown). "
                "Using it to constrain another metric with the same QPOs is circular inference; "
                "pass your own prior or drop the source.",
                CircularInferenceWarning,
                stacklevel=3,
            )
        if ignored_low:
            warnings.warn(
                f"The low-frequency (type-C) QPO of {ignored_low} is ignored: in a spherically "
                "symmetric spacetime the nodal precession frequency vanishes identically, so it "
                "cannot be modelled. Only the twin high-frequency QPOs are fitted.",
                IgnoredDataWarning,
                stacklevel=3,
            )
        unverified = [d.id for d in datasets if not d.verified_by_human]
        if unverified:
            warnings.warn(
                f"Datasets {unverified} have not been verified by a human against their source "
                "(verified_by_human: false). Check the values before publishing results.",
                UnverifiedDataWarning,
                stacklevel=3,
            )

    qpo_obs = [
        (
            d.upper.value,
            d.upper.err_minus,
            d.upper.err_plus,
            d.lower.value,
            d.lower.err_minus,
            d.lower.err_plus,
        )
        for d in qpo_sets
    ]
    param_order = metric.param_names

    def log_likelihood(theta):
        p = (
            jnp.stack(
                [
                    theta[free_index[n]] if n in free_index else jnp.asarray(fixed_values[n])
                    for n in param_order
                ]
            )
            if param_order
            else jnp.zeros((0,))
        )
        total = jnp.asarray(0.0)
        for (i_mass, i_radius), obs in zip(source_index, qpo_obs, strict=True):
            nu_u, nu_l = model.frequencies_core(metric, p, theta[i_radius], theta[i_mass])
            total = total + _split_normal_logpdf(nu_u, obs[0], obs[1], obs[2])
            total = total + _split_normal_logpdf(nu_l, obs[3], obs[4], obs[5])
        if shadow_terms:
            r_sh = shadow_radius_core(metric, p)
            for kind, a, b in shadow_terms:
                if kind == "gaussian":
                    total = total - 0.5 * ((r_sh - a) / b) ** 2 - math.log(b) - _HALF_LOG_2PI
                else:
                    inside = (r_sh >= a) & (r_sh <= b)
                    total = total + jnp.where(inside, -math.log(b - a), -jnp.inf)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    description = {
        "metric": {
            "name": metric.name,
            "parameters": list(metric.param_names),
            "symbolic_f": str(metric.symbolic.f) if metric.is_symbolic else None,
            "reference": metric.reference,
        },
        "fixed": fixed_values,
        "qpo_model": None if model is None else model.name,
        "qpo_model_formula": None if model is None else model.formula,
        "datasets": [
            {
                "id": d.id,
                "object": d.object,
                "data_type": d.data_type,
                "dataset_version": d.dataset_version,
                "sha256": d.sha256,
                "verified_by_human": d.verified_by_human,
            }
            for d in datasets
        ],
        "shadow_likelihood": shadow_likelihood if shadow_sets else None,
        "shadow_level": shadow_level if shadow_sets and shadow_likelihood == "interval" else None,
        "priors": {n: pr.to_dict() for n, pr in zip(names, prior_list, strict=True)},
        "parameter_kinds": dict(zip(names, kinds, strict=True)),
        "circular_mass_priors": circular,
        "ignored_low_frequency_qpos": ignored_low,
    }
    return Problem(
        param_names=names,
        param_labels=labels,
        priors=prior_list,
        kinds=kinds,
        fixed=fixed_values,
        log_likelihood=jax.jit(log_likelihood),
        log_likelihood_batch=jax.jit(jax.vmap(log_likelihood)),
        description=description,
    )


def simulate_qpo_dataset(
    metric,
    qpo_model,
    *,
    name: str,
    mass: float,
    r: float,
    sigma_nu: float | Sequence[float],
    mass_sigma: float,
    rng: np.random.Generator | None = None,
    **params,
) -> QPODataset:
    """Create a synthetic QPO dataset from a known model (for tests and forecasts).

    Parameters
    ----------
    metric : StaticSpherical
    qpo_model : str or QPOModel
    name : str
        Source name; the dataset id becomes ``"synthetic/<name>"``.
    mass : float
        True mass in solar masses.
    r : float
        True orbital radius in units of M.
    sigma_nu : float or (float, float)
        1-sigma uncertainty of the upper and lower frequency in Hz.
    mass_sigma : float
        1-sigma uncertainty of the (independent) mass measurement.
    rng : numpy.random.Generator, optional
        If given, Gaussian noise is added to the frequencies and the mass.
        Without it the data equal the model prediction exactly (a
        noise-free "Asimov" dataset, whose posterior is centred on the truth).
    **params
        True metric parameters.

    Returns
    -------
    QPODataset
    """
    from blackpy.data.registry import parse_dataset

    model = get_qpo_model(qpo_model)
    nu_u, nu_l = (float(v) for v in model.frequencies(metric, r, mass, **params))
    if not (np.isfinite(nu_u) and np.isfinite(nu_l)):
        raise ValueError("The model frequencies are not finite at this radius (inside the ISCO?).")
    sig_u, sig_l = (sigma_nu, sigma_nu) if np.isscalar(sigma_nu) else sigma_nu
    mass_obs = float(mass)
    if rng is not None:
        nu_u += rng.normal(0.0, sig_u)
        nu_l += rng.normal(0.0, sig_l)
        mass_obs += rng.normal(0.0, mass_sigma)

    def entry(value, sigma, unit):
        return {
            "value": float(value),
            "err_minus": float(sigma),
            "err_plus": float(sigma),
            "unit": unit,
            "source": "synthetic",
            "location": "simulated",
        }

    raw = {
        "schema_version": 1,
        "id": f"synthetic/{name}",
        "object": name,
        "data_type": "qpo_pair",
        "dataset_version": "synthetic",
        "verified_by_human": True,
        "description": f"Synthetic data from {metric.name}, model {model.name}.",
        "frequencies": {"upper": entry(nu_u, sig_u, "Hz"), "lower": entry(nu_l, sig_l, "Hz")},
        "mass_prior": {
            **entry(mass_obs, mass_sigma, "Msun"),
            "method": "synthetic",
            "independent": True,
        },
        "intended_models": [model.name],
        "references": {"synthetic": {"citation": "synthetic data generated by BlackPy"}},
        "source_access": "generated in memory",
        "notes": f"truth: mass={mass}, r={r}, params={params}",
    }
    return parse_dataset(raw, origin=f"synthetic/{name}")
