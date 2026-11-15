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

Rotating metrics (:class:`blackpy.metric.StationaryAxisymmetric`)
-----------------------------------------------------------------
* **QPO triplets**: the low-frequency (type-C) QPO is fitted as well, as the
  nodal precession frequency ``|nu_phi - nu_theta|`` at the same radius
  (Gaussian term, like the twin peaks). With the RP model this is the
  relativistic-precession-model triplet likelihood.
* **Spin and deformation bounds**: the likelihood is zero outside the
  metric's declared black hole domain and wherever no event horizon exists
  (``require_black_hole=True``), so the allowed spin range depends on the
  metric and on its other parameters.
* **Shadow**: the shadow of a rotating hole is not a circle and depends on
  the observer's inclination. The compared quantity is the mean radius of
  the contour (or its area-equivalent radius), and the inclination (in
  degrees) is a nuisance parameter named ``"inclination"``.

The log-likelihood is a pure JAX function of the parameter vector and is
compiled once; root finding for the horizon and the shadow happens inside it.
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
    NodalPrecessionUnavailableError,
    NotAsymptoticallyFlatError,
    SpinIgnoredWarning,
    UnverifiedDataWarning,
)
from blackpy.geometry.horizons import static_domain_core
from blackpy.infer.priors import Gaussian, Isotropic, Prior, Uniform
from blackpy.light.rotating import area_radius_core, mean_radius_core, require_separable
from blackpy.light.shadow import shadow_radius_core
from blackpy.orbits.circular import direction_sign, require_equatorial_orbits
from blackpy.orbits.qpo import QPOModel, get_qpo_model

#: Name of the inclination nuisance parameter (degrees) of rotating-metric shadows.
INCLINATION = "inclination"

#: Default prior of the inclination: random orientation, 0 to 90 degrees.
DEFAULT_INCLINATION_PRIOR = (0.0, 90.0)

#: Default prior on the orbital radius of a QPO source, in units of M.
DEFAULT_RADIUS_PRIOR = (1.0, 50.0)


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
        ``"metric"``, ``"mass"``, ``"radius"`` or ``"nuisance"`` (the shadow
        inclination) for each parameter.
    fixed : dict
        Metric parameters held fixed, with their values.
    log_likelihood : callable
        Jit-compiled ``theta -> log L`` for a single parameter vector.
    log_likelihood_batch : callable
        Jit-compiled, vectorised version for arrays of shape ``(n, ndim)``.
    description : dict
        JSON-serialisable summary, stored in the fit result.
    predict : callable
        Jit-compiled ``theta -> model values``, one per fitted datum, in the
        order of ``data_table``.
    predict_batch : callable
        Vectorised version for arrays of shape ``(n, ndim)``.
    data_table : list of dict
        One row per fitted datum: ``label``, ``dataset``, ``kind``
        (``"qpo"``, ``"shadow"`` or ``"shadow_interval"``), and either
        ``value``, ``err_minus``, ``err_plus`` (Gaussian or split-normal
        datum) or ``lower``, ``upper`` (interval). Used by the goodness of
        fit and by the posterior predictive check.
    """

    param_names: list[str]
    param_labels: list[str]
    priors: list[Prior]
    kinds: list[str]
    fixed: dict[str, float]
    log_likelihood: callable
    log_likelihood_batch: callable
    description: dict = field(default_factory=dict)
    predict: callable = None
    predict_batch: callable = None
    data_table: list = field(default_factory=list)

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
    direction: str | int = "prograde",
    use_low_frequency: bool | None = None,
    require_black_hole: bool | None = None,
    shadow_size: str = "mean",
) -> Problem:
    """Assemble parameters, priors and the joint log-likelihood.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
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
        and ``Uniform(1, 50)`` for the radius. With a rotating metric and
        shadow data, ``"inclination"`` (degrees) is sampled as a nuisance
        parameter; its default prior is ``Isotropic(0, 90)``.
    fixed : mapping of str -> float
        Values of metric parameters that are not sampled (default: the
        parameter's default value). ``fixed={"inclination": 30.0}`` fixes
        the shadow inclination instead of sampling it.
    shadow_likelihood : {"gaussian", "interval"}
        Form of the shadow likelihood. ``"interval"`` is flat inside the
        bounds of ``shadow_level`` and zero outside.
    shadow_level : str
        Which bound of the dataset the interval likelihood uses.
    warn : bool
        Emit the integrity warnings (circular mass prior, ignored
        low-frequency QPO, static metric for a rotating source, data not
        verified by a human).
    direction : {"prograde", "retrograde"}
        Rotating metrics: orbit direction of the QPO-emitting matter.
        Counter-rotation can equally be described by a negative spin.
    use_low_frequency : bool, optional
        Fit the low-frequency QPO of triplets as the nodal precession
        frequency. Default: True for rotating metrics, False for static
        ones (where it vanishes identically; asking for it is an error).
    require_black_hole : bool, optional
        Give zero likelihood to parameter values outside the metric's
        declared black hole domain or without an event horizon. Default:
        True for rotating metrics (this is what bounds the spin), False for
        static ones (unchanged v0.1 behaviour).
    shadow_size : {"mean", "area"}
        Rotating metrics: compare the mean radius of the shadow contour
        (default) or its area-equivalent radius with the shadow data.

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
    if shadow_size not in ("mean", "area"):
        raise ValueError("shadow_size must be 'mean' or 'area'.")
    rotating = metric.is_rotating_class
    sign = direction_sign(direction)
    if qpo_sets:
        require_equatorial_orbits(metric)
    if use_low_frequency is None:
        use_low_frequency = rotating
    elif use_low_frequency and not rotating:
        raise NodalPrecessionUnavailableError(
            f"use_low_frequency=True needs a rotating metric: in the static metric "
            f"'{metric.name}' the nodal precession frequency vanishes identically. Use a "
            "rotating metric (e.g. blackpy.metric.Kerr or rotating_from_static(metric, a))."
        )
    if require_black_hole is None:
        require_black_hole = rotating
    if shadow_sets and rotating:
        require_separable(metric)

    names, labels, prior_list, kinds = [], [], [], []
    overlap = set(fixed) & set(priors) & set(metric.param_names)
    if overlap:
        raise ValueError(f"Parameters {sorted(overlap)} have both a prior and a fixed value.")
    uses_inclination = bool(shadow_sets) and rotating
    unknown_fixed = set(fixed) - set(metric.param_names) - ({INCLINATION} if rotating else set())
    if unknown_fixed:
        raise ValueError(f"'fixed' contains unknown metric parameters {sorted(unknown_fixed)}.")
    if INCLINATION in fixed and INCLINATION in priors:
        raise ValueError("'inclination' has both a prior and a fixed value.")
    if not uses_inclination and (INCLINATION in fixed or INCLINATION in priors):
        raise ValueError(
            "'inclination' is only a parameter when a rotating metric is fitted to shadow data"
            + (
                "."
                if rotating
                else f": the shadow of the static metric '{metric.name}' is a circle for every "
                "observer."
            )
        )
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
    source_index, circular, ignored_low, used_low, rotating_sources = [], [], [], [], []
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
            (used_low if (use_low_frequency and d.is_triplet) else ignored_low).append(d.id)
        if not rotating and (d.low is not None or d.spin is not None):
            rotating_sources.append(d.id)
        source_index.append((len(names), len(names) + 1))
        names += [f"M_{slug}", f"r_{slug}"]
        labels += [rf"M_{{\rm {d.object}}}/M_\odot", rf"r_{{\rm {d.object}}}/M"]
        prior_list += [mass_prior, radius_prior]
        kinds += ["mass", "radius"]
    inclination_index, inclination_fixed = None, None
    if uses_inclination:
        if INCLINATION in fixed:
            inclination_fixed = float(fixed[INCLINATION])
            if not 0.0 <= inclination_fixed <= 180.0:
                raise ValueError("The fixed inclination must lie between 0 and 180 degrees.")
        else:
            inclination_index = len(names)
            names.append(INCLINATION)
            labels.append(r"i\ [{}^\circ]")
            prior_list.append(
                priors.pop(INCLINATION, None) or Isotropic(*DEFAULT_INCLINATION_PRIOR)
            )
            kinds.append("nuisance")
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
            reason = (
                "use_low_frequency=False was requested"
                if rotating
                else "in a spherically symmetric spacetime the nodal precession frequency "
                "vanishes identically, so it cannot be modelled"
            )
            warnings.warn(
                f"The low-frequency (type-C) QPO of {ignored_low} is ignored: {reason}. Only "
                "the twin high-frequency QPOs are fitted.",
                IgnoredDataWarning,
                stacklevel=3,
            )
        if rotating_sources:
            warnings.warn(
                f"The static metric '{metric.name}' is fitted to {rotating_sources}, which "
                "rotate (they have a published spin measurement or a nodal-precession QPO). "
                "The extra metric parameters will absorb the effect of the spin, so the result "
                "is not a physical constraint on them. Use a rotating metric, e.g. "
                "blackpy.metric.rotating_from_static(metric, a).",
                SpinIgnoredWarning,
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
    low_obs = [
        (d.low.value, d.low.err_minus, d.low.err_plus) if d.id in used_low else None
        for d in qpo_sets
    ]
    param_order = metric.param_names
    size_core = mean_radius_core if shadow_size == "mean" else area_radius_core

    # One row per fitted datum, in the order in which ``model_values`` returns them.
    data_table = []

    def gaussian_row(label, dataset, kind, value, minus, plus, unit):
        data_table.append(
            {
                "label": label,
                "dataset": dataset.id,
                "kind": kind,
                "value": float(value),
                "err_minus": float(minus),
                "err_plus": float(plus),
                "unit": unit,
            }
        )

    for d, obs, low in zip(qpo_sets, qpo_obs, low_obs, strict=True):
        slug = _slug(d.id)
        gaussian_row(f"nu_upper[{slug}]", d, "qpo", obs[0], obs[1], obs[2], "Hz")
        gaussian_row(f"nu_lower[{slug}]", d, "qpo", obs[3], obs[4], obs[5], "Hz")
        if low is not None:
            gaussian_row(f"nu_low[{slug}]", d, "qpo", *low, "Hz")
    for d, (kind, a, b) in zip(shadow_sets, shadow_terms, strict=True):
        if kind == "gaussian":
            gaussian_row(f"r_sh[{_slug(d.id)}]", d, "shadow", a, b, b, "M")
        else:
            data_table.append(
                {
                    "label": f"r_sh[{_slug(d.id)}]",
                    "dataset": d.id,
                    "kind": "shadow_interval",
                    "lower": float(a),
                    "upper": float(b),
                    "unit": "M",
                }
            )

    def metric_parameters(theta):
        if not param_order:
            return jnp.zeros((0,))
        return jnp.stack(
            [
                theta[free_index[n]] if n in free_index else jnp.asarray(fixed_values[n])
                for n in param_order
            ]
        )

    def model_values(theta):
        """Model prediction for every row of ``data_table``."""
        p = metric_parameters(theta)
        values = []
        for (i_mass, i_radius), low in zip(source_index, low_obs, strict=True):
            mass, radius = theta[i_mass], theta[i_radius]
            nu_u, nu_l = model.frequencies_core(metric, p, radius, mass, sign)
            values += [nu_u, nu_l]
            if low is not None:
                values.append(model.low_frequency_core(metric, p, radius, mass, sign))
        if shadow_terms:
            if rotating:
                inclination = (
                    theta[inclination_index]
                    if inclination_index is not None
                    else jnp.asarray(inclination_fixed)
                )
                r_sh = size_core(metric, p, jnp.radians(inclination))
            else:
                r_sh = shadow_radius_core(metric, p)
            values += [r_sh] * len(shadow_terms)
        return jnp.stack(values)

    def log_likelihood(theta):
        values = model_values(theta)
        total = jnp.asarray(0.0)
        for k, row in enumerate(data_table):
            if row["kind"] == "shadow_interval":
                inside = (values[k] >= row["lower"]) & (values[k] <= row["upper"])
                total = total + jnp.where(inside, -math.log(row["upper"] - row["lower"]), -jnp.inf)
            else:
                total = total + _split_normal_logpdf(
                    values[k], row["value"], row["err_minus"], row["err_plus"]
                )
        if require_black_hole:
            p = metric_parameters(theta)
            allowed = jnp.isfinite(static_domain_core(metric, p)[0])
            if rotating:
                allowed = allowed & metric.in_domain_array(p)
            total = jnp.where(allowed, total, -jnp.inf)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    description = {
        "metric": {
            "name": metric.name,
            "parameters": list(metric.param_names),
            "class": type(metric).__name__,
            "symbolic_f": str(metric.symbolic.f) if metric.is_symbolic and not rotating else None,
            "symbolic_components": [str(e) for e in metric.symbolic.components]
            if metric.is_symbolic and rotating
            else None,
            "spin_parameter": metric.spin_parameter if rotating else None,
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
        "data_table": data_table,
        "parameter_kinds": dict(zip(names, kinds, strict=True)),
        "circular_mass_priors": circular,
        "ignored_low_frequency_qpos": ignored_low,
        "fitted_low_frequency_qpos": used_low,
        "static_metric_for_rotating_sources": rotating_sources,
        "direction": "prograde" if sign > 0 else "retrograde",
        "require_black_hole": bool(require_black_hole),
        "shadow_size": (shadow_size + " radius") if uses_inclination else None,
        "inclination_deg": inclination_fixed
        if inclination_fixed is not None
        else ("sampled" if uses_inclination else None),
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
        predict=jax.jit(model_values),
        predict_batch=jax.jit(jax.vmap(model_values)),
        data_table=data_table,
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
    sigma_low: float | None = None,
    direction: str | int = "prograde",
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
    sigma_low : float, optional
        Rotating metrics: if given, the dataset is a triplet whose
        low-frequency QPO is the nodal precession frequency with this
        1-sigma uncertainty in Hz.
    direction : {"prograde", "retrograde"}
        Rotating metrics: orbit direction.
    **params
        True metric parameters.

    Returns
    -------
    QPODataset
    """
    from blackpy.data.registry import parse_dataset

    model = get_qpo_model(qpo_model)
    options = {"direction": direction} if metric.is_rotating_class else {}
    nu_u, nu_l = (float(v) for v in model.frequencies(metric, r, mass, **options, **params))
    if not (np.isfinite(nu_u) and np.isfinite(nu_l)):
        raise ValueError("The model frequencies are not finite at this radius (inside the ISCO?).")
    nu_low = None
    if sigma_low is not None:
        nu_low = float(model.low_frequency(metric, r, mass, **options, **params))
    sig_u, sig_l = (sigma_nu, sigma_nu) if np.isscalar(sigma_nu) else sigma_nu
    mass_obs = float(mass)
    if rng is not None:
        nu_u += rng.normal(0.0, sig_u)
        nu_l += rng.normal(0.0, sig_l)
        mass_obs += rng.normal(0.0, mass_sigma)
        if nu_low is not None:
            nu_low += rng.normal(0.0, sigma_low)

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
        "data_type": "qpo_pair" if nu_low is None else "qpo_triplet",
        "dataset_version": "synthetic",
        "verified_by_human": True,
        "description": f"Synthetic data from {metric.name}, model {model.name}.",
        "frequencies": {
            "upper": entry(nu_u, sig_u, "Hz"),
            "lower": entry(nu_l, sig_l, "Hz"),
            **({} if nu_low is None else {"low": entry(nu_low, sigma_low, "Hz")}),
        },
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
