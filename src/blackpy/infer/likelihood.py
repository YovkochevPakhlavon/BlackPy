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
  degrees) is a nuisance parameter named ``"inclination_deg"`` (before 1.0:
  ``"inclination"``, still accepted with a :class:`DeprecationWarning`).

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

from blackpy._deprecation import warn_deprecated
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
from blackpy.light.plasma import PlasmaModel, static_shadow_radius_core, with_plasma
from blackpy.light.rotating import (
    BAMBI2019_FACTOR,
    _shape_core,
    check_delta_c_definition,
    require_separable,
    warn_if_definition_mismatch,
)
from blackpy.light.shadow import shadow_radius_core
from blackpy.orbits.circular import direction_sign, require_equatorial_orbits
from blackpy.orbits.qpo import QPOModel, get_qpo_model

#: Name of the inclination nuisance parameter (degrees) of rotating-metric shadows.
#: Every angle in the public API is in radians; a quantity in degrees has ``_deg``
#: in its name (design decision 39). Per-source copies are ``inclination_deg_<source>``.
INCLINATION = "inclination_deg"

#: Name of the same parameter before 1.0 (deprecated; accepted with a warning).
LEGACY_INCLINATION = "inclination"

#: Default prior of the inclination: random orientation, 0 to 90 degrees.
DEFAULT_INCLINATION_PRIOR = (0.0, 90.0)

#: Default prior on the orbital radius of a QPO source, in units of M.
DEFAULT_RADIUS_PRIOR = (1.0, 50.0)


def legacy_inclination_key(key) -> str | None:
    """The 1.0 name of a pre-1.0 inclination parameter name, or None.

    ``"inclination"`` -> ``"inclination_deg"`` and ``"inclination_<source>"`` ->
    ``"inclination_deg_<source>"``; every other name gives None.
    """
    if not isinstance(key, str):
        return None
    if key == LEGACY_INCLINATION:
        return INCLINATION
    prefix = LEGACY_INCLINATION + "_"
    if key.startswith(prefix) and not (key == INCLINATION or key.startswith(INCLINATION + "_")):
        return f"{INCLINATION}_{key[len(prefix) :]}"
    return None


def rename_legacy_inclination(names, what: str, *, stacklevel: int = 4):
    """Map pre-1.0 inclination names in a mapping or sequence to the 1.0 names.

    Each renamed key emits a :class:`DeprecationWarning` naming the new key; a
    name given under both spellings is an error. ``None`` passes through.
    """
    if names is None or isinstance(names, str):
        return names
    is_mapping = isinstance(names, Mapping)
    keys = list(names)
    present = set(keys)
    out_keys = []
    for key in keys:
        new = legacy_inclination_key(key)
        if new is None:
            out_keys.append(key)
            continue
        if new in present:
            raise ValueError(
                f"{what} gives the inclination twice, as '{key}' (deprecated name) and as "
                f"'{new}'. Keep only '{new}' (degrees)."
            )
        warn_deprecated(
            f"The parameter name '{key}' (in {what})",
            f"'{new}' (the inclination is in degrees, so its name carries '_deg')",
            stacklevel=stacklevel,
        )
        out_keys.append(new)
    if is_mapping:
        return {new: names[old] for new, old in zip(out_keys, keys, strict=True)}
    return type(names)(out_keys) if isinstance(names, (list, tuple)) else out_keys


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
        ``"metric"``, ``"plasma"``, ``"mass"``, ``"radius"`` or
        ``"nuisance"`` (a shadow inclination) for each parameter.
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
        (``"qpo"``, ``"shadow"``, ``"shadow_interval"`` or
        ``"circularity"``), and either ``value``, ``err_minus``,
        ``err_plus`` (Gaussian or split-normal datum), ``lower``, ``upper``
        (interval) or ``upper`` (circularity bound). Used by the goodness
        of fit and by the posterior predictive check.
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
        raise ValueError(
            "At least one dataset is required, e.g. data='eht/sgra' or a list of registry "
            "ids (see blackpy.data.list())."
        )
    slugs = [_slug(d.id) for d in datasets]
    if len(set(slugs)) != len(slugs):
        raise ValueError(
            f"Dataset names must be unique, got {slugs}: the last part of each id names the "
            "source parameters (M_<name>, r_<name>). Give simulated datasets different names."
        )
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
    plasma: PlasmaModel | None = None,
    shared: Sequence[str] | None = None,
    per_source: Sequence[str] | None = None,
    shadow_mass_prior: str | Mapping[str, str] | None = None,
    circularity: str | None = None,
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
        shadow data, ``"inclination_deg"`` (the observer inclination in
        degrees) is sampled as a nuisance parameter; its default prior is
        ``Isotropic(0, 90)``. The pre-1.0 name ``"inclination"`` is accepted
        with a :class:`DeprecationWarning`. Per-source
        parameters (see ``per_source``) are named ``<name>_<source>``; a
        prior under the bare name applies to every source without its own.
    fixed : mapping of str -> float
        Values of metric parameters that are not sampled (default: the
        parameter's default value). ``fixed={"inclination_deg": 30.0}``
        fixes the shadow inclination (degrees) instead of sampling it.
    shadow_likelihood : {"gaussian", "interval"}
        Form of the shadow likelihood. ``"interval"`` is flat inside the
        bounds of ``shadow_level`` and zero outside.
    shadow_level : str
        Which bound of the dataset the interval likelihood uses.
    warn : bool
        Emit the integrity warnings (circular mass prior, ignored
        low-frequency QPO, static metric for a rotating source, data not
        verified by a human, circularity bound with another definition).
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
    plasma : blackpy.light.PlasmaModel, optional
        Plasma around the black hole; it changes the shadow size (not the
        QPO frequencies). Its parameters (for example ``omega_p``, the
        plasma frequency in units of the photon frequency at infinity) are
        handled like metric parameters: sampled if they have a prior, fixed
        otherwise (``fixed`` or their default). Parameter values for which
        the plasma is opaque along the path get zero likelihood. A plasma
        parameter is usually degenerate with the metric parameters: the
        prior-dominance and degeneracy guards apply to it.
    shared, per_source : sequence of str, optional
        Metric (or plasma) parameters that have one value for all sources
        (``shared``) or one value per dataset (``per_source``; the copies
        are named ``<name>_<source>``, e.g. ``a_m87``). Metric parameters
        are in units of each object's own mass, so a charge-like parameter
        has no reason to be equal for two black holes. **With two or more
        shadow datasets every sampled metric parameter must be listed in
        one of the two**; the spin is per source unless listed in
        ``shared``, and each shadow source has its own inclination
        (``inclination_deg_<source>``). In all other fits the default is
        ``shared``, as in earlier versions.
    shadow_mass_prior : str or mapping of str -> str, optional
        Which mass prior of a shadow dataset the Gaussian bound on
        ``r_sh/M`` assumes (a key of ``dataset.mass_priors``, e.g.
        ``"gas_dynamics"``), for all shadow datasets or per source
        (``{"m87": "gas_dynamics"}``). Default: the one the dataset's bound
        was published with. A mass prior that is not independent of the
        image (``"eht_ring"`` for M87*) triggers a
        :class:`blackpy.errors.CircularInferenceWarning`.
    circularity : str, optional
        Use the circularity bound of the shadow datasets that have one
        (rotating metrics): parameter values whose shadow has a deviation
        from circularity above the bound get zero likelihood. The value is
        the **name of the definition** of the deviation from circularity
        (see :func:`blackpy.light.deviation_from_circularity`); there is no
        default. If it differs from the definition the bound was derived
        for, a :class:`blackpy.errors.CircularityDefinitionWarning` is
        raised.

    Returns
    -------
    Problem
    """
    priors = dict(rename_legacy_inclination(priors or {}, "priors"))
    fixed = dict(rename_legacy_inclination(fixed or {}, "fixed"))
    shared = rename_legacy_inclination(shared, "shared")
    per_source = rename_legacy_inclination(per_source, "per_source")
    datasets = _resolve_datasets(data)
    plasma_names: tuple[str, ...] = ()
    if plasma is not None:
        if not any(isinstance(d, ShadowDataset) for d in datasets):
            raise ValueError(
                "A plasma only changes the shadow, but no shadow dataset was given: remove "
                "plasma=... or add a shadow dataset (e.g. 'eht/sgra')."
            )
        plasma_names = plasma.param_names
        metric = with_plasma(metric, plasma)
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
    if circularity is not None:
        check_delta_c_definition(circularity)
        if not rotating:
            raise ValueError(
                f"circularity='{circularity}' was requested, but the shadow of the static "
                f"metric '{metric.name}' is a circle for every observer: the bound cannot "
                "constrain it. Use a rotating metric or drop the option."
            )
        if not any(
            d.circularity is not None and d.circularity.upper is not None for d in shadow_sets
        ):
            raise ValueError(
                "circularity was requested, but none of the shadow datasets carries a "
                "circularity bound. Drop circularity=... or add a dataset that has one "
                "(for example 'eht/m87')."
            )

    # ------------------------------------------------------------------ parameter layout
    slugs = [_slug(d.id) for d in datasets]
    shadow_slugs = [_slug(d.id) for d in shadow_sets]
    multi_shadow = len(shadow_sets) >= 2
    shared_set, per_source_set = set(shared or ()), set(per_source or ())
    unknown = (shared_set | per_source_set) - set(metric.param_names)
    if unknown:
        raise ValueError(
            f"shared/per_source name unknown parameters {sorted(unknown)}. Parameters of this "
            f"model: {list(metric.param_names)}."
        )
    if shared_set & per_source_set:
        raise ValueError(
            f"Parameters {sorted(shared_set & per_source_set)} are listed both as shared and "
            "as per_source. List each parameter in only one of the two."
        )
    spin = getattr(metric, "spin_parameter", None)
    if multi_shadow and spin is not None and spin not in shared_set:
        per_source_set.add(spin)
    if per_source_set and len(datasets) < 2:
        raise ValueError(
            "per_source needs at least two datasets; with one dataset every parameter is "
            "simply sampled (drop per_source=...)."
        )

    def copy_name(name: str, slug: str) -> str:
        return f"{name}_{slug}"

    uses_inclination = bool(shadow_sets) and rotating
    inclination_names = {
        slug: (copy_name(INCLINATION, slug) if multi_shadow else INCLINATION)
        for slug in shadow_slugs
    }
    allowed_fixed = set(metric.param_names)
    allowed_fixed |= {copy_name(n, s) for n in per_source_set for s in slugs}
    if rotating:
        allowed_fixed |= {INCLINATION, *inclination_names.values()}
    unknown_fixed = set(fixed) - allowed_fixed
    if unknown_fixed:
        raise ValueError(
            f"'fixed' contains unknown parameters {sorted(unknown_fixed)}. Parameters that can "
            f"be fixed here: {sorted(allowed_fixed)}."
        )
    inclination_keys = {INCLINATION, *inclination_names.values()}
    if not uses_inclination and (inclination_keys & (set(fixed) | set(priors))):
        raise ValueError(
            f"'{INCLINATION}' is only a parameter when a rotating metric is fitted to shadow data"
            + (
                "."
                if rotating
                else f": the shadow of the static metric '{metric.name}' is a circle for every "
                "observer."
            )
        )

    names, labels, prior_list, kinds = [], [], [], []
    base_of: dict[str, str] = {}
    #: layout[slug][parameter] = ("free", index into theta) or ("fixed", value)
    layout: dict[str, dict[str, tuple]] = {slug: {} for slug in slugs}
    fixed_values: dict[str, float] = {}
    undecided = []

    def add_parameter(name, label, prior, kind, base):
        names.append(name)
        labels.append(label)
        prior_list.append(prior)
        kinds.append(kind)
        base_of[name] = base
        return len(names) - 1

    for par in metric.parameters:
        kind = "plasma" if par.name in plasma_names else "metric"
        if par.name in per_source_set:
            generic = priors.get(par.name)
            for d, slug in zip(datasets, slugs, strict=True):
                key = copy_name(par.name, slug)
                if key in fixed and (key in priors):
                    raise ValueError(
                        f"Parameter '{key}' has both a prior and a fixed value: remove it from "
                        "priors (to fix it) or from fixed (to sample it)."
                    )
                prior = priors.pop(key, None)
                if prior is None and key not in fixed:
                    prior = generic
                if prior is not None:
                    label = rf"{par.symbol}^{{\rm {d.object}}}"
                    layout[slug][par.name] = (
                        "free",
                        add_parameter(key, label, prior, kind, par.name),
                    )
                else:
                    value = fixed.get(key, fixed.get(par.name, par.default))
                    if value is None:
                        raise ValueError(
                            f"Per-source parameter '{key}' has no prior, no fixed value and no "
                            f"default. Add priors={{'{key}': ...}} or fixed={{'{key}': value}}."
                        )
                    layout[slug][par.name] = ("fixed", float(value))
                    fixed_values[key] = float(value)
            priors.pop(par.name, None)
            continue
        if par.name in priors:
            if par.name in fixed:
                raise ValueError(
                    f"Parameter '{par.name}' has both a prior and a fixed value: remove it from "
                    "priors (to fix it) or from fixed (to sample it)."
                )
            if multi_shadow and par.name not in shared_set:
                undecided.append(par.name)
            entry = (
                "free",
                add_parameter(par.name, par.symbol, priors.pop(par.name), kind, par.name),
            )
        else:
            value = fixed.get(par.name, par.default)
            if value is None:
                raise ValueError(
                    f"Metric parameter '{par.name}' has no prior, no fixed value and no default. "
                    f"Add priors={{'{par.name}': ...}} or fixed={{'{par.name}': value}}."
                )
            entry = ("fixed", float(value))
            fixed_values[par.name] = float(value)
        for slug in slugs:
            layout[slug][par.name] = entry
    if undecided:
        objects = [d.object for d in shadow_sets]
        raise ValueError(
            f"This fit combines the shadows of {objects}. The sampled parameter(s) {undecided} "
            "are in units of each object's own mass, so they need not be equal for different "
            "black holes: say explicitly whether each is common to all sources, "
            f"shared={undecided}, or has one value per source, per_source={undecided} (the "
            f"copies are then named e.g. '{undecided[0]}_{shadow_slugs[0]}')."
        )

    generic_mass = priors.pop("M", None)
    generic_radius = priors.pop("r", None)
    source_index, circular, ignored_low, used_low, rotating_sources = {}, [], [], [], []
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
        i_mass = add_parameter(
            f"M_{slug}", rf"M_{{\rm {d.object}}}/M_\odot", mass_prior, "mass", f"M_{slug}"
        )
        i_radius = add_parameter(
            f"r_{slug}", rf"r_{{\rm {d.object}}}/M", radius_prior, "radius", f"r_{slug}"
        )
        source_index[slug] = (i_mass, i_radius)

    #: inclination[slug] = ("free", index) or ("fixed", degrees)
    inclination: dict[str, tuple] = {}
    if uses_inclination:
        generic_prior = priors.get(INCLINATION) if multi_shadow else None
        for d, slug in zip(shadow_sets, shadow_slugs, strict=True):
            key = inclination_names[slug]
            if key in fixed and key in priors:
                raise ValueError(
                    f"'{key}' has both a prior and a fixed value: remove it from priors (to fix "
                    "it) or from fixed (to sample it)."
                )
            value = fixed.get(key, fixed.get(INCLINATION) if key not in priors else None)
            if value is not None and key not in priors:
                value = float(value)
                if not 0.0 <= value <= 180.0:
                    raise ValueError(
                        f"The fixed inclination '{key}' = {value} must lie between 0 and 180 "
                        "degrees. It is in degrees (not radians): use math.degrees(...) to "
                        "convert an angle in radians."
                    )
                inclination[slug] = ("fixed", value)
                continue
            prior = priors.pop(key, None) or generic_prior or Isotropic(*DEFAULT_INCLINATION_PRIOR)
            label = r"i\ [{}^\circ]" if not multi_shadow else rf"i_{{\rm {d.object}}}\ [{{}}^\circ]"
            inclination[slug] = ("free", add_parameter(key, label, prior, "nuisance", INCLINATION))
        if multi_shadow:
            priors.pop(INCLINATION, None)
    if priors:
        raise ValueError(
            f"Priors were given for unknown parameters {sorted(priors)}. "
            f"Sampled parameters of this problem: {names}; metric parameters: "
            f"{list(metric.param_names)}."
        )
    if not names:
        raise ValueError("Nothing to sample: give a prior for at least one parameter.")

    # ------------------------------------------------------------------ shadow data
    def reference_parameters(slug):
        out = {}
        for name, (state, value) in layout[slug].items():
            out[name] = float(prior_list[value].ppf(0.5)) if state == "free" else value
        return out

    for slug in shadow_slugs:
        if not metric.is_asymptotically_flat(**reference_parameters(slug)):
            raise NotAsymptoticallyFlatError(
                f"The shadow likelihood compares r_sh/M seen from infinity, but metric "
                f"'{metric.name}' is not asymptotically flat for the prior medians of its "
                "parameters. Restrict the priors to an asymptotically flat range (for example "
                "Lambda = 0), or declare asymptotically_flat=True if the numerical probe is wrong."
            )
    if isinstance(shadow_mass_prior, str):
        chosen_mass_prior = {slug: shadow_mass_prior for slug in shadow_slugs}
    else:
        chosen_mass_prior = dict(shadow_mass_prior or {})
        unknown = set(chosen_mass_prior) - set(shadow_slugs)
        if unknown:
            raise ValueError(
                f"shadow_mass_prior names unknown shadow sources {sorted(unknown)}; shadow "
                f"sources of this fit: {shadow_slugs}."
            )
    shadow_terms, circular_shadow = [], []
    for d, slug in zip(shadow_sets, shadow_slugs, strict=True):
        key = chosen_mass_prior.get(slug)
        if key is not None and shadow_likelihood != "gaussian":
            raise ValueError(
                "shadow_mass_prior selects the Gaussian bound obtained with a given mass prior; "
                "it cannot be combined with shadow_likelihood='interval'."
            )
        if d.mass_prior_is_independent(key) is False:
            circular_shadow.append((d.id, key or "/".join(d.assumed_mass_priors)))
        if shadow_likelihood == "gaussian":
            shadow_terms.append(("gaussian", *d.gaussian(key)))
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
        if circular_shadow:
            detail = ", ".join(f"'{key}' of {dataset_id}" for dataset_id, key in circular_shadow)
            warnings.warn(
                f"The shadow bound in units of M rests on a mass prior that is not independent "
                f"of the image: {detail}. That mass was obtained from the same horizon-scale "
                "image by assuming a Kerr black hole, so using it to test another metric with "
                "the shadow size is circular inference. Choose an independent mass prior with "
                "shadow_mass_prior=... (see dataset.mass_priors).",
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

    # ------------------------------------------------------------------ data table
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

    low_obs = {}
    for d in qpo_sets:
        slug = _slug(d.id)
        gaussian_row(
            f"nu_upper[{slug}]", d, "qpo", d.upper.value, d.upper.err_minus, d.upper.err_plus, "Hz"
        )
        gaussian_row(
            f"nu_lower[{slug}]", d, "qpo", d.lower.value, d.lower.err_minus, d.lower.err_plus, "Hz"
        )
        low_obs[slug] = d.id in used_low
        if low_obs[slug]:
            gaussian_row(
                f"nu_low[{slug}]", d, "qpo", d.low.value, d.low.err_minus, d.low.err_plus, "Hz"
            )
    circularity_bounds = {}
    for d, slug, (kind, a, b) in zip(shadow_sets, shadow_slugs, shadow_terms, strict=True):
        if kind == "gaussian":
            gaussian_row(f"r_sh[{slug}]", d, "shadow", a, b, b, "M")
        else:
            data_table.append(
                {
                    "label": f"r_sh[{slug}]",
                    "dataset": d.id,
                    "kind": "shadow_interval",
                    "lower": float(a),
                    "upper": float(b),
                    "unit": "M",
                }
            )
        bound = d.circularity
        if circularity is not None and bound is not None and bound.upper is not None:
            if warn:
                warn_if_definition_mismatch(
                    circularity, bound.definition, f"The circularity bound of '{d.id}'"
                )
            circularity_bounds[slug] = float(bound.upper)
            data_table.append(
                {
                    "label": f"delta_c[{slug}]",
                    "dataset": d.id,
                    "kind": "circularity",
                    "upper": float(bound.upper),
                    "definition": circularity,
                    "bound_definition": bound.definition,
                    "unit": "",
                }
            )

    # ------------------------------------------------------------------ model
    param_order = metric.param_names

    def metric_parameters(theta, slug):
        if not param_order:
            return jnp.zeros((0,))
        values = []
        for name in param_order:
            state, value = layout[slug][name]
            values.append(theta[value] if state == "free" else jnp.asarray(value))
        return jnp.stack(values)

    def shadow_values(theta, slug):
        """Shadow size and (if requested) deviation from circularity of one source."""
        p = metric_parameters(theta, slug)
        if not rotating:
            size = (
                static_shadow_radius_core(metric, p)
                if plasma is not None
                else shadow_radius_core(metric, p)
            )
            return size, None
        state, value = inclination[slug]
        degrees = theta[value] if state == "free" else jnp.asarray(value)
        observables, asymmetry = _shape_core(metric, p, jnp.radians(degrees), jnp.inf, n=128)
        size = observables.mean_radius if shadow_size == "mean" else observables.area_radius
        if circularity == "jp_asymmetry":
            shape = asymmetry
        elif circularity == "bambi2019":
            shape = BAMBI2019_FACTOR * observables.delta_c
        else:
            shape = observables.delta_c
        return size, shape

    def model_values(theta):
        """Model prediction for every row of ``data_table``."""
        values = []
        for d in qpo_sets:
            slug = _slug(d.id)
            p = metric_parameters(theta, slug)
            i_mass, i_radius = source_index[slug]
            mass, radius = theta[i_mass], theta[i_radius]
            nu_u, nu_l = model.frequencies_core(metric, p, radius, mass, sign)
            values += [nu_u, nu_l]
            if low_obs[slug]:
                values.append(model.low_frequency_core(metric, p, radius, mass, sign))
        cache = {}
        for slug in shadow_slugs:
            key = (tuple(layout[slug].items()), inclination.get(slug))
            if key not in cache:
                cache[key] = shadow_values(theta, slug)
            size, shape = cache[key]
            values.append(size)
            if slug in circularity_bounds:
                values.append(shape)
        return jnp.stack(values)

    distinct_layouts = {tuple(layout[slug].items()): slug for slug in slugs}

    def log_likelihood(theta):
        values = model_values(theta)
        total = jnp.asarray(0.0)
        for k, row in enumerate(data_table):
            if row["kind"] == "shadow_interval":
                inside = (values[k] >= row["lower"]) & (values[k] <= row["upper"])
                total = total + jnp.where(inside, -math.log(row["upper"] - row["lower"]), -jnp.inf)
            elif row["kind"] == "circularity":
                total = total + jnp.where(values[k] <= row["upper"], 0.0, -jnp.inf)
            else:
                total = total + _split_normal_logpdf(
                    values[k], row["value"], row["err_minus"], row["err_plus"]
                )
        if require_black_hole:
            for slug in distinct_layouts.values():
                p = metric_parameters(theta, slug)
                allowed = jnp.isfinite(static_domain_core(metric, p)[0])
                if rotating:
                    allowed = allowed & metric.in_domain_array(p)
                total = jnp.where(allowed, total, -jnp.inf)
        return jnp.where(jnp.isfinite(total), total, -jnp.inf)

    inclination_record = {
        inclination_names[slug]: ("sampled" if state == "free" else value)
        for slug, (state, value) in inclination.items()
    }
    single_inclination = None
    if uses_inclination and not multi_shadow:
        single_inclination = inclination_record[INCLINATION]
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
        "plasma": None
        if plasma is None
        else {
            "model": type(plasma).__name__,
            "name": plasma.name,
            "parameters": list(plasma_names),
        },
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
        "shadow_mass_priors": {
            slug: chosen_mass_prior.get(slug) or list(d.assumed_mass_priors)
            for d, slug in zip(shadow_sets, shadow_slugs, strict=True)
        },
        "circular_shadow_mass_priors": [list(item) for item in circular_shadow],
        "circularity": None
        if circularity is None
        else {"definition": circularity, "bounds": circularity_bounds},
        "priors": {n: pr.to_dict() for n, pr in zip(names, prior_list, strict=True)},
        "data_table": data_table,
        "parameter_kinds": dict(zip(names, kinds, strict=True)),
        "parameter_base": dict(base_of),
        "shared": sorted(shared_set),
        "per_source": sorted(per_source_set),
        "circular_mass_priors": circular,
        "ignored_low_frequency_qpos": ignored_low,
        "fitted_low_frequency_qpos": used_low,
        "static_metric_for_rotating_sources": rotating_sources,
        "direction": "prograde" if sign > 0 else "retrograde",
        "require_black_hole": bool(require_black_hole),
        "shadow_size": (shadow_size + " radius") if uses_inclination else None,
        "inclination_deg": single_inclination,
        "inclinations_deg": inclination_record or None,
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


def inclination_prior(dataset, *, low: float = 0.0, high: float = 90.0) -> Gaussian:
    """Gaussian prior on the inclination (degrees) from a dataset's measured value.

    The registry stores a measured inclination (for M87* the viewing angle of
    the jet) but the likelihood never uses it on its own: equating the jet
    axis with the spin axis of the black hole is an assumption. Pass the
    result explicitly, e.g. ``priors={"inclination_deg": inclination_prior(m87)}``.

    Parameters
    ----------
    dataset : ShadowDataset or registry id
    low, high : float
        Truncation of the Gaussian, in degrees.
    """
    if isinstance(dataset, str):
        dataset = load(dataset)
    measured = getattr(dataset, "inclination", None)
    if measured is None or not measured.is_complete:
        raise BlackPyError(
            f"Dataset '{dataset.id}' has no measured inclination with an uncertainty. Pass your "
            "own prior, e.g. Gaussian(mean, sigma, low=0, high=90) in degrees."
        )
    return Gaussian(measured.value, measured.sigma, low=low, high=high)


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
        raise ValueError(
            f"The model frequencies are not finite at r = {r}: probably inside the ISCO or "
            "where no circular orbit exists. Choose a larger r (see blackpy.orbits.isco)."
        )
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


def simulate_shadow_dataset(
    metric,
    *,
    name: str,
    sigma: float,
    rng: np.random.Generator | None = None,
    inclination_deg: float | None = None,
    shadow_size: str = "mean",
    **params,
) -> ShadowDataset:
    r"""Create a synthetic shadow-size dataset from a known model (tests and forecasts).

    The shadow equivalent of :func:`simulate_qpo_dataset`. The simulated
    measurement is the shadow radius in units of M seen from infinity: the
    critical impact parameter for a static metric, the mean (or
    area-equivalent) radius of the contour for a rotating one.

    Parameters
    ----------
    metric : StaticSpherical or StationaryAxisymmetric
        Must be asymptotically flat; a rotating metric must have separable
        null geodesics.
    name : str
        Source name; the dataset id becomes ``"synthetic/<name>"``.
    sigma : float
        1-sigma uncertainty of ``r_sh / M``.
    rng : numpy.random.Generator, optional
        If given, Gaussian noise of width ``sigma`` is added. Without it the
        dataset is noise-free.
    inclination_deg : float, optional
        Rotating metrics: inclination of the observer in **degrees** (the
        unit of the ``inclination_deg`` fit parameter). Required for them.
        The pre-1.0 keyword ``inclination`` (also degrees) is accepted with a
        :class:`DeprecationWarning`.
    shadow_size : {"mean", "area"}
        Rotating metrics: which radius is "measured".
    **params
        True metric parameters.

    Returns
    -------
    ShadowDataset
        With 1-sigma and 2-sigma bounds and the fractional deviation
        ``delta = r_sh / (3 sqrt(3)) - 1``, so that both the Gaussian and
        the interval likelihood can be used.
    """
    from blackpy.data.registry import parse_dataset
    from blackpy.light.shadow import shadow_radius

    if LEGACY_INCLINATION in params:
        if inclination_deg is not None:
            raise ValueError(
                "simulate_shadow_dataset() got both 'inclination' (deprecated name) and "
                "'inclination_deg'. Keep only inclination_deg=<degrees>."
            )
        warn_deprecated(
            "The keyword 'inclination' of simulate_shadow_dataset()",
            "inclination_deg=<degrees>",
        )
        inclination_deg = params.pop(LEGACY_INCLINATION)
    if not sigma > 0:
        raise ValueError(
            f"sigma must be positive, got {sigma}: give the 1-sigma uncertainty of r_sh/M."
        )
    if metric.is_rotating_class:
        if inclination_deg is None:
            raise TypeError(
                f"The shadow of the rotating metric '{metric.name}' depends on the observer: "
                "pass inclination_deg=<degrees>."
            )
        if not 0.0 <= float(inclination_deg) <= 180.0:
            raise ValueError(
                f"inclination_deg = {inclination_deg} must lie between 0 and 180 degrees; use "
                "math.degrees(...) to convert an angle in radians."
            )
        truth = shadow_radius(
            metric, inclination=math.radians(inclination_deg), kind=shadow_size, **params
        )
    else:
        truth = shadow_radius(metric, **params)
    truth = float(truth)
    if not math.isfinite(truth):
        raise ValueError(
            f"The model has no shadow for the parameters {params}: no event horizon or no "
            "unstable photon sphere / photon region. Check blackpy.geometry.horizons and "
            "blackpy.light.photon_spheres, and choose other true parameters."
        )
    value = truth + (0.0 if rng is None else float(rng.normal(0.0, sigma)))
    scale = 3.0 * math.sqrt(3.0)
    raw = {
        "schema_version": 1,
        "id": f"synthetic/{name}",
        "object": name,
        "data_type": "shadow_bound",
        "dataset_version": "synthetic",
        "verified_by_human": True,
        "description": f"Synthetic shadow size from {metric.name}.",
        "quantity": "r_sh/M",
        "bounds": {
            "1sigma": {"lower": value - sigma, "upper": value + sigma, "location": "simulated"},
            "2sigma": {
                "lower": value - 2.0 * sigma,
                "upper": value + 2.0 * sigma,
                "location": "simulated",
            },
        },
        "fractional_deviation": {
            "value": value / scale - 1.0,
            "err_minus": sigma / scale,
            "err_plus": sigma / scale,
            "unit": "",
            "source": "synthetic",
            "location": "simulated",
        },
        "references": {"synthetic": {"citation": "synthetic data generated by BlackPy"}},
        "source_access": "generated in memory",
        "notes": f"truth: r_sh/M={truth}, inclination_deg={inclination_deg}, params={params}",
    }
    return parse_dataset(raw, origin=f"synthetic/{name}")
