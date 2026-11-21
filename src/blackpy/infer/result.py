"""Fit results: posterior samples plus everything needed to reproduce them."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from blackpy._deprecation import warn_deprecated
from blackpy._version import __version__
from blackpy.errors import BlackPyError, BlackPyWarning, PoorFitWarning
from blackpy.infer.likelihood import legacy_inclination_key
from blackpy.infer.priors import Prior, prior_from_dict
from blackpy.infer.quality import (
    POOR_FIT_P_VALUE,
    GoodnessOfFit,
    PosteriorPredictive,
    posterior_predictive,
)

#: A parameter is flagged "prior dominated" if the Kolmogorov-Smirnov distance
#: between its marginal posterior and its prior is below this value.
PRIOR_DOMINATED_KS = 0.1


def _json_default(obj):
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float) and not math.isfinite(obj):
        return str(obj)
    return str(obj)


#: Format of the ``.npz`` result bundle written by :meth:`FitResult.save`.
#: 0: versions 0.1 to 0.3 (no ``bundle_format`` field); 1: since 1.0.
BUNDLE_FORMAT = 1


def _rename_names(obj, renames: dict[str, str]):
    """Rename parameter names in a JSON record: dict keys, strings and "a, b" pairs."""
    if isinstance(obj, dict):
        return {_rename_names(k, renames): _rename_names(v, renames) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_rename_names(v, renames) for v in obj]
    if isinstance(obj, str):
        if obj in renames:
            return renames[obj]
        if ", " in obj:
            parts = obj.split(", ")
            if any(part in renames for part in parts):
                return ", ".join(renames.get(part, part) for part in parts)
    return obj


def _upgrade_format_0(record: dict) -> dict:
    """Bring a format-0 (pre-1.0) record to format 1: rename the inclination columns."""
    renames = {}
    for name in record["param_names"]:
        new = legacy_inclination_key(name)
        if new is not None:
            renames[name] = new
    if renames:
        # The base name of every per-source inclination was "inclination" as well.
        renames.setdefault("inclination", "inclination_deg")
        record = _rename_names(record, renames)
    record["bundle_format"] = BUNDLE_FORMAT
    record.setdefault("metadata", {})["loaded_from_bundle_format"] = 0
    return record


def poor_fit_message(p_value: float, dof: int) -> str:
    """Text of the :class:`blackpy.errors.PoorFitWarning`."""
    return (
        f"Poor fit: the posterior predictive p-value of the chi-square discrepancy is "
        f"{p_value:.2g} (threshold {POOR_FIT_P_VALUE}, {dof} degree(s) of freedom). The model "
        "does not reproduce the data within their uncertainties, so its posterior and any "
        "Bayes factor in its favour describe a model that fails. Inspect "
        "print(result.goodness_of_fit()) for the pulls; check the data uncertainties, the "
        "priors, or use a different model."
    )


@dataclass
class BayesFactor:
    """Result of a model comparison.

    Attributes
    ----------
    log_bf : float
        ``ln B_10 = ln Z(full model) - ln Z(null model)``. Positive values
        favour the full (extended) model, negative values the null model.
    log_bf_err : float or None
        Uncertainty of ``log_bf`` (statistical only).
    method : str
        ``"evidence ratio"`` (nested sampling) or ``"Savage-Dickey"``.
    null : dict
        The null hypothesis, e.g. ``{"Q": 0.0}``.
    is_lower_limit : bool
        True if the posterior density at the null value is below the
        resolution of the samples, so ``log_bf`` is only a lower limit.
    notes : list of str
        Caveats that apply to this number.
    favoured_fit_is_poor : bool or None
        True if the model favoured by this Bayes factor fails the fit-quality
        check (a :class:`blackpy.errors.PoorFitWarning` is then raised: a
        Bayes factor ranks models, it does not say that the better one
        describes the data). None if the check could not be made.
    """

    log_bf: float
    log_bf_err: float | None
    method: str
    null: dict
    is_lower_limit: bool = False
    notes: list[str] = field(default_factory=list)
    favoured_fit_is_poor: bool | None = None

    @property
    def interpretation(self) -> str:
        """Verbal scale of Kass & Raftery (1995), J. Am. Stat. Assoc. 90, 773,
        Sec. 3.2, in terms of ``2 ln B``: 0-2 not worth more than a bare mention, 2-6
        positive, 6-10 strong, >10 very strong."""
        favoured = "full model" if self.log_bf > 0 else "null model"
        two_ln_b = 2.0 * abs(self.log_bf)
        if two_ln_b < 2:
            return "inconclusive (not worth more than a bare mention)"
        strength = "positive" if two_ln_b < 6 else "strong" if two_ln_b < 10 else "very strong"
        return f"{strength} evidence for the {favoured}"

    def __str__(self) -> str:
        err = "" if self.log_bf_err is None else f" +- {self.log_bf_err:.2f}"
        relation = ">=" if self.is_lower_limit else "="
        text = (
            f"ln B_10 {relation} {self.log_bf:.2f}{err} ({self.method}; null: {self.null}): "
            f"{self.interpretation}"
        )
        return "\n  note: ".join([text, *self.notes])


@dataclass
class FitResult:
    """Posterior samples and the full reproducibility record of a fit.

    Attributes
    ----------
    samples : numpy.ndarray, shape (n_samples, ndim)
        Equally weighted posterior samples (after burn-in and thinning for
        emcee; resampled to equal weights for dynesty).
    log_prob : numpy.ndarray
        Log posterior (emcee) or log likelihood (dynesty) of each sample.
    param_names, param_labels : list of str
        Parameter names and LaTeX labels.
    priors : dict of str -> Prior
    sampler : str
        ``"emcee"`` or ``"dynesty"``.
    log_evidence, log_evidence_err : float or None
        Natural log of the Bayesian evidence and its uncertainty (dynesty).
    config : dict
        Model, data (ids, versions, file hashes), priors, likelihood options.
    metadata : dict
        BlackPy and dependency versions, sampler settings, seed, timing.
    diagnostics : dict
        Convergence, prior-dominance and fit-quality diagnostics.
    predictions : numpy.ndarray or None, shape (n_samples, n_data)
        Model value of every fitted datum (rows of ``config["data_table"]``)
        for every posterior sample. Stored so that the posterior predictive
        check also works on a result loaded from disk.
    """

    samples: np.ndarray
    log_prob: np.ndarray
    param_names: list[str]
    param_labels: list[str]
    priors: dict[str, Prior]
    sampler: str
    log_evidence: float | None = None
    log_evidence_err: float | None = None
    config: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    diagnostics: dict = field(default_factory=dict)
    predictions: np.ndarray | None = None

    # ------------------------------------------------------------------ summaries

    def __getitem__(self, name: str) -> np.ndarray:
        """Samples of one parameter."""
        new = legacy_inclination_key(name)
        if name not in self.param_names and new in self.param_names:
            warn_deprecated(f"The parameter name '{name}'", f"'{new}'", stacklevel=3)
            name = new
        if name not in self.param_names:
            raise KeyError(f"No parameter '{name}'. Available: {self.param_names}.")
        return self.samples[:, self.param_names.index(name)]

    def correlation(self, first: str, second: str, method: str = "pearson") -> float:
        """Posterior correlation coefficient between two sampled parameters.

        Parameters
        ----------
        first, second : str
            Parameter names.
        method : {"pearson", "spearman"}
            Linear or rank correlation.
        """
        x, y = self[first], self[second]
        if method == "pearson":
            return float(np.corrcoef(x, y)[0, 1])
        if method == "spearman":
            from scipy import stats

            return float(stats.spearmanr(x, y).statistic)
        raise ValueError("method must be 'pearson' or 'spearman'.")

    def summary(self, level: float = 0.68) -> dict[str, dict[str, float]]:
        """Posterior median and central credible interval of every parameter.

        Parameters
        ----------
        level : float
            Probability content of the interval (default 0.68).

        Returns
        -------
        dict
            ``name -> {"median", "lower", "upper", "err_minus", "err_plus",
            "mean", "std", "prior_dominated"}``.
        """
        tail = 50.0 * (1.0 - level)
        out = {}
        dominated = self.diagnostics.get("prior_dominated", {})
        for i, name in enumerate(self.param_names):
            lo, med, hi = np.percentile(self.samples[:, i], [tail, 50.0, 100.0 - tail])
            out[name] = {
                "median": float(med),
                "lower": float(lo),
                "upper": float(hi),
                "err_minus": float(med - lo),
                "err_plus": float(hi - med),
                "mean": float(np.mean(self.samples[:, i])),
                "std": float(np.std(self.samples[:, i])),
                "prior_dominated": bool(dominated.get(name, False)),
            }
        return out

    def __str__(self) -> str:
        lines = [f"FitResult ({self.sampler}, {self.samples.shape[0]} samples)"]
        for name, s in self.summary().items():
            flag = "   [posterior ~ prior: NOT a constraint]" if s["prior_dominated"] else ""
            lines.append(
                f"  {name:<24} = {s['median']:.5g} +{s['err_plus']:.3g} -{s['err_minus']:.3g}{flag}"
            )
        if self.log_evidence is not None:
            lines.append(f"  ln Z = {self.log_evidence:.3f} +- {self.log_evidence_err:.3f}")
        if "converged" in self.diagnostics:
            lines.append(f"  converged: {self.diagnostics['converged']}")
        if "goodness_of_fit" in self.diagnostics:
            lines.append("  fit quality: " + self.goodness_of_fit().summary_line())
            check = self.diagnostics.get("posterior_predictive", {})
            if check.get("p_value") is not None:
                flag = "   [POOR FIT]" if check.get("poor_fit") else ""
                lines.append(
                    f"  posterior predictive p = {check['p_value']:.3g} (conservative, not "
                    f"uniform; warning below {check['threshold']}){flag}"
                )
        spin = self.diagnostics.get("spin_parameter")
        for name, corr in self.diagnostics.get("spin_correlations", {}).items():
            flag = "   [strongly degenerate]" if self.diagnostics["spin_degenerate"][name] else ""
            lines.append(
                f"  corr({spin}, {name}) = {corr['pearson']:+.3f} "
                f"(rank {corr['spearman']:+.3f}){flag}"
            )
        for pair, corr in self.diagnostics.get("plasma_correlations", {}).items():
            flag = "   [strongly degenerate]" if corr["degenerate"] else ""
            lines.append(
                f"  corr({pair}) = {corr['pearson']:+.3f} (rank {corr['spearman']:+.3f}){flag}"
            )
        return "\n".join(lines)

    # ------------------------------------------------------------------ fit quality

    def goodness_of_fit(self) -> GoodnessOfFit:
        r"""Pulls, :math:`\chi^2` and degrees of freedom at the best-fit point.

        The best-fit point minimises the :math:`\chi^2` discrepancy (the sum
        of the squared pulls of all fitted data and of all informative
        Gaussian priors) inside the prior support; it was located when the
        fit ran, starting from the best posterior sample.

        **Degrees of freedom.** Number of fitted data (frequencies, shadow
        sizes) plus one for each informative Gaussian prior (for example an
        independent mass measurement, which is a datum in all but name),
        minus the number of free parameters. This is exact for a model that
        is linear in its parameters with Gaussian data; a parameter that the
        data do not constrain, or that sits on a prior boundary, uses less
        than one degree of freedom, so the count is then a lower bound.

        The number of degrees of freedom **can be zero or negative**: each
        QPO source has its own mass and radius, so a two-frequency fit with
        one metric parameter has none. In that case the result reports
        "no degrees of freedom: goodness of fit is undefined" and
        ``p_value`` is None.

        Returns
        -------
        GoodnessOfFit
            Print it for a table of the pulls.
        """
        record = self.diagnostics.get("goodness_of_fit")
        if record is None:
            raise BlackPyError(
                "This result carries no fit-quality record (it was produced by BlackPy < 0.3). "
                "Run the fit again with this version."
            )
        return GoodnessOfFit.from_dict(record)

    def posterior_predictive(
        self, n: int = 2000, seed: int | None = None, *, warn: bool = True
    ) -> PosteriorPredictive:
        r"""Posterior predictive check (Gelman, Meng & Stern 1996, Statistica Sinica 6, 733).

        ``n`` parameter vectors are drawn from the posterior; for each one,
        replicated data are simulated from the model with the stated
        (split-normal) uncertainties, and a replicated measurement is drawn
        for every informative Gaussian prior. The result holds the
        replicated data, a p-value for every datum,
        :math:`P(y^{\rm rep}_k \ge y_k \mid y)`, and the p-value of the
        :math:`\chi^2` discrepancy,
        :math:`P(X^2(y^{\rm rep};\theta) \ge X^2(y;\theta) \mid y)`.

        **The p-value is not uniform under the true model.** The data are
        used twice (to obtain the posterior and to be compared with it), so
        it concentrates around 0.5 and is conservative: Rubin (1996, comment
        on Gelman et al.) and Meng (1994, Ann. Stat. 22, 1142), who shows
        that the frequency of ``p <= alpha`` is "often close to but less
        than" ``alpha`` and never above ``2 alpha``. Consequently a value
        near 0.5 does **not** certify the fit (with few degrees of freedom
        it is what one always gets), while a small value is strong evidence
        against the model. The warning threshold is 0.05
        (:data:`blackpy.infer.POOR_FIT_P_VALUE`); a stricter one would
        almost never fire for a conservative p-value.

        With no degrees of freedom the check is undefined: ``p_value`` is
        None and no warning is issued.

        Parameters
        ----------
        n : int
            Number of posterior draws.
        seed : int, optional
        warn : bool
            Raise a :class:`blackpy.errors.PoorFitWarning` when the p-value
            is below the threshold.

        Returns
        -------
        PosteriorPredictive
        """
        if self.predictions is None or "data_table" not in self.config:
            raise BlackPyError(
                "This result carries no model predictions (it was produced by BlackPy < 0.3), "
                "so replicated data cannot be simulated. Run the fit again with this version."
            )
        check = posterior_predictive(
            self.config["data_table"],
            self.param_names,
            self.priors,
            self.samples,
            self.predictions,
            n,
            seed,
            threshold=POOR_FIT_P_VALUE,
        )
        if warn and check.poor_fit:
            warnings.warn(poor_fit_message(check.p_value, check.dof), PoorFitWarning, stacklevel=2)
        return check

    # ------------------------------------------------------------------ model comparison

    def savage_dickey(self, null: dict[str, float]) -> BayesFactor:
        r"""Savage-Dickey density ratio for a nested null model.

        For a null model obtained by fixing parameters :math:`\theta` of the
        full model to :math:`\theta_0`, and priors that factorise,

        .. math:: B_{01} = \frac{p(\theta_0 \mid D)}{\pi(\theta_0)}

        (Dickey 1971, Ann. Math. Stat. 42, 204; Verdinelli & Wasserman 1995,
        J. Am. Stat. Assoc. 90, 614). The posterior density is estimated with
        a Gaussian kernel density estimate of the samples (Silverman's
        rule-of-thumb bandwidth for a single parameter). If
        :math:`\theta_0` lies on a prior boundary the samples are reflected
        about the boundary first (otherwise the estimate is biased low by a
        factor of about two), and a warning is issued.

        Parameters
        ----------
        null : dict
            Null values, e.g. ``{"Q": 0.0}``.

        Returns
        -------
        BayesFactor
            With ``log_bf = ln B_10 = -ln B_01``.
        """
        names = list(null)
        for name in names:
            if name not in self.param_names:
                raise KeyError(
                    f"'{name}' is not a sampled parameter of this fit. Sampled: {self.param_names}."
                )
        values = np.array([float(null[n]) for n in names])
        log_prior = sum(float(self.priors[n].logpdf(v)) for n, v in zip(names, values, strict=True))
        if not np.isfinite(log_prior):
            raise ValueError(
                f"The null value {null} lies outside the prior support, where the posterior "
                "density is zero by construction. Choose a null value inside the prior range, "
                "or refit with a prior that contains it."
            )
        notes = []
        boundary = [
            n for n, v in zip(names, values, strict=True) if self.priors[n].is_on_boundary(v)
        ]
        data = np.column_stack([self[n] for n in names])
        if data.shape[0] > 20000:
            data = data[:: int(np.ceil(data.shape[0] / 20000))]

        def log_density(block: np.ndarray) -> tuple[float, float]:
            """Log posterior density at the null value and the kernel-resolution floor."""
            if len(names) == 1:
                # Explicit 1-D Gaussian KDE evaluated at the null value. The bandwidth
                # is Silverman's rule of thumb on the *unreflected* samples: taking it
                # from the reflected (bimodal) sample would over-smooth by the distance
                # between the two mirror images and bias the density high.
                x = block[:, 0]
                spread = min(np.std(x), stats.iqr(x) / 1.349)
                bandwidth = 0.9 * spread * x.size ** (-0.2)
                kernel = np.exp(-0.5 * ((values[0] - x) / bandwidth) ** 2)
                # Reflection about the boundary doubles the kernel sum at the boundary.
                factor = 2.0 if boundary else 1.0
                peak = factor / (x.size * bandwidth * math.sqrt(2.0 * math.pi))
                return float(peak * np.sum(kernel)), peak
            kde = stats.gaussian_kde(block.T)
            det = float(np.linalg.det(kde.covariance))
            floor = 1.0 / (block.shape[0] * math.sqrt((2.0 * math.pi) ** len(names) * det))
            return float(kde(values[:, None])[0]), floor

        if boundary and len(names) == 1:
            message = (
                f"Savage-Dickey: the null value {names[0]} = {values[0]} lies on a prior "
                "boundary. The posterior density there is estimated with a boundary-reflected "
                "KDE; treat the Bayes factor as approximate."
            )
            warnings.warn(message, BlackPyWarning, stacklevel=2)
            notes.append("null value on a prior boundary: boundary-reflected KDE used")
        elif boundary:
            message = (
                f"Savage-Dickey: null values of {boundary} lie on a prior boundary and no "
                "reflection is applied for a multi-parameter null; the posterior density is "
                "biased low, so ln B_10 is biased high."
            )
            warnings.warn(message, BlackPyWarning, stacklevel=2)
            notes.append("multi-parameter null on a prior boundary: KDE biased, no reflection")

        density, floor = log_density(data)
        is_lower_limit = not density > floor
        if is_lower_limit:
            notes.append(
                "posterior density at the null value is below the resolution of the samples; "
                "ln B_10 is a lower limit"
            )
            density = floor
        log_bf = log_prior - math.log(density)
        # Statistical uncertainty from five disjoint sub-samples.
        err = None
        if not is_lower_limit and data.shape[0] >= 500:
            parts = [log_density(data[k::5])[0] for k in range(5)]
            if all(p > 0 for p in parts):
                err = float(np.std(np.log(parts), ddof=1) / math.sqrt(5.0))
        return BayesFactor(
            log_bf=float(log_bf),
            log_bf_err=err,
            method="Savage-Dickey",
            null={n: float(v) for n, v in zip(names, values, strict=True)},
            is_lower_limit=is_lower_limit,
            notes=notes,
        )

    # ------------------------------------------------------------------ persistence

    def save(self, path) -> Path:
        """Save the result as a single ``.npz`` reproducibility bundle.

        The bundle holds the samples and a JSON record with the configuration,
        priors, package versions, sampler settings, seed and dataset versions,
        and the bundle format (``"bundle_format"``, currently
        :data:`BUNDLE_FORMAT`). No pickle is used, so the file is safe to share.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination; the suffix ``.npz`` is added if missing.

        Returns
        -------
        pathlib.Path
            The path that was written.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        record = {
            "bundle_format": BUNDLE_FORMAT,
            "param_names": self.param_names,
            "param_labels": self.param_labels,
            "priors": {n: p.to_dict() for n, p in self.priors.items()},
            "sampler": self.sampler,
            "log_evidence": self.log_evidence,
            "log_evidence_err": self.log_evidence_err,
            "config": self.config,
            "metadata": self.metadata,
            "diagnostics": self.diagnostics,
        }
        arrays = {"samples": self.samples, "log_prob": self.log_prob}
        if self.predictions is not None:
            arrays["predictions"] = self.predictions
        np.savez_compressed(
            path, record=np.array(json.dumps(record, default=_json_default)), **arrays
        )
        return path

    @classmethod
    def load(cls, path) -> FitResult:
        """Load a result saved with :meth:`save`.

        Bundles written before 1.0 have no ``"bundle_format"`` field and are
        read as format 0; their inclination parameter (``inclination``,
        ``inclination_<source>``) is renamed to ``inclination_deg``
        (``inclination_deg_<source>``). A bundle written by a newer BlackPy
        with an unknown format is refused.

        Parameters
        ----------
        path : str or pathlib.Path
            A ``.npz`` file written by :meth:`save`.

        Returns
        -------
        FitResult
        """
        with np.load(Path(path), allow_pickle=False) as bundle:
            record = json.loads(str(bundle["record"]))
            samples = np.array(bundle["samples"])
            log_prob = np.array(bundle["log_prob"])
            predictions = np.array(bundle["predictions"]) if "predictions" in bundle else None
        fmt = record.get("bundle_format", 0)
        if not isinstance(fmt, int) or fmt > BUNDLE_FORMAT or fmt < 0:
            raise BlackPyError(
                f"'{path}' is a result bundle of format {fmt}, but this BlackPy "
                f"({__version__}) reads formats 0 to {BUNDLE_FORMAT} only. Upgrade BlackPy "
                f"(pip install --upgrade blackpy-gr) to the version that wrote it: "
                f"{record.get('metadata', {}).get('versions', {}).get('blackpy', 'unknown')}."
            )
        if fmt == 0:
            record = _upgrade_format_0(record)
        return cls(
            samples=samples,
            log_prob=log_prob,
            param_names=list(record["param_names"]),
            param_labels=list(record["param_labels"]),
            priors={n: prior_from_dict(spec) for n, spec in record["priors"].items()},
            sampler=record["sampler"],
            log_evidence=record["log_evidence"],
            log_evidence_err=record["log_evidence_err"],
            config=record["config"],
            metadata=record["metadata"],
            diagnostics=record["diagnostics"],
            predictions=predictions,
        )
