"""Fit results: posterior samples plus everything needed to reproduce them."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy import stats

from blackpy.errors import BlackPyWarning
from blackpy.infer.priors import Prior, prior_from_dict

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
    """

    log_bf: float
    log_bf_err: float | None
    method: str
    null: dict
    is_lower_limit: bool = False
    notes: list[str] = field(default_factory=list)

    @property
    def interpretation(self) -> str:
        """Verbal scale of Kass & Raftery (1995), J. Am. Stat. Assoc. 90, 773,
        in terms of ``2 ln B``: 0-2 not worth more than a bare mention, 2-6
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
        Convergence and prior-dominance diagnostics.
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

    # ------------------------------------------------------------------ summaries

    def __getitem__(self, name: str) -> np.ndarray:
        """Samples of one parameter."""
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
        spin = self.diagnostics.get("spin_parameter")
        for name, corr in self.diagnostics.get("spin_correlations", {}).items():
            flag = "   [strongly degenerate]" if self.diagnostics["spin_degenerate"][name] else ""
            lines.append(
                f"  corr({spin}, {name}) = {corr['pearson']:+.3f} "
                f"(rank {corr['spearman']:+.3f}){flag}"
            )
        return "\n".join(lines)

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
            raise ValueError(f"The null value {null} lies outside the prior support.")
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
        priors, package versions, sampler settings, seed and dataset versions.
        No pickle is used, so the file is safe to share.
        """
        path = Path(path)
        if path.suffix != ".npz":
            path = path.with_suffix(".npz")
        record = {
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
        np.savez_compressed(
            path,
            samples=self.samples,
            log_prob=self.log_prob,
            record=np.array(json.dumps(record, default=_json_default)),
        )
        return path

    @classmethod
    def load(cls, path) -> FitResult:
        """Load a result saved with :meth:`save`."""
        with np.load(Path(path), allow_pickle=False) as bundle:
            record = json.loads(str(bundle["record"]))
            samples = np.array(bundle["samples"])
            log_prob = np.array(bundle["log_prob"])
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
        )
