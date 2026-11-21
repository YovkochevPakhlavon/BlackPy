r"""Fit quality: pulls, :math:`\chi^2`, degrees of freedom, posterior predictive checks.

Bayes factors rank models; they do not say whether the best model describes
the data at all. This module answers that question for every data type,
because it only needs the model prediction for each fitted datum.

Discrepancy
-----------
For data :math:`y_k` with uncertainties :math:`\sigma_k` and model values
:math:`m_k(\theta)` the *pull* is :math:`(m_k - y_k)/\sigma_k`, where
:math:`\sigma_k` is the split-normal uncertainty used by the likelihood
(:math:`\sigma_{k,+}` if the model lies above the datum, :math:`\sigma_{k,-}`
otherwise). The discrepancy is the :math:`\chi^2` discrepancy of Gelman, Meng
& Stern (1996, Statistica Sinica 6, 733), Eq. (8):

.. math:: X^2(y; \theta) = \sum_k \frac{(y_k - m_k(\theta))^2}{\sigma_k^2} .

An informative Gaussian prior (for example an independent mass measurement)
is an additional measurement of the parameter it constrains, and is treated
as one more datum with "model value" equal to that parameter.

Degrees of freedom
------------------
.. math::

    N_{\rm dof} = N_{\rm data} + N_{\rm Gaussian\ priors} - N_{\rm free\ parameters}.

For a model that is linear in its parameters and Gaussian data the minimum
of :math:`X^2` then follows a :math:`\chi^2` distribution with
:math:`N_{\rm dof}` degrees of freedom. The count is exact only in that
case: a parameter that the data cannot constrain, or that sits on a prior
boundary, does not use up a full degree of freedom.

:math:`N_{\rm dof}` can be zero or negative. Every QPO source brings its own
mass and orbital radius, so two frequencies and a mass measurement fitted
with a mass, a radius and one metric parameter leave nothing over. The
goodness of fit is then **undefined** and no p-value is reported.

Posterior predictive check
--------------------------
Following Gelman, Meng & Stern (1996), section 2.3: for posterior draws
:math:`\theta_j`, replicated data :math:`y^{\rm rep}_j` are simulated from
the model at :math:`\theta_j` with the stated uncertainties, and the
posterior predictive p-value of the realized discrepancy is the fraction of
draws with :math:`X^2(y^{\rm rep}_j; \theta_j) \ge X^2(y; \theta_j)` (their
Eq. (7)). Because :math:`X^2(y^{\rm rep}; \theta)` is exactly
:math:`\chi^2_N` distributed given :math:`\theta`, the same number is also
obtained without simulation noise by averaging classical p-values over the
posterior (their Eq. (9)).

**These p-values are not uniform under the true model.** The data are used
twice, to obtain the posterior and to be compared with it, so the p-value is
"centered at .5 but less variable than a uniform random variable, and thus
will have conservative operating characteristics" (Rubin 1996, comment on
Gelman et al., Statistica Sinica 6, 787). Meng (1994, Ann. Stat. 22, 1142,
Sec. 5, Lemma 1 and Eq. (5.6)) shows that the frequency with which a
posterior predictive p-value falls below :math:`\alpha` is "often close to
but less than" :math:`\alpha` and never exceeds :math:`2\alpha` (when the
true parameters are distributed as the prior). A value near 0.5 is therefore
*not* evidence of a good fit when
there are few degrees of freedom; a small value is strong evidence of a bad
one. The warning threshold :data:`POOR_FIT_P_VALUE` is 0.05: with a
conservative p-value a stricter threshold would almost never fire, and the
false-alarm rate under the true model stays below the nominal 5%.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

from blackpy.infer.priors import Gaussian

#: Posterior predictive p-value of the chi-square discrepancy below which a
#: :class:`blackpy.errors.PoorFitWarning` is raised.
POOR_FIT_P_VALUE = 0.05

UNDEFINED = "no degrees of freedom: goodness of fit is undefined"

#: Kinds of data-table rows that are bounds, not Gaussian data, and have no pull.
_NO_PULL = ("shadow_interval", "circularity")


def gaussian_rows(data_table) -> list[int]:
    """Indices of the rows of a data table that have a Gaussian (split-normal) form."""
    return [k for k, row in enumerate(data_table) if row["kind"] not in _NO_PULL]


def prior_rows(param_names, priors) -> list[tuple[int, str, float, float]]:
    """Informative Gaussian priors as extra data: ``(parameter index, name, mean, sigma)``."""
    out = []
    for i, name in enumerate(param_names):
        prior = priors[name] if isinstance(priors, dict) else priors[i]
        if isinstance(prior, Gaussian):
            out.append((i, name, prior.mean, prior.sigma))
    return out


def split_sigma(model, value, err_minus, err_plus):
    """Uncertainty used by the split-normal likelihood for a model value."""
    return np.where(np.asarray(model) >= value, err_plus, err_minus)


def pulls(data_table, param_names, priors, theta, predictions):
    """Pulls ``(model - data) / sigma`` of the Gaussian data rows and the Gaussian priors.

    Parameters
    ----------
    data_table : list of dict
        Rows of :attr:`blackpy.infer.Problem.data_table`.
    param_names : sequence of str
    priors : mapping or sequence of Prior
    theta : array, shape (..., ndim)
        Parameter vector(s).
    predictions : array, shape (..., n_rows)
        Model values for every row of ``data_table`` at ``theta``.

    Returns
    -------
    labels : list of str
    observed, model, sigma, pull : numpy.ndarray, shape (..., n_terms)
    """
    theta = np.asarray(theta, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    labels, observed, model, sigma = [], [], [], []
    for k in gaussian_rows(data_table):
        row = data_table[k]
        labels.append(row["label"])
        observed.append(np.full(predictions.shape[:-1], row["value"]))
        model.append(predictions[..., k])
        sigma.append(
            split_sigma(predictions[..., k], row["value"], row["err_minus"], row["err_plus"])
        )
    for i, name, mean, width in prior_rows(param_names, priors):
        labels.append(f"prior[{name}]")
        observed.append(np.full(theta.shape[:-1], mean))
        model.append(theta[..., i])
        sigma.append(np.full(theta.shape[:-1], width))
    if not labels:
        empty = np.zeros((*theta.shape[:-1], 0))
        return labels, empty, empty, empty, empty
    observed, model, sigma = (np.stack(v, axis=-1) for v in (observed, model, sigma))
    return labels, observed, model, sigma, (model - observed) / sigma


@dataclass(frozen=True)
class GoodnessOfFit:
    r"""Pulls and :math:`\chi^2` of a fit at one parameter point.

    Attributes
    ----------
    point : dict of str -> float
        The parameter values (the best-fit point unless stated otherwise).
    labels : list of str
        One label per term: the fitted data, then ``prior[<name>]`` for each
        informative Gaussian prior.
    observed, model, sigma, pulls : numpy.ndarray
        Datum, model value, split-normal uncertainty used, and the pull
        ``(model - observed) / sigma`` for every term.
    chi2 : float
        Sum of the squared pulls.
    n_data, n_gaussian_priors, n_parameters : int
        The three counts that enter the degrees of freedom.
    dof : int
        ``n_data + n_gaussian_priors - n_parameters`` (may be zero or negative).
    p_value : float or None
        Probability that a :math:`\chi^2_{\rm dof}` variable exceeds
        ``chi2``; **None when ``dof <= 0``** (goodness of fit undefined).
        It assumes a linear model and Gaussian data and is meaningful only
        at the best-fit point.
    is_best_fit : bool
        Whether ``point`` minimises ``chi2`` inside the prior support.
    notes : tuple of str
        Caveats (for example interval likelihoods that have no pull).
    """

    point: dict
    labels: list
    observed: np.ndarray
    model: np.ndarray
    sigma: np.ndarray
    pulls: np.ndarray
    chi2: float
    n_data: int
    n_gaussian_priors: int
    n_parameters: int
    is_best_fit: bool = True
    notes: tuple = ()

    @property
    def dof(self) -> int:
        return self.n_data + self.n_gaussian_priors - self.n_parameters

    @property
    def defined(self) -> bool:
        """False when there are no degrees of freedom."""
        return self.dof > 0

    @property
    def p_value(self) -> float | None:
        if not self.defined or not self.is_best_fit:
            return None
        return float(stats.chi2.sf(self.chi2, self.dof))

    @property
    def reduced_chi2(self) -> float | None:
        """``chi2 / dof``; None when there are no degrees of freedom."""
        return self.chi2 / self.dof if self.defined else None

    def summary_line(self) -> str:
        """One line for ``print(result)``."""
        counts = (
            f"{self.n_data} data + {self.n_gaussian_priors} Gaussian prior(s) - "
            f"{self.n_parameters} parameters"
        )
        if not self.defined:
            return f"chi2 = {self.chi2:.3g}; {UNDEFINED} ({counts} = {self.dof})"
        tail = f", p = {self.p_value:.3g}" if self.p_value is not None else ""
        return f"chi2 = {self.chi2:.3g} for {self.dof} dof ({counts}){tail}"

    def __str__(self) -> str:
        where = "best-fit point" if self.is_best_fit else "given point"
        lines = [f"Goodness of fit at the {where}: " + self.summary_line()]
        lines.append(f"  {'term':<26} {'data':>12} {'model':>12} {'sigma':>10} {'pull':>8}")
        for k, label in enumerate(self.labels):
            lines.append(
                f"  {label:<26} {self.observed[k]:12.6g} {self.model[k]:12.6g} "
                f"{self.sigma[k]:10.4g} {self.pulls[k]:+8.3f}"
            )
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)

    def to_dict(self) -> dict:
        """JSON-serialisable form, stored in ``FitResult.diagnostics``."""
        return {
            "point": {k: float(v) for k, v in self.point.items()},
            "labels": list(self.labels),
            "observed": np.asarray(self.observed).tolist(),
            "model": np.asarray(self.model).tolist(),
            "sigma": np.asarray(self.sigma).tolist(),
            "pulls": np.asarray(self.pulls).tolist(),
            "chi2": float(self.chi2),
            "n_data": int(self.n_data),
            "n_gaussian_priors": int(self.n_gaussian_priors),
            "n_parameters": int(self.n_parameters),
            "dof": int(self.dof),
            "p_value": self.p_value,
            "is_best_fit": bool(self.is_best_fit),
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, record: dict) -> GoodnessOfFit:
        return cls(
            point=dict(record["point"]),
            labels=list(record["labels"]),
            observed=np.asarray(record["observed"], dtype=float),
            model=np.asarray(record["model"], dtype=float),
            sigma=np.asarray(record["sigma"], dtype=float),
            pulls=np.asarray(record["pulls"], dtype=float),
            chi2=float(record["chi2"]),
            n_data=int(record["n_data"]),
            n_gaussian_priors=int(record["n_gaussian_priors"]),
            n_parameters=int(record["n_parameters"]),
            is_best_fit=bool(record["is_best_fit"]),
            notes=tuple(record.get("notes", ())),
        )


def _notes(data_table) -> tuple:
    notes = []
    intervals = [row["label"] for row in data_table if row["kind"] == "shadow_interval"]
    if intervals:
        notes.append(
            f"{intervals} use the interval (top-hat) likelihood, which has no pull: they are "
            "not counted as data in chi2 or in the degrees of freedom. Use shadow_likelihood="
            "'gaussian' for a goodness-of-fit statement that includes them."
        )
    bounds = [row["label"] for row in data_table if row["kind"] == "circularity"]
    if bounds:
        notes.append(
            f"{bounds} are upper bounds (allowed / excluded), not measurements: they are not "
            "counted as data in chi2 or in the degrees of freedom."
        )
    return tuple(notes)


def goodness_at(problem, theta, *, is_best_fit: bool = False) -> GoodnessOfFit:
    """Pulls and chi-square of ``problem`` at the parameter vector ``theta``."""
    theta = np.asarray(theta, dtype=float)
    predictions = np.asarray(problem.predict(theta))
    priors = problem.priors
    labels, observed, model, sigma, pull = pulls(
        problem.data_table, problem.param_names, priors, theta, predictions
    )
    n_prior = len(prior_rows(problem.param_names, priors))
    return GoodnessOfFit(
        point=dict(zip(problem.param_names, theta.tolist(), strict=True)),
        labels=labels,
        observed=observed,
        model=model,
        sigma=sigma,
        pulls=pull,
        chi2=float(np.sum(pull**2)),
        n_data=len(labels) - n_prior,
        n_gaussian_priors=n_prior,
        n_parameters=problem.ndim,
        is_best_fit=is_best_fit,
        notes=_notes(problem.data_table),
    )


def best_fit(problem, start) -> np.ndarray:
    r"""Minimise the :math:`\chi^2` discrepancy inside the prior support.

    The search starts from ``start`` (the best posterior sample) and uses the
    Nelder-Mead simplex method, which needs no derivatives and is not
    disturbed by the hard edges of the prior support or of the black hole
    domain. Points outside the support, or with zero likelihood, are rejected.
    """
    from scipy.optimize import minimize

    start = np.asarray(start, dtype=float)

    def objective(theta) -> float:
        if not np.isfinite(problem.log_prior(theta)[0]):
            return 1e300
        if not np.isfinite(float(problem.log_likelihood(theta))):
            return 1e300
        value = goodness_at(problem, theta).chi2
        return value if np.isfinite(value) else 1e300

    result = minimize(
        objective,
        start,
        method="Nelder-Mead",
        options={"maxiter": 600 * problem.ndim, "xatol": 1e-10, "fatol": 1e-12, "adaptive": True},
    )
    return result.x if result.fun <= objective(start) else start


@dataclass(frozen=True)
class PosteriorPredictive:
    r"""Posterior predictive check of a fit (Gelman, Meng & Stern 1996).

    Attributes
    ----------
    labels : list of str
        The fitted data, then ``prior[<name>]`` for informative Gaussian
        priors (their replicated "measurement" is drawn around the
        parameter value, with the width of the prior).
    observed : numpy.ndarray, shape (n_terms,)
    replicated : numpy.ndarray, shape (n, n_terms)
        Replicated data: model prediction of posterior draw ``j`` plus noise
        drawn from the stated (split-normal) uncertainties.
    datum_p_values : numpy.ndarray, shape (n_terms,)
        Posterior predictive p-value of each datum,
        :math:`P(y^{\rm rep}_k \ge y_k \mid y)`. Values near 0 or 1 mark a
        datum that the model cannot reproduce.
    discrepancy_observed, discrepancy_replicated : numpy.ndarray, shape (n,)
        :math:`X^2(y; \theta_j)` and :math:`X^2(y^{\rm rep}_j; \theta_j)`.
    p_value : float or None
        Posterior predictive p-value of the :math:`\chi^2` discrepancy,
        computed without simulation noise as the posterior mean of the
        classical p-value (Gelman et al. 1996, Eq. (9)). **None when there
        are no degrees of freedom.**
    p_value_simulated : float
        The same quantity estimated from the replicated data (the fraction
        of draws with ``discrepancy_replicated >= discrepancy_observed``);
        agrees with ``p_value`` within the simulation error
        ``sqrt(p (1 - p) / n)``.
    dof : int
        Degrees of freedom of the fit (see :class:`GoodnessOfFit`).
    threshold : float
        Value of ``p_value`` below which the fit is called poor.

    Notes
    -----
    The p-value is **not uniformly distributed under the true model**: the
    data are used twice, so it concentrates around 0.5 and is conservative
    (Rubin 1996; Meng 1994, who bounds the frequency of ``p <= alpha`` by
    ``2 alpha``). A value near 0.5 is not evidence of a good fit; a small
    value is strong evidence of a bad one.
    """

    labels: list
    observed: np.ndarray
    replicated: np.ndarray
    datum_p_values: np.ndarray
    discrepancy_observed: np.ndarray
    discrepancy_replicated: np.ndarray
    p_value_simulated: float
    p_value_average: float
    dof: int
    threshold: float = POOR_FIT_P_VALUE

    @property
    def defined(self) -> bool:
        return self.dof > 0

    @property
    def p_value(self) -> float | None:
        return self.p_value_average if self.defined else None

    @property
    def poor_fit(self) -> bool:
        """True if the check is defined and the p-value is below the threshold."""
        return self.defined and self.p_value_average < self.threshold

    def __str__(self) -> str:
        n = self.replicated.shape[0]
        if not self.defined:
            head = f"Posterior predictive check ({n} draws): {UNDEFINED} (dof = {self.dof})"
        else:
            verdict = "POOR FIT" if self.poor_fit else "no evidence of misfit"
            head = (
                f"Posterior predictive check ({n} draws): p = {self.p_value_average:.3g} for the "
                f"chi2 discrepancy (simulated: {self.p_value_simulated:.3g}); {verdict} at "
                f"threshold {self.threshold}. Note: p is conservative, not uniform."
            )
        lines = [head, f"  {'term':<26} {'data':>12} {'replicated (16-84%)':>26} {'p':>7}"]
        low, high = np.percentile(self.replicated, [15.87, 84.13], axis=0)
        for k, label in enumerate(self.labels):
            lines.append(
                f"  {label:<26} {self.observed[k]:12.6g} {low[k]:12.6g} .. {high[k]:<10.6g} "
                f"{self.datum_p_values[k]:7.3f}"
            )
        return "\n".join(lines)


def draw_split_normal(rng: np.random.Generator, centre, err_minus, err_plus):
    """Draw data around ``centre`` from the noise model of the split-normal likelihood.

    The likelihood uses ``err_plus`` when the model lies above the datum, so a
    datum *below* the model value is drawn with ``err_plus`` and a datum above
    it with ``err_minus``; the two sides are chosen with probabilities
    proportional to their widths.
    """
    centre = np.asarray(centre, dtype=float)
    below = rng.uniform(size=centre.shape) < err_plus / (err_minus + err_plus)
    deviate = np.abs(rng.standard_normal(centre.shape))
    return np.where(below, centre - err_plus * deviate, centre + err_minus * deviate)


def posterior_predictive(
    data_table,
    param_names,
    priors,
    samples,
    predictions,
    n: int,
    seed: int | None,
    threshold: float = POOR_FIT_P_VALUE,
) -> PosteriorPredictive:
    """Posterior predictive check from posterior samples and their model predictions.

    Parameters
    ----------
    data_table, param_names, priors
        Description of the problem (see :func:`pulls`).
    samples : array, shape (n_samples, ndim)
        Equally weighted posterior samples.
    predictions : array, shape (n_samples, n_rows)
        Model values of every row of ``data_table`` for each sample.
    n : int
        Number of posterior draws (with replacement if larger than the
        number of samples).
    seed : int or None
    threshold : float
        Poor-fit threshold recorded in the result.
    """
    rng = np.random.default_rng(seed)
    samples = np.asarray(samples, dtype=float)
    predictions = np.asarray(predictions, dtype=float)
    if n < 1:
        raise ValueError("n must be a positive number of posterior draws.")
    pick = rng.choice(samples.shape[0], size=n, replace=n > samples.shape[0])
    theta, pred = samples[pick], predictions[pick]
    labels, observed, model, sigma, pull = pulls(data_table, param_names, priors, theta, pred)
    n_prior = len(prior_rows(param_names, priors))
    n_terms = len(labels)
    dof = n_terms - samples.shape[1]

    # Replicated data: the model value of each draw plus noise with the stated widths.
    widths_minus, widths_plus = [], []
    for k in gaussian_rows(data_table):
        widths_minus.append(data_table[k]["err_minus"])
        widths_plus.append(data_table[k]["err_plus"])
    for _, _, _, width in prior_rows(param_names, priors):
        widths_minus.append(width)
        widths_plus.append(width)
    widths_minus, widths_plus = np.asarray(widths_minus), np.asarray(widths_plus)
    replicated = draw_split_normal(rng, model, widths_minus, widths_plus)
    sigma_rep = np.where(model >= replicated, widths_plus, widths_minus)
    discrepancy_rep = np.sum(((model - replicated) / sigma_rep) ** 2, axis=-1)
    discrepancy_obs = np.sum(pull**2, axis=-1)
    if n_terms:
        average = float(np.mean(stats.chi2.sf(discrepancy_obs, n_terms)))
    else:
        average = math.nan
    del n_prior
    return PosteriorPredictive(
        labels=labels,
        observed=observed[0] if n_terms else np.zeros(0),
        replicated=replicated,
        datum_p_values=np.mean(replicated >= observed, axis=0),
        discrepancy_observed=discrepancy_obs,
        discrepancy_replicated=discrepancy_rep,
        p_value_simulated=float(np.mean(discrepancy_rep >= discrepancy_obs)),
        p_value_average=average,
        dof=int(dof),
        threshold=threshold,
    )
