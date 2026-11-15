"""Posterior plots and tables."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

from blackpy.infer.result import FitResult


def corner(
    result: FitResult,
    *,
    truths: Mapping[str, float] | None = None,
    parameters: Sequence[str] | None = None,
    **kwargs,
):
    """Corner plot of the posterior samples (wrapper around ``corner.corner``).

    Parameters
    ----------
    result : FitResult
    truths : mapping, optional
        Reference values by parameter name, drawn as lines.
    parameters : sequence of str, optional
        Subset (and order) of parameters to show. Default: all.
    **kwargs
        Passed on to ``corner.corner``.

    Returns
    -------
    matplotlib.figure.Figure
    """
    import corner as corner_module

    names = list(parameters) if parameters is not None else list(result.param_names)
    idx = [result.param_names.index(n) for n in names]
    labels = [f"${result.param_labels[i]}$" for i in idx]
    truth_list = None if truths is None else [truths.get(n) for n in names]
    options = {
        "labels": labels,
        "truths": truth_list,
        "quantiles": [0.16, 0.5, 0.84],
        "show_titles": True,
        "title_fmt": ".3g",
    }
    options.update(kwargs)
    return corner_module.corner(result.samples[:, idx], **options)


def _format(median: float, plus: float, minus: float) -> str:
    """Format ``median^{+plus}_{-minus}`` with two significant digits in the errors."""
    smallest = min(abs(plus), abs(minus))
    if smallest <= 0 or not math.isfinite(smallest):
        return f"{median:.4g}"
    decimals = max(0, 1 - int(math.floor(math.log10(smallest))))
    return f"{median:.{decimals}f}^{{+{plus:.{decimals}f}}}_{{-{minus:.{decimals}f}}}"


def latex_table(
    results: FitResult | Mapping[str, FitResult], *, level: float = 0.68, caption: str | None = None
) -> str:
    r"""Posterior summary as a LaTeX table.

    Each cell is the posterior median with the central credible interval of
    probability ``level``. Parameters whose posterior is indistinguishable
    from the prior are marked with a dagger: they are **not** constraints.

    Parameters
    ----------
    results : FitResult or mapping of str -> FitResult
        One result, or several (columns) keyed by a column title.
    level : float
        Probability content of the credible interval.
    caption : str, optional

    Returns
    -------
    str
        A complete ``table`` environment.
    """
    if isinstance(results, FitResult):
        results = {results.sampler: results}
    names, labels = [], {}
    for res in results.values():
        for name, label in zip(res.param_names, res.param_labels, strict=True):
            if name not in labels:
                names.append(name)
                labels[name] = label
    summaries = {title: res.summary(level) for title, res in results.items()}
    any_dagger = False
    lines = [
        r"\begin{table}",
        r"\centering",
        r"\begin{tabular}{l" + "c" * len(results) + "}",
        r"\hline",
        "Parameter & " + " & ".join(results) + r" \\",
        r"\hline",
    ]
    for name in names:
        cells = []
        for title in results:
            s = summaries[title].get(name)
            if s is None:
                cells.append("--")
                continue
            cell = "$" + _format(s["median"], s["err_plus"], s["err_minus"]) + "$"
            if s["prior_dominated"]:
                cell += r"$^\dagger$"
                any_dagger = True
            cells.append(cell)
        lines.append(f"${labels[name]}$ & " + " & ".join(cells) + r" \\")
    if any(res.log_evidence is not None for res in results.values()):
        cells = [
            "--"
            if res.log_evidence is None
            else f"${res.log_evidence:.2f} \\pm {res.log_evidence_err:.2f}$"
            for res in results.values()
        ]
        lines += [r"\hline", r"$\ln Z$ & " + " & ".join(cells) + r" \\"]
    lines += [r"\hline", r"\end{tabular}"]
    text = caption or f"Posterior medians and {100 * level:.0f}\\% credible intervals."
    if any_dagger:
        text += r" $^\dagger$Posterior indistinguishable from the prior: not a constraint."
    lines += [rf"\caption{{{text}}}", r"\end{table}"]
    return "\n".join(lines)


def residuals(result: FitResult, *, n: int = 2000, seed: int | None = 0, ax=None):
    r"""Residual and posterior predictive plot of a fit.

    For every fitted datum, and for every informative Gaussian prior (which
    acts as a datum), the plot shows in units of the datum's uncertainty:

    * the pull at the best-fit point, ``(model - data) / sigma`` (black dot);
    * the central 68% and 95% ranges of posterior predictive replications of
      the datum, relative to the observed value (bars), i.e. where a new
      measurement would fall if the model were true;
    * the bands :math:`\pm 1` and :math:`\pm 2`.

    A datum whose observed value (the zero line) lies outside its 95% bar is
    one the model cannot reproduce. The title gives :math:`\chi^2`, the
    degrees of freedom and the posterior predictive p-value, or states that
    the goodness of fit is undefined.

    Parameters
    ----------
    result : FitResult
        A result produced by BlackPy 0.3 or later.
    n : int
        Number of posterior predictive draws.
    seed : int, optional
    ax : matplotlib axes, optional

    Returns
    -------
    matplotlib.axes.Axes
    """
    import matplotlib.pyplot as plt
    import numpy as np

    quality = result.goodness_of_fit()
    check = result.posterior_predictive(n, seed, warn=False)
    count = len(quality.labels)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4.5, 1.1 * count + 2.0), 4.0))
    x = np.arange(count)
    sigma = np.asarray(quality.sigma)
    scaled = (check.replicated - check.observed) / sigma
    lo95, lo68, hi68, hi95 = np.percentile(scaled, [2.275, 15.865, 84.135, 97.725], axis=0)
    ax.axhspan(-2, 2, color="0.92", lw=0)
    ax.axhspan(-1, 1, color="0.84", lw=0)
    ax.axhline(0.0, color="k", lw=0.8)
    ax.vlines(x, lo95, hi95, color="C0", lw=2, label="replicated data, 95%")
    ax.vlines(x, lo68, hi68, color="C0", lw=6, alpha=0.6, label="replicated data, 68%")
    ax.plot(x, quality.pulls, "ko", label="best-fit pull (model - data)")
    ax.set_xticks(x)
    ax.set_xticklabels(quality.labels, rotation=30, ha="right")
    ax.set_xlim(-0.6, count - 0.4)
    ax.set_ylabel(r"(value $-$ data) / $\sigma$")
    if check.defined:
        title = (
            rf"$\chi^2 = {quality.chi2:.3g}$ for {quality.dof} dof; "
            f"posterior predictive p = {check.p_value:.3g}"
        )
    else:
        title = "no degrees of freedom: goodness of fit is undefined"
    ax.set_title(title, fontsize="medium")
    ax.legend(fontsize="small")
    return ax
