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
