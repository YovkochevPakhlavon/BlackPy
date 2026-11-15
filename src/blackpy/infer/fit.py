"""The :class:`Fit` driver: emcee and dynesty behind one interface."""

from __future__ import annotations

import datetime
import platform
import time
import warnings
from collections.abc import Mapping

import numpy as np
from scipy import stats

from blackpy._version import __version__
from blackpy.errors import (
    BlackPyError,
    ConvergenceWarning,
    DegeneracyWarning,
    PriorDominatedWarning,
)
from blackpy.infer.likelihood import Problem, build_problem
from blackpy.infer.priors import Prior
from blackpy.infer.result import PRIOR_DOMINATED_KS, BayesFactor, FitResult

SAMPLERS = ("emcee", "dynesty")

#: emcee's recommended criterion: the chain must be longer than 50 autocorrelation times.
AUTOCORR_FACTOR = 50.0

#: Number of autocorrelation times discarded as burn-in.
BURN_IN_FACTOR = 5.0

#: Posterior correlation between the spin and another metric parameter above which the
#: two are reported as strongly degenerate.
DEGENERACY_CORRELATION = 0.9


def _versions() -> dict:
    import dynesty
    import emcee
    import jax
    import scipy
    import sympy

    return {
        "blackpy": __version__,
        "jax": jax.__version__,
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sympy": sympy.__version__,
        "emcee": emcee.__version__,
        "dynesty": dynesty.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


class Fit:
    """Bayesian fit of a metric to QPO and/or shadow data.

    Parameters
    ----------
    model : StaticSpherical or StationaryAxisymmetric
        The spacetime.
    data : dataset, registry id, or sequence of them
        E.g. ``["qpo/gro_j1655_40", "eht/sgra"]``.
    qpo_model : str or QPOModel, optional
        QPO model (required with QPO data), e.g. ``"RP"``.
    priors : mapping of str -> Prior
        See :func:`blackpy.infer.build_problem` for the naming rules.
    sampler : {"emcee", "dynesty"}
        Sampling backend.
    fixed : mapping, optional
        Fixed values of metric parameters that are not sampled.
    shadow_likelihood, shadow_level
        Options of the shadow likelihood, see :func:`blackpy.infer.build_problem`.
    direction, use_low_frequency, require_black_hole, shadow_size
        Options for rotating metrics (orbit direction, nodal-precession QPO
        of triplets, restriction to black holes, shadow size definition), see
        :func:`blackpy.infer.build_problem`.
    **sampler_kwargs
        emcee: ``nwalkers`` (default ``max(32, 4 ndim)``), ``nsteps`` (5000),
        ``max_steps`` (= ``nsteps``; if larger, sampling continues in blocks
        of ``nsteps`` until converged), ``initial_state``, ``moves``,
        ``progress``.
        dynesty: ``nlive`` (500), ``dlogz`` (0.1), ``bound`` ("multi"),
        ``sample`` ("rwalk"), ``maxcall``, ``print_progress``.

    Examples
    --------
    >>> fit = Fit(rn, [data], qpo_model="RP", priors={"Q": Uniform(0, 1)})   # doctest: +SKIP
    >>> result = fit.run(seed=1)                                             # doctest: +SKIP
    >>> print(fit.bayes_factor(null={"Q": 0.0}))                             # doctest: +SKIP
    """

    def __init__(
        self,
        model,
        data,
        qpo_model=None,
        priors: Mapping[str, Prior] | None = None,
        sampler: str = "emcee",
        *,
        fixed: Mapping[str, float] | None = None,
        shadow_likelihood: str = "gaussian",
        shadow_level: str = "2sigma",
        direction: str | int = "prograde",
        use_low_frequency: bool | None = None,
        require_black_hole: bool | None = None,
        shadow_size: str = "mean",
        **sampler_kwargs,
    ):
        if sampler not in SAMPLERS:
            raise ValueError(f"sampler must be one of {SAMPLERS}, got '{sampler}'.")
        self.model = model
        self.data = data
        self.qpo_model = qpo_model
        self.priors = dict(priors or {})
        self.sampler = sampler
        self.fixed = dict(fixed or {})
        self.shadow_likelihood = shadow_likelihood
        self.shadow_level = shadow_level
        self.sampler_kwargs = dict(sampler_kwargs)
        self.options = {
            "shadow_likelihood": shadow_likelihood,
            "shadow_level": shadow_level,
            "direction": direction,
            "use_low_frequency": use_low_frequency,
            "require_black_hole": require_black_hole,
            "shadow_size": shadow_size,
        }
        self.problem: Problem = build_problem(
            model, data, qpo_model, self.priors, fixed=self.fixed, **self.options
        )
        self.result: FitResult | None = None
        self._seed: int | None = None
        self._null_results: dict = {}

    @property
    def param_names(self) -> list[str]:
        """Names of the sampled parameters."""
        return self.problem.param_names

    def log_likelihood(self, **theta) -> float:
        """Log-likelihood at a point given by keyword parameters (all sampled ones)."""
        missing = set(self.param_names) - set(theta)
        if missing:
            raise TypeError(f"Missing parameters {sorted(missing)}. Needed: {self.param_names}.")
        vector = np.array([float(theta[n]) for n in self.param_names])
        return float(self.problem.log_likelihood(vector))

    # ------------------------------------------------------------------ running

    def run(self, seed: int | None = None) -> FitResult:
        """Run the sampler.

        Parameters
        ----------
        seed : int, optional
            Seed of the random number generator. With a seed the run is
            reproducible on the same machine and package versions.

        Returns
        -------
        FitResult
            Also stored as ``self.result``.
        """
        self._seed = seed
        start = time.perf_counter()
        if self.sampler == "emcee":
            samples, log_prob, extra, settings = self._run_emcee(seed)
        else:
            samples, log_prob, extra, settings = self._run_dynesty(seed)
        problem = self.problem
        diagnostics = dict(extra.pop("diagnostics", {}))
        diagnostics.update(self._prior_dominance(samples))
        diagnostics.update(self._spin_degeneracy(samples))
        metadata = {
            "versions": _versions(),
            "sampler_settings": settings,
            "seed": seed,
            "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(
                timespec="seconds"
            ),
            "runtime_seconds": round(time.perf_counter() - start, 3),
        }
        config = dict(problem.description)
        config["sampler"] = self.sampler
        self.result = FitResult(
            samples=samples,
            log_prob=log_prob,
            param_names=list(problem.param_names),
            param_labels=list(problem.param_labels),
            priors=dict(zip(problem.param_names, problem.priors, strict=True)),
            sampler=self.sampler,
            log_evidence=extra.get("log_evidence"),
            log_evidence_err=extra.get("log_evidence_err"),
            config=config,
            metadata=metadata,
            diagnostics=diagnostics,
        )
        return self.result

    def _initial_state(self, rng: np.random.Generator, nwalkers: int) -> np.ndarray:
        """Starting positions: a small ball around the posterior maximum.

        The maximum is located by taking the best of a large batch of prior
        draws and polishing it with a Nelder-Mead search. Starting all
        walkers in one compact ball avoids walkers that stay stuck in minor
        local maxima for the whole run (observed when starting from scattered
        prior draws). For a genuinely multimodal posterior pass
        ``initial_state`` explicitly or use nested sampling.
        """
        from scipy.optimize import minimize

        problem = self.problem
        ndim = problem.ndim
        n_draw = max(4000, 200 * nwalkers)
        candidates = problem.sample_prior(rng, n_draw)
        log_post = problem.log_posterior_batch(candidates)
        if not np.isfinite(log_post).any():
            raise BlackPyError(
                f"None of {n_draw} prior draws has a finite likelihood. The priors probably "
                "cover only unphysical regions (e.g. orbital radii inside the ISCO): change "
                "them, or pass initial_state=<array of shape (nwalkers, ndim)>."
            )
        best = candidates[int(np.argmax(log_post))]

        def log_posterior(theta) -> float:
            log_prior = float(problem.log_prior(theta)[0])
            if not np.isfinite(log_prior):
                return -np.inf
            value = log_prior + float(problem.log_likelihood(theta))
            return value if np.isfinite(value) else -np.inf

        def objective(theta) -> float:
            value = log_posterior(theta)
            return -value if np.isfinite(value) else 1e300

        polished = minimize(
            objective,
            best,
            method="Nelder-Mead",
            options={"maxiter": 400 * ndim, "xatol": 1e-8, "fatol": 1e-8},
        )
        centre = polished.x if polished.fun < objective(best) else best
        widths = np.array([0.5 * (pr.ppf(0.84) - pr.ppf(0.16)) for pr in problem.priors])
        state = np.empty((nwalkers, ndim))
        for i in range(nwalkers):
            for _ in range(1000):
                trial = centre + 1e-4 * widths * rng.standard_normal(ndim)
                if np.isfinite(log_posterior(trial)):
                    break
            else:
                raise BlackPyError(
                    "Could not place the walkers around the posterior maximum; pass "
                    "initial_state=<array of shape (nwalkers, ndim)>."
                )
            state[i] = trial
        return state

    def _run_emcee(self, seed):
        import emcee

        problem, kw = self.problem, dict(self.sampler_kwargs)
        ndim = problem.ndim
        nwalkers = int(kw.pop("nwalkers", max(32, 4 * ndim)))
        nwalkers += nwalkers % 2
        if nwalkers < 2 * ndim:
            raise ValueError(f"emcee needs nwalkers >= 2 * ndim = {2 * ndim}, got {nwalkers}.")
        nsteps = int(kw.pop("nsteps", 5000))
        max_steps = int(kw.pop("max_steps", nsteps))
        progress = bool(kw.pop("progress", False))
        initial = kw.pop("initial_state", None)
        moves = kw.pop("moves", None)
        if kw:
            raise TypeError(f"Unknown emcee options {sorted(kw)}.")
        rng = np.random.default_rng(seed)
        state = (
            self._initial_state(rng, nwalkers) if initial is None else np.asarray(initial, float)
        )
        if state.shape != (nwalkers, ndim):
            raise ValueError(
                f"initial_state must have shape {(nwalkers, ndim)}, got {state.shape}."
            )
        # vectorize=True: emcee passes all walkers of a move at once, and the
        # likelihood evaluates them with a single jit-compiled vmap call.
        sampler = emcee.EnsembleSampler(
            nwalkers, ndim, problem.log_posterior_batch, vectorize=True, moves=moves
        )
        sampler.random_state = np.random.RandomState(
            None if seed is None else int(rng.integers(0, 2**31 - 1))
        ).get_state()

        total, converged, tau = 0, False, np.full(ndim, np.nan)
        while total < max(max_steps, nsteps):
            state = sampler.run_mcmc(state, nsteps, progress=progress)
            total += nsteps
            tau = np.asarray(sampler.get_autocorr_time(tol=0), dtype=float)
            converged = bool(np.all(np.isfinite(tau)) and total > AUTOCORR_FACTOR * np.max(tau))
            if converged:
                break
        if np.all(np.isfinite(tau)):
            # Burn-in of five autocorrelation times (walkers start in a compact ball and
            # must spread over the posterior); thinning by half the shortest one, as in
            # the emcee documentation (Foreman-Mackey et al. 2013, PASP 125, 306).
            burn_in = int(np.ceil(BURN_IN_FACTOR * np.max(tau)))
            thin = max(1, int(np.floor(0.5 * np.min(tau))))
        else:
            burn_in, thin = total // 2, 1
        burn_in = min(burn_in, total // 2)
        samples = sampler.get_chain(discard=burn_in, thin=thin, flat=True)
        log_prob = sampler.get_log_prob(discard=burn_in, thin=thin, flat=True)
        if not converged:
            warnings.warn(
                f"emcee chain not converged: {total} steps is less than {AUTOCORR_FACTOR:.0f} "
                f"times the integrated autocorrelation time (max tau = {np.nanmax(tau):.1f}). "
                "Increase nsteps or set max_steps; do not quote constraints from this chain.",
                ConvergenceWarning,
                stacklevel=3,
            )
        diagnostics = {
            "converged": converged,
            "autocorr_time": dict(zip(problem.param_names, tau.tolist(), strict=True)),
            "n_steps": total,
            "steps_over_tau": float(total / np.nanmax(tau)) if np.any(np.isfinite(tau)) else None,
            "burn_in": burn_in,
            "thin": thin,
            "acceptance_fraction": float(np.mean(sampler.acceptance_fraction)),
            "n_independent_estimate": float(nwalkers * (total - burn_in) / np.nanmax(tau))
            if np.any(np.isfinite(tau))
            else None,
        }
        settings = {
            "nwalkers": nwalkers,
            "nsteps": nsteps,
            "max_steps": max_steps,
            "vectorize": True,
            "moves": "default (StretchMove)" if moves is None else str(moves),
            "initialisation": "ball around posterior maximum" if initial is None else "user",
        }
        return samples, log_prob, {"diagnostics": diagnostics}, settings

    def _run_dynesty(self, seed):
        import dynesty

        problem, kw = self.problem, dict(self.sampler_kwargs)
        nlive = int(kw.pop("nlive", 500))
        dlogz = float(kw.pop("dlogz", 0.1))
        bound = kw.pop("bound", "multi")
        # Random-walk proposals by default: QPO posteriors are narrow, curved
        # degeneracies between mass and radius, for which dynesty's uniform
        # sampling in ellipsoids needed ~8x more likelihood calls in our tests.
        sample = kw.pop("sample", "rwalk")
        maxcall = kw.pop("maxcall", None)
        print_progress = bool(kw.pop("print_progress", False))
        if kw:
            raise TypeError(f"Unknown dynesty options {sorted(kw)}.")
        rng = np.random.default_rng(seed)
        log_likelihood = problem.log_likelihood

        def loglike(theta):
            value = float(log_likelihood(theta))
            # dynesty needs finite numbers; -1e300 marks excluded points.
            return value if np.isfinite(value) else -1e300

        sampler = dynesty.NestedSampler(
            loglike,
            problem.prior_transform,
            problem.ndim,
            nlive=nlive,
            bound=bound,
            sample=sample,
            rstate=rng,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            sampler.run_nested(dlogz=dlogz, maxcall=maxcall, print_progress=print_progress)
        res = sampler.results
        weights = np.exp(res.logwt - res.logz[-1])
        weights = weights / weights.sum()
        idx = dynesty.utils.resample_equal(np.arange(res.samples.shape[0]), weights, rstate=rng)
        samples = np.asarray(res.samples)[idx]
        log_like = np.asarray(res.logl)[idx]
        diagnostics = {
            "converged": True,
            "n_iterations": int(res.niter),
            "n_likelihood_calls": int(np.sum(res.ncall)),
            "sampling_efficiency_percent": float(np.ravel(res.eff)[0]),
            "effective_sample_size": float(1.0 / np.sum(weights**2)),
            "stopping_criterion": f"dlogz < {dlogz}",
        }
        extra = {
            "log_evidence": float(res.logz[-1]),
            "log_evidence_err": float(res.logzerr[-1]),
            "diagnostics": diagnostics,
        }
        settings = {
            "nlive": nlive,
            "dlogz": dlogz,
            "bound": bound,
            "sample": sample,
            "maxcall": maxcall,
        }
        return samples, log_like, extra, settings

    def _prior_dominance(self, samples: np.ndarray) -> dict:
        """Compare each marginal posterior with its prior (Kolmogorov-Smirnov distance)."""
        problem = self.problem
        distance, dominated = {}, {}
        for i, (name, prior) in enumerate(zip(problem.param_names, problem.priors, strict=True)):
            d = float(stats.kstest(samples[:, i], prior.cdf).statistic)
            distance[name] = d
            dominated[name] = d < PRIOR_DOMINATED_KS
        flagged = [
            n
            for n, k in zip(problem.param_names, problem.kinds, strict=True)
            if dominated[n] and k == "metric"
        ]
        if flagged:
            warnings.warn(
                f"The posterior of {flagged} is indistinguishable from its prior (KS distance "
                f"below {PRIOR_DOMINATED_KS}): the data do not constrain it. Do not report the "
                "credible interval as a constraint; it only reflects the prior.",
                PriorDominatedWarning,
                stacklevel=3,
            )
        return {"prior_ks_distance": distance, "prior_dominated": dominated}

    def _spin_degeneracy(self, samples: np.ndarray) -> dict:
        """Posterior correlation between the spin and every other sampled metric parameter.

        Both the linear (Pearson) and the rank (Spearman) correlation are
        recorded; the rank correlation also sees curved degeneracies. A
        parameter is flagged when either exceeds ``DEGENERACY_CORRELATION``
        in absolute value.
        """
        problem = self.problem
        spin = getattr(self.model, "spin_parameter", None)
        if spin is None or spin not in problem.param_names:
            return {}
        i_spin = problem.param_names.index(spin)
        correlations, degenerate = {}, {}
        for i, (name, kind) in enumerate(zip(problem.param_names, problem.kinds, strict=True)):
            if kind != "metric" or i == i_spin:
                continue
            pearson = float(np.corrcoef(samples[:, i_spin], samples[:, i])[0, 1])
            spearman = float(stats.spearmanr(samples[:, i_spin], samples[:, i]).statistic)
            correlations[name] = {"pearson": pearson, "spearman": spearman}
            degenerate[name] = max(abs(pearson), abs(spearman)) >= DEGENERACY_CORRELATION
        flagged = [n for n, bad in degenerate.items() if bad]
        if flagged:
            detail = ", ".join(
                f"{n}: r = {correlations[n]['pearson']:+.3f} "
                f"(rank {correlations[n]['spearman']:+.3f})"
                for n in flagged
            )
            warnings.warn(
                f"The spin '{spin}' is strongly degenerate with {flagged} (posterior "
                f"correlation {detail}). The data constrain a combination of these parameters, "
                "not each of them: report the joint posterior (corner plot), not the marginal "
                "intervals, and consider adding data that break the degeneracy.",
                DegeneracyWarning,
                stacklevel=3,
            )
        return {
            "spin_parameter": spin,
            "spin_correlations": correlations,
            "spin_degenerate": degenerate,
        }

    # ------------------------------------------------------------------ model comparison

    def bayes_factor(self, null: Mapping[str, float], method: str | None = None) -> BayesFactor:
        """Bayes factor of the fitted model against a nested null model.

        Parameters
        ----------
        null : mapping
            Null hypothesis: metric parameters fixed to given values, e.g.
            ``{"Q": 0.0}``.
        method : {"evidence", "savage-dickey"}, optional
            Default: ``"evidence"`` for dynesty (the null model is fitted
            automatically with the same settings and the evidences are
            compared) and ``"savage-dickey"`` for emcee (density ratio at the
            null value, valid for nested models).

        Returns
        -------
        BayesFactor
            ``log_bf = ln B_10 > 0`` favours the full model.
        """
        if self.result is None:
            raise BlackPyError("Call run() before bayes_factor().")
        null = {k: float(v) for k, v in null.items()}
        kinds = self.problem.description["parameter_kinds"]
        not_metric = [k for k in null if kinds.get(k) != "metric"]
        if not_metric:
            sampled = [n for n, k in kinds.items() if k == "metric"]
            raise ValueError(
                f"The null hypothesis must fix sampled metric parameters; {not_metric} are not. "
                f"Sampled metric parameters: {sampled}."
            )
        if method is None:
            method = "evidence" if self.sampler == "dynesty" else "savage-dickey"
        if method == "savage-dickey":
            return self.result.savage_dickey(null)
        if method != "evidence":
            raise ValueError("method must be 'evidence' or 'savage-dickey'.")
        if self.sampler != "dynesty":
            raise BlackPyError(
                "The evidence ratio needs nested sampling: use sampler='dynesty', or "
                "method='savage-dickey' for an emcee fit."
            )
        key = tuple(sorted(null.items()))
        if key not in self._null_results:
            priors = {k: v for k, v in self.priors.items() if k not in null}
            null_fit = Fit(
                self.model,
                self.data,
                self.qpo_model,
                priors,
                sampler="dynesty",
                fixed={**self.fixed, **null},
                **self.options,
                **self.sampler_kwargs,
            )
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")  # the same warnings were shown for the full fit
                null_seed = None if self._seed is None else self._seed + 1
                self._null_results[key] = null_fit.run(seed=null_seed)
        null_result = self._null_results[key]
        log_bf = self.result.log_evidence - null_result.log_evidence
        err = float(np.hypot(self.result.log_evidence_err, null_result.log_evidence_err))
        return BayesFactor(
            log_bf=float(log_bf),
            log_bf_err=err,
            method="evidence ratio",
            null=null,
            notes=[
                f"ln Z(full) = {self.result.log_evidence:.3f}, "
                f"ln Z(null) = {null_result.log_evidence:.3f}"
            ],
        )
