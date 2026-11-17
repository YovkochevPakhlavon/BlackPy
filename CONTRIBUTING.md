# Contributing to BlackPy

BlackPy is "metric in, constraints out": a feature belongs here if it feeds a
likelihood, makes a fit more trustworthy, or widens the set of metrics that
can be fitted. Conventions that every change must follow are in
[`CLAUDE.md`](CLAUDE.md); the reasoning behind them is in
[`DESIGN_NOTES.md`](DESIGN_NOTES.md).

## Install

```bash
git clone https://github.com/YovkochevPakhlavon/BlackPy
cd BlackPy
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev,docs]"     # Windows: .venv\Scripts\python
```

Python 3.10 or newer. Work only inside the project's `.venv`.

## Tests and linter

```bash
.venv/bin/python -m pytest -m "not slow" -n 4 --dist loadfile   # quick subset, about 2 minutes
.venv/bin/python -m pytest -n 4 --dist loadfile                 # full suite (slow tests included)
.venv/bin/python -m pytest -m "not slow"                        # the same without pytest-xdist
.venv/bin/ruff check src tests examples
.venv/bin/ruff format --check src tests examples
.venv/bin/python -m mkdocs build --strict
```

- `--dist loadfile` keeps the tests of one file on one worker, so that the
  compiled functions cached on the session-scoped metric fixtures are reused.
- Mark a test `@pytest.mark.slow` if it runs a sampler, a symbolic 4D
  computation, a benchmark, or compiles many metric/observable combinations
  (more than about ten seconds on a laptop).
- Never report a test as passing without having run it.

## Code

- English everywhere. Type hints on public functions, NumPy-style docstrings
  with units and literature references.
- The numerical core is JAX with 64-bit floats. Derivatives come from
  automatic differentiation only: **no finite differences**. sympy is used
  only for metric input and for checks.
- Every observable is a pure core function `xxx_core(metric, p, *args,
  **static_options)` plus a thin public wrapper that calls
  `metric.evaluate(...)`. Likelihoods call core functions.
- Every formula needs a source that was actually read, or a derivation in
  `docs/theory.md`, and a test in at least one known limit. If a reference
  could not be read, say so; never cite or quote a number from memory
  without marking it.
- A feature that is out of scope stays a stub that raises
  `NotImplementedError` with a pointer to the roadmap. Do not implement it
  half-way.
- New dependencies must be justified in `DESIGN_NOTES.md`.

## Catalogue metrics: one file, one reference, one test

A new built-in metric is

1. **one file** in `src/blackpy/metric/catalogue/`, containing a factory
   function that returns a `StaticSpherical` or a `StationaryAxisymmetric`;
2. **one reference**, in the module docstring: the paper that defines the
   metric, which you have read, with the equation numbers you transcribed;
3. **at least one test** in a known limit (for example: the parameter set to
   zero gives Schwarzschild or Kerr; a horizon, photon sphere or ISCO value
   printed in the paper).

Then add the factory to `CATALOGUE` (static metrics) or `ROTATING_CATALOGUE`
(rotating metrics) in `catalogue/__init__.py`. Generated or deformed metrics
are models: their docstring must say that the source has to be checked with
`blackpy.geometry.effective_source`.

## Registry data

The rules for datasets under `src/blackpy/data/registry/` are strict and are
described once, in [`docs/data.md`](docs/data.md). In short: no number from
memory, the exact location of every value, `null` plus a `todo` entry for
anything that could not be read from the original source, and
`verified_by_human: false` until the maintainer has checked the file by hand.
Tests may check the schema and the loading of a dataset, never its numbers;
reproductions of published results go to `examples/`.

## Commits and pull requests

- Small commits, one topic each, with the tests of that topic passing.
- Describe in the pull request which sources you read and which checks you
  ran. If a formula or number could not be verified, say so explicitly.
