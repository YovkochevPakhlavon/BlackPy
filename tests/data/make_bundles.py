"""Regenerate the result-bundle fixtures of ``tests/test_bundles.py`` (not run by pytest).

* ``bundle_format_1.npz``: written by this checkout (format 1, since 1.0)::

      .venv/bin/python tests/data/make_bundles.py 1

* ``bundle_format_0.npz``: written by BlackPy 0.3.0 (format 0, no ``bundle_format``
  field, inclination column named ``inclination``). It was made from commit 0f23707
  (0.3.0 plus two fixes) checked out in a separate work tree::

      git worktree add /tmp/blackpy-0.3 0f23707
      PYTHONPATH=/tmp/blackpy-0.3/src .venv/bin/python tests/data/make_bundles.py 0
      git worktree remove /tmp/blackpy-0.3

Both are a short emcee run of the Kerr metric on a synthetic shadow (spin and
inclination sampled, a few hundred samples): small, but with every field of a real
bundle. The run is not converged; the fixtures test loading, not physics.
"""

import sys
import warnings
from pathlib import Path

import numpy as np

import blackpy as bp
from blackpy.infer import Fit, Uniform, simulate_shadow_dataset

fmt = int(sys.argv[1])
here = Path(__file__).parent
kerr = bp.metric.Kerr()
if fmt == 0:
    data = simulate_shadow_dataset(kerr, name="img", sigma=0.3, inclination=60.0, a=0.5)
else:
    data = simulate_shadow_dataset(kerr, name="img", sigma=0.3, inclination_deg=60.0, a=0.5)
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fit = Fit(kerr, data, priors={"a": Uniform(0.0, 0.99)}, nwalkers=16, nsteps=600)
    result = fit.run(seed=3)
keep = np.arange(0, result.samples.shape[0], max(1, result.samples.shape[0] // 300))[:300]
result.samples = result.samples[keep]
result.log_prob = result.log_prob[keep]
result.predictions = result.predictions[keep]
path = result.save(here / f"bundle_format_{fmt}.npz")
print(path, path.stat().st_size, "bytes", result.param_names, result.samples.shape)
