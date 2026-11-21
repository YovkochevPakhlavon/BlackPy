# Verification reports

Each development session of BlackPy ends with a verification report: what was
tested and with which tolerances, which sources were read for each formula and
dataset, the benchmarks, the known limitations, and the points of the
specification that need a decision. A later report describes only what changed;
earlier reports stay valid for everything they cover.

| Report | Version | Content |
|---|---|---|
| [`v1.0.md`](v1.0.md) | 1.0.0rc1 | **Current.** Frozen public API, units and names, error-message audit, wheel install, minimum dependency versions, documentation tests, benchmarks |
| [`v0.3.md`](v0.3.md) | 0.3.0 | Fit quality, plasma, M87\*, RZ/KRZ, Fisher forecasts, lensing, thin disks, catalogue additions |
| [`v0.2.md`](v0.2.md) | 0.2.0 | Rotating metrics: horizons, curvature, orbits, nodal precession, separable shadows, triplet and rotating-shadow likelihoods |
| [`v0.1.md`](v0.1.md) | 0.1.0 | Static metrics, numerical toolbox, QPO and shadow likelihoods, data registry, samplers |

The reports record how each registry value was read by Claude. The owner has
since checked all eight datasets against their papers and set
`verified_by_human: true` in each (2026-11-19; see `v1.0.md`, section 7). The
statements "not verified by a human" in `v0.1.md` to `v0.3.md` describe the
state at those releases.
