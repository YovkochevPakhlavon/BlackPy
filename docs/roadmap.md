# Roadmap

BlackPy is **"metric in, constraints out"**. A feature is scheduled if it
feeds a likelihood, makes a fit more trustworthy, or widens the set of
metrics that can be fitted. Features outside that scope keep a reserved
name that raises `NotImplementedError` (or refuses with a message), so that
code written against them fails clearly instead of returning a number that
silently assumes something.

Version 1.0 is a stabilisation release: it adds no physics. Everything below
is either planned for a later 1.x release or out of scope.

| Feature | Reserved interface | Status |
|---|---|---|
| Original Newman–Janis algorithm | `blackpy.metric.future.newman_janis` | **not planned**: `blackpy.metric.rotating_from_static` implements the Azreg-Aïnou version, which covers it |
| Black hole thermodynamics | `blackpy.thermo` (`hawking_temperature`, `entropy`, `heat_capacity`) | **not scheduled**: there are no thermodynamic data, so nothing would feed a likelihood |
| Quasinormal modes | `blackpy.qnm.quasinormal_modes` | **not scheduled**: without field equations only test-field modes can be computed, and those cannot be compared with ringdown observations for non-vacuum metrics |
| S2 orbit data | none yet (a new `data_type` in the registry; needs ODE integration of timelike geodesics) | later |
| Lensing by rotating metrics (equatorial plane) | the lensing functions refuse rotating metrics | later |
| Gradient-based sampler (NUTS) | none | later; needs gradients through the root finders (implicit differentiation) and an optional dependency |
| Event horizon when `g^rr = 0` depends on θ | `blackpy.geometry.horizon_profile` reports the case | later: solve `g^rr + g^θθ (∂H/∂θ)² = 0` |
| Shadows of non-separable metrics | `NotSeparableError` | **out of scope**: needs numerical ray tracing (for example Gradus.jl) |
| Continuum-fitting and iron-line likelihoods | none | **out of scope**: they need ray tracing of the disk image |

Thin-disk and lensing results are **predictions**: they do not enter a
likelihood, because there are no data for the relativistic images and the
disk observables would need ray tracing.

All observable functions take the abstract `Metric` as first argument, so
new metric classes can be added without changing the public API.
