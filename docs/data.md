# Data registry

The registry lives in `src/blackpy/data/registry/`, one YAML file per dataset.

```python
import blackpy as bp
bp.data.list()                       # table with verification status and open TODOs
d = bp.data.load("qpo/xte_j1550_564")
```

## Rules

1. **No number is written from memory.** A value enters a file only if it was
   read from the cited source; the `location` field records the table,
   figure, equation or section, and `source_access` records how and when the
   source was read.
2. A value that could not be read from an accessible original is `null` and
   has a `todo` entry.
3. Every dataset starts with `verified_by_human: false`. Only the maintainer
   flips it to `true`, by hand, after checking the paper.
4. Mass priors state whether they are **independent** of QPO modelling.

## Schema (version 1)

| Field | Meaning |
|---|---|
| `id`, `object`, `data_type` | identifier, source name, `qpo_pair` / `qpo_triplet` / `shadow_bound` |
| `dataset_version` | bumped whenever a value changes; stored in every fit result together with the file hash |
| `verified_by_human` | see rule 3 |
| `frequencies.upper/lower/low` | `value`, `err_minus`, `err_plus`, `unit`, `source`, `location`; `low` is the type-C QPO of a triplet (fitted as nodal precession with rotating metrics) |
| `spin` (optional) | published spin of the source with `method`, `independent`, `source`, `location`; it marks the source as rotating and is never used as a prior automatically |
| `published_fit` (optional) | other published results of the same fit (e.g. RPM `mass` and `radius`), for comparison in the examples only |
| `simultaneous` | whether the peaks come from the same data selection |
| `mass_prior` | value with uncertainties, `method`, `independent`, `source`, `location` |
| `intended_models` | QPO models used in the source papers |
| `alternatives` | other published measurements, for reference |
| `bounds`, `fractional_deviation` | shadow datasets: interval bounds and the Gaussian summary |
| `assumed_mass_priors`, `mass_priors` (optional) | shadow datasets: the mass measurements of the source, each with `independent` (is it independent of the horizon-scale image?), `theta_g` and, where published, the `fractional_deviation` obtained with it; and the keys the default bound assumes |
| `measured` (optional) | shadow datasets: what the experiment measured (ring diameter, image-based `theta_g`), for the record |
| `distance`, `inclination` (optional) | shadow datasets: measurements with sources; the inclination is never used as a prior automatically |
| `circularity` (optional) | shadow datasets: `upper`, the `definition` of the deviation from circularity the bound was derived for, and what it `applies_to` |
| `other_values`, `preferred_reason` (optional, on any measurement) | other published values of the same quantity that disagree with the main one, each with its `location`, and why the main one is preferred |
| `open_questions` (optional) | unresolved questions about published values (cannot be closed by re-reading the paper) |
| `references` | citation, DOI, arXiv id, role |
| `source_access`, `notes`, `todo` | provenance and open issues |

## Changes in version 0.3

Read by Claude on 2026-11-15 to 2026-11-17 from the arXiv versions (EHT
M87\* Papers I and VI from the arXiv PDFs). **None has been verified by a
human.**

| Dataset | Change | Source and location |
|---|---|---|
| `qpo/gro_j1655_40` (version 3) | Table 3 values of spin 0.286 ± 0.003, mass 5.307 ± 0.066, radius 5.677 ± 0.035 are the main entries; the text values (0.290, 5.31, 5.68) are kept under `other_values` | Motta et al. (2014), Table 3; abstract, Sec. 4.1 and 5 |
| `qpo/xte_j1859_226` (version 3) | open question: published RPM uncertainties 2.7 times larger than propagation gives | Motta et al. (2022) |
| `eht/sgra` (version 2) | mass priors `keck` (3.964 ± 0.054 × 10⁶ M☉) and `vlti` (4.297 ± 0.042 × 10⁶ M☉) with \(\theta_g\) and \(\delta\) per prior; distance 8.277 ± 0.034 kpc | Do et al. (2019); GRAVITY Collaboration (2022); EHT Sgr A\* Paper VI, Sec. 2.1; Vagnozzi et al. (2023) |
| `eht/m87` (new) | 68% bound 4.31–6.08 on \(r_{\rm sh}/M\); \(\delta = -0.01 \pm 0.17\) (stellar dynamics); ring diameter 42 ± 3 µas; \(\theta_g = 3.8 \pm 0.4\) µas; masses from stellar dynamics (independent), gas dynamics (independent, in tension) and the EHT ring (**not** independent); distance 16.8 ± 0.8 Mpc; jet inclination 17.2 ± 3.3°; circularity < 0.10 (`fractional_rms`) | EHT (2019) Papers I and VI; Kocherlakota et al. (2021), Eq. (14); Gebhardt et al. (2011); Walsh et al. (2013); Mertens et al. (2016) |

The M87\* entry records the chain from the measured ring to a bound on the
shadow size (theory section 20). A fit uses the stellar-dynamics bound by
default; `shadow_mass_prior=` selects another, and the image-based mass
triggers a `CircularInferenceWarning`.

## Status of version 0.2

All values below were read from the arXiv versions of the papers (ar5iv HTML
rendering) on 2026-11-14; bibliographic details were checked against Crossref.
**None has been verified by a human yet.**

Added in version 0.2 (read on 2026-11-15, dataset version 2), for comparison
with BlackPy's own fits (example 6); these are Kerr QPO fits and **not**
independent measurements:

| Dataset | Published RPM result | Source and location |
|---|---|---|
| `qpo/gro_j1655_40` | \(a = 0.290 \pm 0.003\), \(M = 5.31 \pm 0.07\,M_\odot\), \(R = 5.68 \pm 0.04\,R_g\) | Motta et al. (2014), abstract and Sec. 4.1; their Table 3 gives the spin as 0.286 ± 0.003 |
| `qpo/xte_j1859_226` | \(a = 0.149 \pm 0.005\), \(M = 7.85 \pm 0.46\,M_\odot\), \(R = 6.85 \pm 0.18\,R_g\) | Motta et al. (2022), abstract, Sec. 3 and Table 3 |

| Dataset | Values | Source and location | Mass prior |
|---|---|---|---|
| `qpo/gro_j1655_40` | 441 ± 2, 298 ± 4, 17.3 ± 0.1 Hz | Motta et al. (2014), Table 2, Sample B1 | 5.40 ± 0.30 M☉, Beer & Podsiadlowski (2002), Table 4 (dynamical) |
| `qpo/xte_j1859_226` | 227.5 (+2.1/−2.4), 128.6 (+1.6/−1.8), 3.65 ± 0.01 Hz | Motta et al. (2022), Table 2, triplet | 7.8 ± 1.9 M☉, Yanes-Rizo et al. (2022), abstract (dynamical) |
| `qpo/xte_j1550_564` | 268 ± 3, 188 ± 3 Hz | Miller et al. (2001), Fig. 4 caption | 9.10 ± 0.61 M☉, Orosz et al. (2011), abstract and Sec. IV.1 (dynamical) |
| `qpo/grs_1915_105` | 69.2 ± 0.15, 41.5 ± 0.4 Hz | Strohmayer (2001), Sec. 2 | 12.4 (+2.0/−1.8) M☉, Reid et al. (2014), abstract |
| `qpo/h1743_322` | 241 ± 3, 165 (+9/−5) Hz | Homan et al. (2005), Sec. 3 | **null** (no dynamical mass found) |
| `qpo/m82_x1` | 5.07 ± 0.06, 3.32 ± 0.06 Hz | Pasham et al. (2014), abstract | 415 ± 63 M☉, same paper: **not independent** (RPM fit) |
| `eht/sgra` | 4.55–5.22 (1σ), 4.21–5.56 (2σ) for \(r_{\rm sh}/M\) | Vagnozzi et al. (2023), Eqs. (5)–(8) | – |

### Open issues

- **XTE J1550-564**: the pair 276 ± 3 / 184 ± 5 Hz used in many QPO papers was
  not found in the original papers accessed (Remillard et al. 2002 give only
  harmonic fits and "near 184 and 276 Hz"). The registry uses the
  independently fitted pair of Miller et al. (2001).
- **GRS 1915+105**: the 113/168 Hz pair is listed under `alternatives` without
  uncertainties (only a review was accessible); the default pair is the
  41/67 Hz detection of Strohmayer (2001).
- **H1743-322**: no mass prior. A fit needs an explicit user prior.
- **M82 X-1**: the only mass estimates come from the QPOs themselves; using
  them is circular and triggers a warning.
- **Sgr A\***: the paper gives lower limits 4.55 and 4.21, whereas 4.54 and
  4.20 are sometimes quoted.

## Your own data

Write a YAML file with the same schema and load it with
`blackpy.data.load_file(path)`; or create synthetic data with
`blackpy.infer.simulate_qpo_dataset(...)`.
