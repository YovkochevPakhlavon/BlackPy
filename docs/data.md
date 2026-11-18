# Data registry

The registry lives in `src/blackpy/data/registry/`, one YAML file per dataset.

```python
import blackpy as bp

bp.data.list()  # table with verification status and open TODOs
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

## Your own data

A dataset of your own is a YAML file with the same schema; BlackPy validates
it when it is loaded and every fit records its file hash, exactly as for the
registry.

1. **Copy a registry file of the same type** as a template: a QPO pair or
   triplet from `qpo/`, a shadow bound from `eht/`. The folder is
   `blackpy.data.registry_path()`.
2. **Fill in the fields.** Required: `schema_version: 1`, `id` (any
   `"group/name"`; the last part names the source parameters, e.g. `M_name`),
   `object`, `data_type`, `dataset_version`, `verified_by_human`,
   `references` (at least one, with a `citation`), and the data block
   (`frequencies` and `mass_prior` for QPOs, `bounds` and
   `fractional_deviation` for shadows). Every measurement has `value`,
   `err_minus`, `err_plus`, `unit`, `source` (a key of `references`) and
   `location` (where in the paper the number is printed). Unknown values are
   `null`.
3. **State whether the mass prior is independent** of the QPO modelling
   (`independent: true/false`); a non-independent prior triggers
   `CircularInferenceWarning` in fits.
4. **Load it** with `blackpy.data.load_file(path)`; a schema error raises
   `RegistryError` naming the field. Pass the dataset object (or a list of
   them) to `Fit`, `build_problem` or `forecast` like a registry id.

A minimal QPO pair (the numbers are made up for the example):

```python
import tempfile
import textwrap
from pathlib import Path

import blackpy as bp

text = textwrap.dedent("""
    schema_version: 1
    id: mine/source_x
    object: Source X
    data_type: qpo_pair
    dataset_version: "1"
    verified_by_human: true
    frequencies:
      upper: {value: 300.0, err_minus: 3.0, err_plus: 3.0, unit: Hz, source: me, location: "Table 1"}
      lower: {value: 200.0, err_minus: 4.0, err_plus: 4.0, unit: Hz, source: me, location: "Table 1"}
    simultaneous: true
    mass_prior: {value: 8.0, err_minus: 1.0, err_plus: 1.0, unit: Msun, method: dynamical,
                 independent: true, source: me, location: "Section 2"}
    intended_models: [RP]
    references:
      me: {citation: "Me et al. (2026), in preparation"}
    source_access: "own measurement"
""")
path = Path(tempfile.mkdtemp()) / "source_x.yaml"
path.write_text(text)
mine = bp.data.load_file(path)
print(mine.id, mine.upper, mine.lower, mine.mass_prior.value, mine.mass_prior.independent)
fit = bp.infer.Fit(
    bp.metric.Kerr(), mine, qpo_model="RP", priors={"a": bp.infer.Uniform(0.0, 0.998)}
)
print(fit.param_names)
```

```text title="Output"
mine/source_x 300 +- 3 Hz 200 +- 4 Hz 8.0 True
['a', 'M_source_x', 'r_source_x']
```

The same in memory, without a file: `blackpy.data.parse_dataset(dict)`.
For synthetic data from a known model use
`blackpy.infer.simulate_qpo_dataset(...)` or `simulate_shadow_dataset(...)`.
Datasets you create set `verified_by_human` themselves; the field of the
registry files is changed only by the maintainer, by hand.
