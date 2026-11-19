# TO_DO: finalising BlackPy 1.0 (owner's checklist)

Everything on this list can only be done by the owner. It was written at the
end of the 1.0 session (2026-11-19) from what was found in that session; the
code is labelled **1.0.0rc1**. Work through the sections in order; each item
says what, why, how, and how to tell it is done. Delete this file after the
release.

Reference: the session's report is `verification/v1.0.md`; decisions are in
`DESIGN_NOTES.md` (section 9 and "Status at 1.0").

## 1. Decisions to confirm

For each item: my recommendation in **bold**. Record the decision in
`DESIGN_NOTES.md` (section 5 lists the open ones) and close the item.

### From `DESIGN_NOTES.md`, section 5

- [ ] **Default QPO pair for XTE J1550-564 and GRS 1915+105.** The registry uses
  the independently fitted pairs it could read (Miller et al. 2001: 268/188 Hz;
  Strohmayer 2001: 41.5/69.2 Hz); the widely used 276/184 and 168/113 Hz pairs
  were not found in an accessible original. **Keep the current defaults until the
  originals are read (section 2: Remillard 2004 and the 276/184 Hz source); then
  decide.** Done when the YAML `todo` entries are closed.
- [x] **Should unverified data become an error once the registry is checked?**
  Owner decision 4 keeps `UnverifiedDataWarning` for 1.0. **Keep the warning in
  1.x** (an error would be a breaking change); revisit for 2.0. *Decided 2026-11-19: DESIGN_NOTES decision 45.*
- [x] **Should metric parameter bounds optionally restrict to the black hole
  range for observables?** **No for 1.x**: observables stay defined for naked
  singularities (reported as horizonless), and fits of rotating metrics already
  restrict through `require_black_hole=True`. *Decided 2026-11-19: DESIGN_NOTES decision 46.*
- [x] **Repository URL.** Done: `https://github.com/YovkochevPakhlavon/BlackPy`
  is in `pyproject.toml`, `CITATION.cff` and `mkdocs.yml`. Confirm, then delete
  this item from section 5 of `DESIGN_NOTES.md`. *Decided 2026-11-19: DESIGN_NOTES decision 47.*
- [ ] **GRO J1655-40 spin: 0.290 (text) or 0.286 (Table 3)?** Dataset version 3
  already stores the Table 3 value 0.286 as the main entry and keeps 0.290 under
  `other_values`. **Keep 0.286** (it is also the exact RPM solution); confirm
  against the MNRAS version (section 2).
- [x] **Delta C: Bambi et al. (2019) or Johannsen & Psaltis (2010)?** Since 0.3
  every use names its definition (`fractional_rms` default, `jp_asymmetry`,
  `bambi2019`). **Keep `fractional_rms` as default**: it is the only one fully
  specified by a published formula with its normalisation. *Decided 2026-11-19: DESIGN_NOTES decision 48.*
- [x] **Merge `ROTATING_CATALOGUE` into `CATALOGUE`?** Owner decision 5: no for
  1.x. **Revisit only for 2.0.** *Decided 2026-11-19: DESIGN_NOTES decision 49.*

### From `verification/v0.3.md`, section 7

- [ ] **The factor 4 of the `bambi2019` definition** is empirical (it reproduces
  Fig. 1 of Bambi et al. 2019 to the accuracy of the figure; their printed
  Eq. (5) does not). **Keep it labelled empirical**; if you can, ask the authors
  or find the normalisation in a later paper, and update theory section 17.
- [x] **Which M87\* mass prior should a published bound use?** The bound
  4.31–6.08 M exists only for the stellar-dynamics prior; the gas-dynamics mass
  gives delta = 0.78 ± 0.3 (2.6 sigma tension with any Schwarzschild-like
  shadow). **Keep stellar dynamics as default and always print which prior was
  used** (as `shadow_mass_prior` does now). *Decided 2026-11-19: DESIGN_NOTES decision 50.*
- [x] **KRZ truncation**: BlackPy keeps the full Kerr polar function (the
  printed truncation is not Kerr off the equator) and the printed w_01 of the
  rotating dilaton black hole has the wrong sign. **Keep BlackPy's choice**; the
  theory page documents it. *Decided 2026-11-19: DESIGN_NOTES decision 51.*
- [x] **RZ coefficients of Reissner–Nordström** were derived (the paper does not
  treat RN). **Accept**, or check against a later paper that tabulates them. *Decided 2026-11-19: DESIGN_NOTES decision 52.*
- [x] **NUTS sampler** (not attempted; needs implicit differentiation of the root
  finders). **Roadmap item for a later 1.x**, as an optional extra. *Decided 2026-11-19: DESIGN_NOTES decision 53.*

### From the registry (`open_questions`)

- [ ] **Sgr A\* Keck numbers**: Do et al. (2019) as read give M = 3.964e6 Msun,
  R0 = 7946 pc; EHT Paper VI quotes 3.951e6 and 7935 pc; Vagnozzi et al. (2023)
  print 3.951 and 7.953 kpc. **Check the journal version of Do et al. (2019)**;
  the difference (0.3% in mass) is far below the 6.5% shadow uncertainty.
- [ ] **XTE J1859+226 published RPM uncertainties** are about 2.7 times larger
  than error propagation gives. **Leave the frequencies as they are** (they are
  unaffected); if possible ask the authors. Only the comparison in example 06 is
  affected.

### Deferred in this session (API, A3; packaging, B2)

These would be breaking changes; recorded here as **decisions for 2.0**:

- [ ] **`mass` versus `mass_msun`**: `orbits.frequencies_hz`, `QPOModel.frequencies`,
  `plot.profile`, `simulate_qpo_dataset` take `mass` (solar masses); `constants`
  and `lensing` take `mass_msun`. **2.0: use `mass_msun` everywhere** (or add the
  alias with a deprecation period in a 1.x minor release).
- [ ] **`distance_kpc` (strong_field_observables) versus `distance_lens_kpc`
  (einstein_ring_angle)**. Both are fine in context. **No change.**
- [ ] **`n`** means contour points (shadows), quadrature nodes (lensing, disks)
  and predictive draws (`plot.residuals`). **2.0: rename to `n_contour`,
  `n_quad`, `n_draws`.**
- [ ] **Positional `r_obs`** in `shadow_contour`, `shadow_observables`,
  `shadow_angular_radius`, but keyword-only in `plot.shadow`. **2.0: make it
  keyword-only everywhere.**
- [x] **`strong_field_observables(dataset=...)`** reads registry masses and
  distances without an `UnverifiedDataWarning` (it is a prediction, not a fit).
  **Recommendation: add the warning in 1.1** (adding a warning is not breaking). *Done before 1.0.0 (2026-11-19): it warns now, and so do `plot.shadow(band=...)` and `inclination_prior`.*
- [ ] **Shadow `bounds` in the registry have no `source` field** (the source is
  named in the `location` text). **Add `source:` to the bounds in a future
  dataset version** (schema-compatible).
- [ ] **Lower bounds of the dependencies** were verified as declared (jax 0.4.30,
  numpy 1.24, scipy 1.10, sympy 1.12, matplotlib 3.7, corner 2.2, pyyaml 6.0,
  emcee 3.1, dynesty 2.1) on Python 3.10 with the quick subset; lower versions
  were not explored. **Keep them.** Facts on Python 3.10 (from PyPI metadata):
  JAX 0.6.2 is the last release for Python 3.10; JAX ≥ 0.7.0 requires Python ≥
  3.11. So Python 3.10 users get JAX ≤ 0.6.2. **Keep 3.10** (owner decision 6);
  when JAX 0.6.2 becomes too old, dropping 3.10 is a minor-release decision.

### Owner decisions at the top of the 1.0 specification

- [ ] None turned out to be impossible or harmful. Two notes: GitHub Pages for a
  **private** repository needs a paid GitHub plan (section 4); and the
  `inclination` → `inclination_deg` rename was implemented with the old name
  accepted everywhere it could be passed (priors, fixed, shared/per_source,
  keyword arguments, fiducial points, `FitResult[...]`, format-0 bundles).

## 2. Registry verification, value by value

Generated by `scripts/registry_checklist.py` (re-run it after any registry
change: `.venv/bin/python scripts/registry_checklist.py`). For each value: open
the paper, find the location, compare the number, its uncertainties and their
signs, and the unit. Specific items the earlier sessions flagged:

- Remillard (2004), AIP Conf. Proc. 714, 13, for GRS 1915+105 (113/168 Hz with
  uncertainties) — not obtained.
- The original source of the 276 ± 3 / 184 ± 5 Hz pair of XTE J1550-564.
- Independent mass priors for H1743-322 (none) and M82 X-1 (only a QPO-based one).
- The signs of the asymmetric M87\* uncertainties in Tables 7 and 9 of EHT Paper
  VI (read from a PDF text extraction).
- The three M87\* distance papers (Bird et al. 2010; Blakeslee et al. 2009;
  Cantiello et al. 2018) — not read.
- The Sgr A\* lower limits 4.55/4.21 against the published CQG version of
  Vagnozzi et al. (2023) (4.54/4.20 are sometimes quoted).
- The body of Yanes-Rizo et al. (2022) for the XTE J1859+226 mass (abstract only).

**How to flip `verified_by_human`.** Only the owner does this, by hand, never
Claude: when every value of a file is confirmed, edit the file and change
`verified_by_human: false` to `true` (change nothing else). If a value was
wrong, correct it, update its `location`/`source_access`, and increase
`dataset_version` by one. Then run the quick tests (they check schema and
loading, not numbers) and commit with a message naming the paper versions you
checked. `UnverifiedDataWarning` disappears for that dataset. The tests accept
either state of the flag (checked on a copy of the registry with every flag set
to `true`), and documentation snippets that list `# warns: UnverifiedDataWarning`
still pass when the warning disappears; once every dataset a snippet uses is
verified, you may delete that word from its `# warns:` line.

### Dataset `eht/m87` (M87*, shadow_bound, dataset version 2)

File: `src/blackpy/data/registry/eht/m87.yaml`. verified_by_human: **True**.

How it was read: eht2019i and eht2019vi: text extracted from the arXiv PDFs (arxiv.org/pdf/1906.11238 and 1906.11243; no ar5iv rendering exists); all other references: ar5iv HTML renderings of the arXiv versions (ar5iv.labs.arxiv.org). Read by Claude on 2026-11-16. The journal versions were not compared. Titles, journals, volumes and article numbers of all references were confirmed with Crossref (api.crossref.org) on 2026-11-16.

Values:

- [x] `bounds.1sigma`: lower = 4.31, upper = 6.08. Source: not recorded in the entry (see the location and the references below). Location: kocherlakota2021, Eq. (14): 4.31 M <= r_sh <= 6.08 M (68% confidence), i.e. 3 sqrt(3) (1 +- 0.17) M with the stellar-dynamics mass prior
- [x] `fractional_deviation`: value = -0.01, err_minus = 0.17, err_plus = 0.17. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L6 (Paper VI) (doi:10.3847/2041-8213/ab1141, arXiv:1906.11243). Location: Sec. 9.5, text after Eq. (32): 'delta = -0.01 +- 0.17 (68% credible intervals) for the stellar' dynamics prior; delta is defined in Eq. (31) as theta_g / theta_dyn - 1
- [x] `mass_priors.stellar_dynamics`: value = 6.2, err_minus = 0.6, err_plus = 1.1 [1e9 Msun]. Source: EHT Paper I, ApJL 875, L1, Table 1, row 'M(stars)' (journal version; dataset version 2, owner's choice; Paper VI Table 9 prints 6.14 +1.07 -0.62)
- [x] `mass_priors.stellar_dynamics.other_values[0]`: value = 6.6, err_minus = 0.4, err_plus = 0.4. Source: Gebhardt, Adams, Richstone, et al. (2011), Astrophys. J. 729, 119 (doi:10.1088/0004-637X/729/2/119, arXiv:1101.1954). Location: gebhardt2011, abstract and Sec. 6: '(6.6 +- 0.4) x 1e9 Msun', for an assumed distance of 17.9 Mpc (Sec. 1)
- [x] `mass_priors.stellar_dynamics.theta_g`: value = 3.62, err_minus = 0.34, err_plus = 0.6 [muas]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L6 (Paper VI) (doi:10.3847/2041-8213/ab1141, arXiv:1906.11243). Location: Table 9, row 'Stellar dynamics, theta_dyn': 3.62 +0.60 -0.34 muas; also Table 7
- [x] `mass_priors.stellar_dynamics.fractional_deviation`: value = -0.01, err_minus = 0.17, err_plus = 0.17. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L6 (Paper VI) (doi:10.3847/2041-8213/ab1141, arXiv:1906.11243). Location: Sec. 9.5: delta = -0.01 +- 0.17 for the stellar-dynamics prior
- [x] `mass_priors.gas_dynamics`: value = 3.5, err_minus = 0.3, err_plus = 0.9 [1e9 Msun]. Source: EHT Paper I, ApJL 875, L1, Table 1, row 'M(gas)' (journal version; dataset version 2, owner's choice; Paper VI Table 9 prints 3.45 +0.85 -0.26)
- [x] `mass_priors.gas_dynamics.other_values[0]`: value = 3.5, err_minus = 0.7, err_plus = 0.9. Source: Walsh, Barth, Ho & Sarzi (2013), Astrophys. J. 770, 86 (doi:10.1088/0004-637X/770/2/86, arXiv:1304.7273). Location: walsh2013, abstract: 'M_BH = (3.5 +0.9 -0.7) x 1e9 Msun (68% confidence)', for D = 17.9 Mpc (Sec. 1)
- [x] `mass_priors.gas_dynamics.theta_g`: value = 2.05, err_minus = 0.16, err_plus = 0.48 [muas]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L6 (Paper VI) (doi:10.3847/2041-8213/ab1141, arXiv:1906.11243). Location: Table 9, row 'Gas dynamics, theta_dyn': 2.05 +0.48 -0.16 muas; also Table 7
- [x] `mass_priors.gas_dynamics.fractional_deviation`: value = 0.78, err_minus = 0.3, err_plus = 0.3. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L6 (Paper VI) (doi:10.3847/2041-8213/ab1141, arXiv:1906.11243). Location: Sec. 9.5: 'delta = 0.78 +- 0.3 for the gas dynamics' prior
- [x] `mass_priors.eht_ring`: value = 6.5, err_minus = 0.7, err_plus = 0.7 [1e9 Msun]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: eht2019i, Table 1 and Sec. 7: M = (6.5 +- 0.7) x 1e9 Msun; eht2019vi, Table 7, row 'Combined measurements from this work'
- [x] `mass_priors.eht_ring.theta_g`: value = 3.8, err_minus = 0.4, err_plus = 0.4 [muas]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: eht2019i, Table 1: theta_g = GM/(D c^2) = 3.8 +- 0.4 muas; eht2019vi, Table 7
- [x] `measured.ring_diameter`: value = 42.0, err_minus = 3.0, err_plus = 3.0 [muas]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: Abstract, Sec. 7 and Table 1, row 'Ring diameter d': 42 +- 3 muas
- [x] `measured.theta_g`: value = 3.8, err_minus = 0.4, err_plus = 0.4 [muas]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: Sec. 7 and Table 1: theta_g = GM/(D c^2) = 3.8 +- 0.4 muas; eht2019vi, Table 7, last row
- [x] `distance`: value = 16.8, err_minus = 0.8, err_plus = 0.8 [Mpc]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: Sec. 1 and Table 1: D = (16.8 +- 0.8) Mpc
- [x] `inclination`: value = 17.2, err_minus = 3.3, err_plus = 3.3 [deg]. Source: Mertens, Lobanov, Walker & Hardee (2016), Astron. Astrophys. 595, A54 (doi:10.1051/0004-6361/201628829, arXiv:1608.05063). Location: Eq. (4): theta = 17.2 +- 3.3 deg, from the apparent speeds of the jet and the counter-jet between 0.5 and 1 mas from the core and their intensity ratio
- [x] `inclination.other_values[0]`: value = 17.0, err_minus = null, err_plus = null [deg]. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: eht2019i, Sec. 6: 'adopting an inclination of 17 deg between the approaching jet and the line of sight (Walker et al. 2018)'; walker2018, Sec. 1: 'we adopt the smaller viewing angle ... of theta ~ 17 deg'
- [x] `circularity`: upper = 0.1. Source: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238). Location: Sec. 7: the axis ratio of the ring is at most 4:3, 'this corresponds to a 10% deviation from circularity in terms of root-mean-square distance from an average radius'; Sec. 8 repeats it; Table 1: axial ratio < 4:3

References (check the bibliographic details too):

- [x] `eht2019i`: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L1 (Paper I) (doi:10.3847/2041-8213/ab0ec7, arXiv:1906.11238)
- [x] `eht2019vi`: Event Horizon Telescope Collaboration (2019), Astrophys. J. Lett. 875, L6 (Paper VI) (doi:10.3847/2041-8213/ab1141, arXiv:1906.11243)
- [x] `psaltis2020`: Psaltis, Medeiros, Christian, et al. (EHT Collaboration) (2020), Phys. Rev. Lett. 125, 141104 (doi:10.1103/PhysRevLett.125.141104, arXiv:2010.01055)
- [x] `kocherlakota2021`: Kocherlakota, Rezzolla, Falcke, et al. (EHT Collaboration) (2021), Phys. Rev. D 103, 104047 (doi:10.1103/PhysRevD.103.104047, arXiv:2105.09343)
- [x] `gebhardt2011`: Gebhardt, Adams, Richstone, et al. (2011), Astrophys. J. 729, 119 (doi:10.1088/0004-637X/729/2/119, arXiv:1101.1954)
- [x] `walsh2013`: Walsh, Barth, Ho & Sarzi (2013), Astrophys. J. 770, 86 (doi:10.1088/0004-637X/770/2/86, arXiv:1304.7273)
- [x] `mertens2016`: Mertens, Lobanov, Walker & Hardee (2016), Astron. Astrophys. 595, A54 (doi:10.1051/0004-6361/201628829, arXiv:1608.05063)
- [x] `walker2018`: Walker, Hardee, Davies, Ly & Junor (2018), Astrophys. J. 855, 128 (doi:10.3847/1538-4357/aaafcc, arXiv:1802.06166)

TODO items of the file:

- [x] Human check of every number against the published ApJL versions, in particular the signs of the asymmetric uncertainties in Tables 7 and 9 of Paper VI (read from a PDF text extraction). *Done by the owner on 2026-11-19: Paper I against the journal version (Table 1; the signs agree), the rest against arXiv; removed from the file's `todo`.*
- [x] No 2-sigma bound on r_sh/M was found in the papers read; only the 68% interval is stored, so shadow_likelihood='interval' needs shadow_level='1sigma'. *A known gap, not a value to check: it stays in the file's `todo` and does not block the flag.*
- [x] The three distance papers combined by the EHT (Bird et al. 2010; Blakeslee et al. 2009; Cantiello et al. 2018) were not read. *Not needed (owner, 2026-11-19): BlackPy uses the EHT's combined 16.8 +- 0.8 Mpc, checked in the journal version; removed from the file's `todo`.*

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/eht/m87.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `eht/sgra` (Sgr A*, shadow_bound, dataset version 2)

File: `src/blackpy/data/registry/eht/sgra.yaml`. verified_by_human: **True**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared. Mass priors, distance and per-prior deviations (dataset version 2): do2019, gravity2020, gravity2022, eht2022vi and vagnozzi2023 read from their ar5iv HTML renderings on 2026-11-16 by Claude; bibliographic data of do2019, gravity2020, gravity2022 and eht2022vi confirmed with Crossref on 2026-11-16; the journal versions were not compared

Values:

- [x] `bounds.1sigma`: lower = 4.55, upper = 5.22. Source: not recorded in the entry (see the location and the references below). Location: Eq. (7)
- [x] `bounds.2sigma`: lower = 4.21, upper = 5.56. Source: not recorded in the entry (see the location and the references below). Location: Eq. (8)
- [x] `fractional_deviation`: value = -0.06, err_minus = 0.065, err_plus = 0.065. Source: Vagnozzi, Roy, Tsai, Visinelli, et al. (2023), Class. Quantum Grav. 40, 165007 (doi:10.1088/1361-6382/acd97b, arXiv:2205.07787). Location: Eq. (5): delta ~ -0.060 +- 0.065, with r_sh/M = 3 sqrt(3) (delta + 1)
- [ ] `mass_priors.keck`: value = 3.964, err_minus = 0.054, err_plus = 0.054 [1e6 Msun]. Source: Do, Hees, Ghez, et al. (2019), Science 365, 664 (doi:10.1126/science.aav8137, arXiv:1907.10731). Location: Main text (results) and Supplementary text ('fit 2'): 'If we assume GR is true, then M_BH = (3.964 +- 0.047 +- 0.026) x 1e6 Msun and R_0 = 7,946 +- 50 +- 32 pc' (statistical, then systematic from a jackknife analysis)
- [ ] `mass_priors.keck.other_values[0]`: value = 3.951, err_minus = 0.047, err_plus = 0.047. Source: Event Horizon Telescope Collaboration (2022), Astrophys. J. Lett. 930, L17 (Sgr A* Paper VI) (doi:10.3847/2041-8213/ac6756, arXiv:2311.09484). Location: eht2022vi, Sec. 2.1: 'M = (3.951 +- 0.047) x 1e6 Msun' with 'R_0 = (7935 +- 50) pc', attributed to the posterior of Do et al. (2019) that assumes general relativity; the same mass is in Table 1 of vagnozzi2023
- [ ] `mass_priors.keck.other_values[1]`: value = 3.984, err_minus = 0.064, err_plus = 0.064. Source: Do, Hees, Ghez, et al. (2019), Science 365, 664 (doi:10.1126/science.aav8137, arXiv:1907.10731). Location: do2019, main text and Table 1: 'M_BH = (3.984 +- 0.058 +- 0.026) x 1e6 Msun and R_0 = 7,971 +- 59 +- 32 pc', for the fit with the redshift parameter free
- [ ] `mass_priors.keck.theta_g`: value = 4.92, err_minus = 0.032, err_plus = 0.032 [muas]. Source: Event Horizon Telescope Collaboration (2022), Astrophys. J. Lett. 930, L17 (Sgr A* Paper VI) (doi:10.3847/2041-8213/ac6756, arXiv:2311.09484). Location: eht2022vi, Sec. 2.1: 'theta_g = (4.92 +- 0.03 +- 0.01) muas (Keck)'
- [ ] `mass_priors.keck.fractional_deviation`: value = -0.04, err_minus = 0.1, err_plus = 0.09. Source: Vagnozzi, Roy, Tsai, Visinelli, et al. (2023), Class. Quantum Grav. 40, 165007 (doi:10.1088/1361-6382/acd97b, arXiv:2205.07787). Location: vagnozzi2023, section 'Methodology', list before Eq. (5): 'Keck: delta = -0.04 +0.09 -0.10', quoting eht2022vi (Table 2 there lists -0.04 +0.09 -0.10)
- [ ] `mass_priors.vlti`: value = 4.297, err_minus = 0.042, err_plus = 0.042 [1e6 Msun]. Source: GRAVITY Collaboration (2022), Astron. Astrophys. 657, L12 (doi:10.1051/0004-6361/202142465, arXiv:2112.07478). Location: Sec. 6 (Conclusions): 'R_0 = (8277 +- 9) pc and M = (4.297 +- 0.012) x 1e6 Msun (statistical errors, see Gravity Coll. 2021 for a discussion of the systematics that are ~30 pc for R_0 and ~40,000 Msun for M)'
- [ ] `mass_priors.vlti.other_values[0]`: value = 4.261, err_minus = 0.012, err_plus = 0.012. Source: GRAVITY Collaboration (2020), Astron. Astrophys. 636, L5 (doi:10.1051/0004-6361/202037813, arXiv:2004.07187). Location: gravity2020, Appendix E, table 'Best-fit orbit parameters': M = 4.261 +- 0.012 x 1e6 Msun, R_0 = 8246.7 +- 9.3 pc (statistical)
- [ ] `mass_priors.vlti.theta_g`: value = 5.125, err_minus = 0.022, err_plus = 0.022 [muas]. Source: Event Horizon Telescope Collaboration (2022), Astrophys. J. Lett. 930, L17 (Sgr A* Paper VI) (doi:10.3847/2041-8213/ac6756, arXiv:2311.09484). Location: eht2022vi, Sec. 2.1: 'theta_g = (5.125 +- 0.009 +- 0.020) muas (VLTI)'
- [ ] `mass_priors.vlti.fractional_deviation`: value = -0.08, err_minus = 0.09, err_plus = 0.09. Source: Vagnozzi, Roy, Tsai, Visinelli, et al. (2023), Class. Quantum Grav. 40, 165007 (doi:10.1088/1361-6382/acd97b, arXiv:2205.07787). Location: vagnozzi2023, section 'Methodology', list before Eq. (5): 'VLTI: delta = -0.08 +0.09 -0.09', quoting eht2022vi (Table 2 there lists -0.08 +0.09 -0.09)
- [ ] `distance`: value = 8.277, err_minus = 0.034, err_plus = 0.034 [kpc]. Source: GRAVITY Collaboration (2022), Astron. Astrophys. 657, L12 (doi:10.1051/0004-6361/202142465, arXiv:2112.07478). Location: Sec. 6 (Conclusions): 'R_0 = (8277 +- 9) pc' (statistical); systematic error 33 pc from eht2022vi, Sec. 2.1 ('R_0 = (8277 +- 9 +- 33) pc')
- [ ] `distance.other_values[0]`: value = 7.946, err_minus = 0.059, err_plus = 0.059. Source: Do, Hees, Ghez, et al. (2019), Science 365, 664 (doi:10.1126/science.aav8137, arXiv:1907.10731). Location: do2019, main text and Supplementary text ('fit 2'): 'R_0 = 7,946 +- 50 +- 32 pc' assuming general relativity

References (check the bibliographic details too):

- [ ] `vagnozzi2023`: Vagnozzi, Roy, Tsai, Visinelli, et al. (2023), Class. Quantum Grav. 40, 165007 (doi:10.1088/1361-6382/acd97b, arXiv:2205.07787)
- [ ] `eht2022vi`: Event Horizon Telescope Collaboration (2022), Astrophys. J. Lett. 930, L17 (Sgr A* Paper VI) (doi:10.3847/2041-8213/ac6756, arXiv:2311.09484)
- [ ] `do2019`: Do, Hees, Ghez, et al. (2019), Science 365, 664 (doi:10.1126/science.aav8137, arXiv:1907.10731)
- [ ] `gravity2022`: GRAVITY Collaboration (2022), Astron. Astrophys. 657, L12 (doi:10.1051/0004-6361/202142465, arXiv:2112.07478)
- [ ] `gravity2020`: GRAVITY Collaboration (2020), Astron. Astrophys. 636, L5 (doi:10.1051/0004-6361/202037813, arXiv:2004.07187)

TODO items of the file:

- [ ] Human check of Eqs. (5)-(8) against the published CQG version (lower limits 4.55/4.21 vs 4.54/4.20).
- [x] Human check of the mass priors and distances added in dataset version 2 against the published papers; the number of the GRAVITY (2020) orbit table in the journal version was not determined. *Done by the owner on 2026-11-19: do2019 Table 1 and the GRAVITY first pages against the journals, the rest against arXiv; removed from the file's `todo`.*

Open questions:

- [ ] Which Keck numbers are right? Do et al. (2019), as read from ar5iv, give M = 3.964 x 1e6 Msun and R_0 = 7946 pc for the fit that assumes general relativity, whereas EHT Sgr A* Paper VI (Sec. 2.1) quotes 3.951 x 1e6 Msun and 7935 pc for it, and Vagnozzi et al. (2023, Table 1) print 3.951 and 7.953 kpc. The differences (0.3% in mass) are far below the 6.5% uncertainty of the shadow bound, so the default likelihood is unaffected.

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/eht/sgra.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `qpo/gro_j1655_40` (GRO J1655-40, qpo_triplet, dataset version 3)

File: `src/blackpy/data/registry/qpo/gro_j1655_40.yaml`. verified_by_human: **True**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared; spin and published_fit (text values and Table 3): read again from the ar5iv HTML rendering of the arXiv version (arXiv:1309.3652) on 2026-11-15 by Claude for dataset version 3; the journal version was not compared

Values:

- [x] `frequencies`: lower = {'value': 298.0, 'err_minus': 4.0, 'err_plus': 4.0, 'unit': 'Hz', 'source': 'motta2014', 'location': "Table 2, block 'Sample B1 observations', column 'lower HFQPO', nu [Hz]"}, upper = {'value': 441.0, 'err_minus': 2.0, 'err_plus': 2.0, 'unit': 'Hz', 'source': 'motta2014', 'location': "Table 2, block 'Sample B1 observations', column 'upper HFQPO', nu [Hz]"}. Source: not recorded in the entry (see the location and the references below). Location: (no location) *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `frequencies.upper`: value = 441.0, err_minus = 2.0, err_plus = 2.0 [Hz]. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Table 2, block 'Sample B1 observations', column 'upper HFQPO', nu [Hz] *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `frequencies.lower`: value = 298.0, err_minus = 4.0, err_plus = 4.0 [Hz]. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Table 2, block 'Sample B1 observations', column 'lower HFQPO', nu [Hz] *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `frequencies.low`: value = 17.3, err_minus = 0.1, err_plus = 0.1 [Hz]. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Table 2, block 'Sample B1 observations', column 'Type-C QPO', nu [Hz] *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `spin`: value = 0.286, err_minus = 0.003, err_plus = 0.003. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Table 3 ('Black hole parameters and associated quantities as measured from the RPM'), row 'Spin': mean value 0.286, standard deviation 0.003 *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `spin.other_values[0]`: value = 0.29, err_minus = 0.003, err_plus = 0.003. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Abstract, Sec. 4.1 ('a = 0.290 +- 0.003'), caption of Fig. 3 and Sec. 5 *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `published_fit.mass`: value = 5.307, err_minus = 0.066, err_plus = 0.066 [Msun]. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Table 3, row 'Mass (Solar masses)': mean value 5.307, standard deviation 0.066 *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `published_fit.mass.other_values[0]`: value = 5.31, err_minus = 0.07, err_plus = 0.07. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Abstract, Sec. 4.1 ('M = (5.31 +- 0.07) Msun'), caption of Fig. 3 and Sec. 5 *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `published_fit.radius`: value = 5.677, err_minus = 0.035, err_plus = 0.035 [G M / c^2]. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Table 3, row 'Radius (Gravitational radii)': mean value 5.677, standard deviation 0.035 *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `published_fit.radius.other_values[0]`: value = 5.68, err_minus = 0.04, err_plus = 0.04. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Sec. 4.1 ('R = (5.68 +- 0.04) Rg') *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `published_fit.radius.other_values[1]`: value = 5.68, err_minus = 0.03, err_plus = 0.03. Source: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652). Location: Sec. 5 ('R = 5.68 +- 0.03 Rg') *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `mass_prior`: value = 5.4, err_minus = 0.3, err_plus = 0.3 [Msun]. Source: Beer & Podsiadlowski (2002), MNRAS 331, 351 (doi:10.1046/j.1365-8711.2002.05189.x, arXiv:astro-ph/0109136). Location: Table 4 ('Best-fitting model parameters for Nova Sco'), row M_1; also the abstract

References (check the bibliographic details too):

- [x] `motta2014`: Motta, Belloni, Stella, Munoz-Darias & Fender (2014), MNRAS 437, 2554 (doi:10.1093/mnras/stt2068, arXiv:1309.3652) *Checked by Claude against the journal PDF (MNRAS 437, 2554) supplied by the owner, 2026-11-19.*
- [x] `beer2002`: Beer & Podsiadlowski (2002), MNRAS 331, 351 (doi:10.1046/j.1365-8711.2002.05189.x, arXiv:astro-ph/0109136)

TODO items of the file:

- [x] Human check of Table 2 of Motta et al. (2014) against the published MNRAS version. *Done 2026-11-19 against the journal version; removed from the file's `todo`.*
- [x] Human check of the RPM results (abstract, Sec. 4.1, Table 3) and of the 0.290 / 0.286 discrepancy against the published MNRAS version. Since dataset version 3 the Table 3 values are the main entries and the text values are stored under other_values. *Done 2026-11-19 against the journal version; removed from the file's `todo`.*

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/qpo/gro_j1655_40.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `qpo/grs_1915_105` (GRS 1915+105, qpo_pair, dataset version 1)

File: `src/blackpy/data/registry/qpo/grs_1915_105.yaml`. verified_by_human: **True**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared

Values:

- [x] `frequencies`: lower = {'value': 41.5, 'err_minus': 0.4, 'err_plus': 0.4, 'unit': 'Hz', 'source': 'strohmayer2001', 'location': "Section 2 (data analysis): 'well fit by Gaussians centered at nu_0 = 41.5 +- 0.4 and 69.2 +- 0.15 Hz'"}, upper = {'value': 69.2, 'err_minus': 0.15, 'err_plus': 0.15, 'unit': 'Hz', 'source': 'strohmayer2001', 'location': "Section 2 (data analysis): 'well fit by Gaussians centered at nu_0 = 41.5 +- 0.4 and 69.2 +- 0.15 Hz'"}. Source: not recorded in the entry (see the location and the references below). Location: (no location)
- [x] `frequencies.upper`: value = 69.2, err_minus = 0.15, err_plus = 0.15 [Hz]. Source: Strohmayer (2001), ApJ 554, L169 (doi:10.1086/321720, arXiv:astro-ph/0105338). Location: Section 2 (data analysis): 'well fit by Gaussians centered at nu_0 = 41.5 +- 0.4 and 69.2 +- 0.15 Hz'
- [x] `frequencies.lower`: value = 41.5, err_minus = 0.4, err_plus = 0.4 [Hz]. Source: Strohmayer (2001), ApJ 554, L169 (doi:10.1086/321720, arXiv:astro-ph/0105338). Location: Section 2 (data analysis): 'well fit by Gaussians centered at nu_0 = 41.5 +- 0.4 and 69.2 +- 0.15 Hz'
- [x] `mass_prior`: value = 12.4, err_minus = 1.8, err_plus = 2.0 [Msun]. Source: Reid, McClintock, Steiner, Steeghs, et al. (2014), ApJ 796, 2 (doi:10.1088/0004-637X/796/1/2, arXiv:1409.2453). Location: Abstract ('a revised estimate for the mass of the black hole of 12.4 +2.0 -1.8 Msun')
- [x] `alternatives[0]`: lower = {'value': 113.0, 'err_minus': None, 'err_plus': None, 'unit': 'Hz'}, upper = {'value': 168.0, 'err_minus': None, 'err_plus': None, 'unit': 'Hz'}. Source: Remillard & McClintock (2006), ARA&A 44, 49 (doi:10.1146/annurev.astro.44.051905.092532, arXiv:astro-ph/0606352). Location: (no location)
- [x] `alternatives[0].upper`: value = 168.0, err_minus = null, err_plus = null [Hz]. Source: Remillard & McClintock (2006), ARA&A 44, 49 (doi:10.1146/annurev.astro.44.051905.092532, arXiv:astro-ph/0606352). Location: (no location)
- [x] `alternatives[0].lower`: value = 113.0, err_minus = null, err_plus = null [Hz]. Source: Remillard & McClintock (2006), ARA&A 44, 49 (doi:10.1146/annurev.astro.44.051905.092532, arXiv:astro-ph/0606352). Location: (no location)

References (check the bibliographic details too):

- [x] `strohmayer2001`: Strohmayer (2001), ApJ 554, L169 (doi:10.1086/321720, arXiv:astro-ph/0105338)
- [x] `reid2014`: Reid, McClintock, Steiner, Steeghs, et al. (2014), ApJ 796, 2 (doi:10.1088/0004-637X/796/1/2, arXiv:1409.2453)
- [x] `remillard_mcclintock2006`: Remillard & McClintock (2006), ARA&A 44, 49 (doi:10.1146/annurev.astro.44.051905.092532, arXiv:astro-ph/0606352)

TODO items of the file:

- [x] Obtain Remillard (2004), AIP Conf. Proc. 714, 13 and record the 113/168 Hz frequencies with their uncertainties.

- [x] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/qpo/grs_1915_105.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `qpo/h1743_322` (H1743-322, qpo_pair, dataset version 1)

File: `src/blackpy/data/registry/qpo/h1743_322.yaml`. verified_by_human: **False**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared

Values:

- [ ] `frequencies`: lower = {'value': 165.0, 'err_minus': 5.0, 'err_plus': 9.0, 'unit': 'Hz', 'source': 'homan2005', 'location': "Section 3 (high-frequency QPOs): 'This gave a frequency of 165 +9 -5 Hz ... (a 3 sigma detection)'", 'note': '3 sigma detection with the Q value fixed to 6.'}, upper = {'value': 241.0, 'err_minus': 3.0, 'err_plus': 3.0, 'unit': 'Hz', 'source': 'homan2005', 'location': "Section 3 (high-frequency QPOs): 'A fit to the second part (see Fig. 7) gives nu_max = 241 +- 3'"}. Source: not recorded in the entry (see the location and the references below). Location: (no location)
- [ ] `frequencies.upper`: value = 241.0, err_minus = 3.0, err_plus = 3.0 [Hz]. Source: Homan, Miller, Wijnands, van der Klis, et al. (2005), ApJ 623, 383 (doi:10.1086/424994, arXiv:astro-ph/0406334). Location: Section 3 (high-frequency QPOs): 'A fit to the second part (see Fig. 7) gives nu_max = 241 +- 3'
- [ ] `frequencies.lower`: value = 165.0, err_minus = 5.0, err_plus = 9.0 [Hz]. Source: Homan, Miller, Wijnands, van der Klis, et al. (2005), ApJ 623, 383 (doi:10.1086/424994, arXiv:astro-ph/0406334). Location: Section 3 (high-frequency QPOs): 'This gave a frequency of 165 +9 -5 Hz ... (a 3 sigma detection)'
- [ ] `mass_prior`: value = null, err_minus = null, err_plus = null [Msun]. Source: not recorded in the entry (see the location and the references below). Location: (no location)
- [ ] `alternatives[0]`: lower = {'value': 166.0, 'err_minus': 5.0, 'err_plus': 5.0, 'unit': 'Hz'}, upper = {'value': 242.0, 'err_minus': 3.0, 'err_plus': 3.0, 'unit': 'Hz'}. Source: Remillard, McClintock, Orosz & Levine (2006), ApJ 637, 1002 (doi:10.1086/498556, arXiv:astro-ph/0407025). Location: (no location)
- [ ] `alternatives[0].upper`: value = 242.0, err_minus = 3.0, err_plus = 3.0 [Hz]. Source: Remillard, McClintock, Orosz & Levine (2006), ApJ 637, 1002 (doi:10.1086/498556, arXiv:astro-ph/0407025). Location: (no location)
- [ ] `alternatives[0].lower`: value = 166.0, err_minus = 5.0, err_plus = 5.0 [Hz]. Source: Remillard, McClintock, Orosz & Levine (2006), ApJ 637, 1002 (doi:10.1086/498556, arXiv:astro-ph/0407025). Location: (no location)

References (check the bibliographic details too):

- [ ] `homan2005`: Homan, Miller, Wijnands, van der Klis, et al. (2005), ApJ 623, 383 (doi:10.1086/424994, arXiv:astro-ph/0406334)
- [ ] `remillard2006`: Remillard, McClintock, Orosz & Levine (2006), ApJ 637, 1002 (doi:10.1086/498556, arXiv:astro-ph/0407025)

TODO items of the file:

- [ ] Mass prior: null. Find an independent mass estimate (and record whether it relies on QPO or jet modelling), or keep the source out of joint fits.

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/qpo/h1743_322.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `qpo/m82_x1` (M82 X-1, qpo_pair, dataset version 1)

File: `src/blackpy/data/registry/qpo/m82_x1.yaml`. verified_by_human: **False**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared

Values:

- [ ] `frequencies`: lower = {'value': 3.32, 'err_minus': 0.06, 'err_plus': 0.06, 'unit': 'Hz', 'source': 'pasham2014', 'location': "Abstract; main text ('two power spectral peaks at 3.32 +- 0.06 Hz ... and 5.07 +- 0.06 Hz'); Figure 1 caption"}, upper = {'value': 5.07, 'err_minus': 0.06, 'err_plus': 0.06, 'unit': 'Hz', 'source': 'pasham2014', 'location': "Abstract; main text ('two power spectral peaks at 3.32 +- 0.06 Hz ... and 5.07 +- 0.06 Hz'); Figure 1 caption"}. Source: not recorded in the entry (see the location and the references below). Location: (no location)
- [ ] `frequencies.upper`: value = 5.07, err_minus = 0.06, err_plus = 0.06 [Hz]. Source: Pasham, Strohmayer & Mushotzky (2014), Nature 513, 74 (doi:10.1038/nature13710, arXiv:1501.03180). Location: Abstract; main text ('two power spectral peaks at 3.32 +- 0.06 Hz ... and 5.07 +- 0.06 Hz'); Figure 1 caption
- [ ] `frequencies.lower`: value = 3.32, err_minus = 0.06, err_plus = 0.06 [Hz]. Source: Pasham, Strohmayer & Mushotzky (2014), Nature 513, 74 (doi:10.1038/nature13710, arXiv:1501.03180). Location: Abstract; main text ('two power spectral peaks at 3.32 +- 0.06 Hz ... and 5.07 +- 0.06 Hz'); Figure 1 caption
- [ ] `mass_prior`: value = 415.0, err_minus = 63.0, err_plus = 63.0 [Msun]. Source: Pasham, Strohmayer & Mushotzky (2014), Nature 513, 74 (doi:10.1038/nature13710, arXiv:1501.03180). Location: Abstract ('using the relativistic precession model, from which we get a value of 415 +- 63 Msun')

References (check the bibliographic details too):

- [ ] `pasham2014`: Pasham, Strohmayer & Mushotzky (2014), Nature 513, 74 (doi:10.1038/nature13710, arXiv:1501.03180)

TODO items of the file:

- [ ] No independent mass prior. Use a broad uninformative prior instead, or exclude the source from tests of non-Kerr metrics.

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/qpo/m82_x1.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `qpo/xte_j1550_564` (XTE J1550-564, qpo_pair, dataset version 1)

File: `src/blackpy/data/registry/qpo/xte_j1550_564.yaml`. verified_by_human: **False**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared

Values:

- [ ] `frequencies`: lower = {'value': 188.0, 'err_minus': 3.0, 'err_plus': 3.0, 'unit': 'Hz', 'source': 'miller2001', 'location': "Figure 4 caption ('The lower-frequency QPO: 188 +- 3 Hz'); also Section 3", 'note': '3.5 sigma single-trial detection.'}, upper = {'value': 268.0, 'err_minus': 3.0, 'err_plus': 3.0, 'unit': 'Hz', 'source': 'miller2001', 'location': "Figure 4 caption ('The higher-frequency QPO: 268 +- 3 Hz'); also Section 3"}. Source: not recorded in the entry (see the location and the references below). Location: (no location)
- [ ] `frequencies.upper`: value = 268.0, err_minus = 3.0, err_plus = 3.0 [Hz]. Source: Miller, Wijnands, Homan, Belloni, et al. (2001), ApJ 563, 928 (doi:10.1086/324027, arXiv:astro-ph/0105371). Location: Figure 4 caption ('The higher-frequency QPO: 268 +- 3 Hz'); also Section 3
- [ ] `frequencies.lower`: value = 188.0, err_minus = 3.0, err_plus = 3.0 [Hz]. Source: Miller, Wijnands, Homan, Belloni, et al. (2001), ApJ 563, 928 (doi:10.1086/324027, arXiv:astro-ph/0105371). Location: Figure 4 caption ('The lower-frequency QPO: 188 +- 3 Hz'); also Section 3
- [ ] `mass_prior`: value = 9.1, err_minus = 0.61, err_plus = 0.61 [Msun]. Source: Orosz, Steiner, McClintock, Torres, et al. (2011), ApJ 730, 75 (doi:10.1088/0004-637X/730/2/75, arXiv:1101.2499). Location: Abstract and Section IV.1 ('The adopted value for the mass of the black hole is 9.10 +- 0.61 Msun (Table 2)')
- [ ] `alternatives[1]`: lower = null, upper = null. Source: not recorded in the entry (see the location and the references below). Location: (no location)

References (check the bibliographic details too):

- [ ] `miller2001`: Miller, Wijnands, Homan, Belloni, et al. (2001), ApJ 563, 928 (doi:10.1086/324027, arXiv:astro-ph/0105371)
- [ ] `remillard2002`: Remillard, Muno, McClintock & Orosz (2002), ApJ 580, 1030 (doi:10.1086/343791, arXiv:astro-ph/0202305)
- [ ] `orosz2011`: Orosz, Steiner, McClintock, Torres, et al. (2011), ApJ 730, 75 (doi:10.1088/0004-637X/730/2/75, arXiv:1101.2499)

TODO items of the file:

- [ ] Locate the original source of the widely used pair 276 +- 3 Hz / 184 +- 5 Hz (probably Remillard et al. 2002 figures or Remillard & McClintock 2006) and decide which pair should be the default.

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/qpo/xte_j1550_564.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

### Dataset `qpo/xte_j1859_226` (XTE J1859+226, qpo_triplet, dataset version 3)

File: `src/blackpy/data/registry/qpo/xte_j1859_226.yaml`. verified_by_human: **False**.

How it was read: full text read from the ar5iv HTML rendering of the arXiv version (ar5iv.labs.arxiv.org), title/DOI checked against the arXiv API, accessed 2026-11-14 by Claude; the journal version was not compared; spin and published_fit: read from the ar5iv HTML rendering of the arXiv version on 2026-11-15 by Claude (re-read the same day for dataset version 3, which only adds open_questions); the journal version was not compared

Values:

- [ ] `frequencies`: lower = {'value': 128.6, 'err_minus': 1.8, 'err_plus': 1.6, 'unit': 'Hz', 'source': 'motta2022', 'location': "Table 2, block 'Triplet', row 'Lower HFQPO', column 'Frequency [Hz]'"}, upper = {'value': 227.5, 'err_minus': 2.4, 'err_plus': 2.1, 'unit': 'Hz', 'source': 'motta2022', 'location': "Table 2, block 'Triplet', row 'Upper HFQPO', column 'Frequency [Hz]'"}. Source: not recorded in the entry (see the location and the references below). Location: (no location)
- [ ] `frequencies.upper`: value = 227.5, err_minus = 2.4, err_plus = 2.1 [Hz]. Source: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376). Location: Table 2, block 'Triplet', row 'Upper HFQPO', column 'Frequency [Hz]'
- [ ] `frequencies.lower`: value = 128.6, err_minus = 1.8, err_plus = 1.6 [Hz]. Source: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376). Location: Table 2, block 'Triplet', row 'Lower HFQPO', column 'Frequency [Hz]'
- [ ] `frequencies.low`: value = 3.65, err_minus = 0.01, err_plus = 0.01 [Hz]. Source: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376). Location: Table 2, block 'Triplet', row 'Type-C', column 'Frequency [Hz]'
- [ ] `spin`: value = 0.149, err_minus = 0.005, err_plus = 0.005. Source: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376). Location: Abstract, Sec. 3 ('a BH spin a* = 0.149 +- 0.005') and Table 3
- [ ] `published_fit.mass`: value = 7.85, err_minus = 0.46, err_plus = 0.46 [Msun]. Source: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376). Location: Abstract, Sec. 3 ('a BH mass M = (7.85 +- 0.46) Msun') and Table 3
- [ ] `published_fit.radius`: value = 6.85, err_minus = 0.18, err_plus = 0.18 [G M / c^2]. Source: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376). Location: Sec. 3 ('a QPO emission radius R_QPO = (6.85 +- 0.18) Rg') and Table 3
- [ ] `mass_prior`: value = 7.8, err_minus = 1.9, err_plus = 1.9 [Msun]. Source: Yanes-Rizo, Torres, Casares, Motta, et al. (2022), MNRAS 517, 1476 (doi:10.1093/mnras/stac2719, arXiv:2209.10395). Location: Abstract ('we obtained an updated black hole mass of M_1 = 7.8 +- 1.9 Msun'); the same value is quoted by Motta et al. (2022), Sections 4 and 5

References (check the bibliographic details too):

- [ ] `motta2022`: Motta, Belloni, Stella, Pappas, et al. (2022), MNRAS 517, 1469 (doi:10.1093/mnras/stac2142, arXiv:2209.10376)
- [ ] `yanesrizo2022`: Yanes-Rizo, Torres, Casares, Motta, et al. (2022), MNRAS 517, 1476 (doi:10.1093/mnras/stac2719, arXiv:2209.10395)

TODO items of the file:

- [ ] The dynamical mass was read from the abstract of Yanes-Rizo et al. (2022) only; confirm it in the body of the paper.
- [ ] Human check of Table 2 of Motta et al. (2022) against the published MNRAS version.
- [ ] Human check of the RPM results (abstract, Sec. 3, Table 3) against the published MNRAS version.

Open questions:

- [ ] Published RPM uncertainties are about 2.7 times larger than error propagation gives. Motta et al. (2022; abstract, Sec. 3, Table 3) quote M = 7.85 +- 0.46 Msun, a = 0.149 +- 0.005, R = 6.85 +- 0.18 Rg and state that they follow Motta et al. (2014) at every step. Propagating the Table 2 frequency uncertainties through the same RPM equations (Monte Carlo with split-normal errors, and independently a BlackPy fit with flat priors) gives +- 0.175 Msun, +- 0.0018 and +- 0.067 Rg: smaller by factors 2.6, 2.8 and 2.7, nearly the same factor for all three quantities, while the central values agree. For GRO J1655-40 the same procedure reproduces the published uncertainties. The origin of the larger published errors is not known (see verification/v0.2.md, section 2.7, and examples/06_rpm_published_results.py). The frequencies in this file are unaffected; only the published_fit uncertainties are in question.

- [ ] When every item above is confirmed: set `verified_by_human: true` in `src/blackpy/data/registry/qpo/xte_j1859_226.yaml` by hand and bump nothing else (a changed value bumps `dataset_version`).

## 3. Formulas and sources not fully verified

From `verification/v0.1.md` to `v0.3.md` (and theory.md). For each: read the
source if you can get it, then change the status in `docs/theory.md` and in the
current report in `verification/`.

- [ ] **Einstein ring formula** (`lensing.einstein_ring_angle`): the standard lens
  equation, written **from memory** (marked in the docstring).
- [ ] **Factor 4 of `bambi2019`**: empirical, from the pixels of Fig. 1 of Bambi
  et al. (2019).
- [ ] **Page–Thorne prefactor** 3/(2M) of Eq. (15n): fixed by the Newtonian limit
  because the "3" is unreadable in the scan of Page & Thorne (1974).
- [ ] **Novikov & Thorne (1973)** (Les Houches): not accessed.
- [ ] **Hawking & Ellis (1973)**, section 4.3 (classification of T, energy
  conditions): not accessed; the classification follows Martín-Moruno & Visser
  (2018), which was read.
- [ ] **GMGHS originals** (Gibbons & Maeda 1988; Garfinkle, Horowitz & Strominger
  1991): not accessed; the metric is the a = 0 limit of Sen (1992).
- [ ] **Meng (1994)**, Ann. Stat. 22, 1142: abstract only (posterior predictive
  p-values are conservative).
- [ ] **Bardeen, Press & Teukolsky (1972)**: not accessible; the Kerr ISCO formula
  was read from Motta et al. (2014), Eq. (6); r_mb, r_ph, E, L closed forms and
  the ZAMO tetrad quoted from memory (all reproduced by the general code).
- [ ] **Bardeen (1973)** (Les Houches): not accessible; shadow formulas read from
  Perlick & Tsupko (2022).
- [ ] **Kerr (1963) and Newman et al. (1965)**: originals not read (Kerr from
  Johannsen 2013; Kerr–Newman checked in limits and symbolically).
- [ ] **Original QPO-model papers** (Stella & Vietri 1999; Abramowicz & Kluźniak
  2001; Čadež et al. 2008; Kato 2004, 2008; Bursa 2005; Stuchlík et al. 2013):
  not read; the formulas come from the papers that use them.
- [ ] **From memory in v0.1**: Stuchlík & Hledík (1999) value Lambda = 4/5625
  (reproduced numerically); emcee's 50-tau criterion; Savage–Dickey (Dickey
  1971; Verdinelli & Wasserman 1995); the Kass & Raftery (1995) scale.
- [ ] **RZ convergence rates**: Fig. 1 of Rezzolla & Zhidenko (2014) could not be
  accessed; one statement disagrees by one order.

## 4. Publishing the repository and running CI

- [ ] **Push** this session's commits: `git push origin main` (Claude never
  pushes). Done when GitHub shows the commit "Version 1.0.0rc1".
- [ ] **Watch CI** (Actions tab, workflow "CI"): jobs `lint`, `quick` (Linux,
  macOS, Windows; Python 3.10–3.13), `docs`, `wheel`, `min-deps`. The full suite
  is a separate, manually started workflow (below). The
  workflows have never run on GitHub.
  - `min-deps` fails: open the log; if a dependency resolution fails, compare with
    `.github/min-requirements.txt` (tested locally on Python 3.10.20, macOS
    arm64). If a test fails only there, it is a real incompatibility: raise the
    lower bound in `pyproject.toml` and the pin in `.github/min-requirements.txt`
    to the next version that passes.
  - `quick` fails on Windows only: compare with the last local Windows run
    (below); typical causes are paths and encodings.
  - A `test_docs` failure on Linux in an "Output" block: the printed digits differ
    by round-off between platforms. Round that snippet's print further (in
    `docs/user_guide.md`), then `.venv/bin/python scripts/update_doc_outputs.py
    docs/user_guide.md` and commit.
- [ ] **Run the full suite once** by hand: Actions → "Full test suite" → Run
  workflow (`.github/workflows/full.yml`; manual only, timeout 180 min, because
  the former `full` job of `ci.yml` timed out after 60 min on GitHub). Done when
  it passes; locally it takes about 20 minutes with four workers.
- [ ] **Enable the weekly full run when the repository becomes public**: in
  `.github/workflows/full.yml` uncomment the `schedule:` block (two lines).
- [ ] **macOS runners were not available** on the private repository's last CI
  run. If the `quick` jobs on `macos-latest` stay queued, remove `macos-latest`
  from the `os:` matrix in `.github/workflows/ci.yml` (macOS is covered by the
  local runs in `verification/v1.0.md`) and put it back once runners are
  available or the repository is public.
- [ ] **Run the examples workflow once** by hand (Actions → "Examples" → Run
  workflow); it also runs weekly. Done when all 15 pass.
- [ ] **On the Windows laptop, re-run the check on the final code with a fresh
  `.venv`** (the last Windows run, full suite and all examples passed, tested
  commit 0f23707, i.e. the code before this session):
  ```
  git pull
  Remove-Item -Recurse -Force .venv
  powershell -ExecutionPolicy Bypass -File scripts\check_windows.ps1
  powershell -ExecutionPolicy Bypass -File scripts\check_windows.ps1 -SkipInstall -Full -Examples
  ```
  Then check `build\windows-check\SUMMARY.txt` ("All steps passed."). Optionally
  also `-SkipInstall -ExamplesQuick`.
- [ ] **Enable GitHub Pages**: Settings → Pages → Build and deployment → Source:
  **GitHub Actions**. The workflow `docs.yml` deploys on every push to `main`.
  For a private repository Pages needs a paid plan (GitHub Pro/Team); otherwise
  make the repository public first. Done when
  https://yovkochevpakhlavon.github.io/BlackPy/ shows the documentation; the
  error messages of BlackPy point to its `roadmap/` page.
- [ ] **Decide whether the repository becomes public** (needed for free Pages, for
  PyPI's project links to work for others, and for Zenodo).

## 5. Releasing

- [ ] **Create the PyPI project with trusted publishing.** On pypi.org (logged in):
  Account → Publishing → "Add a new pending publisher": project name
  `blackpy-gr`, owner `YovkochevPakhlavon`, repository `BlackPy`, workflow
  `release.yml`, environment `pypi`. Do the same on test.pypi.org with
  environment `testpypi`. In GitHub: Settings → Environments → create `pypi` and
  `testpypi` (optionally with required reviewers).
- [ ] **Dry run on TestPyPI**: Actions → "Release" → Run workflow (manual runs
  publish to TestPyPI). Then in a fresh environment:
  `python -m pip install -i https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple blackpy-gr==1.0.0rc1`
  and `python -c "import blackpy; print(blackpy.__version__)"`.
- [ ] **Change the version to 1.0.0** (after sections 1–4 are done): in
  `src/blackpy/_version.py` (`__version__ = "1.0.0"`), `CITATION.cff`
  (`version`, `date-released`), `CHANGELOG.md` (rename `[1.0.0rc1]` or add a
  `[1.0.0]` entry with the date), the version row of `CLAUDE.md`, the status line
  of `README.md`; and in `pyproject.toml` the classifier to
  `Development Status :: 5 - Production/Stable`. Run the quick tests, commit.
- [ ] **Tag and release**: `git tag v1.0.0 && git push origin v1.0.0`. The
  release workflow checks that the tag matches `_version.py`, builds and
  publishes to PyPI. Then create the GitHub release from the tag (Releases →
  Draft a new release), pasting the changelog entry. Done when
  `pip install blackpy-gr` installs 1.0.0.
- [ ] **Zenodo**: link the GitHub repository at zenodo.org (GitHub → enable the
  repository) **before** creating the GitHub release, so the release gets a DOI.
  Then add the DOI to `CITATION.cff` (`doi:` and an `identifiers` entry, see the
  commented placeholder) and a badge or citation line in `README.md`; add your
  ORCID to `CITATION.cff` (`orcid:` under the author, placeholder in comments).

## 6. After the release (optional)

- [ ] A JOSS submission (`CONTRIBUTING.md` anticipates it): needs a public
  repository, a `paper.md`, and the release DOI.
- [ ] Announce the package (arXiv software note, mailing lists, a talk).
- [ ] Material for MkDocs announces end of life in May 2027 (message printed by
  `mkdocs build`); its successor Zensical reads `mkdocs.yml`. Plan the switch
  for a 1.x documentation update.
- [ ] Delete this file.
