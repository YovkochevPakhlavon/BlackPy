# Theory and formulas

Sections 1 to 11 cover static, spherically symmetric metrics (version 0.1).
Sections 12 to 16 cover stationary, axisymmetric (rotating) metrics
(version 0.2). Sections 17 onwards were added in version 0.3. All formulas
are written in LaTeX so that they can be reused. **The section numbers are
stable**: code and docstrings cite them ("theory section 18").

## Topic index

| Topic | Sections |
|---|---|
| Conventions, units, catalogue | [1](#1-conventions), [12](#12-rotating-metrics-conventions-and-catalogue), [25](#25-further-catalogue-metrics) |
| Horizons, curvature, effective source, energy conditions | [2](#2-horizons), [3](#3-curvature-and-effective-stress-energy), [4](#4-energy-conditions), [13](#13-geometry-of-rotating-metrics) |
| Circular orbits, ISCO, epicyclic frequencies | [5](#5-circular-timelike-orbits), [6](#6-epicyclic-frequencies-and-the-isco), [7](#7-frequencies-in-physical-units), [14](#14-equatorial-circular-orbits-and-epicyclic-frequencies-rotating) |
| QPO models | [8](#8-qpo-models) |
| Photon spheres and shadows | [9](#9-photon-spheres-and-shadow), [15](#15-separable-null-geodesics-and-the-shadow-of-rotating-black-holes), [17](#17-deviation-from-circularity-three-definitions), [19](#19-shadows-in-a-plasma) |
| Numerical methods | [10](#10-numerical-methods) |
| Statistics, fits, model comparison | [11](#11-statistics), [16](#16-statistics-for-rotating-metrics), [18](#18-goodness-of-fit-and-posterior-predictive-checks), [20](#20-several-shadow-sources-m87-and-the-role-of-the-mass-prior) |
| Parametrised metrics (RZ, KRZ) | [21](#21-parametrised-metrics-rezzollazhidenko-and-krz) |
| Fisher forecasts | [22](#22-fisher-matrix-forecasts) |
| Lensing | [23](#23-gravitational-lensing-static-metrics) |
| Thin disks | [24](#24-thin-accretion-disks) |

Every formula used by BlackPy is listed here with its source or derivation
and the test that checks it. Status labels:

- **derived**: derived for BlackPy (derivation below) and checked in known limits;
- **verified symbolically**: confirmed with sympy against the textbook tensor
  definitions, for arbitrary functions \(f, g, h\);
- **transcribed**: copied from the cited source, which was read for this purpose.

## 1. Conventions

- Metric signature \((-,+,+,+)\), coordinates \((t, r, \theta, \phi)\).
- Geometric units \(G = c = 1\). Lengths are in units of the black hole mass
  \(M\), so the Schwarzschild metric is \(f = 1 - 2/r\).
- Line element
  \[ ds^2 = -f(r)\,dt^2 + \frac{dr^2}{g(r)} + h(r)\left(d\theta^2 + \sin^2\theta\,d\phi^2\right), \]
  with defaults \(g = f\), \(h = r^2\). Primes denote \(d/dr\).
- Einstein equations \(G^\mu{}_\nu = 8\pi T^\mu{}_\nu\).
- Christoffel symbols and Riemann tensor as in Misner, Thorne & Wheeler,
  *Gravitation* (1973):
  \(R^a{}_{bcd} = \partial_c\Gamma^a_{bd} - \partial_d\Gamma^a_{bc} + \Gamma^a_{ce}\Gamma^e_{bd} - \Gamma^a_{de}\Gamma^e_{bc}\),
  \(R_{bd} = R^a{}_{bad}\).

## 2. Horizons

A Killing horizon sits where \(g(r) = 0\) and \(f(r) = 0\). BlackPy finds all
roots of \(g\) on the scan range and classifies them by the sign of \(g'\):

| Kind | Condition |
|---|---|
| event | \(g\) goes from negative to positive (static region outside); the outermost such root is *the* event horizon |
| extremal | double root with \(g \ge 0\) on both sides |
| inner | \(g\) goes from positive to negative, inside the event horizon |
| cosmological | \(g\) goes from positive to negative, outside the event horizon |
| not a horizon | \(g = 0\) but \(f \ne 0\) (e.g. a wormhole throat) |

**Event horizon = outermost black hole horizon, not the cosmological
horizon.** The "outermost root" is deliberately *not* used as the definition:
in Schwarzschild–de Sitter the outermost root is the cosmological horizon.

A minimum of \(g\) with \(|g| \le 10^{-10}\) (option `degenerate_atol`) is
reported as an extremal horizon. The exact extremal values of the regular
black holes follow from \(g = g' = 0\): for Bardeen \(g' = 0\) gives
\(r^2 = 2q^2\) and then \(g = 0\) gives \(q_c = 4/(3\sqrt3)\),
\(r_{\rm ext} = 4\sqrt6/9\); for Hayward \(g' = 0\) gives \(r^3 = 4\ell^2\)
and then \(\ell_c = 4/(3\sqrt3)\), \(r_{\rm ext} = 4/3\). Both are derived
with sympy in the test-suite and reproduced by the root finder to
\(10^{-10}\).

Tests: Schwarzschild \(r_h = 2\); Reissner–Nordström
\(r_\pm = 1 \pm \sqrt{1 - Q^2}\), the extremal double root at \(Q = 1\) and
the horizonless case \(Q > 1\); Schwarzschild–de Sitter; pure de Sitter.

## 3. Curvature and effective stress-energy

**Status: derived, verified symbolically** (`tests/test_geometry.py::test_closed_form_curvature_matches_full_tensor_computation`).

Define the four independent orthonormal-frame curvature components

\[
\begin{aligned}
E_1 &= g\left(\frac{f''}{2f} - \frac{f'^2}{4f^2}\right) + \frac{g' f'}{4f}, &
E_2 &= \frac{g f' h'}{4 f h}, \\
E_3 &= g\left(\frac{h''}{2h} - \frac{h'^2}{4h^2}\right) + \frac{g' h'}{4h}, &
E_4 &= \frac{1}{h} - \frac{g h'^2}{4h^2}.
\end{aligned}
\]

Then

\[
G^t{}_t = 2E_3 - E_4, \qquad G^r{}_r = 2E_2 - E_4, \qquad
G^\theta{}_\theta = G^\phi{}_\phi = E_1 + E_2 + E_3,
\]

\[
R = -2E_1 - 4E_2 - 4E_3 + 2E_4, \qquad
K \equiv R_{abcd}R^{abcd} = 4E_1^2 + 8E_2^2 + 8E_3^2 + 4E_4^2 .
\]

The symbolic path (`blackpy.geometry.symbolic`) does not use these closed
forms: it computes Christoffel symbols, Riemann, Ricci and Einstein tensors
directly from the metric. The numerical path evaluates the closed forms with
automatic differentiation.

**Stress-energy.** In the static region (\(f, g > 0\)) the effective source is
an anisotropic fluid \(T^\mu{}_\nu = \mathrm{diag}(-\rho, p_r, p_t, p_t)\):

\[
\rho = -\frac{G^t{}_t}{8\pi}, \qquad p_r = \frac{G^r{}_r}{8\pi}, \qquad p_t = \frac{G^\theta{}_\theta}{8\pi}.
\]

Checks: Schwarzschild gives \(\rho = p_r = p_t = 0\) and \(K = 48/r^6\);
Reissner–Nordström gives the Coulomb field
\(\rho = -p_r = p_t = Q^2/(8\pi r^4)\) (Gaussian-geometrized charge) and
\(K = 48/r^6 - 96Q^2/r^7 + 56Q^4/r^8\); Schwarzschild–de Sitter gives
\(\rho = -p = \Lambda/8\pi\).

## 4. Energy conditions

**Status: transcribed** from Martín-Moruno & Visser, "Classical and
semi-classical energy conditions" (arXiv:1702.05915), section 2.1, where the
conditions for a Hawking–Ellis type I tensor are listed as necessary and
sufficient constraints. That text was read for version 0.3 (2026-11-15). The
original, Hawking & Ellis, *The Large Scale Structure of Space-Time* (1973),
section 4.3, was not accessed. For the anisotropic fluid of a static,
spherically symmetric metric the three principal pressures are
\(p_r, p_t, p_t\):

| Condition | Requirement (\(p_i \in \{p_r, p_t\}\)) |
|---|---|
| NEC | \(\rho + p_i \ge 0\) |
| WEC | NEC and \(\rho \ge 0\) |
| SEC | NEC and \(\rho + p_r + 2p_t \ge 0\) |
| DEC | \(\rho \ge \lvert p_i \rvert\) |

They are evaluated on a radial grid outside the event horizon, with a
round-off tolerance \(10^{-10}/h(r)\). Checks: Schwarzschild and
Reissner–Nordström satisfy all four; a positive cosmological constant
violates only the SEC; the Hayward metric satisfies the WEC but violates the
DEC where \(r^3 > 4\ell^2\), because \(p_t/\rho = 2(r^3 - \ell^2)/(r^3 + 2\ell^2)\).

## 5. Circular timelike orbits

**Status: derived.**

With \(\theta = \pi/2\), the Lagrangian
\(2\mathcal{L} = -f\dot t^2 + \dot r^2/g + h\dot\phi^2\) has the conserved
energy and angular momentum per unit mass

\[ E = f\,\dot t, \qquad L = h\,\dot\phi . \]

The radial geodesic equation for a circular orbit
(\(\dot r = \ddot r = 0\)) is
\(\Gamma^r_{tt}\dot t^2 + \Gamma^r_{\phi\phi}\dot\phi^2 = 0\) with
\(\Gamma^r_{tt} = g f'/2\) and \(\Gamma^r_{\phi\phi} = -g h'/2\), hence the
angular velocity measured by a static observer at infinity:

\[ \boxed{\;\Omega_\phi^2 \equiv \left(\frac{d\phi}{dt}\right)^2 = \frac{f'}{h'}\;} \]

It does not depend on \(g\). The normalisation \(u^\mu u_\mu = -1\) gives
\((u^t)^{-2} = f - h\,\Omega_\phi^2 \equiv D\), therefore

\[ E = \frac{f}{\sqrt{D}}, \qquad L = \frac{h\,\Omega_\phi}{\sqrt{D}}, \qquad D = f - \frac{h f'}{h'} . \]

**Existence.** A circular timelike geodesic exists where
\(f > 0\), \(g > 0\), \(h' > 0\), \(f' \ge 0\) and \(D > 0\). Since
\(D\,h' = f h' - h f'\), the last condition fails at a photon sphere
(section 7); \(f' = 0\) defines a *static radius* (e.g.
\(r = (3/\Lambda)^{1/3}\) in Schwarzschild–de Sitter).

**Marginally bound orbit.** \(E = 1\) is equivalent to
\(B(r) \equiv (f^2 - f)\,h' + h f' = 0\). It is only defined for
asymptotically flat metrics; BlackPy refuses it otherwise.

Checks: Schwarzschild \(\Omega_\phi^2 = r^{-3}\),
\(E = (1 - 2/r)/\sqrt{1 - 3/r}\), \(L = \sqrt{r}/\sqrt{1 - 3/r}\),
\(r_{\rm mb} = 4\); the same quantities in isotropic coordinates
(\(f \ne g\), \(h \ne r^2\)).

## 6. Epicyclic frequencies and the ISCO

**Status: derived**, checked against the Schwarzschild limit, against an
independent effective-potential computation for Reissner–Nordström, and
against Schwarzschild in isotropic coordinates.

### Radial

The normalisation condition for equatorial motion reads

\[ \dot r^2 = g(r)\,U(r), \qquad U(r) = \frac{E^2}{f} - 1 - \frac{L^2}{h} . \]

Differentiating with respect to proper time, \(\ddot r = \tfrac12 (gU)'\).
On a circular orbit \(U = U' = 0\), so a small perturbation
\(r = r_0 + \delta r\) at fixed \(E\) and \(L\) obeys

\[ \ddot{\delta r} = \tfrac12\, g\, U''(r_0)\, \delta r \quad\Rightarrow\quad \omega_r^2 = -\tfrac12\, g\, U'' \]

(per unit proper time). Using \(E^2 = f^2 (u^t)^2\),
\(L^2 = h^2 \Omega_\phi^2 (u^t)^2\) and converting to coordinate time with
\(\Omega_r = \omega_r / u^t\):

\[
\Omega_r^2 = \frac{g}{2}\left[f'' - \frac{2f'^2}{f} + \frac{f'}{h'}\left(\frac{2h'^2}{h} - h''\right)\right]
\]

or, with a common denominator,

\[
\boxed{\;\Omega_r^2 = \frac{g\,N}{2 f h h'}, \qquad
N = f f'' h h' - 2 f'^2 h h' + f f' \left(2h'^2 - h h''\right)\;}
\]

Orbits are radially stable where \(N > 0\). BlackPy scans the polynomial-like
function \(N\), which has no poles at the horizon.

**ISCO.** Marginally stable orbits are the roots of \(N\) at which a circular
orbit exists. A root where \(N\) increases outward (unstable inside, stable
outside) is an inner edge (ISCO); one where \(N\) decreases is an outer edge
(OSCO), as in Schwarzschild–de Sitter. `blackpy.orbits.isco` returns the
outermost inner edge; `marginally_stable_orbits` returns all of them. For
\(h = r^2\) the condition \(N = 0\) reduces to the familiar

\[ r f f'' - 2 r f'^2 + 3 f f' = 0 . \]

### Vertical

The \(\theta\) equation is
\(\frac{d}{d\tau}(h\dot\theta) = h \sin\theta\cos\theta\,\dot\phi^2\).
Linearising about \(\theta = \pi/2\) on a circular orbit gives
\(\ddot{\delta\theta} = -\dot\phi^2\,\delta\theta\), i.e.

\[ \boxed{\;\Omega_\theta = \Omega_\phi\;} \]

exactly, for every spherically symmetric metric. Consequently the nodal
(Lense–Thirring) precession frequency \(\nu_\phi - \nu_\theta\) vanishes
identically, and low-frequency QPOs cannot be modelled with a static metric.
With rotation \(\Omega_\theta \ne \Omega_\phi\); see section 14.

### Checks

- Schwarzschild: \(\Omega_r^2 = \Omega_\phi^2 (1 - 6/r)\), \(r_{\rm ISCO} = 6\).
- Reissner–Nordström: \(r_{\rm ISCO}\) solves
  \(r^3 - 6r^2 + 9Q^2 r - 4Q^4 = 0\) (which follows from the \(h = r^2\)
  condition above); \(r_{\rm ISCO}(Q = 1) = 4\).
- Reissner–Nordström: \(\Omega_r^2 = f^2 V''/(2E^2)\) with
  \(V = f(1 + L^2/r^2)\), derived independently with sympy in the test-suite.
- Schwarzschild–de Sitter: ISCO and OSCO merge at \(\Lambda = 4/5625\)
  (Stuchlík & Hledík 1999, Phys. Rev. D 60, 044006). **Status: verified**
  against the journal version (copy supplied by the owner, read 2026-11-21),
  Sec. II.A: with \(y = \Lambda M^2/3\) (their Eq. 2) the marginally stable
  orbits are \(y_{\rm ms}(r) = (r - 6)/[r^3(4r - 15)]\) (Eq. 23), whose
  maximum is at \(r = 15/2\) with \(y_{c({\rm ms})} = 12/15^4\) (Eq. 24), i.e.
  \(\Lambda = 36/15^4 = 4/5625\). It was also derived independently:
  for \(f = 1 - 2/r - \Lambda r^2/3\) the numerator of \(N\) is
  \(-6\,(4\Lambda r^4 - 15\Lambda r^3 - 3r + 18)\). ISCO and OSCO merge where
  it has a double root in \(r\); solving it together with its \(r\)-derivative
  (sympy) gives two solutions. The merger is
  \(r = 15/2\), \(\Lambda = 4/5625 = 36/15^4\), with \(f = 18/25 > 0\). The
  other solution, \(r = 3\), \(\Lambda = 1/9\), has \(f = 0\): it is the
  degenerate (Nariai) horizon, not a merger of orbits. The bracket is
  \(3\,(4yr^4 - 15yr^3 - r + 6)\), the expression of their Eq. (22). BlackPy
  reproduces the critical value numerically.

## 7. Frequencies in physical units

\[ \nu = \frac{\tilde\Omega}{2\pi}\,\frac{c^3}{G M} \]

where \(\tilde\Omega\) is the dimensionless angular frequency (in units of
\(1/M\)). BlackPy uses \(c = 299\,792\,458\) m/s (exact),
\(G = 6.67430\times10^{-11}\) m³ kg⁻¹ s⁻² (CODATA 2018, Tiesinga et al. 2021,
Rev. Mod. Phys. 93, 025010; unchanged in CODATA 2022) and the nominal solar
mass parameter \(GM_\odot = 1.3271244\times10^{20}\) m³ s⁻² (IAU 2015
Resolution B3, arXiv:1510.07674; Prša et al. 2016, AJ 152, 41, Table 1). The
conversion uses \(GM_\odot\) directly. **Status: verified.** \(c\) and \(G\)
are tested against `scipy.constants` (which ships CODATA 2022 since SciPy
1.15); the nominal \(GM_\odot\) was read from both arXiv texts. This gives
\(c^3/(2\pi GM_\odot) = 32\,312.5\) Hz.

Check: the Schwarzschild ISCO orbital frequency of a 10 M☉ black hole is
\(6^{-3/2} c^3/(2\pi\, GM_\odot \cdot 10) = 219.86\) Hz.

## 8. QPO models

**Status: verified** against Table 1 of Stuchlík & Kološ (2016), A&A 586,
A130 (arXiv:1603.07366), p. 2: first transcribed from the arXiv version,
then compared on 2026-11-21 with the journal version (copy supplied by the
owner). All eleven rows, and the low-frequency column \(\nu_\phi - \nu_\theta\),
agree with the table below and with `blackpy.orbits.qpo`.

| Model | \(\nu_U\) | \(\nu_L\) |
|---|---|---|
| RP (relativistic precession) | \(\nu_\phi\) | \(\nu_\phi - \nu_r\) |
| RP1 | \(\nu_\theta\) | \(\nu_\phi - \nu_r\) |
| RP2 (total precession) | \(\nu_\phi\) | \(\nu_\theta - \nu_r\) |
| ER (epicyclic resonance) | \(\nu_\theta\) | \(\nu_r\) |
| ER1 | \(\nu_\theta\) | \(\nu_\theta - \nu_r\) |
| ER2 | \(\nu_\theta - \nu_r\) | \(\nu_r\) |
| ER3 | \(\nu_\theta + \nu_r\) | \(\nu_\theta\) |
| ER4 | \(\nu_\theta + \nu_r\) | \(\nu_\theta - \nu_r\) |
| ER5 | \(\nu_r\) | \(\nu_\theta - \nu_r\) |
| TD (tidal disruption) | \(\nu_\phi + \nu_r\) | \(\nu_\phi\) |
| WD (warped disc) | \(2\nu_\phi - \nu_r\) | \(2(\nu_\phi - \nu_r)\) |

In all of them the same paper assigns the low-frequency QPO to
\(\nu_\phi - \nu_\theta\). The original proposals, as cited there: RP – Stella
& Vietri (1999), Stella, Vietri & Morsink (1999); RP1 – Bursa (2005); RP2 –
Stuchlík, Kotrlová & Török (2013); ER – Abramowicz & Kluźniak (2001); TD –
Čadež, Calvani & Kostić (2008); WD – Kato (2004, 2008). *These original
papers were not read for this project.*

In spherical symmetry \(\nu_\theta = \nu_\phi\), so RP = RP1 = RP2 = ER1 and
ER3 = TD.

**RP model, checked against the paper that fits it.** Motta et al. (2014),
MNRAS 437, 2554 (arXiv:1309.3652), Sec. 3, identify the upper HFQPO with
\(\nu_\phi\), the lower HFQPO with the periastron precession
\(\nu_{\rm per} = \nu_\phi - \nu_r\) (their Eq. 4) and the type-C QPO with the
fundamental of the nodal precession \(\nu_{\rm nod} = \nu_\phi - \nu_\theta\)
(their Eq. 5), not with \(2\nu_{\rm nod}\) as in the original neutron-star
version; Eq. (7) writes the three as functions of \(r\), \(a\) and \(M\). This
agrees with the RP row of the table and with the low-frequency assignment.
**Status: verified** for the RP model: Sec. 3 and Eqs. (4), (5) and (7) were
read on 2026-11-21 in the journal version (copy supplied by the owner) and in
the arXiv v2 HTML. In both, Eq. (7) prints the prefactor as \(v_\phi\), a
typesetting slip for \(\nu_\phi\).

Check: the 3:2 resonance radii in Schwarzschild are \(r = 27/4\) (RP) and
\(r = 54/5\) (ER).

## 9. Photon spheres and shadow

**Status: derived**; the shadow formula is the standard result reviewed by
Perlick & Tsupko (2022), Phys. Rep. 947, 1 (arXiv:2105.07101). *The review was
not re-read for this project; the formula was re-derived below.*

Null geodesics obey \(\dot r^2 = g\,(E^2/f - L^2/h)\). Circular photon orbits
are the stationary points of \(f/h\):

\[ P(r) \equiv f' h - f h' = 0 . \]

A maximum of \(f/h\) (\(P' < 0\)) is an unstable photon sphere, a minimum
(\(P' > 0\)) a stable one. The critical impact parameter is

\[ b_c = \sqrt{\frac{h(r_{\rm ph})}{f(r_{\rm ph})}} . \]

**Static observer at radius \(r_O\).** A photon with impact parameter
\(b = L/E\) makes an angle \(\alpha\) with the radial direction in the
observer's orthonormal frame, with
\(\sin\alpha = b\sqrt{f(r_O)/h(r_O)}\) (ratio of the tangential to the total
momentum). The edge of the shadow is \(b = b_c\):

\[ \boxed{\;\sin\alpha_{\rm sh} = b_c \sqrt{\frac{f(r_O)}{h(r_O)}}\;} \]

For \(r_O > r_{\rm ph}\), \(\alpha_{\rm sh} = \arcsin(\ldots)\); for
\(r_O < r_{\rm ph}\), \(\alpha_{\rm sh} = \pi - \arcsin(\ldots)\).

**Observer at infinity.** In an asymptotically flat spacetime
\(r_{\rm sh} = \lim r_O\,\alpha_{\rm sh} = b_c\).

**Non-asymptotically-flat metrics.** There is no observer at infinity
(Schwarzschild–de Sitter has a cosmological horizon). `shadow_radius` then
requires a finite `r_obs` and applies to a *static* observer; observers
comoving with the cosmic expansion see a different angle (Perlick, Tsupko &
Bisnovatyi-Kogan 2018, Phys. Rev. D 97, 104062).

Checks: Schwarzschild \(r_{\rm ph} = 3\), \(b_c = 3\sqrt3\), Synge's formula
\(\sin^2\alpha_{\rm sh} = 27(1 - 2/r_O)/r_O^2\); Reissner–Nordström
\(r_{\rm ph} = (3 + \sqrt{9 - 8Q^2})/2\) and \(b_c = 4\) at \(Q = 1\);
Schwarzschild–de Sitter \(r_{\rm ph} = 3\),
\(b_c = 3\sqrt3/\sqrt{1 - 9\Lambda}\); two photon spheres (one stable) for
\(1 < Q^2 < 9/8\).

## 10. Numerical methods

- **Derivatives**: forward-mode automatic differentiation (`jax.jvp`),
  nested for higher orders. No finite differences anywhere.
- **Root finding** (`blackpy._numerics.find_roots`): the function is
  evaluated on a logarithmic scan grid; every sign change is refined with a
  safeguarded Newton iteration (the `rtsafe` scheme of *Numerical Recipes*,
  section 9.4). In addition the extrema of the function are located (roots of
  its derivative) and both sides of each extremum are searched, which
  recovers pairs of roots closer than the grid spacing. An extremum where
  \(\lvert F\rvert \le 10^{-10}\) is reported as a degenerate (double) root.
  Outputs have fixed length (NaN padded) so that everything can be compiled
  and vectorised.
- **Quadrature**: Gauss–Legendre (reserved for lensing integrals).

## 11. Statistics

**Likelihood.** For each QPO source with measured \(\nu_U^{\rm obs}\),
\(\nu_L^{\rm obs}\):

\[ \ln\mathcal{L} = \sum_{k \in \{U, L\}} \left[\ln\frac{\sqrt{2/\pi}}{\sigma_{k,-} + \sigma_{k,+}} - \frac{(\nu_k^{\rm model} - \nu_k^{\rm obs})^2}{2\sigma_k^2}\right], \]

with \(\sigma_k = \sigma_{k,+}\) if the model lies above the measurement and
\(\sigma_{k,-}\) otherwise (split normal; it reduces to the ordinary Gaussian
for symmetric errors). The model frequencies depend on the shared metric
parameters and on the source's own mass \(M_i\) and radius \(r_i\). The
shadow term is Gaussian in \(r_{\rm sh}/M\) (mean \(3\sqrt3(1 + \delta)\),
standard deviation \(3\sqrt3\,\sigma_\delta\)) or a top-hat on the published
interval. The joint log-likelihood is the sum over datasets.

**Convergence (emcee).** The integrated autocorrelation time \(\tau\) is
estimated by emcee; a chain counts as converged if its length exceeds
\(50\,\tau_{\max}\). Burn-in is \(5\,\tau_{\max}\) and thinning
\(\tau_{\min}/2\) (Foreman-Mackey et al. 2013, PASP 125, 306, and the emcee
documentation). **Status of the 50 \(\tau\) criterion: verified** against the
emcee tutorial "Autocorrelation analysis & convergence"
(emcee.readthedocs.io, stable, read 2026-11-21), which finds that chains
longer than about \(50\,\tau\) are often sufficient and advises not to trust
an estimate of \(\tau\) with fewer than \(F\tau\) samples for some
\(F \ge 50\). The burn-in and thinning factors are BlackPy conventions; that
page does not give them (*not checked against a source*).

**Bayes factors.** With nested sampling, \(\ln B_{10} = \ln Z_1 - \ln Z_0\)
from two runs. With MCMC, the Savage–Dickey density ratio for a nested null
\(\theta = \theta_0\) and factorising priors (Dickey 1971, Ann. Math. Stat.
42, 204; Verdinelli & Wasserman 1995, J. Am. Stat. Assoc. 90, 614):

\[ B_{01} = \frac{p(\theta_0 \mid D)}{\pi(\theta_0)} . \]

**Status: verified** (journal versions supplied by the owner, read
2026-11-21). Dickey (1971), Sec. 3, Theorem "Savage's Density Ratio",
Eqs. (3.8)–(3.9), p. 208: if the prior of the other parameters under the null
equals the conditional prior of the full model at \(\theta = \theta_0\), the
likelihood ratio of the null equals the ratio of posterior to prior density
at \(\theta_0\). Verdinelli & Wasserman (1995), Sec. 2.1, Eq. (1), p. 615,
state the same with \(B = m_0/m\) (Sec. 1), i.e. \(B_{01}\), and give the
correction factor when the condition fails. In BlackPy the priors are
specified parameter by parameter (they factorise) and the null fixes
\(\theta\) without changing the other priors, so the condition holds.

The posterior density is a Gaussian kernel estimate; if \(\theta_0\) is on a
prior boundary the samples are reflected about it. If no samples reach
\(\theta_0\), only a lower limit on \(\ln B_{10}\) is reported. The verbal
scale is that of Kass & Raftery (1995), J. Am. Stat. Assoc. 90, 773: \(2\ln B\)
of 0–2 "not worth more than a bare mention", 2–6 "positive", 6–10 "strong",
above 10 "very strong". **Status: verified** against Sec. 3.2, p. 777 (second
table; JSTOR scan supplied by the owner, read 2026-11-21). The paper states
the scale as evidence against \(H_0\) and notes that Bayes factors can equally
give evidence for it; BlackPy applies it to \(2|\ln B_{10}|\) and names the
favoured model.

**Prior dominance.** A parameter is flagged when the Kolmogorov–Smirnov
distance between its marginal posterior and its prior is below 0.1; such a
credible interval reflects the prior and is not a constraint.

## 12. Rotating metrics: conventions and catalogue

**Line element.** Stationary, axisymmetric metrics are written in
Boyer–Lindquist-like coordinates \((t, r, \theta, \phi)\):

\[
ds^2 = g_{tt}\,dt^2 + 2g_{t\phi}\,dt\,d\phi + g_{\phi\phi}\,d\phi^2
       + g_{rr}\,dr^2 + g_{\theta\theta}\,d\theta^2 ,
\]

with all components functions of \((r, \theta)\). Units \(G = c = M = 1\).
Shorthand used below:

\[
A \equiv -g_{tt}, \qquad B \equiv g_{t\phi}, \qquad C \equiv g_{\phi\phi}, \qquad
\varrho^2 \equiv B^2 + AC = g_{t\phi}^2 - g_{tt}g_{\phi\phi} .
\]

\(\varrho^2\) is minus the determinant of the \((t, \phi)\) block; the inverse
metric is \(g^{tt} = -C/\varrho^2\), \(g^{t\phi} = B/\varrho^2\),
\(g^{\phi\phi} = A/\varrho^2\). The squared lapse of zero-angular-momentum
observers is \(N^2 = \varrho^2/C\) and their angular velocity
\(\omega = -B/C\). A static metric is the special case \(B = 0\), \(A = f\),
\(g_{rr} = 1/g\), \(g_{\theta\theta} = h\), \(C = h\sin^2\theta\)
(`StaticSpherical.as_axisymmetric()`).

**Reflection symmetry.** Equatorial orbits require invariance under
\(\theta \to \pi - \theta\). It is checked symbolically (difference of each
component under the substitution) or numerically on a grid.

**Kerr.** **Status: transcribed** from Eq. (1) of Johannsen (2013), Phys.
Rev. D 88, 044002 (arXiv:1501.02809), **and verified symbolically** to solve
\(G^\mu{}_\nu = 0\). With \(\Sigma = r^2 + a^2\cos^2\theta\),
\(\Delta = r^2 - 2r + a^2\):

\[
g_{tt} = -\left(1 - \frac{2r}{\Sigma}\right), \quad
g_{t\phi} = -\frac{2ar\sin^2\theta}{\Sigma}, \quad
g_{rr} = \frac{\Sigma}{\Delta}, \quad
g_{\theta\theta} = \Sigma, \quad
g_{\phi\phi} = \left(r^2 + a^2 + \frac{2a^2r\sin^2\theta}{\Sigma}\right)\sin^2\theta .
\]

**Kerr–Newman.** **Status: derived / verified.** The Kerr components with
\(2r \to 2r - Q^2\), \(\Delta = r^2 - 2r + a^2 + Q^2\). Checks: reduces to
Kerr (\(Q = 0\)) and to Reissner–Nordström (\(a = 0\)); equals the metric
generated from Reissner–Nordström by the procedure below; its source is
traceless (\(R = 0\) symbolically) with \(T^r{}_r = -Q^2/(8\pi\Sigma^2)\).

**Johannsen metric.** **Status: transcribed** from Eqs. (51)–(56) of
Johannsen (2013), at the lowest non-vanishing order of the deviation
functions:

\[
\begin{aligned}
g_{tt} &= -\frac{\tilde\Sigma\,[\Delta - a^2A_2^2\sin^2\theta]}{D^2}, &
g_{t\phi} &= -\frac{a\,[(r^2 + a^2)A_1A_2 - \Delta]\,\tilde\Sigma\sin^2\theta}{D^2}, \\
g_{rr} &= \frac{\tilde\Sigma}{\Delta A_5}, \qquad g_{\theta\theta} = \tilde\Sigma, &
g_{\phi\phi} &= \frac{\tilde\Sigma\sin^2\theta\,[(r^2 + a^2)^2A_1^2 - a^2\Delta\sin^2\theta]}{D^2},
\end{aligned}
\]

\[
D = (r^2 + a^2)A_1 - a^2A_2\sin^2\theta, \quad
A_1 = 1 + \frac{\alpha_{13}}{r^3}, \quad A_2 = 1 + \frac{\alpha_{22}}{r^2}, \quad
A_5 = 1 + \frac{\alpha_{52}}{r^2}, \quad
\tilde\Sigma = r^2 + a^2\cos^2\theta + \frac{\epsilon_3}{r} .
\]

The exterior is regular for (their Eqs. (74), (75), (77), (78))

\[
\alpha_{13},\ \epsilon_3 > -\left(1 + \sqrt{1 - a^2}\right)^3, \qquad
\alpha_{22},\ \alpha_{52} > -\left(1 + \sqrt{1 - a^2}\right)^2 ,
\]

which, with \(|a| \le 1\), is the metric's declared black hole domain. The
event horizon is the Kerr horizon (their Sec. III.1). The metric does not
solve the Einstein equations; BlackPy reports its effective source instead
of assuming one (for \(\alpha_{13} = 0.5\), \(a = 0.5\) it violates all four
energy conditions and is not of Hawking–Ellis type I everywhere).

**Rotating generator.** **Status: transcribed** from Azreg-Aïnou (2014),
Phys. Rev. D 90, 064041 (arXiv:1405.2569), Eqs. (16) and (18). For a static
seed \(ds^2 = -f\,dt^2 + dr^2/g + h\,d\Omega^2\) (their \(G = f\), \(F = g\),
\(H = h\)):

\[
ds^2 = \frac{\Psi}{\rho^2}\left[-\left(1 - \frac{2\tilde f}{\rho^2}\right)dt^2
       + \frac{\rho^2}{\Delta}dr^2 - \frac{4a\tilde f\sin^2\theta}{\rho^2}dt\,d\phi
       + \rho^2d\theta^2 + \frac{\Sigma\sin^2\theta}{\rho^2}d\phi^2\right],
\]

\[
K = \sqrt{g/f}\,h, \quad \rho^2 = K + a^2\cos^2\theta, \quad 2\tilde f = K - gh, \quad
\Delta = gh + a^2, \quad \Sigma = (K + a^2)^2 - a^2\Delta\sin^2\theta .
\]

The conformal factor \(\Psi\) is not fixed by the algorithm. For \(g = f\),
\(h = r^2\) the paper takes \(\Psi = r^2 + a^2\cos^2\theta\) (its Eq. (23)).
BlackPy's default \(\Psi = h + a^2\cos^2\theta\) reproduces this and satisfies
\(\lim_{a\to0}\Psi = h\) in general; for other seeds it is a choice of
BlackPy, not of the paper, and can be replaced. **The result is a model**:
neither field equations nor a physical source are guaranteed. Checks:
Schwarzschild gives Kerr and Reissner–Nordström gives Kerr–Newman (symbolic
identities); for the rotating Hayward metric the energy density and
pressures agree with the paper's Eq. (31),
\(\epsilon = -p_r = 2(r\tilde f' - \tilde f)/\rho^4\),
\(p_\theta = p_\phi = -p_r - \tilde f''/\rho^2\) (in units
\(G_{\mu\nu} = T_{\mu\nu}\)).

## 13. Geometry of rotating metrics

**Horizons.** BlackPy finds the roots of \(g^{rr} = 1/g_{rr}\) at a given
polar angle and classifies them as in section 2, with \(N^2\) in the role of
\(f\) (a root where \(N^2 \ne 0\) is not a horizon). For Kerr-like metrics
\(g^{rr} = \Delta/\Sigma\). The direction of the sign change is read from the
sign of \(g^{rr}\) on both sides of the root, because \(g_{rr}\) is infinite at
the root. In general the horizon \(r = H(\theta)\) solves

\[ g^{rr} + g^{\theta\theta}\left(\frac{dH}{d\theta}\right)^2 = 0 \]

(Johannsen 2013, Eq. (58)), so the surface \(g^{rr} = 0\) is the event horizon
only if it is \(\theta\)-independent; `horizon_profile` reports this, and the
general equation is not solved. Checks: Kerr
\(r_\pm = 1 \pm \sqrt{1 - a^2}\) to \(10^{-12}\); Kerr–Newman
\(r_\pm = 1 \pm \sqrt{1 - a^2 - Q^2}\).

**Ergosurface.** The outermost root of \(g_{tt} = 0\). Kerr: \(g_{tt} = 0
\Leftrightarrow r^2 - 2r + a^2\cos^2\theta = 0\), i.e.
\(r_E = 1 + \sqrt{1 - a^2\cos^2\theta}\) (derived).

**Curvature.** Christoffel symbols, Riemann, Ricci and Einstein tensors from
the definitions of section 1, with the derivatives in \(r\) and \(\theta\) by
automatic differentiation; \(K = R_{abcd}R^{abcd}\). The symbolic path does
the same computation with sympy in the variable \(x = \cos\theta\), in which
Kerr-like metrics are rational functions, under a time limit.

**Kerr Kretschmann scalar.** **Status: derived** with the symbolic path
(test `test_kerr_is_vacuum_and_kretschmann_closed_form`), not quoted:

\[
K = \frac{48\,(r^2 - a^2x^2)\left[(r^2 + a^2x^2)^2 - 16a^2r^2x^2\right]}{(r^2 + a^2x^2)^6},
\qquad x = \cos\theta .
\]

The numerical (autodiff) value agrees with it to \(10^{-10}\). For
Kerr–Newman the same computation gives
\(K = 8\,[\,7Q^4(r^4 + a^4x^4) - 34Q^4a^2r^2x^2 - 12Q^2r(r^4 - 10a^2r^2x^2 + 5a^4x^4)
+ 6(r^2 - a^2x^2)(r^4 - 14a^2r^2x^2 + a^4x^4)\,]/\Sigma^6\).

**Effective source.** \(T^\mu{}_\nu = G^\mu{}_\nu/8\pi\) is classified at
each grid point by its eigenvalue structure in an orthonormal frame
following the Hawking–Ellis classification. **Status: transcribed** from
Martín-Moruno & Visser (2018), "Essential core of the Hawking–Ellis types",
Class. Quantum Grav. 35, 125003 (arXiv:1802.00865), section 2, read for
version 0.3 (the classification was implemented from memory in version 0.2;
the comparison with this source found no difference). The book of Hawking &
Ellis (1973, section 4.3) itself was not accessed. In an orthonormal frame:

| Type | Characterisation (Martín-Moruno & Visser 2018) | Eigenvalues |
|---|---|---|
| I | a timelike eigenvector, which is then unique, and three spacelike ones; diagonalisable | \(\{-\rho, p_1, p_2, p_3\}\) |
| II | a null eigenvector for a double eigenvalue; defective (three eigenvectors) | \(\{-\mu, -\mu, p_2, p_3\}\) |
| III | a null eigenvector for a triple eigenvalue; defective (two eigenvectors) | \(\{-\rho, -\rho, -\rho, p_3\}\) |
| IV | no causal eigenvector; two real (spacelike) eigenvectors only | \(\{-\rho + if, -\rho - if, p_2, p_3\}\) |

BlackPy reports:

| Result | Condition tested |
|---|---|
| vacuum | all frame components below \(10^{-7}\sqrt{K}/8\pi\) |
| perfect fluid | type I (real eigenvalues, a complete set of eigenvectors, exactly one timelike direction among them) with three equal pressures |
| anisotropic fluid | type I with unequal pressures |
| neither | complex eigenvalues (type IV) or a defective matrix (types II, III) |

For type I, \(\rho = -\lambda_{\rm timelike}\) and the principal pressures
\(p_i\) are the other eigenvalues; a degenerate eigenvalue is assigned by the
signature of the metric restricted to its eigenspace. Energy conditions:
NEC \(\rho + p_i \ge 0\); WEC adds \(\rho \ge 0\); SEC adds
\(\rho + \sum_i p_i \ge 0\); DEC \(\rho \ge |p_i|\). Checks: Kerr vacuum;
Kerr–Newman \(\rho = -p_1 = p_2 = p_3 = Q^2/(8\pi\Sigma^2)\); a cosmological
constant is a perfect fluid violating only the SEC; null dust and a tensor
with complex eigenvalues are "neither".

## 14. Equatorial circular orbits and epicyclic frequencies (rotating)

**Status: derived**; checked against the Kerr closed forms, against the
static formulas of sections 5–6, and against the direct effective-potential
definition evaluated symbolically for the Johannsen metric.

**Angular velocity.** For a circular equatorial orbit the radial geodesic
equation is \(\partial_r g_{\mu\nu}u^\mu u^\nu = 0\), i.e.
\(A' - 2B'\Omega - C'\Omega^2 = 0\) with \(\Omega = d\phi/dt\):

\[ \boxed{\;\Omega_\pm = \frac{-B' \pm \sqrt{B'^2 + A'C'}}{C'}\;} \]

(the same expression as Eq. (83) of Johannsen 2013). The upper sign is
called *prograde* (positive \(\phi\) direction), the lower *retrograde*; a
retrograde orbit for spin \(a\) is a prograde orbit for spin \(-a\).

**Energy and angular momentum.** From \(u^\mu u_\mu = -1\),

\[
(u^t)^{-2} = D \equiv A - 2B\Omega - C\Omega^2, \qquad
E = -u_t = \frac{e}{\sqrt D}, \quad L = u_\phi = \frac{\ell}{\sqrt D}, \qquad
e \equiv A - B\Omega, \quad \ell \equiv B + C\Omega .
\]

An orbit exists where \(\varrho^2 > 0\), \(g_{rr} > 0\), \(C' > 0\),
\(B'^2 + A'C' \ge 0\) and \(D > 0\); \(D = 0\) is the equatorial circular
photon orbit.

**Epicyclic frequencies.** With conserved \(E\), \(L\) the normalisation of
the four-velocity reads

\[
g_{rr}\dot r^2 + g_{\theta\theta}\dot\theta^2 = -\,\mathcal U(r, \theta), \qquad
\mathcal U \equiv g^{tt}E^2 - 2g^{t\phi}EL + g^{\phi\phi}L^2 + 1
= 1 - \frac{CE^2 + 2BEL - AL^2}{\varrho^2} .
\]

On the circular orbit \(\mathcal U = \partial_r\mathcal U =
\partial_\theta\mathcal U = 0\) (the last by reflection symmetry).
Differentiating along the orbit and linearising,
\(\ddot{\delta r} = -\frac{\partial_r^2\mathcal U}{2g_{rr}}\delta r\) and
\(\ddot{\delta\theta} = -\frac{\partial_\theta^2\mathcal U}{2g_{\theta\theta}}\delta\theta\)
in proper time; dividing by \((u^t)^2 = 1/D\) gives the frequencies seen from
infinity (the definition of Johannsen 2013, Eqs. (92)–(97)):

\[
\Omega_r^2 = \frac{D\,\partial_r^2\mathcal U}{2g_{rr}}, \qquad
\Omega_\theta^2 = \frac{D\,\partial_\theta^2\mathcal U}{2g_{\theta\theta}} .
\]

Write \(\mathcal U = 1 - n/(\varrho^2 D)\) with
\(n \equiv Ce^2 + 2Be\ell - A\ell^2\) (at fixed \(e\), \(\ell\)). The orbit
conditions are \(n = \varrho^2D\) and \(n' = D(\varrho^2)'\), hence
\(\partial_r^2\mathcal U = [D(\varrho^2)'' - n'']/(\varrho^2D)\) and

\[
\boxed{\;\Omega_r^2 = \frac{N_r}{2g_{rr}\varrho^2}, \qquad
N_r = D(\varrho^2)'' - C''e^2 - 2B''e\ell + A''\ell^2\;}
\]

\[
\boxed{\;\Omega_\theta^2 = \frac{N_\theta}{2g_{\theta\theta}\varrho^2}, \qquad
N_\theta = D(\varrho^2)_{\theta\theta} - C_{\theta\theta}e^2
           - 2B_{\theta\theta}e\ell + A_{\theta\theta}\ell^2\;}
\]

where a subscript \(\theta\theta\) is a second polar derivative at
\(\theta = \pi/2\). These forms have no pole at the photon orbit. For a static
metric \(N_r = N/h'\) with \(N\) of section 6, and
\(N_\theta = 2fh^2\Omega^2\) gives \(\Omega_\theta = \Omega_\phi\).

**ISCO, marginally bound orbit.** The ISCO is the outermost root of \(N_r\)
with \(N_r\) increasing outward at which an orbit exists; the marginally
bound orbit is \(E = 1\), i.e. \(e^2 - D = 0\).

**Kerr closed forms** (\(M = 1\); prograde for \(a > 0\)). **Status:
verified against a published source**: Ingram & Motta (2014), MNRAS 444,
2065 (arXiv:1408.0884), Eqs. (1)–(3), and Motta et al. (2014), MNRAS 437,
2554 (arXiv:1309.3652), Eqs. (1)–(3), both read from the arXiv versions:

\[
\nu_\phi = \frac{1}{2\pi}\,\frac{1}{r^{3/2} + a}, \qquad
\nu_r^2 = \nu_\phi^2\left(1 - \frac{6}{r} + \frac{8a}{r^{3/2}} - \frac{3a^2}{r^2}\right), \qquad
\nu_\theta^2 = \nu_\phi^2\left(1 - \frac{4a}{r^{3/2}} + \frac{3a^2}{r^2}\right) .
\]

BlackPy's general formulas reproduce them to \(10^{-13}\) (\(\nu_\phi\)) and
\(10^{-10}\) (squared epicyclic frequencies).

**Kerr ISCO and the other circular-orbit radii.** **Status: verified**
against Bardeen, Press & Teukolsky (1972), ApJ 178, 347 (ADS scan supplied
by the owner, read 2026-11-21), Sec. II, Eqs. (2.12)–(2.21); upper signs
are prograde. The ISCO was first read from Motta et al. 2014, Eq. (6), and
checked to solve \(1 - 6/r + 8a/r^{3/2} - 3a^2/r^2 = 0\); it is BPT
Eq. (2.21):

\[
r_{\rm ISCO} = 3 + Z_2 \mp \sqrt{(3 - Z_1)(3 + Z_1 + 2Z_2)}, \quad
Z_1 = 1 + (1 - a^2)^{1/3}\left[(1 + a)^{1/3} + (1 - a)^{1/3}\right], \quad
Z_2 = \sqrt{3a^2 + Z_1^2} .
\]

Reproduced to \(3\times10^{-13}\) for \(a \in [-0.999, 0.999]\). The other
closed forms, all reproduced numerically by the general code:
\(r_{\rm mb} = 2 \mp a + 2\sqrt{1 \mp a}\) (BPT Eq. 2.19),
\(r_{\rm ph} = 2\{1 + \cos[\tfrac23\arccos(\mp a)]\}\) (Eq. 2.18),
\(E = (r^{3/2} - 2r^{1/2} \pm a)/[r^{3/4}(r^{3/2} - 3r^{1/2} \pm 2a)^{1/2}]\)
(Eq. 2.12),
\(L = \pm(r^2 \mp 2ar^{1/2} + a^2)/[r^{3/4}(r^{3/2} - 3r^{1/2} \pm 2a)^{1/2}]\)
(Eq. 2.13) and \(\Omega = \pm 1/(r^{3/2} \pm a)\) (Eq. 2.16).

**Nodal precession.** \(\nu_{\rm nod} = \nu_\phi - \nu_\theta\), with
\(\nu_\phi > 0\); positive for co-rotating orbits. QPO models compare
\(|\nu_{\rm nod}|\) with the low-frequency QPO. For `StaticSpherical` metrics
the request is refused, because it vanishes identically.

**Relativistic precession model, analytic solution.** **Status:
transcribed** from Ingram & Motta (2014), Eqs. (4)–(9). With
\(\Gamma = (1 - \nu_{\rm per}/\nu_\phi)^2\),
\(\Delta = (1 - \nu_{\rm nod}/\nu_\phi)^2\):

\[
r = \frac23\,\frac{6 - \Delta - 5\Gamma + 2\sqrt{2(\Delta - \Gamma)(3 - \Delta - 2\Gamma)}}{(\Delta + \Gamma - 2)^2},
\qquad
a = \frac{r^{3/2}}{4}\left[\Delta + \Gamma - 2 + \frac6r\right],
\qquad
M = \frac{c^3}{2\pi GM_\odot}\,\frac{1}{\nu_\phi\,(r^{3/2} + a)} .
\]

Check: it inverts the Kerr frequencies computed by the general code to
\(10^{-9}\).

## 15. Separable null geodesics and the shadow of rotating black holes

**Status: derived**; checked against Bardeen's Kerr formulas and against the
radial potential of the Johannsen metric.

**Separability.** Let \(W = g_{\theta\theta}\). Null geodesics are
conformally invariant, so the Hamilton–Jacobi equation
\(g^{\mu\nu}p_\mu p_\nu = 0\) may be multiplied by \(W\). With
\(p_t = -E\), \(p_\phi = L\):

\[
(Wg^{rr})\,p_r^2 + p_\theta^2 + (Wg^{tt})E^2 - 2(Wg^{t\phi})EL + (Wg^{\phi\phi})L^2 = 0 .
\]

It separates if

\[
Wg^{tt} = T_r(r) + T_\theta(\theta), \quad
Wg^{t\phi} = P_r(r) + P_\theta(\theta), \quad
Wg^{\phi\phi} = Q_r(r) + Q_\theta(\theta), \quad
Wg^{rr} = \Delta_r(r),
\]

a sufficient condition that BlackPy tests numerically (vanishing mixed
derivatives \(\partial_r\partial_\theta\) of the first three and
\(\partial_\theta\) of the last, by automatic differentiation). The split is
fixed by \(T_\theta = P_\theta = Q_\theta = 0\) at \(\theta = \pi/2\). With
\(\xi = L/E\) and the separation constant \(\eta = \mathcal K/E^2\):

\[
\Delta_r\frac{p_r^2}{E^2} = \mathcal R(r) \equiv -T_r + 2P_r\xi - Q_r\xi^2 - \eta, \qquad
\frac{p_\theta^2}{E^2} = \Theta(\theta) \equiv \eta - T_\theta + 2P_\theta\xi - Q_\theta\xi^2 .
\]

For Kerr \(T_r = a^2 - (r^2 + a^2)^2/\Delta\),
\(P_r = a - a(r^2 + a^2)/\Delta\), \(Q_r = 1 - a^2/\Delta\),
\(T_\theta = -a^2\cos^2\theta\), \(P_\theta = 0\), \(Q_\theta = \cot^2\theta\),
which gives the familiar
\(\mathcal R = [(r^2 + a^2 - a\xi)^2 - \Delta((\xi - a)^2 + \eta)]/\Delta\) and
\(\Theta = \eta + a^2\cos^2\theta - \xi^2\cot^2\theta\), with \(\eta\) Carter's
constant. Metrics that fail the test are refused: they need numerical ray
tracing, which is out of scope (an external option is Gradus.jl).

**Spherical photon orbits.** \(\mathcal R = \mathcal R' = 0\) at \(r_p\):
\(Q_r'\xi^2 - 2P_r'\xi + T_r' = 0\). The physical root is the one of smaller
magnitude, written in the numerically stable form

\[
\boxed{\;\xi(r_p) = \frac{T_r'}{P_r' + \mathrm{sgn}(P_r')\sqrt{P_r'^2 - Q_r'T_r'}}, \qquad
\eta(r_p) = -T_r + 2P_r\xi - Q_r\xi^2\;}
\]

(the other root is \(\xi = (r^2 + a^2)/a\) in Kerr, for which \(\Theta < 0\)).
For Kerr this reproduces Bardeen's result, in the form given by Perlick &
Tsupko (2022), Phys. Rep. 947, 1 (arXiv:2105.07101), Eq. (56):

\[
\xi = \frac{-r_p^2(r_p - 3) - a^2(r_p + 1)}{a(r_p - 1)}, \qquad
\eta = \frac{r_p^3\left[4a^2 - r_p(r_p - 3)^2\right]}{a^2(r_p - 1)^2} .
\]

*Bardeen (1973) itself (Les Houches lectures) was not accessible; the
formulas were read from the review.*

**Observer at infinity.** For inclination \(i\) (their Eq. (55)):

\[ \alpha = -\frac{\xi}{\sin i}, \qquad \beta = \pm\sqrt{\Theta(i)} . \]

The contour is traced by \(r_p \in [r_{\min}, r_{\max}]\), the two roots of
\(\Theta(i) = 0\) nearest to the polar orbit \(T_r' = 0\) (which has
\(\xi = 0\) and exists for every spin). Edge-on, \(r_{\min}\) and
\(r_{\max}\) are the prograde and retrograde equatorial photon orbits.

**Slow rotation.** At \(a = 0\) every \(\xi\) belongs to the same radius and
\(\xi(r_p)\) is \(0/0\). With \(m \equiv |P_r'|\,r_s^2\) at the polar orbit
\(r_s\) (\(m = 18a\) for Kerr): for \(m < 10^{-5}\) the shadow is the circle
of radius \(\sqrt{-T_r(r_s)}\), neglecting a shift \(2a\sin i < 1.2\times10^{-6}\);
for \(m < 2\times10^{-4}\) the contour is traced but the shape integrals use
the displaced circle (neglected terms \(O(a^2) < 2\times10^{-10}\)).

**Observer at finite distance.** For a zero-angular-momentum observer at
\((r_O, i)\) the locally measured photon energy is
\(E_{\rm loc} = E(1 - \omega\xi)/N\) and the direction cosines are
\(n_{\hat\phi} = \xi/(\sqrt{C}\,E_{\rm loc}/E)\),
\(n_{\hat\theta} = \pm\sqrt{\Theta(i)}/(\sqrt{g_{\theta\theta}}\,E_{\rm loc}/E)\).
The frame is the locally nonrotating frame of Bardeen, Press & Teukolsky
(1972), Eq. (3.1), \(e_{(t)} = e^{-\nu}(\partial_t + \omega\partial_\phi)\),
\(e_{(\phi)} = e^{-\psi}\partial_\phi\), with \(e^{2\nu} = N^2\),
\(e^{2\psi} = C\) and \(\omega = -g_{t\phi}/g_{\phi\phi}\) (**verified**
against the scan, 2026-11-21). BlackPy returns \(\alpha = -R_On_{\hat\phi}\), \(\beta = R_On_{\hat\theta}\)
with \(R_O = \sqrt{g_{\theta\theta}(r_O, i)}\). These tend to Bardeen's
coordinates as \(r_O \to \infty\) and reduce to
\(\sqrt{h(r_O)}\sin\alpha_{\rm sh}\) of section 9 for a static metric. The
observer must be outside the photon region.

**Shape observables.** With \(\ell(\varphi)\) the distance of the contour
from its geometric centre \((\alpha_G, 0)\),
\(\alpha_G = \int\alpha\,dA/\int dA\):

\[
R_A = \sqrt{A/\pi}, \qquad
\langle\ell\rangle = \frac{1}{2\pi}\oint\ell\,d\varphi, \qquad
\Delta C = \frac{1}{\bar R}\sqrt{\frac{1}{2\pi}\oint(\ell - \bar R)^2d\varphi}, \quad
\bar R^2 = \frac{1}{2\pi}\oint\ell^2d\varphi .
\]

\(\Delta C\) is **transcribed** from Bambi, Freese, Vagnozzi & Visinelli
(2019), Phys. Rev. D 100, 044057 (arXiv:1904.12983), Eqs. (4)–(5). Since
\(\tfrac12\oint\ell^2d\varphi\) is the enclosed area, \(\bar R = R_A\)
exactly and \(\Delta C = \sqrt{2(1 - \langle\ell\rangle/R_A)}\) (derived;
tested). The integrals use the midpoint rule in the contour parameter
\(t\), \(r_p = \tfrac12(r_{\max} + r_{\min}) - \tfrac12(r_{\max} - r_{\min})\cos t\),
which converges exponentially (64 points give \(10^{-10}\)).

Three definitions of the deviation from circularity are in use and are
implemented; see section 17.

Checks: \(a = 0\) gives the circle of radius \(3\sqrt3\); the Kerr contour
agrees with the formulas above to \(10^{-10}\); the horizontal width shrinks
from \(10.39\) to \(9.11\) between \(a = 0\) and \(0.998\) edge-on while the
height stays \(2\cdot3\sqrt3\) to \(10^{-5}\); the Kerr mean radius ranges
from 4.830 to 5.196 (7.0%) over all spins up to 0.998 and all inclinations,
consistent with "the radius ranges from ∼4.8 to ∼5.2 \(GM/c^2\)" and the
"∼8% range" of EHT Collaboration (2022), Sgr A\* Paper VI (arXiv:2311.09484).

## 16. Statistics for rotating metrics

**Triplet likelihood.** For a rotating metric and a QPO triplet a third
split-normal term is added for the low-frequency QPO, with model value
\(|\nu_\phi - \nu_\theta|\) at the same radius. With the RP model this is the
relativistic-precession-model likelihood.

**Spin bounds.** The likelihood is zero outside the metric's declared black
hole domain and where no event horizon exists, so the allowed spin range
depends on the metric and on its other parameters (\(|a| \le 1\) for Kerr,
\(a^2 + Q^2 \le 1\) for Kerr–Newman, the regularity bounds of section 12 for
Johannsen).

**Shadow likelihood.** The model value is the mean radius
\(\langle\ell\rangle\) (optionally \(R_A\)) for an observer at infinity at
inclination \(i\), a nuisance parameter in degrees with default prior
\(p(i) \propto \sin i\) on \([0°, 90°]\).

**Degeneracy guard.** After sampling, the Pearson and Spearman correlation
between the spin and every other sampled metric parameter is recorded; if
either exceeds 0.9 in absolute value a `DegeneracyWarning` is raised.

**Static metric for a rotating source.** Fitting a `StaticSpherical` metric
to a source with a published spin or a nodal-precession QPO raises a
`SpinIgnoredWarning`.

**Coverage.** With true parameters drawn from the priors and Gaussian noise
on the data, the central 68% credible interval contains the truth in 68% of
the cases. A slow test checks this on 100 noisy datasets (observed 73, the
binomial standard deviation being 4.7).

## 17. Deviation from circularity: three definitions

The literature uses different quantities under the name "deviation from
circularity". They differ by factors of 2 to 4, so a bound such as
\(\Delta C < 0.1\) has no meaning without its definition.
`blackpy.light.deviation_from_circularity(..., definition=...)` implements
three; every likelihood or plot that uses \(\Delta C\) takes the definition
explicitly, and a `CircularityDefinitionWarning` is raised when a bound is
combined with a definition other than the one it was derived for.

Notation as in section 15: \(\ell(\varphi)\) is the distance of the contour
from its geometric centre \((\alpha_G, 0)\) and
\(\bar R^2 = \frac{1}{2\pi}\oint\ell^2 d\varphi\) (so \(\bar R = R_A\)).

**`"fractional_rms"` (default). Status: transcribed.**

\[ \Delta C = \frac{1}{\bar R}\sqrt{\frac{1}{2\pi}\oint(\ell - \bar R)^2 d\varphi} . \]

This is Eq. (5) of Bambi, Freese, Vagnozzi & Visinelli (2019), Phys. Rev. D
100, 044057 (arXiv:1904.12983), and Eq. (28) of Shaikh, Pal, Pathak & Joshi
(arXiv:2102.04299), who call it "the fractional RMS distance from the average
radius". Both were read. Largest Kerr value: 0.050 (\(a = 0.998\), edge-on).
It is the default because it is the only one of the three that is fully
specified by a published formula *including* its normalisation.

**`"jp_asymmetry"`. Status: transcribed, with one choice.** Johannsen &
Psaltis (2010), Astrophys. J. 718, 446 (arXiv:1005.1931), Eqs. (3)–(8):

\[
D = \frac{|x'_{\max} + x'_{\min}|}{2}, \qquad
R = \sqrt{(x' - D)^2 + y'^2}, \qquad
\langle R\rangle = \frac{1}{2\pi}\oint R\,d\alpha, \qquad
A = 2\sqrt{\frac{1}{2\pi}\oint(R - \langle R\rangle)^2 d\alpha} .
\]

The centre is the midpoint of the horizontal extent, not the centroid of the
area; \(\alpha\) is the polar angle about that centre (the paper writes
\(\tan\alpha = y'/x'\); BlackPy takes \(x'\) relative to the centre, which is
the only reading for which \(\langle R\rangle\) is a mean radius). The paper
defines \(A\) as a length and remarks that only values "in units of the ring
diameter" \(L = 2\langle R\rangle\) matter. BlackPy returns the dimensionless
\(A/\langle R\rangle\), i.e. twice the rms relative to the mean radius, which
is the quantity that equals 0.10 for a near-extremal Kerr hole seen edge-on
("a maximum asymmetry of 10%", EHT M87\* Paper VI); \(A/L\) is half of it.
Check: the paper's own fit, its Eq. (10), \(A/M = 0.36\,(a/M)^3\sin^{3/2}i\)
for Kerr, agrees with BlackPy's \(A\) within 15% (a test).

**`"bambi2019"`. Status: empirical.** \(4\times\) the fractional rms.
Bambi et al. (2019) print Eq. (5) above, state that "the EHT limit" is
\(\Delta C < 0.1\), and show the excluded region of Kerr parameter space in
their Fig. 1. Their abstract says that the bound excludes Kerr black holes
with \(\theta_{\rm obs} \gtrsim 45^\circ\) for \(0.95 \lesssim a_* \le 1\) and
leaves \(\theta_{\rm obs}\) unbounded for \(a_* \lesssim 0.9\). Eq. (5) as
printed cannot do that: it never exceeds 0.050 for a Kerr black hole. To
find what was plotted, the black curve of Fig. 1 was read from the pixels of
the published figure (192 points; one pixel is 0.003 in \(a_*\) and
\(0.26^\circ\)) and candidate quantities were evaluated along it:

| Quantity | Value along the published curve | Its own 0.1 contour at \(a_* = 0.92,\ 0.95,\ 0.97,\ 0.99\) |
|---|---|---|
| published curve (Fig. 1) | – | \(64.0^\circ,\ 54.1^\circ,\ 48.1^\circ,\ 42.4^\circ\); reaches \(90^\circ\) at \(a_* = 0.881\) |
| fractional rms, Eq. (5) | 0.024 to 0.025 | never reaches 0.1 |
| \(2\times\) fractional rms | 0.048 to 0.051 | only for \(a_* > 0.998\), edge-on |
| **\(4\times\) fractional rms** | **0.096 to 0.101** | \(64.2^\circ,\ 53.5^\circ,\ 47.6^\circ,\ 41.9^\circ\); \(90^\circ\) at \(a_* = 0.889\) |
| unnormalised rms in units of \(M\) | 0.121 to 0.125 | \(51.1^\circ,\ 44.8^\circ,\ 40.8^\circ,\ 36.9^\circ\); \(90^\circ\) at \(a_* = 0.829\) |

Only \(4\times\) the fractional rms reproduces the published curve, to
within the accuracy with which the figure can be read (the angles differ by
0.2° to 0.6°). It also matches the colour scale of the figure, which
saturates at 0.20 in the corner \(a_* \to 1\), \(\theta_{\rm obs} = 90^\circ\):
BlackPy gives 0.20 at \(a_* = 0.998\). The origin of the factor 4 is not known; it is not
any of the three normalisations one would guess (fractional, twice, or
unnormalised). The statements in the text of the paper are looser than its
figure: "for \(a_* = 0.9\), \(\theta_{\rm obs} \gtrsim 70^\circ\) is excluded"
agrees with the figure (72.5°) and with the \(4\times\) quantity (74.5°), but
"for \(a_* = 0.95\), \(\theta_{\rm obs} \gtrsim 45^\circ\)" does not agree
with the figure (54°); 45° happens to be where the *unnormalised* rms reaches
0.1 at that spin, which however misses the figure everywhere else and gives
56°, not 70°, at \(a_* = 0.9\). BlackPy therefore defines `"bambi2019"` by
the figure. It is a reproduction device: use it only to compare with that
paper and with papers that adopted its exclusion region.

**Which definition does a published "10%" bound refer to?** It depends on
the paper, and it must be recorded with the bound (the registry field
`circularity.definition`, section 20). The EHT statement for M87\* is in
Paper I (ApJL 875, L1, sections 7 and 8): the axis ratio of the emission
ring is below 4:3, which "corresponds to a 10% deviation from circularity in
terms of root-mean-square distance from an average radius". This is a
fractional rms (an ellipse with axis ratio 4:3 has fractional rms 0.101), so
the bound is recorded with the definition `"fractional_rms"`, with two
caveats: it refers to the emission ring, not to the shadow boundary, and it
is derived from an axis ratio. Paper VI (section 7.4) measures yet another
quantity, the fractional spread (standard deviation over mean) of the ring
diameter along different orientations. Bambi et al. (2019) compare the same
"10%" with the quantity of their Fig. 1, which is four times the fractional
rms; using `"bambi2019"` with the EHT bound therefore triggers the warning.
With the matching definition the EHT bound excludes no Kerr black hole
(largest Kerr value 0.050).

Tests (`tests/test_circularity.py`): relations between the definitions; the
Johannsen–Psaltis quantity against a direct polygon computation
(\(2\times10^{-5}\)) and against their fit formula; the published curve of
Bambi et al. (angles within 1.5°, edge-on spin within 0.01); the three other
normalisations shown *not* to reproduce it.

## 18. Goodness of fit and posterior predictive checks

A Bayes factor (section 11) ranks models. It does not say whether the better
model describes the data. Two tools answer that; both work on any fit
result, for every data type and for both samplers, because they only need
the model value \(m_k(\theta)\) of every fitted datum \(y_k\).

**Pulls and \(\chi^2\). Status: standard; discrepancy transcribed.** The
pull of datum \(k\) is \((m_k - y_k)/\sigma_k\) with the split-normal
\(\sigma_k\) of the likelihood (\(\sigma_{k,+}\) if \(m_k \ge y_k\), else
\(\sigma_{k,-}\)). Their sum of squares is the \(\chi^2\) discrepancy of
Gelman, Meng & Stern (1996), Statistica Sinica 6, 733, Eq. (8),

\[ X^2(y;\theta) = \sum_k \frac{(y_k - m_k(\theta))^2}{\sigma_k^2}, \]

and \(-2\ln\mathcal{L} = X^2 + \text{const}\) for the likelihood of
section 11. An informative Gaussian prior
\(\mathcal{N}(\mu, s)\) on a parameter \(\theta_i\) (for example an
independent mass measurement) is a measurement of that parameter and is
included as one more term, \((\theta_i - \mu)^2/s^2\). The best-fit point
minimises \(X^2\) inside the prior support (Nelder–Mead, started from the
best posterior sample).

**Degrees of freedom. Status: convention, derived.**

\[ N_{\rm dof} = N_{\rm data} + N_{\rm Gaussian\ priors} - N_{\rm free\ parameters} . \]

Justification: if the model is linear in its parameters and all terms are
Gaussian with the stated widths, then, with the mean of each Gaussian prior
regarded as a noisy measurement of the true parameter value, the vector of
\(N = N_{\rm data} + N_{\rm Gaussian\ priors}\) pulls at the true parameters
is a standard normal vector and minimising over \(P\) parameters removes
\(P\) of its components: \(X^2_{\min} \sim \chi^2_{N-P}\). Uniform and other
non-Gaussian priors are not counted as data (they carry no measurement) but
their parameters are counted as free. Limits of the convention, stated in
the docstrings: it is exact only in the linear Gaussian case; a parameter
that the data do not constrain, or that sits on a prior boundary, uses less
than one degree of freedom, so the count is then a lower bound; a datum with
an interval (top-hat) likelihood has no pull and is not counted.

\(N_{\rm dof}\) **can be zero or negative.** Each QPO source brings its own
mass and orbital radius, so a two-frequency fit with a mass prior and one
metric parameter has \(2 + 1 - 3 = 0\). Then \(X^2_{\min} = 0\) whatever the
data: the fit is exact and says nothing. BlackPy reports "no degrees of
freedom: goodness of fit is undefined", gives no p-value, and never divides
by \(N_{\rm dof}\). For \(N_{\rm dof} > 0\) the classical p-value
\(P(\chi^2_{N_{\rm dof}} \ge X^2_{\min})\) is reported as an approximation.

**Posterior predictive check. Status: transcribed** from Gelman, Meng &
Stern (1996), sections 2.2 to 2.4 (read in full). For posterior draws
\(\theta_j\): (1) draw replicated data \(y^{\rm rep}_j\) from the sampling
distribution at \(\theta_j\) (model values plus split-normal noise with the
stated widths; a replicated measurement \(\mathcal{N}(\theta_{i,j}, s)\) for
each Gaussian prior); (2) compute \(X^2(y;\theta_j)\) and
\(X^2(y^{\rm rep}_j;\theta_j)\). The posterior predictive p-value of the
realized discrepancy (their Eq. (7)) is the fraction of draws with
\(X^2(y^{\rm rep}_j;\theta_j) \ge X^2(y;\theta_j)\). Because
\(X^2(y^{\rm rep};\theta) \sim \chi^2_N\) exactly given \(\theta\), the same
number follows without simulation noise from their Eq. (9),

\[ p_b = \int P\!\left(\chi^2_N \ge X^2(y;\theta)\right) p(\theta \mid y)\,d\theta
       \simeq \frac1J\sum_j P\!\left(\chi^2_N \ge X^2(y;\theta_j)\right), \]

which BlackPy uses for the decision; the simulated estimate is returned as
well and agrees within \(\sqrt{p(1-p)/J}\). The p-value of datum \(k\) is
\(P(y^{\rm rep}_k \ge y_k \mid y)\).

**The p-value is not uniform under the true model.** The data are used
twice. In the discussion printed with Gelman et al. (1996), Rubin (p. 790)
shows that \(p_b\) is "centered at .5 but less variable than a uniform
random variable, and thus will have conservative operating characteristics".
Meng (1994), Ann. Stat. 22, 1142, states in the abstract that the Type I
error of an \(\alpha\)-level posterior predictive test "is often close to but
less than \(\alpha\) and will never exceed \(2\alpha\)". **Status: verified**
against the journal version (JSTOR scan supplied by the owner, read
2026-11-21): Sec. 5, Theorem 1 (p. 1156) and Lemma 1, Eqs. (5.4)–(5.6)
(p. 1157), where the probability is taken under the prior predictive
distribution (5.2), i.e. with the parameters drawn from their prior under
the null. Two consequences are built in:

- a value near 0.5 is *not* evidence of a good fit (with few degrees of
  freedom it is what one always gets; with none, the check is undefined);
- the warning threshold is \(p_b < 0.05\) (`POOR_FIT_P_VALUE`). For a
  conservative p-value a stricter threshold would almost never fire, and at
  0.05 the false-alarm rate stays below the nominal value: 1 of 100
  simulated true-model fits triggered the warning (posterior predictive
  p-values between 0.19 and 0.63 for 98 of them).

**`PoorFitWarning`** is raised by `Fit.run` and by
`FitResult.posterior_predictive` when \(p_b < 0.05\) and \(N_{\rm dof} > 0\).
`Fit.bayes_factor` checks the model that the Bayes factor favours and raises
the warning if it fails; for the Savage–Dickey ratio the null model is not
sampled, so its \(X^2_{\min}\) is found by optimisation and the classical
p-value is used (stated in the result's notes).

Tests (`tests/test_fit_quality.py`, `tests/test_infer.py`): pulls and
\(\chi^2\) at a fixed point against a hand computation; "undefined" for zero
degrees of freedom; a Schwarzschild fit of data simulated from Kerr with
\(a = 0.9\) gives \(X^2_{\min} > 100\) for one degree of freedom and the
warning; false-alarm rate over 60 simulated fits; identical results from
emcee and dynesty; the Bayes-factor path; the bundle round trip.

## 19. Shadows in a plasma

**Sources (all read for version 0.3):** Perlick, Tsupko & Bisnovatyi-Kogan
(2015), Phys. Rev. D 92, 104031 (arXiv:1507.04217), "PTB"; Perlick & Tsupko
(2017), Phys. Rev. D 95, 104003 (arXiv:1702.08768), "PT"; Perlick & Tsupko
(2022), Phys. Rep. 947, 1 (arXiv:2105.07101), section IX.

**Medium.** A cold, non-magnetised, pressureless electron–ion plasma with
refractive index \(n^2 = 1 - \omega_p(x)^2/\omega(x)^2\), where \(\omega\) is
the photon frequency measured by a static observer. Rays are the solutions
of Hamilton's equations for
\(\mathcal{H} = \tfrac12\left(g^{\mu\nu}p_\mu p_\nu + \omega_p^2\right) = 0\).
The constant of motion \(\omega_0 = -p_t\) is the frequency at infinity, and
\(\omega(r) = \omega_0/\sqrt{f}\) (PTB Eq. (13)). BlackPy describes the
plasma by the dimensionless profile

\[ X(r, \theta) = \frac{\omega_p(r, \theta)^2}{\omega_0^2} . \]

Profiles: homogeneous (\(X = \omega_p^2/\omega_0^2\)); power law
\(X = \omega_{p,\rm ref}^2\,(r_{\rm ref}/r)^k\) (PTB Eq. (57) with
\(\beta_0 = \omega_{p,\rm ref}^2\) for \(r_{\rm ref} = M\); the index is named
`k`, not `q`, which is the Bardeen parameter); any radial callable; and the
separable class for rotating metrics (below).

### 19.1 Static metrics. Status: transcribed (PTB), notation \(A = f\), \(D = h\)

\[ h_{\rm pl}(r)^2 = \frac{h}{f}\left(1 - fX\right) \qquad \text{(PTB Eq. (17))} \]

- **Propagation condition** (PTB Eq. (14)): a ray with constant of motion
  \(\omega_0\) exists only where \(fX < 1\), i.e. where the local photon
  frequency exceeds the plasma frequency. A ray with impact parameter \(b\)
  is further restricted to \(h_{\rm pl}(r)^2 \ge b^2\) (PTB Eq. (16)).
- **Photon spheres** (PTB Eq. (32)): \(\tfrac{d}{dr}h_{\rm pl}^2 = 0\); a
  minimum is unstable.
- **Shadow** of a static observer at \(r_O\) outside the photon sphere
  (PTB Eq. (43)):
  \[ \sin^2\alpha_{\rm sh} = \frac{h_{\rm pl}(r_{\rm ph})^2}{h_{\rm pl}(r_O)^2} . \]
  BlackPy returns \(\sqrt{h(r_O)}\sin\alpha_{\rm sh}\) at finite distance
  and, for an asymptotically flat metric, the limit
  \(r_{\rm sh} = h_{\rm pl}(r_{\rm ph})/n_\infty\) with
  \(n_\infty^2 = 1 - X(\infty)\) (\(n_\infty = 1\) unless the plasma extends
  to infinity; \(X(\infty)\) is evaluated at \(r = 10^{12}\)).

**Which photon sphere.** A ray from infinity is captured if and only if
\(b^2 < h_{\rm pl}(r)^2\) everywhere outside the horizon, so the edge of the
shadow is the *smallest* minimum of \(h_{\rm pl}^2\) between the horizon and
the observer (derived from PTB Eq. (16)). For a black hole with a monotonic
profile there is one minimum.

**Refusal.** BlackPy evaluates \(fX\) and \(h_{\rm pl}^2\) on 64 radii
between the photon sphere and the observer (or the outer edge of the scan
range) and raises `PlasmaPropagationError` if \(fX \ge 1\) or
\(h_{\rm pl}^2 < b_c^2\) anywhere, or if \(X(\infty) \ge 1\). Inside a
likelihood such parameter values get zero probability instead.

Checks against the papers: Schwarzschild in a homogeneous plasma,
\(r_{\rm ph} = [3 - 4x + \sqrt{9 - 8x}]/[2(1 - x)]\) with \(x = X\) (solution
of PTB Eq. (33)) and PTB Eq. (46), to \(10^{-11}\); power law with \(k = 2\):
\(r_{\rm ph} = 3\) exactly and \(r_{\rm sh} = \sqrt{27 - \beta_0}\); general
\(k\): the first-order results PTB Eqs. (58)–(59), with an error that scales
as \(\beta_0^2\); PT Eq. (78): for \(k = 1\) the plasma enlarges the shadow
for \(3 < r_O < 6\) and shrinks it beyond, and a homogeneous plasma enlarges
it for every observer.

### 19.2 Rotating metrics. Status: PT's result, re-derived in BlackPy's separability framework

On Kerr the Hamilton–Jacobi equation with plasma separates if and only if
(PT Eq. (26))

\[ \omega_p(r, \theta)^2 = \frac{f_r(r) + f_\theta(\theta)}{r^2 + a^2\cos^2\theta} . \]

In the notation of section 15 (\(W = g_{\theta\theta}\)), multiplying
\(\mathcal{H} = 0\) by \(2W/E^2\) gives

\[ (T_r + T_\theta) - 2(P_r + P_\theta)\xi + (Q_r + Q_\theta)\xi^2
   + \Delta_r\frac{p_r^2}{E^2} + \frac{p_\theta^2}{E^2} + W X = 0 , \]

so the equation separates exactly when \(W X = F_r(r) + F_\theta(\theta)\),
and the plasma then only replaces

\[ T_r \to T_r + F_r, \qquad T_\theta \to T_\theta + F_\theta . \]

For Kerr \(W = r^2 + a^2\cos^2\theta\) and the condition is PT Eq. (26). All
formulas of section 15 (spherical photon orbits, the photon region, the
contour) hold with this replacement, which is how the code implements it:
`with_plasma(metric, plasma)` returns the metric with one extra hook, and
the separable-shadow code is otherwise unchanged. A profile for which
\(\partial_r\partial_\theta(WX) \ne 0\) (tested numerically, tolerance
\(10^{-8}\)) is refused with `NotSeparableError`; this includes every
non-constant profile that depends on \(r\) only (PT; review section IX).
A homogeneous plasma is separable on Kerr-like metrics
(\(F_r = X r^2\), \(F_\theta = X a^2\cos^2\theta\), PT Eq. (36)).

**Observer.** At infinity the celestial coordinates are Bardeen's divided by
the refractive index there: \(\alpha = -\xi/(n_\infty\sin i)\),
\(\beta = \pm\sqrt{\Theta(i)}/n_\infty\). For the ZAMO at finite distance the
locally measured energy \(E_{\rm loc}\) of section 15 is replaced by the
magnitude of the spatial momentum, \(\sqrt{E_{\rm loc}^2 - \omega_p^2}\) (the
ray direction is that of the momentum). **Derived**; it reduces to PTB
Eq. (43) for a static metric, which is tested. (PT use a different observer,
their Eq. (45), and stereographic coordinates, so their figures are not
compared point by point.)

**Propagation.** For seven points of the contour the radial potential
\(\mathcal{R}(r)\) must be non-negative on 64 radii between the spherical
photon orbit and the observer, and \(n^2 > 0\) at the observer.

Checks against PT: the constants of motion of spherical light rays, PT
Eqs. (39)–(40) with the upper sign, for \(f_r = Cr^{3/2}\) and a
\(\theta\)-dependent \(f_\theta\) (with \(\xi = -p_\varphi/\omega_0\) and
\(K = \eta + (\xi - a)^2 + f_\theta(\pi/2)\) relating their constants to
BlackPy's), to \(10^{-10}\); the Schwarzschild case with a
\(\theta\)-dependent plasma, PT Eqs. (69)–(72): the shadow is a circle whose
size depends on the observer's polar angle; the homogeneous model against
PT Eq. (36) written out explicitly; \(X \to 0\) reproduces Kerr in vacuum to
\(10^{-12}\); static metrics sent through the general path reproduce
section 19.1 to \(10^{-11}\).

### 19.3 Inference

`Fit(..., plasma=PowerLawPlasma(k=1.5), priors={"omega_p": ...})` adds the
plasma parameters to the shadow likelihood (kind `"plasma"`). Both a plasma
that thins out and most metric deformations shrink the shadow, so one shadow
size constrains only a combination. The prior-dominance guard covers plasma
parameters, and a `DegeneracyWarning` is raised when the posterior
correlation (Pearson or Spearman) between a plasma parameter and a metric
parameter reaches 0.9; both are tested. Relevance: for Sgr A\* and M87\* at
230 GHz the plasma frequency near the horizon is far below the observing
frequency, so the effect matters only at low radio frequencies.

## 20. Several shadow sources: M87\* and the role of the mass prior

### 20.1 From the M87\* image to a bound on \(r_{\rm sh}/M\). Status: transcribed, chain documented

The registry entry `eht/m87` records each step with its location. Sources
read: EHT Collaboration (2019), Paper I (ApJL 875, L1) and Paper VI (ApJL
875, L6), from the arXiv PDFs; Psaltis et al. (2020), Phys. Rev. Lett. 125,
141104; Kocherlakota et al. (2021), Phys. Rev. D 103, 104047; Gebhardt et
al. (2011); Walsh et al. (2013); Mertens et al. (2016); Walker et al. (2018).

1. **Measured:** the diameter of the emission ring, \(d = 42 \pm 3\,\mu\)as
   (Paper I, Table 1). This is not the shadow.
2. **Calibrated:** the angular gravitational radius
   \(\theta_g = GM/(c^2D) = d/\alpha\), with \(\alpha\) calibrated on GRMHD
   simulations *in the Kerr metric* (Paper I, section 7, Table 1).
3. **Compared** with an independent mass-to-distance ratio
   \(\theta_{\rm dyn}\) through (Paper VI, Eq. (31))
   \[ \delta \equiv \frac{\theta_g}{\theta_{\rm dyn}} - 1 . \]
   The result depends on the mass prior (Paper VI, section 9.5 and
   Table 9): consistent with zero for the stellar-dynamical mass, and
   \(+0.78\) for the gas-dynamical one.
4. **Interpreted** as the fractional deviation of the shadow radius from
   the Schwarzschild value, \(r_{\rm sh}/M = 3\sqrt3\,(1 + \delta)\) (Psaltis
   et al. 2020; Kocherlakota et al. 2021, Eqs. (13)–(14)).

Consequences built into the code:

- A shadow bound in units of \(M\) always assumes a mass prior. Each
  `ShadowDataset` lists its `mass_priors`, each with `independent`
  (true/false), and names the ones its default bound assumes
  (`assumed_mass_priors`). `shadow_mass_prior="gas_dynamics"` selects the
  bound obtained with another prior. If no \(\delta\) was published for the
  chosen prior it is computed from step 3 with the stored \(\theta_g\) and
  \(\theta_{\rm dyn}\), the symmetrised relative uncertainties being added in
  quadrature (a Gaussian approximation).
- A mass obtained from the image itself (`eht_ring` for M87\*) is not
  independent: steps 2 and 3 then compare the image with itself. Using it
  raises a `CircularInferenceWarning`.
- The stored inclination of M87\* is the viewing angle of the *jet*
  (Mertens et al. 2016, Eq. (4)). Equating it with the spin axis is an
  assumption, so the likelihood never uses it by itself;
  `blackpy.infer.inclination_prior(dataset)` turns it into a prior on
  request.
- The circularity bound carries its definition (section 17) and is applied
  only when `circularity="<definition>"` is given: parameter values with a
  larger deviation get zero likelihood. It is a bound, not a measurement,
  so it has no pull and adds no degree of freedom (section 18).

For Sgr A\* the registry now also stores the Keck (Do et al. 2019) and
VLTI (GRAVITY Collaboration 2022) masses and distances whose \(\delta\)
values Vagnozzi et al. (2023) average, with the angular gravitational radii
of EHT Sgr A\* Paper VI (2022). The numbers quoted for the Keck analysis
differ slightly between the three papers; all are stored with their
locations (an open question in the file).

### 20.2 Shared and per-source parameters

Metric parameters are dimensionless, in units of each object's own mass: a
charge-to-mass ratio or a deformation has no reason to be equal for two
black holes, whereas a coupling constant of a theory, expressed in suitable
units, might be. BlackPy does not decide. With two or more shadow datasets
every sampled metric parameter must be listed in `shared=[...]` or in
`per_source=[...]`; otherwise the fit is refused with a message that states
the choice. Per-source copies are named `<name>_<source>`. The spin is per
source unless it is listed in `shared`, and each shadow source has its own
inclination `inclination_deg_<source>` (degrees; isotropic prior by default). The black
hole domain is applied to the parameters of every source. With one shadow
dataset nothing changes with respect to version 0.2.

The joint log-likelihood is the sum over the rows of the data table; row
\(k\) of source \(s\) is compared with the model evaluated with the
parameter vector of that source.

Tests (`tests/test_shadow_sources.py`): schema of both registry files
(structure only, no numbers); the deviation for a chosen mass prior against
a hand computation on a synthetic file; refusal without an explicit choice;
the joint likelihood for static and rotating metrics against hand
computations; the circular-inference and circularity-definition warnings; a
simulated two-source dataset recovers the injected parameters. Reproduced
in `examples/11_m87_and_sgra_shadows.py`: the Reissner–Nordström shadow
leaves the 68% interval of M87\* at \(Q/M = 0.904\); Kocherlakota et al.
(2021) quote \(q \lesssim 0.90\).

## 21. Parametrised metrics: Rezzolla–Zhidenko and KRZ

These are families of metrics, not solutions of field equations. Their
parameters measure deviations from Schwarzschild and Kerr, so a fit
constrains deviation parameters instead of one theory at a time. Whatever
source such a metric would need has to be checked
(`blackpy.geometry.energy_conditions`, `effective_source`), never assumed.

### 21.1 Rezzolla–Zhidenko (static). Status: transcribed

Rezzolla & Zhidenko (2014), Phys. Rev. D 90, 084009 (arXiv:1407.3086), read
for version 0.3; their Eqs. (1)–(7), (14)–(19):

\[
ds^2 = -N^2dt^2 + \frac{B^2}{N^2}dr^2 + r^2d\Omega^2, \qquad
x = 1 - \frac{r_0}{r}, \qquad N^2 = x\,A(x),
\]
\[
A = 1 - \epsilon(1-x) + (a_0 - \epsilon)(1-x)^2 + \tilde A(x)(1-x)^3, \qquad
B = 1 + b_0(1-x) + \tilde B(x)(1-x)^2,
\]
\[
\tilde A = \cfrac{a_1}{1 + \cfrac{a_2x}{1 + \cfrac{a_3x}{1 + \dots}}}, \qquad
\tilde B = \cfrac{b_1}{1 + \cfrac{b_2x}{1 + \cfrac{b_3x}{1 + \dots}}} .
\]

In BlackPy's notation \(f = N^2\), \(g = N^2/B^2\), \(h = r^2\). "Order
\(n\)" keeps \(a_1 \dots a_n\), \(b_1 \dots b_n\) and sets
\(a_{n+1} = b_{n+1} = 0\).

**Unit of length.** The horizon radius \(r_0\) is a parameter of the
expansion and \(\epsilon = 2M/r_0 - 1\) (their Eq. (14)). BlackPy measures
lengths in units of the ADM mass, hence
\[ r_0 = \frac{2}{1 + \epsilon} . \]
This is the event horizon provided \(A(x) > 0\) on \([0, 1]\) (their
Eq. (5)); with only \(\epsilon\) non-zero that requires \(\epsilon < 1/2\).
Otherwise the outer zero of \(A\) is the horizon, which the horizon finder
returns (tested).

**PPN coefficients** (their Eqs. (15)–(18)):
\(a_0 = (\beta - \gamma)(1 + \epsilon)^2/2\),
\(b_0 = (\gamma - 1)(1 + \epsilon)/2\). `ppn="gr"` (default) sets
\(a_0 = b_0 = 0\), which assumes that the solar-system bounds on \(\beta\),
\(\gamma\) (their Eq. (10)) also hold for the black hole; `ppn="free"`
makes them parameters. The choice matters: Reissner–Nordström has
\(a_0 = Q^2/r_0^2 \ne 0\) and cannot be represented with `ppn="gr"`.

**Coefficients of a given metric** (`rz_coefficients`). **Derived.** For
\(ds^2 = -f\,dt^2 + d\rho^2/g + h\,d\Omega^2\) the areal radius is
\(r = \sqrt{h(\rho)}\), so \(N^2 = f\) and \(B^2 = (f/g)\,4h/h'^2\). With the
operator \(D = (dx/d\rho)^{-1}d/d\rho\) (autodiff), the Taylor coefficients
in \(x\) of \(f\), \(g\) and \(4h/h'^2\) at the horizon give those of
\(A = N^2/x\) and of \(B\) by series division, product and square root; the
series of \(\tilde A = [A - 1 + \epsilon(1-x) - (a_0-\epsilon)(1-x)^2](1-x)^{-3}\)
and \(\tilde B\) are converted to continued fractions. The asymptotic
coefficients are \(\epsilon = dA/dx|_{x=1}\),
\(a_0 = \epsilon + \tfrac12 d^2A/dx^2|_{x=1}\), \(b_0 = -dB/dx|_{x=1}\); the
derivatives are evaluated by autodiff at 50, 100, 200 and 400 horizon radii
and extrapolated to \(1 - x = 0\) with the cubic through the four points
(accuracy about \(10^{-10}\); directly at infinity autodiff yields
\(0\cdot\infty\), and at larger radii the second derivative loses accuracy
to cancellation).

Checks:

- all parameters zero is Schwarzschild to \(10^{-12}\);
- **dilaton black hole** (their section IV): the helper reproduces the
  printed \(\epsilon, a_0, b_0, a_1, b_1, b_2\) (their Eqs. (30)–(31)) to
  \(10^{-9}\) and \(a_2\) to \(5\times10^{-6}\) (it is the ratio of two small
  numbers);
- **Reissner–Nordström** is not treated in the paper. **Derived:** with
  \(u = r_0/r\), \(N^2 = 1 - (1+\epsilon)u + (Q^2/r_0^2)u^2\) vanishes at
  \(u = 1\), so \(Q^2/r_0^2 = \epsilon\) and
  \(N^2 = x[1 - \epsilon(1-x)]\): \(a_0 = \epsilon\), all \(a_i = b_i = 0\).
  RN is represented exactly at lowest order; \(\beta = 1 + Q^2/2\);
- **convergence** against the exact dilaton results, their Eqs. (37) and
  (43), for orders 1 to 4. Impact parameter of the photon orbit at
  \(b = \mu/2\): \(9.2\times10^{-5}\), \(1.5\times10^{-5}\),
  \(2.6\times10^{-7}\), \(2.2\times10^{-9}\) (units of \(\mu\)); maximum
  relative error of \(g_{tt}\) at \(b = \mu\): \(2.9\times10^{-4}\),
  \(4.9\times10^{-5}\), \(3.6\times10^{-6}\), \(9.2\times10^{-8}\); ISCO
  frequency at \(b = \mu/2\): \(1.6\times10^{-6}\) down to
  \(2.8\times10^{-10}\). The paper quotes \(\sim10^{-4}\mu\) for the impact
  parameter at \(b \sim \mu/2\) with \(a_2 = 0\), \(3\times10^{-4}\) and
  \(3\times10^{-6}\) for \(g_{tt}\) at \(b = \mu\), and \(\sim10^{-6}\) for the
  ISCO frequency; these agree with orders 1 and 3 here. The paper's
  "\(\sim10^{-6}\mu\) even for \(b \sim \mu\)" for the impact parameter with
  \(a_4 = 0\) is reached here one order later (\(1.2\times10^{-5}\) at order
  3, \(2.2\times10^{-7}\) at order 4); its Fig. 1 could not be accessed to
  compare point by point.

**Published bound reproduced** (example 12): Psaltis et al. (2020) find
\(-1.2 < a_1 < 1.3\) from the M87\* shadow size (17%, \(\epsilon = a_0 = 0\));
BlackPy gives \(-1.23 < a_1 < 1.25\).

### 21.2 KRZ (rotating). Status: transcribed, with two corrections

Konoplya, Rezzolla & Zhidenko (2016), Phys. Rev. D 93, 064015
(arXiv:1602.02378), read for version 0.3. Line element (their Eq. (7)):

\[
ds^2 = -\frac{N^2 - W^2\sin^2\theta}{K^2}dt^2 - 2Wr\sin^2\theta\,dt\,d\phi
       + K^2r^2\sin^2\theta\,d\phi^2
       + \Sigma\left(\frac{B^2}{N^2}dr^2 + r^2d\theta^2\right),
\qquad \Sigma = 1 + \frac{a^2\cos^2\theta}{r^2} .
\]

`blackpy.metric.KRZ()` uses the lowest-order form that the paper recommends
for practical use (its Appendix A, Eqs. (86)–(89), Table 3): reflection
symmetric, second order in \(\cos\theta\), first order of the continued
fractions in the radial direction, with
\(\epsilon_2 = b_{20} = w_{20} = k_{20} = 0\), \(k_{00} = a^2/r_0^2\),
\(w_{00} = 2a/r_0^2\), \(\epsilon_0 = 2/r_0 - 1\) (units of \(M\)). The kept
coefficients are \(\epsilon_0\), \(a_{00}\), \(b_{00}\), \(a_{01}\),
\(b_{01}\), \(w_{01}\), \(a_{20}\), \(a_{21}\), \(k_{21}\), \(w_{21}\),
\(b_{21}\). BlackPy's parameters are their **deviations from the Kerr
values** (`d_eps0`, `d_a20`, `d_a21`, `d_k21`; the others vanish for Kerr),
so that all parameters zero is Kerr; `krz_deviations` converts coefficients
as printed. The explicit functions are in the docstring of `KRZ`.

**Correction 1: the printed lowest-order form is Kerr only on the equator.**
With the Kerr coefficients of Appendix A.1
(\(a_{20} = 2a^2/r_0^3\), \(a_{21} = -a^4/r_0^4\),
\(k_{21} = a^4/r_0^4 - 2a^2/r_0^3\)), Eqs. (86)–(89) give
\(N^2 = \Delta/r^2 + (a^4/r_0^4)(r_0/r)^3(1 - r_0/r)\cos^2\theta\), which is
not the Kerr value off the equatorial plane; at \(r = 3.5\), \(\theta = 0.7\),
\(a = 0.6\) the components \(g_{tt}\), \(g_{\phi\phi}\), \(g_{rr}\) differ from
Kerr by \(3\times10^{-4}\) to \(2\times10^{-3}\) (verified symbolically; a
test). The reason: for Kerr, \(K^2 - aW/r = 1 + [a^2/r^2 +
(a^4/r^4 - 2a^2/r^3)\cos^2\theta]/\Sigma\), and the \(r^{-4}\) term needs the
next two coefficients of the continued fraction of \(\tilde K_2\). BlackPy
keeps the full Kerr function, \(K_2 = a^4/r^4 - 2a^2/r^3 + \delta k_{21}
r_0^3/r^3\), which is a member of the KRZ family (\(k_{22}\), \(k_{23}\) tied
to \(k_{21}\)). Then the \(\cos^2\theta\) term of \(N^2\) is
\([(\delta k_{21} + \delta a_{20})r_0^3/r^3 + \delta a_{21}r_0^4/r^4]\) and
zero deviations give Kerr to \(10^{-12}\) everywhere.

**Correction 2: the sign of \(w_{01}\) of the rotating dilaton black
hole.** The paper prints (Eq. (60f), Appendix A.2)
\(w_{01} = +2a(\mu+b)(b + r_0 - \sqrt{r_0^2 + b^2})/r_0^3\). Evaluating the
paper's own definitions (\(W_0(x) = w_{00}(1-x)^2 + \tilde W_0(1-x)^3\), so
\(w_{01} = W(r_0) - w_{00}\)) on its metric (6) with the coordinate
conditions (16)–(19) gives \(W(r_0) = 2aM\rho_h/r_0^3\) with
\(\rho_h = \sqrt{r_0^2 + b^2} - b\), hence
\(w_{01} = -2aM(b + r_0 - \sqrt{r_0^2 + b^2})/r_0^3\): the opposite sign.
With the corrected sign the lowest-order KRZ metric reproduces the exact
ISCO frequency of the rotating dilaton black hole to \(5\times10^{-5}\)
(\(a = 0.5\), \(b = 0.05\)) and \(1\times10^{-3}\) (\(b = 0.2\)); with the
printed sign the error is 3% and 18%. Both are tested. Only the arXiv
version (ar5iv rendering) was read; the journal version was not compared.

**Properties.** Reflection symmetric, so equatorial orbits, both epicyclic
frequencies and all QPO models are available (tested with a deformed metric
and a triplet likelihood). Null geodesics do not separate for generic
deviations: the numerical separability test fails and the analytic shadow
is refused with `NotSeparableError`; nothing is special-cased. For generic
deviations the surface \(N^2 = 0\) depends on \(\theta\); it is a sphere if
\(\delta a_{20} + \delta a_{21} + \delta k_{21} = 0\) (the paper's condition
\(a_{20} + a_{21} + k_{21} = 0\)), which `horizon_profile` reports.

## 22. Fisher-matrix forecasts

**Status: standard; derived for the likelihood of sections 11 and 16.** For
independent Gaussian data with model values \(m_k(\theta)\) and priors
\(\pi_i\),

\[
F_{ij} = \sum_k \frac{1}{\sigma_k^2}\frac{\partial m_k}{\partial\theta_i}
         \frac{\partial m_k}{\partial\theta_j}
         - \delta_{ij}\frac{\partial^2\ln\pi_i}{\partial\theta_i^2},
\qquad \mathrm{Cov} = F^{-1} .
\]

The first term is minus the Hessian of \(\ln\mathcal{L}\) when the data
equal the model at the fiducial point, so only the uncertainties of a
dataset matter, not its central values. The prior term is \(1/s^2\) for a
Gaussian of width \(s\), zero for a uniform prior, \(-1/\theta^2\) for a
log-uniform one and \((\pi/180)^2/\sin^2 i\) for the isotropic inclination
prior (degrees). The Jacobian \(\partial m_k/\partial\theta_i\) is obtained
by forward-mode automatic differentiation of the same functions the
likelihood uses, root finding included; no finite differences.

**Traceability.** The whole log-likelihood is twice differentiable by
forward-mode autodiff for QPO pairs, QPO triplets and shadow data, with
static and rotating metrics: minus its autodiff Hessian equals \(F\) from
the Jacobian to \(10^{-8}\) (tested). Reverse-mode differentiation does not
work, because the safeguarded Newton iteration is a `while_loop`; this is
why the optional gradient-based sampler of the specification was not added
(see the [roadmap](roadmap.md)).

**Degenerate directions.** \(F\) is diagonalised in units of the prior
widths (half the central 68% interval of each prior). An eigenvalue below
\(10^{-8}\) of the largest is a degenerate direction: it is reported as an
eigenvector in parameter names, and every parameter with a component along
it gets an infinite forecast uncertainty. A singular matrix is a result,
not an error. This complements the correlation-based guard of section 16,
which needs samples and misses non-monotonic degeneracies.

**Limits (stated in the docstring, warned about at run time).** Gaussian
approximation at the fiducial point; invalid where a prior boundary
truncates the posterior (`ForecastWarning` if the fiducial value is within
one forecast \(\sigma\) of a boundary); meaningless for parameters that the
data alone constrain less than the prior does (`PriorDominatedWarning`).

Tests (`tests/test_forecast.py`): a linear Gaussian model, covariance exact
to \(10^{-10}\); a singular case with its degenerate direction; the Hessian
identity above; a Kerr triplet, where the forecast \(\sigma\) of spin, mass
and radius equal the emcee posterior values within 1% (example 13:
0.002255 vs 0.002266 for the spin); the Kerr spin from a shadow size, which
must warn.

## 23. Gravitational lensing (static metrics)

**No data exist for the strong-deflection observables** (relativistic
images have not been resolved), so this part produces predictions, not
constraints. Rotating metrics are not covered.

### 23.1 Exact deflection angle. Status: transcribed formula, derived numerical scheme

For closest approach \(r_0\) (Bozza 2002, Phys. Rev. D 66, 103001,
Eqs. (6)–(7); with a plasma, Perlick, Tsupko & Bisnovatyi-Kogan 2015,
Eq. (20)):

\[
\alpha(r_0) = 2\int_{r_0}^\infty \frac{dr}{\sqrt{g\,h\,[H(r)/H(r_0) - 1]}} - \pi,
\qquad H = \frac{h}{f}\left(1 - fX\right),
\]

with \(X = \omega_p^2/\omega_0^2\) (\(X = 0\) in vacuum) and
\(H(r_0) = (L/E)^2\). Following Tsukamoto (2017), Phys. Rev. D 95, 064035,
Eqs. (28)–(44), the variable \(z = 1 - r_0/r\) turns it into
\(\int_0^1 2r_0\,G^{-1/2}dz\) with
\(G = [H/H_0 - 1]\,g\,h\,(1-z)^4 = c_1z + c_2z^2 + \dots\). The integrand
has a \(z^{-1/2}\) singularity that becomes logarithmic as \(c_1 \to 0\)
(photon sphere). BlackPy integrates \(2r_0/\sqrt{c_1z + c_2z^2}\) in closed
form,

\[
\int_0^1\frac{dz}{\sqrt{c_1z + c_2z^2}} =
\frac{2}{\sqrt{c_1}}\times
\begin{cases}
\operatorname{arcsinh}\sqrt{k}/\sqrt{k}, & k = c_2/c_1 > 0,\\
\arcsin\sqrt{-k}/\sqrt{-k}, & k < 0,
\end{cases}
\]

(the first case is Tsukamoto's Eq. (44); the second is needed in the weak
field, where \(c_2 < 0\)), and the bounded remainder numerically
(Gauss–Legendre in \(s = \sqrt z\)). One more precaution is needed:
\(H(r)/H(r_0) - 1\) is the difference of nearly equal numbers for small
\(z\), so for \(z < 10^{-3}\) the difference between \(G\) and its two
leading terms is taken from the Taylor coefficients \(c_3, c_4\). All
coefficients come from nested forward-mode autodiff. The impact parameter
is \(b = \sqrt{H(r_0)}/n_\infty\), the distance of the asymptote from the
centre.

Checks: Schwarzschild
\(\alpha = 4M/b + \tfrac{15\pi}{4}(M/b)^2 + \tfrac{128}{3}(M/b)^3 +
\tfrac{3465\pi}{64}(M/b)^4\) and Reissner–Nordström
\(A_2 = \tfrac{3\pi}{4}(5 - q^2)\), \(A_3 = \tfrac{128}{3} - 16q^2\)
(Keeton & Petters 2005, Phys. Rev. D 72, 104006, Eqs. (24), (25), (52)),
with residuals of the size of the next term from \(b = 30\) to \(10^6\).

### 23.2 Weak deflection from the Gauss–Bonnet theorem. Status: derived from Gibbons & Werner (2008)

Gibbons & Werner (2008), Class. Quantum Grav. 25, 235009, Eq. (5):
\(\alpha = -\iint_{D}K\,dS\) over the region of the optical manifold
outside the ray, with the straight line \(r = b/\sin\phi\) as boundary to
leading order (their Eq. (14)). For the optical metric
\(dt^2 = E\,dr^2 + \mathcal{G}\,d\phi^2\), \(E = 1/(fg)\),
\(\mathcal{G} = h/f\), the curvature form is
\(K\,dS = -\partial_r[E^{-1/2}\partial_r\sqrt{\mathcal{G}}]\,dr\,d\phi\), so
the radial integral is exact (the bracket tends to 1 at infinity):

\[ \alpha_{\rm GB}(b) = \int_0^\pi\left[1 - \sqrt{fg}\,\frac{d}{dr}\sqrt{\frac{h}{f}}\right]_{r=b/\sin\phi}d\phi . \]

Schwarzschild: \(4M/b\) at first order (their result); at second order the
straight-line boundary gives \(\tfrac{3\pi}{4}(M/b)^2\) instead of the exact
\(\tfrac{15\pi}{4}(M/b)^2\). It is a first-order method, used as an
independent check of the exact integral (tested for Schwarzschild,
Reissner–Nordström and Bardeen).

### 23.3 Strong deflection limit. Status: transcribed (Tsukamoto 2017; Bozza 2002)

\[ \alpha(b) = -\bar a\ln\left(\frac{b}{b_c} - 1\right) + \bar b + O\big((b - b_c)\ln(b - b_c)\big) \]

With \(A = f\), \(B = 1/g\), \(C = h\) at the photon sphere \(r_m\)
(Tsukamoto 2017, Eqs. (54)–(55)):

\[
\bar a = \sqrt{\frac{2B_mA_m}{C_m''A_m - C_mA_m''}}, \qquad
\bar b = \bar a\ln\left[r_m^2\left(\frac{C_m''}{C_m} - \frac{A_m''}{A_m}\right)\right]
         + I_R(r_m) - \pi,
\]
\[ I_R(r_m) = \int_0^1\left[\frac{2r_m}{\sqrt{G(z)}} - \frac{2r_m}{\sqrt{c_2}\,z}\right]dz . \]

BlackPy evaluates them as \(\bar a = r_m/\sqrt{c_2}\) and
\(\bar b = \bar a\ln(r_m^2H_m''/H_m) + I_R - \pi\), identical in vacuum
(at the photon sphere \(H''/H = C''/C - A''/A\)) and valid with a plasma.

Observables for a lens at distance \(D_{OL}\) (Bozza 2002, Eqs. (85), (89),
(90)): \(\theta_\infty = b_c/D_{OL}\),
\(s = \theta_\infty e^{(\bar b - 2\pi)/\bar a}\), \(r = e^{2\pi/\bar a}\),
quoted as \(r_m = 2.5\log_{10}r\). The weak-field Einstein ring is
\(\theta_E = \sqrt{4GM D_{LS}/(c^2D_{OL}D_{OS})}\). **Status: verified**
against Narayan & Bartelmann (1996), *Lectures on Gravitational Lensing*,
astro-ph/9606001v2, Sec. 2.1.4, Eq. (20) (point mass, \(M(\theta_E) = M\);
read 2026-11-21). The distances there are angular-diameter distances, and
in general \(D_{LS} \ne D_{OS} - D_{OL}\) (their Sec. 2.1.3);
`einstein_ring_angle` uses the Euclidean difference, which is adequate for
lenses and sources inside the Galaxy and nearby galaxies, not at
cosmological distances.

Checks:

- Schwarzschild: \(\bar a = 1\),
  \(\bar b = \ln[216(7 - 4\sqrt3)] - \pi = -0.40023\) (Bozza, Eqs. (47)–(50)),
  reproduced to \(10^{-9}\);
- Reissner–Nordström: Tsukamoto's exact \(\bar a\), \(\bar b\), \(b_c\) (his
  section III.2), including the extremal values \(b_c = 4\),
  \(\bar a = \sqrt2\), \(\bar b = 2\sqrt2\ln[4(2 - \sqrt2)] - \pi\), to
  \(2\times10^{-8}\); and every entry of Bozza's Table 1 for
  \(q = Q/2M = 0.1 \dots 0.4\) (\(\bar a\) = 1.005, 1.02, 1.052, 1.123;
  \(\bar b\) = −0.3993, −0.3972, −0.3965, −0.4136) to the printed digits;
- the exact integral approaches the limit as \(b \to b_c\): the difference
  is \(2.5\times10^{-2}\), \(3\times10^{-3}\), \(4\times10^{-5}\),
  \(7\times10^{-8}\) at \(b/b_c - 1 = 10^{-2}, 10^{-3}, 10^{-5}, 10^{-8}\);
  at Bozza's outermost image (\(u - u_m = 0.003264\) in units of \(2M\))
  the exact angle is \(2\pi\) and the limit is off by 0.061% (he quotes
  "about 0.06%");
- Bozza's Table 1 observables for \(M = 2.8\times10^6M_\odot\) at 8.5 kpc:
  \(\theta_\infty = 16.90\,\mu\)as (printed 16.87), \(s = 0.0211\),
  \(r_m = 6.82\).

### 23.4 Deflection in a homogeneous plasma. Status: transcribed

Weak field (Bisnovatyi-Kogan & Tsupko 2010, Mon. Not. R. Astron. Soc. 404,
1790, Eq. (38)), with \(b\) the geometric impact parameter:

\[ \hat\alpha = \frac{R_S}{b}\left(1 + \frac{1}{1 - \omega_e^2/\omega^2}\right) . \]

The exact angle of section 23.1 with \(X = \omega_e^2/\omega^2\) reproduces
it with a relative error of order \(M/b\) (0.3% at \(b = 1000M\)), reduces
to the vacuum angle for \(X = 0\), and approaches its own strong deflection
limit (all tested).

## 24. Thin accretion disks

**Sources:** Page & Thorne (1974), Astrophys. J. 191, 499 ("PT"), read from
the scanned journal version at the NASA ADS (the text was extracted from a
scan; see the note on the prefactor below). Novikov & Thorne (1973, in
*Black Holes*, Les Houches) was not accessible.

**Radiative efficiency.** \(\eta = 1 - E(r_{\rm in})\), with \(E = -u_t\) of
the circular geodesic at the inner edge, by default the ISCO. For Kerr
(PT Eq. (15g), \(x^2 = r/M\)):
\(E = (1 - 2x^{-2} + a x^{-3})/\sqrt{1 - 3x^{-2} + 2a x^{-3}}\).
Schwarzschild: \(1 - \sqrt{8/9} = 0.0572\); prograde Kerr tends to
\(1 - 1/\sqrt3 = 0.4226\) as \(a \to 1\) (approached slowly: 0.340 at
\(a = 0.999\), 0.403 at \(a = 0.99999\)).

**Flux. Status: transcribed (PT Eqs. (11b), (12)); the metric factor is
derived.** PT write the metric near the equator as
\(ds^2 = -e^{2\nu}dt^2 + e^{2\psi}(d\phi - \omega dt)^2 + e^{2\mu}dr^2 + dz^2\)
(their Eq. (1)) and find

\[
F(r) = \frac{\dot M_0}{4\pi}e^{-(\nu+\psi+\mu)}f, \qquad
f = -\frac{\Omega_{,r}}{(E - \Omega L)^2}\int_{r_{\rm in}}^r(E - \Omega L)\,L_{,r}\,dr .
\]

Comparing with BlackPy's components at \(\theta = \pi/2\):
\(e^{2\psi} = g_{\phi\phi}\), \(e^{2\mu} = g_{rr}\), and the lapse
\(e^{2\nu} = (g_{t\phi}^2 - g_{tt}g_{\phi\phi})/g_{\phi\phi}\), so

\[ e^{\nu+\psi+\mu} = \sqrt{(g_{t\phi}^2 - g_{tt}g_{\phi\phi})\,g_{rr}}, \]

which is \(\sqrt{fh/g}\) for a static metric and \(r\) for Kerr (PT
Eq. (15d)). The vertical coordinate \(z\) of PT is proper height, so the
result does not depend on \(g_{\theta\theta}\). \(L_{,r}\) and
\(\Omega_{,r}\) come from autodiff, the integral from Gauss–Legendre
quadrature in \(\ln r\). Units: \(F M^2/\dot M_0\). The effective
temperature profile is \(F^{1/4}\) (\(\sigma T_{\rm eff}^4 = F\)),
returned divided by its maximum; the radius of maximum flux is the zero of
\(dF/dr\) (autodiff) outside the inner edge.

**Kerr closed form (PT Eqs. (14), (15n)).** With \(x_0 = \sqrt{r_{\rm ms}}\)
and \(x_{1,2,3}\) the roots of \(x^3 - 3x + 2a = 0\),

\[
f = \frac{3}{2M}\frac{1}{x^2(x^3 - 3x + 2a)}\Bigl[x - x_0 - \tfrac32 a\ln\frac{x}{x_0}
 - \sum_{i}\frac{3(x_i - a)^2}{x_i(x_i - x_j)(x_i - x_k)}\ln\frac{x - x_i}{x_0 - x_i}\Bigr] .
\]

The factor 3 of the prefactor is lost in the scanned text; it was fixed by
the Newtonian limit \(F \to 3\dot M_0/(8\pi r^3)\), which the general
formula reproduces independently. With it, the general formula agrees with
the closed form to \(10^{-9}\) for spins \(-0.7 \dots 0.998\) from
\(1.001\,r_{\rm ISCO}\) to \(1000\,r_{\rm ISCO}\), which confirms the
remaining terms read from the scan.

**Inner edge.** The ISCO. A metric with no ISCO (e.g. Reissner–Nordström
with \(Q \ge 1.125\)) or with several marginally stable orbits that are
inner edges of stable regions is refused; the user can pass `r_in`
explicitly.

Checks (`tests/test_disk.py`): efficiencies against PT Eq. (15g) at the
Bardeen–Press–Teukolsky ISCO up to \(a = 0.9999\), retrograde = opposite
spin, monotonic approach to \(1 - 1/\sqrt3\); the Kerr flux against
Eq. (15n) on a grid of radii and spins, including the radius and value of
the maximum; zero flux at the inner edge; \(F r^3 \to 3/(8\pi)\) with a
correction \(\propto r^{-1/2}\); the general (rotating) code through
`as_axisymmetric()` equals the static code to \(10^{-10}\).

**Not provided:** continuum-fitting and iron-line likelihoods. They need the
image of the disk on the sky (light bending, Doppler and gravitational
shifts), i.e. ray tracing, which is outside BlackPy's scope.

## 25. Further catalogue metrics

GMGHS, Simpson–Visser, Kerr–Sen, rotating Bardeen and rotating Hayward
(added in version 0.3).

Each entry is one file in `src/blackpy/metric/catalogue/`, with its
reference and tests in `tests/test_catalogue_v03.py`.

**GMGHS** (`GMGHS`, static). Gibbons & Maeda (1988) and Garfinkle, Horowitz
& Strominger (1991) were not accessible when the metric was added (both
predate arXiv; Gibbons & Maeda was read later, see below). The metric is the
\(a = 0\) limit of Sen's Einstein-frame metric (Sen 1992, hep-th/9204046,
read), which Kocherlakota et al. (2021, footnote 2) identify with this
solution. With \(M = 1\), Sen's Eqs. (17)–(18) give \(m = 1 - Q^2/2\) and
\(m\sinh^2(\alpha/2) = M - m = Q^2/2 \equiv b\), hence

\[ ds^2 = -\frac{\rho - 2m}{\rho + 2b}dt^2 + \frac{\rho + 2b}{\rho - 2m}d\rho^2 + \rho(\rho + 2b)\,d\Omega^2 , \]

the form of Rezzolla & Zhidenko (2014, section IV) with \(\mu = m\). Here
\(h \ne \rho^2\). Checks: Schwarzschild at \(Q = 0\); horizon \(\rho = 2m\);
photon-orbit impact parameter and ISCO frequency of RZ Eqs. (37), (43) to
\(10^{-10}\).

**Status: verified** against Gibbons & Maeda (1988), Nucl. Phys. B 298, 741
(copy supplied by the owner, read 2026-11-21), Sec. 4.1 (electrically
charged case), Eqs. (4.15)–(4.19), with \(D = 4\), \(g_2 = 1\) and
\(\kappa^2 = 8\pi G\) (their Sec. 2). Eq. (4.17) then reads
\(ds^2 = -\frac{r - r_H}{r + \alpha}dt^2 + \frac{r + \alpha}{r - r_H}dr^2 + r(r + \alpha)\,d\Omega^2\),
the line element above with \(r = \rho\), \(r_H = 2m\), \(\alpha = 2b\).
Eqs. (4.18)–(4.19) give \(r_H + \alpha = 2M\), i.e. \(m + b = M\), and
Eq. (4.15) gives \(Q^2_{\rm GM} = 4\pi M\alpha\) in their rationalised units,
i.e. \(\alpha = Q^2/M\) for the Gaussian charge used here: \(b = Q^2/2\) at
\(M = 1\) (checked with sympy). Garfinkle, Horowitz & Strominger (1991) and
its erratum **were not accessed**.

**Simpson–Visser** (`SimpsonVisser`, static). Simpson & Visser (2019),
JCAP 02, 042 (arXiv:1812.07114, read), Eq. (2.1):
\(f = g = 1 - 2/\sqrt{r^2 + \ell^2}\), \(h = r^2 + \ell^2\) (their \(a\) is
called `ell`). Checks against their section 7: horizon
\(\sqrt{4 - \ell^2}\), photon sphere \(\sqrt{9 - \ell^2}\), ISCO
\(\sqrt{36 - \ell^2}\), none of them in \(r > 0\) beyond \(\ell = 2, 3, 6\);
the wormhole configurations are reported as horizonless. Derived: the
shadow radius is \(3\sqrt3\) for every \(\ell < 3\).

**Kerr–Sen** (`KerrSen`, rotating). Sen (1992), Einstein-frame metric
(his Eq. (16) in the arXiv numbering) with \(M = 1\):
\(\Sigma = \rho^2 + a^2\cos^2\theta + 2b\rho\),
\(g_{tt} = -(\rho^2 + a^2\cos^2\theta - 2m\rho)/\Sigma\),
\(g_{t\phi} = -2\rho a\sin^2\theta/\Sigma\),
\(g_{\rho\rho} = \Sigma/(\rho^2 + a^2 - 2m\rho)\), \(g_{\theta\theta} = \Sigma\),
\(g_{\phi\phi} = [(\rho^2+a^2)(\rho^2+a^2\cos^2\theta) + 2m\rho a^2\sin^2\theta
+ 4b\rho(\rho^2+a^2) + 4b^2\rho^2]\sin^2\theta/\Sigma\). Checks: equal to the
form of KRZ (2016) Eq. (6) to \(10^{-12}\); Kerr at \(Q = 0\); GMGHS at
\(a = 0\); horizons \(\rho_\pm = m \pm \sqrt{m^2 - a^2}\) (Sen Eq. (20));
null geodesics separate (tested numerically, not assumed); declared domain
\(|a| \le 1 - Q^2/2\).

**Rotating Bardeen and rotating Hayward** (`RotatingBardeen`,
`RotatingHayward`). Bambi & Modesto (2013), Phys. Lett. B 721, 329
(arXiv:1302.6075, read): the Kerr metric with \(M \to m(r)\), their
Eq. (14) with Eqs. (26) (Bardeen, \(m = (r^2/(r^2+q^2))^{3/2}\)) and (15)
(Hayward, \(m = r^3/(r^3 + g^3)\), \(g^3 = 2l^2\) for BlackPy's `l`). Built
with `rotating_from_static`, which for these seeds gives exactly that
metric (tested to \(10^{-12}\)). They are **generated models**; the
docstrings say so. Declared domain \(|a| \le 1\): necessary for a horizon
because \(m(r) \le 1\) implies \(\Delta \ge r^2 - 2r + a^2\), not
sufficient; the likelihood also requires an event horizon.
