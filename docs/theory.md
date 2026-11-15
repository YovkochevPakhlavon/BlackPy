# Theory and formulas

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

**Status: transcribed** from Hawking & Ellis, *The Large Scale Structure of
Space-Time* (1973), section 4.3, in the form given for anisotropic fluids by
Curiel, "A primer on energy conditions", Einstein Stud. 13 (2017) 43
(arXiv:1405.0403). *Both references are cited from memory of standard
material; the text was not re-read for this project.*

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
identically, and low-frequency QPOs cannot be modelled in the MVP.

### Checks

- Schwarzschild: \(\Omega_r^2 = \Omega_\phi^2 (1 - 6/r)\), \(r_{\rm ISCO} = 6\).
- Reissner–Nordström: \(r_{\rm ISCO}\) solves
  \(r^3 - 6r^2 + 9Q^2 r - 4Q^4 = 0\) (which follows from the \(h = r^2\)
  condition above); \(r_{\rm ISCO}(Q = 1) = 4\).
- Reissner–Nordström: \(\Omega_r^2 = f^2 V''/(2E^2)\) with
  \(V = f(1 + L^2/r^2)\), derived independently with sympy in the test-suite.
- Schwarzschild–de Sitter: ISCO and OSCO merge at \(\Lambda = 4/5625\)
  (Stuchlík & Hledík 1999, Phys. Rev. D 60, 044006; *the value is quoted from
  memory of that paper and reproduced numerically by BlackPy*).

## 7. Frequencies in physical units

\[ \nu = \frac{\tilde\Omega}{2\pi}\,\frac{c^3}{G M} \]

where \(\tilde\Omega\) is the dimensionless angular frequency (in units of
\(1/M\)). BlackPy uses \(c = 299\,792\,458\) m/s (exact),
\(G = 6.67430\times10^{-11}\) m³ kg⁻¹ s⁻² (CODATA 2018; Tiesinga et al. 2021,
Rev. Mod. Phys. 93, 025010) and the nominal solar mass parameter
\(GM_\odot = 1.3271244\times10^{20}\) m³ s⁻² (IAU 2015 Resolution B3; Prša et
al. 2016, AJ 152, 41). The conversion uses \(GM_\odot\) directly.

Check: the Schwarzschild ISCO orbital frequency of a 10 M☉ black hole is
\(6^{-3/2} c^3/(2\pi\, GM_\odot \cdot 10) = 219.86\) Hz.

## 8. QPO models

**Status: transcribed** from Table 1 of Stuchlík & Kološ (2016), A&A 586,
A130 (arXiv:1603.07366), read from the arXiv version.

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
documentation; *cited from memory*).

**Bayes factors.** With nested sampling, \(\ln B_{10} = \ln Z_1 - \ln Z_0\)
from two runs. With MCMC, the Savage–Dickey density ratio for a nested null
\(\theta = \theta_0\) and factorising priors (Dickey 1971, Ann. Math. Stat.
42, 204; Verdinelli & Wasserman 1995, J. Am. Stat. Assoc. 90, 614; *cited
from memory*):

\[ B_{01} = \frac{p(\theta_0 \mid D)}{\pi(\theta_0)} . \]

The posterior density is a Gaussian kernel estimate; if \(\theta_0\) is on a
prior boundary the samples are reflected about it. If no samples reach
\(\theta_0\), only a lower limit on \(\ln B_{10}\) is reported. The verbal
scale is that of Kass & Raftery (1995), J. Am. Stat. Assoc. 90, 773 (*cited
from memory*): \(2\ln B\) of 0–2 "not worth more than a bare mention", 2–6
"positive", 6–10 "strong", above 10 "very strong".

**Prior dominance.** A parameter is flagged when the Kolmogorov–Smirnov
distance between its marginal posterior and its prior is below 0.1; such a
credible interval reflects the prior and is not a constraint.
