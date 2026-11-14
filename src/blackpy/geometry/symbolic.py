"""Symbolic geometry (sympy) for static, spherically symmetric metrics.

Everything is computed directly from the metric tensor with the textbook
definitions (no closed-form shortcuts), so these functions serve as an
independent check of the closed-form numerical expressions in
:mod:`blackpy.geometry.curvature`.

Conventions
-----------
* Signature ``(-, +, +, +)``, coordinates ``(t, r, theta, phi)``.
* Christoffel symbols
  ``Gamma^a_bc = (1/2) g^ad (d_b g_dc + d_c g_db - d_d g_bc)``.
* Riemann tensor
  ``R^a_bcd = d_c Gamma^a_bd - d_d Gamma^a_bc + Gamma^a_ce Gamma^e_bd - Gamma^a_de Gamma^e_bc``
  (Misner, Thorne & Wheeler 1973), Ricci tensor ``R_bd = R^a_bad``.
* Einstein equations ``G^mu_nu = 8 pi T^mu_nu`` with G = c = 1.
"""

from __future__ import annotations

import sympy as sp


def line_element_tensors(f, g, h, r):
    """Curvature tensors of ``ds^2 = -f dt^2 + dr^2/g + h dOmega^2``.

    Parameters
    ----------
    f, g, h : sympy.Expr
        Metric functions of ``r`` (may be undefined ``sympy.Function`` objects).
    r : sympy.Symbol
        Radial coordinate.

    Returns
    -------
    dict
        ``einstein_mixed`` (4x4 matrix ``G^mu_nu``), ``ricci_scalar`` and
        ``kretschmann`` (``R_abcd R^abcd``), all unsimplified beyond
        ``sympy.simplify``.
    """
    theta = sp.Symbol("theta", positive=True)
    t, phi = sp.symbols("t phi", real=True)
    x = (t, r, theta, phi)
    metric = sp.diag(-f, 1 / g, h, h * sp.sin(theta) ** 2)
    inverse = sp.diag(-1 / f, g, 1 / h, 1 / (h * sp.sin(theta) ** 2))
    n = 4

    gamma = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(b, n):
                # Diagonal metric: only d = a contributes.
                val = (
                    sp.Rational(1, 2)
                    * inverse[a, a]
                    * (
                        sp.diff(metric[a, c], x[b])
                        + sp.diff(metric[a, b], x[c])
                        - sp.diff(metric[b, c], x[a])
                    )
                )
                gamma[a][b][c] = gamma[a][c][b] = sp.simplify(val)

    def riemann_up(a, b, c, d):
        val = sp.diff(gamma[a][b][d], x[c]) - sp.diff(gamma[a][b][c], x[d])
        for e in range(n):
            val += gamma[a][c][e] * gamma[e][b][d] - gamma[a][d][e] * gamma[e][b][c]
        return val

    riem = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(c + 1, n):
                    val = sp.simplify(riemann_up(a, b, c, d))
                    riem[a][b][c][d] = val
                    riem[a][b][d][c] = -val

    ricci = sp.zeros(n, n)
    for b in range(n):
        for d in range(n):
            ricci[b, d] = sp.simplify(sum(riem[a][b][a][d] for a in range(n)))
    ricci_scalar = sp.simplify(sum(inverse[a, a] * ricci[a, a] for a in range(n)))

    einstein_mixed = sp.zeros(n, n)
    for a in range(n):
        for b in range(n):
            val = inverse[a, a] * ricci[a, b] - (sp.Rational(1, 2) * ricci_scalar if a == b else 0)
            einstein_mixed[a, b] = sp.simplify(val)

    # Kretschmann: R_abcd R^abcd with R_abcd = g_aa R^a_bcd (diagonal metric).
    kretschmann = sp.S.Zero
    for a in range(n):
        for b in range(n):
            for c in range(n):
                for d in range(n):
                    comp = riem[a][b][c][d]
                    if comp != 0:
                        kretschmann += (
                            metric[a, a] * inverse[b, b] * inverse[c, c] * inverse[d, d] * comp**2
                        )
    kretschmann = sp.simplify(kretschmann)
    return {
        "einstein_mixed": einstein_mixed,
        "ricci_scalar": ricci_scalar,
        "kretschmann": kretschmann,
    }
