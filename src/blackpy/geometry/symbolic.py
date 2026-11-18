"""Symbolic geometry (sympy): static, spherically symmetric metrics and the full
4D computation for stationary, axisymmetric metrics.

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

import signal
import threading
import time

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


# ----------------------------------------------------------------------------- 4D, axisymmetric


class _Deadline:
    """Cooperative time limit, with a hard interrupt (SIGALRM) where available."""

    def __init__(self, seconds: float | None):
        self.seconds = seconds
        self.end = None if seconds is None else time.monotonic() + seconds
        self._previous = None

    def check(self):
        if self.end is not None and time.monotonic() > self.end:
            raise TimeoutError

    def _alarm(self, signum, frame):
        raise TimeoutError

    def __enter__(self):
        usable = (
            self.seconds is not None
            and hasattr(signal, "setitimer")
            and threading.current_thread() is threading.main_thread()
        )
        if usable:
            self._previous = signal.signal(signal.SIGALRM, self._alarm)
            signal.setitimer(signal.ITIMER_REAL, self.seconds)
        return self

    def __exit__(self, *exc):
        if self._previous is not None:
            signal.setitimer(signal.ITIMER_REAL, 0.0)
            signal.signal(signal.SIGALRM, self._previous)
        return False


def axisymmetric_tensors(g_tt, g_tphi, g_phiphi, g_rr, g_thth, r, theta, timeout=None):
    r"""Curvature tensors of a stationary, axisymmetric metric (full 4D, symbolic).

    The computation is done in the coordinate :math:`x = \cos\theta`, in
    which Kerr-like metrics are rational functions of ``(r, x)``; rational
    expressions are simplified with ``sympy.cancel``, which is fast and
    decides reliably whether a component vanishes. Results are returned in
    the coordinates ``(t, r, theta, phi)``.

    Parameters
    ----------
    g_tt, g_tphi, g_phiphi, g_rr, g_thth : sympy.Expr
        Covariant components as functions of ``r`` and ``theta``.
    r, theta : sympy.Symbol
    timeout : float or None
        Time limit in seconds. The limit is checked between tensor
        components everywhere, and enforced with an interrupt on platforms
        that have ``SIGALRM`` (Linux, macOS) when called from the main
        thread. ``None`` means no limit.

    Returns
    -------
    dict
        ``einstein_mixed`` (4x4 matrix ``G^mu_nu``), ``ricci_scalar`` and
        ``kretschmann``.

    Raises
    ------
    TimeoutError
        If the time limit is exceeded.
    """
    x = sp.Symbol("x_cos_theta", real=True)
    to_x = {sp.sin(theta): sp.sqrt(1 - x**2), sp.cos(theta): x}
    exprs = [sp.sympify(e).subs(to_x) for e in (g_tt, g_tphi, g_phiphi, g_rr, g_thth)]
    if any(e.has(theta) for e in exprs):
        raise ValueError(
            "axisymmetric_tensors: theta must enter the metric only through sin(theta) and "
            "cos(theta). Rewrite the components in that form (e.g. tan(theta) as "
            "sin(theta)/cos(theta)), or use the numerical functions (blackpy.geometry.kretschmann, "
            "effective_source)."
        )
    g_tt, g_tphi, g_phiphi, g_rr, g_thth = exprs
    coords = (sp.Symbol("t", real=True), r, x, sp.Symbol("phi", real=True))
    n = 4

    def simp(expr):
        expr = sp.sympify(expr)
        if expr == 0:
            return sp.S.Zero
        if expr.is_rational_function(r, x):
            return sp.cancel(sp.together(expr))
        return sp.simplify(expr)

    with _Deadline(timeout) as deadline:
        metric = sp.zeros(n, n)
        metric[0, 0], metric[0, 3], metric[3, 0], metric[3, 3] = g_tt, g_tphi, g_tphi, g_phiphi
        metric[1, 1], metric[2, 2] = g_rr, simp(g_thth / (1 - x**2))
        det = simp(g_tt * g_phiphi - g_tphi**2)
        inverse = sp.zeros(n, n)
        inverse[0, 0], inverse[3, 3] = simp(g_phiphi / det), simp(g_tt / det)
        inverse[0, 3] = inverse[3, 0] = simp(-g_tphi / det)
        inverse[1, 1], inverse[2, 2] = 1 / g_rr, 1 / metric[2, 2]

        dmetric = [
            [[sp.diff(metric[a, b], coords[c]) for c in range(n)] for b in range(n)]
            for a in range(n)
        ]
        gamma = [[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(b, n):
                    deadline.check()
                    val = sum(
                        inverse[a, d] * (dmetric[d][c][b] + dmetric[d][b][c] - dmetric[b][c][d])
                        for d in range(n)
                        if inverse[a, d] != 0
                    )
                    gamma[a][b][c] = gamma[a][c][b] = simp(val / 2)

        riem = [[[[sp.S.Zero] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
        for a in range(n):
            for b in range(n):
                for c in range(n):
                    for d in range(c + 1, n):
                        deadline.check()
                        val = sp.diff(gamma[a][b][d], coords[c]) - sp.diff(
                            gamma[a][b][c], coords[d]
                        )
                        for e in range(n):
                            val += gamma[a][c][e] * gamma[e][b][d] - gamma[a][d][e] * gamma[e][b][c]
                        val = simp(val)
                        riem[a][b][c][d] = val
                        riem[a][b][d][c] = -val

        ricci = sp.zeros(n, n)
        for b in range(n):
            for d in range(b, n):
                deadline.check()
                ricci[b, d] = ricci[d, b] = simp(sum(riem[a][b][a][d] for a in range(n)))
        ricci_scalar = simp(sum(inverse[a, b] * ricci[a, b] for a in range(n) for b in range(n)))
        einstein_x = sp.zeros(n, n)
        for a in range(n):
            for b in range(n):
                deadline.check()
                val = sum(inverse[a, c] * ricci[c, b] for c in range(n))
                einstein_x[a, b] = simp(val - (ricci_scalar / 2 if a == b else 0))

        # Kretschmann scalar: R_abcd R^abcd = 2 * sum over c < d (antisymmetry in c, d).
        # R_abcd = g_ae R^e_bcd and R^abcd = g^bf g^cg g^dh R^a_fgh; both are simplified
        # component by component and the sum is reduced after every (a, b) block, which
        # keeps the intermediate expressions small.
        pairs = [(c, d) for c in range(n) for d in range(c + 1, n)]
        kretschmann = sp.S.Zero
        for a in range(n):
            for b in range(n):
                block = sp.S.Zero
                for c, d in pairs:
                    deadline.check()
                    low = simp(sum(metric[a, e] * riem[e][b][c][d] for e in range(n)))
                    if low == 0:
                        continue
                    up = simp(
                        sum(
                            inverse[b, f] * inverse[c, g] * inverse[d, h] * riem[a][f][g][h]
                            for f in range(n)
                            for g in range(n)
                            for h in range(n)
                            if inverse[b, f] != 0 and inverse[c, g] != 0 and inverse[d, h] != 0
                        )
                    )
                    block += low * up
                kretschmann = simp(kretschmann + 2 * simp(block))

        # Back to (t, r, theta, phi): x = cos(theta), dx = -sin(theta) dtheta.
        back = {x: sp.cos(theta)}
        jac = -sp.sin(theta)
        einstein = sp.zeros(n, n)
        for a in range(n):
            for b in range(n):
                val = einstein_x[a, b].subs(back)
                if a == 2 and b != 2:
                    val = val / jac
                elif b == 2 and a != 2:
                    val = val * jac
                einstein[a, b] = val
    return {
        "einstein_mixed": einstein,
        "ricci_scalar": ricci_scalar.subs(back),
        "kretschmann": kretschmann.subs(back),
    }
