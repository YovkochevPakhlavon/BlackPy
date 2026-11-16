"""Parametrised metrics: Rezzolla-Zhidenko and KRZ (B4).

"RZ": Rezzolla & Zhidenko (2014, arXiv:1407.3086). "KRZ": Konoplya, Rezzolla & Zhidenko
(2016, arXiv:1602.02378). Both were read; equation numbers refer to the arXiv versions.
"""

import math

import jax.numpy as jnp
import numpy as np
import pytest
import sympy as sp

import blackpy as bp
from blackpy.errors import BlackPyError, NotSeparableError
from blackpy.infer import Uniform, build_problem, simulate_qpo_dataset
from blackpy.metric.parametrized import series_to_continued_fraction

mt = bp.metric


@pytest.fixture(scope="module")
def dilaton():
    """Dilaton black hole of RZ section IV in units of mu (so the ADM mass is 1 + b)."""
    return bp.StaticSpherical(
        lambda r, b=0.0: (r - 2.0) / (r + 2.0 * b),
        h=lambda r, b=0.0: r**2 + 2.0 * b * r,
        name="dilaton (units of mu)",
    )


@pytest.fixture(scope="module")
def krz():
    return mt.KRZ()


@pytest.fixture(scope="module")
def kerr_sen():
    """Rotating dilaton (Kerr-Sen) black hole, KRZ Eqs. (1) and (6), in units of M = mu + b."""
    rho, th, a, b = sp.symbols("r theta a b", real=True)
    mu = 1 - b
    f = (rho**2 - 2 * mu * rho + a**2) / rho**2
    common = rho**2 + 2 * b * rho + a**2 * sp.cos(th) ** 2
    sigma = common / rho**2
    kappa2 = ((rho**2 + 2 * b * rho + a**2) ** 2 - a**2 * sp.sin(th) ** 2 * f * rho**2) / (
        rho**2 * common
    )
    omega = 2 * a / common
    sin2 = sp.sin(th) ** 2
    return bp.StationaryAxisymmetric.from_sympy(
        -(f - omega**2 * sin2) / kappa2,
        -omega * rho * sin2,
        kappa2 * rho**2 * sin2,
        sigma / f,
        sigma * rho**2,
        r=rho,
        theta=th,
        parameters={"a": 0.0, "b": 0.0},
        name="Kerr-Sen (KRZ Eq. 6)",
    )


# ----------------------------------------------------------------------------- RZ


@pytest.mark.parametrize(
    "order, ppn",
    [
        (0, "gr"),
        pytest.param(3, "gr", marks=pytest.mark.slow),
        pytest.param(2, "free", marks=pytest.mark.slow),
    ],
)
def test_rz_with_zero_parameters_is_schwarzschild(order, ppn, schwarzschild):
    """All deviation parameters zero reproduces Schwarzschild to 1e-12 (mandatory)."""
    rz = mt.RezzollaZhidenko(order=order, ppn=ppn)
    expected = 1 + (2 if ppn == "free" else 0) + 2 * order
    assert rz.n_params == expected and rz.param_names[0] == "epsilon"
    assert all(par.default == 0.0 for par in rz.parameters) and rz.is_symbolic
    r = np.geomspace(2.0001, 500.0, 60)
    np.testing.assert_allclose(rz.f(r), schwarzschild.f(r), rtol=1e-12, atol=1e-15)
    np.testing.assert_allclose(rz.g(r), schwarzschild.g(r), rtol=1e-12, atol=1e-15)
    pairs = (
        (bp.geometry.event_horizon, 2.0),
        (bp.light.photon_sphere, 3.0),
        (bp.light.shadow_radius, 3.0 * math.sqrt(3.0)),
        (bp.orbits.isco, 6.0),
        (bp.orbits.marginally_bound_orbit, 4.0),
    )
    for function, value in pairs:
        assert float(function(rz)) == pytest.approx(value, rel=1e-12)
    radii = np.array([6.5, 9.0, 30.0])
    for function in (
        bp.orbits.angular_velocity,
        bp.orbits.radial_epicyclic_frequency,
        bp.orbits.vertical_epicyclic_frequency,
        bp.orbits.specific_energy,
    ):
        np.testing.assert_allclose(function(rz, radii), function(schwarzschild, radii), rtol=1e-12)


def test_rz_horizon_radius_in_units_of_the_mass():
    """r_0 = 2 / (1 + epsilon) in units of the ADM mass M (RZ Eq. (14))."""
    rz = mt.RezzollaZhidenko(order=2)
    epsilon = np.array([-0.3, -0.1, 0.0, 0.2, 0.45])
    np.testing.assert_allclose(
        bp.geometry.event_horizon(rz, epsilon=epsilon), 2.0 / (1.0 + epsilon), rtol=1e-12
    )
    # RZ Eq. (5) requires A(x) > 0 outside r_0. With only epsilon non-zero this means
    # epsilon < 1/2; beyond it A has a zero outside r_0 and that zero is the event horizon.
    assert float(bp.geometry.event_horizon(rz, epsilon=0.6)) > 2.0 / 1.6 + 0.1
    # The mass stays 1 whatever the coefficients: f -> 1 - 2/r.
    for params in ({"epsilon": 0.4}, {"epsilon": -0.2, "a1": 0.3, "a2": 0.5, "b1": -0.2}):
        r = 1.0e7
        assert r * (1.0 - float(rz.f(r, **params))) / 2.0 == pytest.approx(1.0, abs=1e-6)
        assert mt.rz_coefficients(rz, **params).mass == pytest.approx(1.0, abs=1e-9)
        assert mt.rz_coefficients(rz, **params).r0 == pytest.approx(
            2.0 / (1.0 + params["epsilon"]), rel=1e-12
        )
    # g_rr coefficients do not enter photon orbits, the shadow or the ISCO (RZ, section V).
    for function in (bp.light.shadow_radius, bp.orbits.isco):
        assert float(function(rz, a1=0.2, b1=0.4, b2=-0.3)) == pytest.approx(
            float(function(rz, a1=0.2)), rel=1e-12
        )
    with pytest.raises(ValueError, match="order"):
        mt.RezzollaZhidenko(order=-1)
    with pytest.raises(ValueError, match="ppn"):
        mt.RezzollaZhidenko(ppn="solar")
    assert mt.future.rezzolla_zhidenko is mt.RezzollaZhidenko  # the reserved name still works


def test_continued_fraction_from_series():
    # 2 / (1 + 3x / (1 - 0.5x)) = 2 - 6x + 21x^2 - 73.5x^3 + ...
    x = sp.Symbol("x")
    series = sp.series(2 / (1 + 3 * x / (1 - x / 2)), x, 0, 5).removeO()
    coefficients = [float(series.coeff(x, k)) for k in range(5)]
    np.testing.assert_allclose(
        series_to_continued_fraction(coefficients), [2.0, 3.0, -0.5, 0.0, 0.0], atol=1e-10
    )
    assert series_to_continued_fraction([0.0, 1.0, 2.0]) == [0.0, 0.0, 0.0]


def test_rz_coefficients_round_trip():
    """The helper recovers the coefficients of an RZ metric (any order, PPN terms free)."""
    rz = mt.RezzollaZhidenko(order=3, ppn="free")
    truth = {
        "epsilon": 0.15,
        "a0": 0.05,
        "b0": -0.03,
        "a1": 0.3,
        "a2": -0.4,
        "a3": 0.25,
        "b1": 0.2,
        "b2": 0.35,
        "b3": -0.15,
    }
    found = mt.rz_coefficients(rz, order=5, **truth)
    for name, value in found.as_params(3).items():
        assert value == pytest.approx(truth[name], abs=2e-8), name
    assert abs(found.a[3]) < 1e-6 and abs(found.b[3]) < 1e-6  # the fractions terminate
    # PPN parameters, RZ Eqs. (17)-(18).
    e = 1.15
    assert found.gamma == pytest.approx(1.0 + 2.0 * (-0.03) / e, abs=1e-8)
    assert found.beta == pytest.approx(1.0 + 2.0 * (0.05 - 0.03 * e) / e**2, abs=1e-8)


def test_rz_coefficients_of_reissner_nordstrom(rn):
    r"""RN is represented exactly at lowest order: epsilon = a_0 = Q^2 / r_0^2.

    Not printed in the RZ paper. Derivation: with u = r_0 / r,
    N^2 = 1 - (1 + epsilon) u + (Q^2 / r_0^2) u^2 must vanish at u = 1, hence
    Q^2 / r_0^2 = epsilon and N^2 = (1 - u)(1 - epsilon u) = x [1 - epsilon (1 - x)].
    """
    for charge in (0.3, 0.7, 0.95):
        r0 = 1.0 + math.sqrt(1.0 - charge**2)
        found = mt.rz_coefficients(rn, order=3, Q=charge)
        assert found.epsilon == pytest.approx(2.0 / r0 - 1.0, abs=1e-9)
        assert found.a0 == pytest.approx(charge**2 / r0**2, abs=1e-9)
        assert found.a0 == pytest.approx(found.epsilon, abs=1e-9)
        assert abs(found.b0) < 1e-9 and found.r0 == pytest.approx(r0, rel=1e-12)
        assert max(map(abs, found.a + found.b)) < 1e-7
        assert found.gamma == pytest.approx(1.0, abs=1e-9)
        assert found.beta == pytest.approx(1.0 + charge**2 / 2.0, abs=1e-8)
        # The order-0 RZ metric with these coefficients is RN.
        rz = mt.RezzollaZhidenko(order=0, ppn="free")
        params = found.as_params(0)
        r = np.geomspace(r0 * 1.001, 300.0, 40)
        np.testing.assert_allclose(rz.f(r, **params), rn.f(r, Q=charge), rtol=1e-8, atol=1e-10)
        assert float(bp.orbits.isco(rz, **params)) == pytest.approx(
            float(bp.orbits.isco(rn, Q=charge)), rel=1e-8
        )
    # With a_0 fixed to its GR value a charge cannot be represented: the docstring says so.
    assert "Reissner-Nordstrom" in mt.RezzollaZhidenko.__doc__
    with pytest.raises(BlackPyError, match="no event horizon"):
        mt.rz_coefficients(rn, Q=1.2)
    with pytest.raises(TypeError):
        mt.rz_coefficients(mt.Kerr())
    with pytest.raises(BlackPyError, match="asymptotically flat"):
        mt.rz_coefficients(mt.SchwarzschildDeSitter(), Lambda=0.01)


@pytest.mark.parametrize("b", [0.25, 0.5, 1.0])
def test_rz_coefficients_of_the_dilaton_black_hole_match_the_paper(dilaton, b):
    """RZ Eqs. (30) and (31), printed as functions of b / mu (mandatory)."""
    found = mt.rz_coefficients(dilaton, order=4, b=b)
    root = math.sqrt(1.0 + b)
    assert found.mass == pytest.approx(1.0 + b, abs=1e-9)  # M = mu + b, RZ Eq. (29)
    assert found.r0 == pytest.approx(2.0 * root, rel=1e-12)  # r^2 = rho^2 + 2 b rho at rho = 2
    assert found.epsilon == pytest.approx(root - 1.0, abs=1e-9)  # (30a)
    assert found.a0 == pytest.approx(b / 2.0, abs=1e-9)  # (30b)
    assert found.b0 == pytest.approx(0.0, abs=1e-9)  # (30c)
    a1 = 2.0 * root + 1.0 / (1.0 + b / 2.0) - 3.0 - b / 2.0  # (31a)
    b1 = root / (1.0 + b / 2.0) - 1.0  # (31b)
    a2 = (root - 0.5 - b**2 / 4.0 + (b / 2.0) * (root - 1.0)) / (1.0 + b / 2.0) ** 2  # (31c)
    b2 = root / (1.0 + b / 2.0) - 1.0 - b**2 / (b + 2.0) ** 2  # (31d)
    assert found.a[0] == pytest.approx(a1, abs=1e-9)
    assert found.b[0] == pytest.approx(b1, abs=1e-9)
    assert found.b[1] == pytest.approx(b2, abs=1e-8)
    # a_2 = -(d A~/dx at 0) / a_1, and a_1 is small (5e-3 at b = 1): errors are amplified.
    assert found.a[1] == pytest.approx(a2, abs=5e-6)


@pytest.mark.slow
def test_rz_observables_converge_with_the_order(dilaton):
    """Truncated RZ metrics against the exact dilaton results, RZ Eqs. (37) and (43).

    The paper reports (text of sections IV and V.1, read off its Fig. 1): an error of the
    photon-orbit impact parameter of about 1e-4 mu at b = mu/2 when a_2 = 0; a maximum
    relative error of g_tt at b = mu of about 3e-4 and 3e-6 at successive orders; an error
    of the ISCO frequency of about 1e-6 / mu at b = mu/2. Measured here, for orders 1 to 4
    (order n keeps a_1 ... a_n): impact parameter at b = mu/2: 9.2e-5, 1.5e-5, 2.6e-7,
    2.2e-9; g_tt at b = mu: 2.9e-4, 4.9e-5, 3.6e-6, 9.2e-8; ISCO frequency at b = mu/2:
    1.6e-6, 1.3e-7, 1.1e-8, 2.8e-10.
    """
    measured = {}
    for b in (0.5, 1.0):
        coefficients = mt.rz_coefficients(dilaton, order=4, b=b)
        mass = 1.0 + b
        exact_impact = math.sqrt((27 + 36 * b + 8 * b**2 + (9 + 8 * b) ** 1.5) / 2.0)  # (37)
        kappa = (1.0 + b) ** (1.0 / 3.0)
        exact_omega = 0.5 * math.sqrt(kappa / ((1 + kappa) * (1 + kappa + kappa**2) ** 3))  # (43)
        # The exact formulas agree with BlackPy's own results for the exact metric.
        assert float(bp.light.shadow_radius(dilaton, b=b)) == pytest.approx(exact_impact, rel=1e-11)
        r_isco = float(bp.orbits.isco(dilaton, b=b))
        assert float(bp.orbits.angular_velocity(dilaton, r_isco, b=b)) == pytest.approx(
            exact_omega, rel=1e-10
        )
        rho = np.geomspace(2.0002, 2000.0, 300)
        areal = np.sqrt(rho**2 + 2.0 * b * rho)
        exact_gtt = (rho - 2.0) / (rho + 2.0 * b)
        for order in (1, 2, 3, 4):
            rz = mt.RezzollaZhidenko(order=order, ppn="free")
            params = coefficients.as_params(order)
            impact = float(bp.light.shadow_radius(rz, **params)) * mass
            r_isco = float(bp.orbits.isco(rz, **params))
            omega = float(bp.orbits.angular_velocity(rz, r_isco, **params)) / mass
            gtt = np.asarray(rz.f(areal / mass, **params))
            measured[b, order] = (
                abs(impact - exact_impact),
                float(np.max(np.abs(gtt / exact_gtt - 1.0))),
                abs(omega - exact_omega),
            )
    # Rates reported in the paper (order-of-magnitude statements).
    assert measured[0.5, 1][0] < 1.5e-4  # "~1e-4 mu for b ~ mu/2" with a_2 = 0
    assert measured[0.5, 1][2] < 3e-6  # ISCO frequency "~1e-6 for b ~ mu/2"
    assert measured[1.0, 1][1] < 4e-4 and measured[1.0, 3][1] < 5e-6  # g_tt: 3e-4 and 3e-6
    assert measured[1.0, 4][0] < 1e-6 and measured[1.0, 4][2] < 1e-7  # "even for b ~ mu"
    # Monotonic convergence: each order gains at least a factor of three in every quantity.
    for b in (0.5, 1.0):
        for k in range(3):
            for order in (1, 2, 3):
                assert measured[b, order + 1][k] < measured[b, order][k] / 3.0, (b, order, k)


# ----------------------------------------------------------------------------- KRZ


def test_krz_with_zero_deviations_has_the_kerr_components(krz, kerr):
    """All deviation parameters zero reproduces the Kerr metric to 1e-12 (mandatory)."""
    assert krz.param_names[0] == "a" and krz.spin_parameter == "a" and krz.n_params == 12
    assert all(par.role == "deformation" for par in krz.parameters[1:])
    assert krz.is_reflection_symmetric
    r, theta = np.meshgrid(np.geomspace(2.2, 80.0, 12), np.linspace(0.2, 2.9, 7), indexing="ij")
    for a in (0.0, 0.5, -0.7, 0.98):
        for name in ("g_tt", "g_tphi", "g_phiphi", "g_rr", "g_thth"):
            np.testing.assert_allclose(
                getattr(krz, name)(r, theta, a=a),
                getattr(kerr, name)(r, theta, a=a),
                rtol=1e-12,
                atol=1e-14,
                err_msg=f"{name}, a = {a}",
            )
    assert mt.future.konoplya_rezzolla_zhidenko is mt.KRZ
    assert "effective_source" in mt.KRZ.__doc__  # a model: the source must be checked


@pytest.mark.slow
def test_krz_with_zero_deviations_has_the_kerr_observables(krz, kerr):
    """Zero deviations: every observable that is defined equals Kerr's to 1e-12 (mandatory)."""
    spins = np.array([-0.9, -0.3, 0.0, 0.4, 0.8, 0.97])
    np.testing.assert_allclose(
        bp.geometry.event_horizon(krz, a=spins), 1.0 + np.sqrt(1.0 - spins**2), rtol=1e-12
    )
    np.testing.assert_allclose(
        bp.geometry.ergosurface(krz, 1.0, a=spins), bp.geometry.ergosurface(kerr, 1.0, a=spins)
    )
    for direction in ("prograde", "retrograde"):
        np.testing.assert_allclose(
            bp.orbits.isco(krz, direction=direction, a=spins),
            bp.orbits.isco(kerr, direction=direction, a=spins),
            rtol=1e-12,
        )
        np.testing.assert_allclose(
            bp.orbits.photon_orbit(krz, direction=direction, a=spins),
            bp.orbits.photon_orbit(kerr, direction=direction, a=spins),
            rtol=1e-11,
        )
    radii = np.array([7.0, 9.5, 25.0])
    for function in (
        bp.orbits.angular_velocity,
        bp.orbits.radial_epicyclic_frequency,
        bp.orbits.vertical_epicyclic_frequency,
        bp.orbits.specific_energy,
        bp.orbits.specific_angular_momentum,
    ):
        np.testing.assert_allclose(
            function(krz, radii, a=0.6), function(kerr, radii, a=0.6), rtol=1e-12
        )
    assert bool(krz.in_domain(a=0.99))
    with pytest.raises(ValueError, match=r"\[-1.0, 1.0\]"):  # deviations are defined for |a| <= 1
        bp.orbits.isco(krz, a=1.01)


def test_krz_coefficients_of_kerr_as_printed_in_the_paper(krz, kerr):
    """KRZ Appendix A.1. The printed lowest-order form is Kerr on the equator only."""
    a = 0.6
    r0 = 1.0 + math.sqrt(1.0 - a**2)
    printed = {
        "a20": 2 * a**2 / r0**3,
        "a21": -(a**4) / r0**4,
        "k21": a**4 / r0**4 - 2 * a**2 / r0**3,
    }
    deviations = mt.krz_deviations(a, r0=r0, **printed)
    assert deviations["a"] == a
    assert all(abs(v) < 1e-15 for k, v in deviations.items() if k != "a")
    assert mt.krz_deviations(a) == deviations
    # KRZ Eqs. (86)-(89) exactly as printed, with the Kerr coefficients of Appendix A.1.
    eps0, k00, w00 = 2 / r0 - 1, a**2 / r0**2, 2 * a / r0**2

    def printed_form(r, theta):
        cos2, sin2 = math.cos(theta) ** 2, math.sin(theta) ** 2
        sigma = 1 + a**2 * cos2 / r**2
        n2 = (1 - r0 / r) * (1 - eps0 * r0 / r + (k00 - eps0) * r0**2 / r**2) + (
            (printed["k21"] + printed["a20"]) * r0**3 / r**3 + printed["a21"] * r0**4 / r**4
        ) * cos2
        w = w00 * r0**2 / r**2 / sigma
        k2 = 1 + a * w / r + (k00 * r0**2 / r**2 + printed["k21"] * r0**3 / r**3 * cos2) / sigma
        return -(n2 - w**2 * sin2) / k2, k2 * r**2 * sin2, sigma / n2

    for theta, exact in ((math.pi / 2, True), (0.7, False)):
        g_tt, g_phiphi, g_rr = printed_form(3.5, theta)
        errors = [
            abs(g_tt - float(kerr.g_tt(3.5, theta, a=a))),
            abs(g_phiphi - float(kerr.g_phiphi(3.5, theta, a=a))),
            abs(g_rr - float(kerr.g_rr(3.5, theta, a=a))),
        ]
        if exact:
            assert max(errors) < 1e-13
        else:  # off the equatorial plane the printed truncation is not Kerr
            assert min(errors) > 1e-4
    with pytest.raises(ValueError):
        mt.krz_deviations(1.2)


@pytest.mark.slow
@pytest.mark.parametrize(
    "a, b, tolerance", [(0.5, 0.05, 1.5e-4), (0.5, 0.2, 3e-3), (0.9, 0.05, 5e-4)]
)
def test_krz_lowest_order_reproduces_the_rotating_dilaton_black_hole(
    krz, kerr_sen, a, b, tolerance
):
    """KRZ Appendix A.2 / Eq. (60): the non-Kerr example of the paper (mandatory).

    The sign of w_01 is opposite to the one printed in the paper: evaluating the paper's
    own definitions on its metric (6) gives W(r_0) - w_00 = -2 a M (b + r_0 - s) / r_0^3
    with s = sqrt(r_0^2 + b^2). With the printed sign the ISCO frequency is wrong by
    several per cent, which this test also checks.
    """
    exact = kerr_sen
    mu = 1.0 - b
    r0 = math.sqrt((1.0 + math.sqrt(mu**2 - a**2)) ** 2 - b**2)
    root = math.sqrt(r0**2 + b**2)
    assert float(bp.geometry.event_horizon(exact, a=a, b=b)) == pytest.approx(
        mu + math.sqrt(mu**2 - a**2), rel=1e-11
    )
    coefficients = {
        "a00": 2 * b / r0**2,
        "a20": 2 * a**2 / r0**3,
        "a01": 2 * (2 * b**2 + r0**2 + (2 * r0 - 3 * b) * root) / (r0**2 * root)
        - 3 * (r0**2 + a**2) / r0**2,
        "a21": -(a**4 + 2 * a**2 * (b + r0 - root)) / r0**4,
        "k21": (a**4 - 2 * a**2 * (root - b)) / r0**4,
        "b01": r0 / root - 1.0,
    }
    w01 = 2 * a * (b + r0 - root) / r0**3

    def isco_frequency_and_energy(metric, direction, **params):
        radius = float(bp.orbits.isco(metric, direction=direction, **params))
        return (
            float(bp.orbits.angular_velocity(metric, radius, direction=direction, **params)),
            float(bp.orbits.specific_energy(metric, radius, direction=direction, **params)),
        )

    corrected = mt.krz_deviations(a, r0=r0, w01=-w01, **coefficients)
    as_printed = mt.krz_deviations(a, r0=r0, w01=+w01, **coefficients)
    assert float(bp.geometry.event_horizon(krz, **corrected)) == pytest.approx(r0, rel=1e-10)
    for direction in ("prograde", "retrograde"):
        omega, energy = isco_frequency_and_energy(exact, direction, a=a, b=b)
        omega_krz, energy_krz = isco_frequency_and_energy(krz, direction, **corrected)
        assert omega_krz == pytest.approx(omega, rel=tolerance), direction
        assert energy_krz == pytest.approx(energy, abs=tolerance), direction
    omega = isco_frequency_and_energy(exact, "prograde", a=a, b=b)[0]
    wrong = isco_frequency_and_energy(krz, "prograde", **as_printed)[0]
    assert abs(wrong / omega - 1.0) > 50 * tolerance


@pytest.mark.slow
def test_krz_deformed_metric_orbits_work_and_shadow_is_refused(krz, kerr):
    deformed = {"a": 0.5, "a01": 0.15, "w01": 0.05, "d_a20": 0.1, "d_a21": -0.1, "d_eps0": 0.05}
    # Equatorial orbits and QPOs work for the reflection-symmetric deformed metric.
    assert krz.check_reflection_symmetry(**deformed)
    isco = float(bp.orbits.isco(krz, **deformed))
    assert np.isfinite(isco) and abs(isco - float(bp.orbits.isco(kerr, a=0.5))) > 0.05
    frequencies = bp.orbits.frequencies_hz(krz, isco * 1.3, 8.0, **deformed)
    assert all(np.isfinite(float(v)) and float(v) > 0 for v in frequencies)
    assert np.isfinite(
        float(bp.orbits.nodal_precession_frequency(krz, isco * 1.3, 8.0, **deformed))
    )
    data = simulate_qpo_dataset(
        krz,
        "RP",
        name="s",
        mass=8.0,
        r=isco * 1.3,
        sigma_nu=2.0,
        mass_sigma=0.3,
        sigma_low=0.1,
        **deformed,
    )
    problem = build_problem(
        krz,
        data,
        "RP",
        {"a": Uniform(0.0, 0.9), "a01": Uniform(-0.5, 0.5), "r": Uniform(4, 12)},
        fixed={k: v for k, v in deformed.items() if k not in ("a", "a01")},
    )
    assert problem.param_names == ["a", "a01", "M_s", "r_s"]
    truth = np.array([0.5, 0.15, 8.0, isco * 1.3])
    assert float(problem.log_likelihood(truth)) > float(
        problem.log_likelihood(truth + np.array([0.0, 0.1, 0.0, 0.0]))
    )
    # Null geodesics do not separate: the analytic shadow is refused, not special-cased.
    assert not krz.is_separable and krz.check_separability(a=0.6)  # Kerr values do separate
    assert not krz.check_separability(**deformed)
    for function in (bp.light.shadow_contour, bp.light.shadow_observables):
        with pytest.raises(NotSeparableError, match="ray tracing"):
            function(krz, 1.0, a=0.5)
    with pytest.raises(NotSeparableError):
        bp.light.shadow_radius(krz, inclination=1.0, a=0.5)
    # The surface N^2 = 0 is a sphere only if d_a20 + d_a21 + d_k21 = 0 (KRZ, Appendix A).
    spherical = bp.geometry.horizon_profile(krz, **deformed)
    assert spherical.theta_independent
    lumpy = bp.geometry.horizon_profile(krz, **{**deformed, "d_a21": 0.05})
    assert not lumpy.theta_independent
    assert jnp.isfinite(bp.orbits.isco(krz, **{**deformed, "d_a21": 0.05}))
