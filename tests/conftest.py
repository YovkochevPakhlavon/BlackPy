"""Shared fixtures."""

import pytest

import blackpy as bp


@pytest.fixture(scope="session")
def schwarzschild():
    return bp.metric.Schwarzschild()


@pytest.fixture(scope="session")
def rn():
    return bp.metric.ReissnerNordstrom()


@pytest.fixture(scope="session")
def sds():
    return bp.metric.SchwarzschildDeSitter()


@pytest.fixture(scope="session")
def bardeen():
    return bp.metric.Bardeen()


@pytest.fixture(scope="session")
def hayward():
    return bp.metric.Hayward()


@pytest.fixture(scope="session")
def kerr():
    return bp.metric.Kerr()


@pytest.fixture(scope="session")
def kerr_newman():
    return bp.metric.KerrNewman()


@pytest.fixture(scope="session")
def johannsen():
    return bp.metric.Johannsen()
