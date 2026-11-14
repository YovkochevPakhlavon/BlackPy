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
