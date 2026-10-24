"""Shared fixtures. Everything here is synthetic: no real data, no model."""

from __future__ import annotations

import pytest

from mmsearch.fixtures import make_fixture
from mmsearch.gallery import Gallery


@pytest.fixture(scope="session")
def planted() -> Gallery:
    return make_fixture("planted")


@pytest.fixture(scope="session")
def null() -> Gallery:
    return make_fixture("null")


@pytest.fixture
def tiny() -> Gallery:
    return make_fixture("planted", n_per_source=6, captions_per_image=2, dim=16)


@pytest.fixture(scope="session")
def small() -> Gallery:
    """A smaller planted gallery for tests that do not need statistical power."""
    return make_fixture("planted", n_per_source=40, captions_per_image=3)


@pytest.fixture(scope="session")
def filler() -> Gallery:
    return make_fixture("filler")


@pytest.fixture(scope="session")
def shuffled() -> Gallery:
    return make_fixture("shuffled")


@pytest.fixture(scope="session")
def small_filler() -> Gallery:
    """Many exact ties (shared filler tokens) at a size that runs quickly."""
    return make_fixture("filler", n_per_source=40, captions_per_image=3)
