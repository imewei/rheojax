"""Shared fixtures for tests/io/."""

from __future__ import annotations

from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "rheocompass"


@pytest.fixture(scope="session")
def rheocompass_fixtures():
    """Skip tests if rheocompass fixtures are not available."""
    if not FIXTURES_DIR.exists():
        pytest.skip(f"Rheocompass fixtures not found at {FIXTURES_DIR}")
    return FIXTURES_DIR
