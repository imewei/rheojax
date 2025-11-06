"""Basic import test for the rheojax package."""

import sys

import pytest


def test_python_version():
    """Test that Python version is 3.12 or higher."""
    assert sys.version_info >= (3, 12), "Python 3.12+ is required"


def test_rheojax_import():
    """Test that rheojax package can be imported."""
    import rheojax

    assert rheojax is not None
    assert hasattr(rheojax, "__version__")
    assert rheojax.__version__ == "0.2.0"


def test_submodule_imports():
    """Test that all submodules can be imported."""
    from rheojax import (
        core,
        io,
        models,
        pipeline,
        transforms,
        utils,
        visualization,
    )

    # Check that modules exist
    assert core is not None
    assert models is not None
    assert transforms is not None
    assert pipeline is not None
    assert io is not None
    assert visualization is not None
    assert utils is not None


def test_version_info():
    """Test version information structure."""
    import rheojax

    assert hasattr(rheojax, "VERSION_INFO")
    version_info = rheojax.VERSION_INFO
    assert "major" in version_info
    assert "minor" in version_info
    assert "patch" in version_info
    assert version_info["major"] == 0
    assert version_info["minor"] == 2
    assert version_info["patch"] == 0
    # Python 3.12+ is required (specified in pyproject.toml and CLAUDE.md)
    assert version_info["python_requires"] == ">=3.12"
