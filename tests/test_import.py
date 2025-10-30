"""Basic import test for the rheo package."""

import sys

import pytest


def test_python_version():
    """Test that Python version is 3.12 or higher."""
    assert sys.version_info >= (3, 12), "Python 3.12+ is required"


def test_rheo_import():
    """Test that rheo package can be imported."""
    import rheo

    assert rheo is not None
    assert hasattr(rheo, "__version__")
    assert rheo.__version__ == "0.1.0"


def test_submodule_imports():
    """Test that all submodules can be imported."""
    from rheo import (
        core,
        io,
        legacy,
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
    assert legacy is not None


def test_version_info():
    """Test version information structure."""
    import rheo

    assert hasattr(rheo, "VERSION_INFO")
    version_info = rheo.VERSION_INFO
    assert "major" in version_info
    assert "minor" in version_info
    assert "patch" in version_info
    assert version_info["major"] == 0
    assert version_info["minor"] == 1
    assert version_info["patch"] == 0
    # Python 3.12+ is required (specified in pyproject.toml and CLAUDE.md)
    assert version_info["python_requires"] == ">=3.12"
