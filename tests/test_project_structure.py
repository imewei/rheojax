"""
Project structure validation tests for rheo package.

These tests verify that the project setup is correct and all essential
components are accessible. They serve as smoke tests for the project structure.
"""

import importlib.util
import sys
from pathlib import Path

import pytest


class TestProjectStructure:
    """Test suite for validating project structure and setup."""

    def test_package_imports_successfully(self):
        """Test that the main rheo package can be imported."""
        import rheo

        assert rheo is not None
        assert hasattr(rheo, "__version__")

    def test_core_modules_exist(self):
        """Test that all core submodules can be imported."""
        from rheo import core

        # Verify core submodules exist
        assert hasattr(core, "base")
        assert hasattr(core, "data")
        assert hasattr(core, "parameters")
        assert hasattr(core, "registry")

    def test_package_structure_directories(self):
        """Test that all expected package directories exist."""
        # Get package root
        import rheo

        package_root = Path(rheo.__file__).parent

        # Expected directories based on spec
        expected_dirs = [
            "core",
            "models",
            "transforms",
            "pipelines",
            "io",
            "visualization",
            "utils",
        ]

        for dirname in expected_dirs:
            dir_path = package_root / dirname
            assert dir_path.exists(), f"Missing directory: {dirname}"
            assert dir_path.is_dir(), f"Not a directory: {dirname}"

            # Verify each directory has an __init__.py
            init_file = dir_path / "__init__.py"
            assert init_file.exists(), f"Missing __init__.py in {dirname}"

    def test_version_accessible(self):
        """Test that version information is accessible and valid."""
        import rheo

        version = rheo.__version__
        assert version is not None
        assert isinstance(version, str)
        assert len(version) > 0

        # Version should follow semantic versioning pattern (simplified check)
        parts = version.split(".")
        assert len(parts) >= 2, "Version should have at least major.minor"

    def test_jax_version_accessible(self):
        """Test that JAX version information is accessible."""
        import rheo

        jax_version = rheo.__jax_version__
        assert jax_version is not None
        assert isinstance(jax_version, str)
        assert len(jax_version) > 0

    @pytest.mark.skip(
        reason="Development dependencies are optional - check requirements-dev.txt for full list"
    )
    def test_development_dependencies_installed(self):
        """Test that required development dependencies are installed.

        NOTE: This test is skipped by default as these are optional development
        dependencies. To enable this test, install with: pip install -r requirements-dev.txt
        """
        required_dev_packages = [
            "pytest",
            "hypothesis",
            "black",
            "ruff",
            "mypy",
            "sphinx",
        ]

        for package_name in required_dev_packages:
            spec = importlib.util.find_spec(package_name)
            assert (
                spec is not None
            ), f"Development dependency not installed: {package_name}"

    def test_jax_dependency_installed(self):
        """Test that JAX is installed and accessible."""
        try:
            import jax
            import jax.numpy as jnp

            # Verify basic JAX functionality
            array = jnp.array([1.0, 2.0, 3.0])
            assert array.shape == (3,)
            assert jnp.sum(array) == 6.0

        except ImportError as e:
            pytest.fail(f"JAX not properly installed: {e}")

    def test_py_typed_marker_exists(self):
        """Test that py.typed marker file exists for type checking support."""
        import rheo

        package_root = Path(rheo.__file__).parent
        py_typed = package_root / "py.typed"

        assert py_typed.exists(), "Missing py.typed marker file"
        assert py_typed.is_file(), "py.typed should be a file"


class TestDocumentation:
    """Test suite for documentation setup."""

    def test_docs_directory_exists(self):
        """Test that documentation directory exists."""
        # Get project root (parent of rheo package)
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        docs_dir = project_root / "docs"

        assert docs_dir.exists(), "Missing docs directory"
        assert docs_dir.is_dir(), "docs should be a directory"

    def test_sphinx_conf_exists(self):
        """Test that Sphinx configuration exists."""
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        conf_file = project_root / "docs" / "source" / "conf.py"

        assert conf_file.exists(), "Missing Sphinx conf.py"
        assert conf_file.is_file(), "conf.py should be a file"

    def test_sphinx_index_exists(self):
        """Test that Sphinx index.rst exists."""
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        index_file = project_root / "docs" / "source" / "index.rst"

        assert index_file.exists(), "Missing Sphinx index.rst"
        assert index_file.is_file(), "index.rst should be a file"


class TestConfiguration:
    """Test suite for project configuration files."""

    def test_pyproject_toml_exists(self):
        """Test that pyproject.toml exists and is valid."""
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        pyproject = project_root / "pyproject.toml"

        assert pyproject.exists(), "Missing pyproject.toml"

        # Try to parse it
        try:
            import tomllib
        except ImportError:
            # Python < 3.11
            try:
                import tomli as tomllib
            except ImportError:
                pytest.skip("tomllib/tomli not available")

        with open(pyproject, "rb") as f:
            config = tomllib.load(f)

        # Verify essential sections
        assert "project" in config
        assert "name" in config["project"]
        assert config["project"]["name"] == "rheo"

    def test_pytest_ini_exists(self):
        """Test that pytest.ini configuration exists."""
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        pytest_ini = project_root / "pytest.ini"

        assert pytest_ini.exists(), "Missing pytest.ini"

    def test_pre_commit_config_exists(self):
        """Test that pre-commit configuration exists."""
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        pre_commit_config = project_root / ".pre-commit-config.yaml"

        assert pre_commit_config.exists(), "Missing .pre-commit-config.yaml"


class TestCICD:
    """Test suite for CI/CD configuration."""

    @pytest.mark.xfail(
        reason="CI/CD setup not yet implemented (workflows.disabled exists)"
    )
    def test_github_workflows_exist(self):
        """Test that GitHub Actions workflows exist."""
        import rheo

        project_root = Path(rheo.__file__).parent.parent
        workflows_dir = project_root / ".github" / "workflows"

        assert workflows_dir.exists(), "Missing .github/workflows directory"
        assert workflows_dir.is_dir()

        # Check for CI workflow
        ci_workflow = workflows_dir / "ci.yml"
        assert ci_workflow.exists(), "Missing ci.yml workflow"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
