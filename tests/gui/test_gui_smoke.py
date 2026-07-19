"""RheoJAX GUI Smoke Tests.

Comprehensive smoke tests for GUI components without requiring a display.
Tests basic functionality including:
- Module imports (with graceful skips if PySide6 unavailable)
- State management (works without Qt)
- Service instantiation
- Stylesheet loading and validation

Markers:
    gui: All GUI-related tests
    smoke: Critical smoke tests for CI/CD

Run with:
    pytest tests/gui/ -v
    pytest tests/gui/ -v -m gui
    pytest tests/gui/ -v -m smoke
"""

from pathlib import Path

import pytest

# Mark all tests as GUI tests
pytestmark = pytest.mark.gui

# Check if PySide6 is available
try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


# =============================================================================
# Test Module Imports
# =============================================================================


class TestModuleImports:
    """Test that all GUI modules can be imported without errors.

    These tests verify:
    - Core GUI module structure
    - State management components
    - Service layer components
    - Resource components (styles, themes)
    """

    @pytest.mark.smoke
    def test_gui_main_module_imports(self) -> None:
        """Test main GUI module imports."""
        from rheojax.gui.main import main

        assert callable(main), "main function should be callable"

    @pytest.mark.smoke
    def test_gui_state_store_imports_no_qt(self) -> None:
        """Test foundation state module imports (pure Python dataclasses)."""
        from rheojax.gui.foundation.state import AppState, PipelineStep, StepStatus

        assert AppState is not None, "AppState should import successfully"
        assert PipelineStep is not None, "PipelineStep should import successfully"
        assert StepStatus is not None, "StepStatus should import successfully"

    @pytest.mark.smoke
    def test_gui_services_imports(self) -> None:
        """Test all service modules can be imported.

        Services are lazy-loaded, verify they can be accessed
        from the services module.
        """
        from rheojax.gui.services import (
            BayesianService,
            DataService,
            ExportService,
            ModelService,
            PlotService,
            TransformService,
        )

        assert DataService is not None, "DataService should import"
        assert ModelService is not None, "ModelService should import"
        assert BayesianService is not None, "BayesianService should import"
        assert TransformService is not None, "TransformService should import"
        assert PlotService is not None, "PlotService should import"
        assert ExportService is not None, "ExportService should import"

    @pytest.mark.smoke
    def test_gui_styles_imports(self) -> None:
        """Test stylesheet utilities import successfully.

        Stylesheet functions should be available without Qt.
        """
        from rheojax.gui.resources.styles import (
            get_dark_stylesheet,
            get_light_stylesheet,
            get_stylesheet,
        )

        assert callable(get_stylesheet), "get_stylesheet should be callable"
        assert callable(get_light_stylesheet), "get_light_stylesheet should be callable"
        assert callable(get_dark_stylesheet), "get_dark_stylesheet should be callable"

    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
    def test_gui_app_imports(self) -> None:
        """Test surviving app-adjacent modules import (requires PySide6)."""
        from rheojax.gui.workspace import status_bar

        assert status_bar is not None, "status_bar module should import"

    @pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
    def test_gui_utils_imports(self) -> None:
        """Test GUI utility modules import successfully."""
        from rheojax.gui.utils import config, jax_utils

        assert config is not None, "config module should import"
        assert jax_utils is not None, "jax_utils module should import"


# =============================================================================
# Test State Management (No Qt Required)
# =============================================================================


class TestStateManagement:
    """Test state management without Qt dependencies.

    State classes use dataclasses and are independent of Qt,
    so can be thoroughly tested without display.
    """

    @pytest.mark.smoke
    def test_dataset_state_creation(self) -> None:
        """Test DatasetState creation with required fields."""
        from rheojax.gui.foundation.state import DatasetState

        ds = DatasetState(
            id="test-123",
            name="Test Dataset",
            file_path=Path("/test/data.csv"),
            test_mode="oscillation",
        )

        assert ds.id == "test-123", "Dataset ID should match"
        assert ds.name == "Test Dataset", "Dataset name should match"
        assert ds.file_path == Path("/test/data.csv"), "File path should match"
        assert ds.test_mode == "oscillation", "Test mode should be oscillation"
        assert ds.is_modified is False, "New dataset should not be modified"
        assert isinstance(ds.metadata, dict), "Metadata should be a dict"

    @pytest.mark.smoke
    def test_pipeline_step_enum_values(self) -> None:
        """Test PipelineStep enum has expected values."""
        from rheojax.gui.foundation.state import PipelineStep

        expected_steps = {"LOAD", "TRANSFORM", "FIT", "BAYESIAN", "EXPORT"}
        actual_steps = {step.name for step in PipelineStep}

        assert actual_steps == expected_steps, "Pipeline should have all expected steps"

    @pytest.mark.smoke
    def test_step_status_enum_values(self) -> None:
        """Test StepStatus enum has expected values."""
        from rheojax.gui.foundation.state import StepStatus

        expected_statuses = {"PENDING", "ACTIVE", "COMPLETE", "WARNING", "ERROR"}
        actual_statuses = {status.name for status in StepStatus}

        assert actual_statuses == expected_statuses, (
            "StepStatus should have all expected values"
        )

    @pytest.mark.smoke
    def test_parameter_state_creation(self) -> None:
        """Test ParameterState creation and defaults."""
        from rheojax.gui.foundation.state import ParameterState

        param = ParameterState(
            name="G0",
            value=1000.0,
            min_bound=100.0,
            max_bound=10000.0,
        )

        assert param.name == "G0"
        assert param.value == 1000.0
        assert param.min_bound == 100.0
        assert param.max_bound == 10000.0
        assert param.fixed is False
        assert param.unit == ""
        assert param.description == ""

    @pytest.mark.smoke
    def test_parameter_state_with_units(self) -> None:
        """Test ParameterState with unit and description."""
        from rheojax.gui.foundation.state import ParameterState

        param = ParameterState(
            name="tau",
            value=0.1,
            min_bound=0.001,
            max_bound=1.0,
            unit="s",
            description="Relaxation time",
            fixed=True,
        )

        assert param.unit == "s"
        assert param.description == "Relaxation time"
        assert param.fixed is True

    @pytest.mark.smoke
    def test_fit_result_creation(self) -> None:
        """Test FitResult creation from fit."""
        from datetime import datetime

        from rheojax.gui.foundation.state import FitResult

        result = FitResult(
            model_name="Maxwell",
            parameters={"G0": 1000.0, "tau": 0.1},
            chi_squared=0.01,
            success=True,
            message="Converged successfully",
            timestamp=datetime.now(),
            dataset_id="test-123",
            r_squared=0.95,
            mpe=0.05,
            fit_time=0.5,
            num_iterations=42,
            convergence_message="Converged successfully",
        )

        assert result.model_name == "Maxwell"
        assert result.dataset_id == "test-123"
        assert result.r_squared == 0.95
        assert result.num_iterations == 42

    @pytest.mark.smoke
    def test_bayesian_result_single_source(self) -> None:
        """BayesianResult should only be defined in foundation/state.py."""
        from rheojax.gui.foundation.state import BayesianResult as CanonicalBR
        from rheojax.gui.jobs.bayesian_worker import BayesianResult as WorkerBR
        from rheojax.gui.services.bayesian_service import BayesianResult as ServiceBR

        assert CanonicalBR is ServiceBR, (
            "Service BayesianResult should be imported from foundation.state"
        )
        assert CanonicalBR is WorkerBR, (
            "Worker BayesianResult should be imported from foundation.state"
        )

    @pytest.mark.smoke
    def test_bayesian_result_canonical_fields(self) -> None:
        """BayesianResult should have all canonical fields."""
        from datetime import datetime

        from rheojax.gui.foundation.state import BayesianResult

        result = BayesianResult(
            model_name="maxwell",
            dataset_id="ds-001",
            posterior_samples={"G": [1.0, 2.0]},
            summary={"G": {"mean": 1.5}},
            r_hat={"G": 1.001},
            ess={"G": 500.0},
            divergences=0,
            credible_intervals={"G": (0.8, 2.2)},
            mcmc_time=12.5,
            timestamp=datetime.now(),
            num_warmup=500,
            num_samples=1000,
            num_chains=4,
        )
        assert result.sampling_time == 12.5
        assert result.diagnostics["r_hat"] == {"G": 1.001}
        assert result.diagnostics["ess"] == {"G": 500.0}
        assert result.diagnostics["divergences"] == 0

    @pytest.mark.smoke
    def test_protocol_to_test_mode_mapping(self) -> None:
        """Protocol enums should map to test mode strings."""
        from rheojax.core.inventory import Protocol

        expected = {
            Protocol.OSCILLATION: "oscillation",
            Protocol.RELAXATION: "relaxation",
            Protocol.CREEP: "creep",
            Protocol.STARTUP: "startup",
            Protocol.FLOW_CURVE: "flow_curve",
            Protocol.LAOS: "laos",
        }
        for protocol, mode_str in expected.items():
            assert protocol.value == mode_str

    @pytest.mark.smoke
    def test_fractional_model_protocols(self) -> None:
        """Fractional models should only report OSCILLATION, RELAXATION, CREEP."""
        from rheojax.core.inventory import Protocol
        from rheojax.core.registry import ModelRegistry

        info = ModelRegistry.get_info("fractional_zener_ss")
        assert info is not None
        protocol_values = {p.value for p in info.protocols}
        assert protocol_values == {"oscillation", "relaxation", "creep"}


# =============================================================================
# Test Service Instantiation
# =============================================================================


class TestServiceInstantiation:
    """Test that services can be instantiated and have expected methods.

    Services encapsulate business logic for different domains.
    """

    @pytest.mark.smoke
    def test_data_service_instantiation(self) -> None:
        """Test DataService can be instantiated without errors."""
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        assert service is not None, "DataService should instantiate"
        assert hasattr(service, "load_file"), "DataService should have load_file method"

    @pytest.mark.smoke
    def test_model_service_instantiation(self) -> None:
        """Test ModelService can be instantiated and has expected methods."""
        from rheojax.gui.services.model_service import ModelService

        service = ModelService()

        assert service is not None, "ModelService should instantiate"
        assert hasattr(service, "get_available_models"), (
            "Should have get_available_models method"
        )

        # Test get_available_models returns proper structure
        try:
            models = service.get_available_models()
            assert isinstance(models, dict), "Models should be a dict"
        except AttributeError as e:
            # Registry might not be fully initialized in test environment
            pytest.skip(f"Registry not fully initialized: {e}")

    @pytest.mark.smoke
    def test_bayesian_service_instantiation(self) -> None:
        """Test BayesianService can be instantiated."""
        from rheojax.gui.services.bayesian_service import BayesianService

        service = BayesianService()
        assert service is not None, "BayesianService should instantiate"

    @pytest.mark.smoke
    def test_transform_service_instantiation(self) -> None:
        """Test TransformService can be instantiated."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()
        assert service is not None, "TransformService should instantiate"

    @pytest.mark.smoke
    def test_plot_service_instantiation(self) -> None:
        """Test PlotService can be instantiated and has styles."""
        from rheojax.gui.services.plot_service import PlotService

        service = PlotService()

        assert service is not None, "PlotService should instantiate"
        assert hasattr(service, "get_available_styles"), (
            "Should have get_available_styles method"
        )

        styles = service.get_available_styles()
        assert isinstance(styles, list), "Styles should be a list"
        assert "default" in styles, "Should have 'default' style"
        assert "publication" in styles, "Should have 'publication' style"

    @pytest.mark.smoke
    def test_plot_service_colorblind_palette(self) -> None:
        """Test PlotService provides colorblind-safe palette."""
        from rheojax.gui.services.plot_service import PlotService

        service = PlotService()
        palette = service.get_colorblind_palette()

        assert isinstance(palette, list), "Palette should be a list"
        assert len(palette) > 0, "Palette should have colors"
        assert all(isinstance(c, str) and c.startswith("#") for c in palette), (
            "All colors should be hex codes"
        )

    @pytest.mark.smoke
    def test_export_service_instantiation(self) -> None:
        """Test ExportService can be instantiated."""
        from rheojax.gui.services.export_service import ExportService

        service = ExportService()
        assert service is not None, "ExportService should instantiate"


# =============================================================================
# Test Stylesheet Loading and Validation
# =============================================================================


class TestStylesheetLoading:
    """Test stylesheet loading and validation.

    Stylesheets are pure text resources that load without Qt.
    """

    @pytest.mark.smoke
    def test_light_stylesheet_loads(self) -> None:
        """Test light stylesheet loads and has expected content."""
        from rheojax.gui.resources.styles import get_light_stylesheet

        css = get_light_stylesheet()

        assert isinstance(css, str), "Stylesheet should be a string"
        assert len(css) > 100, "Stylesheet should be substantial"
        assert "QMainWindow" in css, "Should have QMainWindow styling"

    @pytest.mark.smoke
    def test_dark_stylesheet_loads(self) -> None:
        """Test dark stylesheet loads and has expected content."""
        from rheojax.gui.resources.styles import (
            get_dark_stylesheet,
            get_light_stylesheet,
        )

        css = get_dark_stylesheet()

        assert isinstance(css, str), "Stylesheet should be a string"
        assert len(css) > 100, "Stylesheet should be substantial"
        assert css != get_light_stylesheet(), "Dark and light sheets should differ"

    @pytest.mark.smoke
    def test_get_stylesheet_light_default(self) -> None:
        """Test get_stylesheet returns light theme by default."""
        from rheojax.gui.resources.styles import get_light_stylesheet, get_stylesheet

        default = get_stylesheet()
        light = get_light_stylesheet()

        assert default == light, "Default should be light theme"

    @pytest.mark.smoke
    def test_get_stylesheet_explicit_light(self) -> None:
        """Test get_stylesheet with explicit light theme."""
        from rheojax.gui.resources.styles import get_light_stylesheet, get_stylesheet

        result = get_stylesheet("light")
        expected = get_light_stylesheet()

        assert result == expected, "Explicit light should match get_light_stylesheet"

    @pytest.mark.smoke
    def test_get_stylesheet_dark(self) -> None:
        """Test get_stylesheet with dark theme."""
        from rheojax.gui.resources.styles import get_dark_stylesheet, get_stylesheet

        result = get_stylesheet("dark")
        expected = get_dark_stylesheet()

        assert result == expected, "Dark theme should match get_dark_stylesheet"

    @pytest.mark.smoke
    def test_get_stylesheet_invalid_theme_raises(self) -> None:
        """Test get_stylesheet raises for invalid theme."""
        from rheojax.gui.resources.styles import get_stylesheet

        with pytest.raises(ValueError, match="Invalid theme"):
            get_stylesheet("invalid_theme")

    @pytest.mark.smoke
    def test_stylesheet_qss_syntax_validation(self) -> None:
        """Test stylesheets contain valid QSS syntax elements."""
        from rheojax.gui.resources.styles import get_stylesheet

        for theme in ["light", "dark"]:
            css = get_stylesheet(theme)

            # Check for common QSS elements
            assert "{" in css and "}" in css, f"{theme}: Should have selector blocks"
            assert "color:" in css or "background:" in css or "border:" in css, (
                f"{theme}: Should have style properties"
            )

    @pytest.mark.smoke
    def test_stylesheet_resource_files_exist(self) -> None:
        """Test stylesheet resource files exist on disk."""
        from rheojax.gui.resources.styles import STYLES_DIR

        light_file = STYLES_DIR / "light.qss"
        dark_file = STYLES_DIR / "dark.qss"

        assert light_file.exists(), f"Light stylesheet should exist at {light_file}"
        assert dark_file.exists(), f"Dark stylesheet should exist at {dark_file}"


# =============================================================================
# Test Qt Integration (Requires PySide6)
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestQtIntegration:
    """Tests that require PySide6 and Qt functionality.

    These tests use a QApplication and test Qt-specific behavior.
    Automatically skipped if PySide6 is not installed.
    """

    @pytest.mark.smoke
    def test_create_workspace_window_reachable_from_entrypoint(self, qapp) -> None:
        """The exact function `rheojax-gui`'s main() calls must build a real WorkspaceWindow.

        This is the entrypoint-level regression test: it proves the
        workspace shell is reachable from the CLI entrypoint, not just
        constructible in isolation (dead code from a user's perspective
        otherwise).
        """
        from rheojax.gui.main import _create_workspace_window
        from rheojax.gui.workspace.window import WorkspaceWindow

        window = _create_workspace_window()

        assert isinstance(window, WorkspaceWindow)
        assert window.mode() == "fit"
        assert window.active_step_count() == 6

        window.deleteLater()

    def test_stylesheet_applies_without_error(self, qapp) -> None:
        """Test stylesheet can be applied to QApplication without error."""
        from rheojax.gui.resources.styles import get_stylesheet

        try:
            stylesheet = get_stylesheet("light")
            qapp.setStyleSheet(stylesheet)
            # If we get here, it worked
            assert True, "Stylesheet should apply without error"
        except Exception as e:
            pytest.fail(f"Stylesheet application raised: {e}")


# =============================================================================
# Test Integration Scenarios
# =============================================================================


class TestIntegrationScenarios:
    """Integration tests combining multiple components.

    Tests realistic usage patterns and workflows.
    """

    @pytest.mark.smoke
    def test_state_and_services_work_together(self) -> None:
        """Test that state management and services can work together."""
        from rheojax.gui.foundation.state import AppState
        from rheojax.gui.services.model_service import ModelService

        # Create state
        state = AppState()
        assert state is not None

        # Create service
        service = ModelService()
        assert service is not None

        # Both should coexist without issues
        try:
            models = service.get_available_models()
            assert isinstance(models, dict)
        except AttributeError:
            # Registry might not be fully initialized in test environment
            pytest.skip("Registry not fully initialized")

    @pytest.mark.smoke
    def test_stylesheet_selection_matches_state_theme(self) -> None:
        """Test that stylesheet selection matches state theme values."""
        from rheojax.gui.foundation.state import UiState
        from rheojax.gui.resources.styles import get_stylesheet

        for theme in ["light", "dark"]:
            state = UiState(theme=theme)
            stylesheet = get_stylesheet(state.theme)

            assert isinstance(stylesheet, str)
            assert len(stylesheet) > 100

    @pytest.mark.smoke
    def test_multiple_services_instantiate(self) -> None:
        """Test all services can be instantiated together."""
        from rheojax.gui.services import (
            BayesianService,
            DataService,
            ExportService,
            ModelService,
            PlotService,
            TransformService,
        )

        services = {
            "data": DataService(),
            "model": ModelService(),
            "bayesian": BayesianService(),
            "transform": TransformService(),
            "plot": PlotService(),
            "export": ExportService(),
        }

        assert len(services) == 6, "All services should be instantiated"
        assert all(service is not None for service in services.values()), (
            "All services should be non-None"
        )


# =============================================================================
# Test Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Test edge cases and error handling in GUI components."""

    @pytest.mark.smoke
    def test_parameter_state_with_zero_value(self) -> None:
        """Test ParameterState handles zero value correctly."""
        from rheojax.gui.foundation.state import ParameterState

        param = ParameterState(
            name="linear_term",
            value=0.0,
            min_bound=-1.0,
            max_bound=1.0,
        )

        assert param.value == 0.0, "Should handle zero value"

    @pytest.mark.smoke
    def test_parameter_state_with_negative_bounds(self) -> None:
        """Test ParameterState handles negative bounds."""
        from rheojax.gui.foundation.state import ParameterState

        param = ParameterState(
            name="offset",
            value=-100.0,
            min_bound=-1000.0,
            max_bound=0.0,
        )

        assert param.value == -100.0, "Should handle negative values"
        assert param.min_bound == -1000.0, "Should handle negative bounds"

    @pytest.mark.smoke
    def test_model_service_get_available_models_not_empty(self) -> None:
        """Test ModelService returns non-empty model list."""
        from rheojax.gui.services.model_service import ModelService

        service = ModelService()
        try:
            models = service.get_available_models()
            assert len(models) > 0, "Should have available models"
        except AttributeError:
            # Registry might not be fully initialized in test environment
            pytest.skip("Registry not fully initialized")

    def test_plot_service_colorblind_palette_unique_colors(self) -> None:
        """Test colorblind palette has unique colors."""
        from rheojax.gui.services.plot_service import PlotService

        service = PlotService()
        palette = service.get_colorblind_palette()

        # Check uniqueness
        assert len(palette) == len(set(palette)), "All colors should be unique"

    @pytest.mark.smoke
    def test_dataset_state_metadata_is_mutable(self) -> None:
        """Test DatasetState metadata can be modified."""
        from rheojax.gui.foundation.state import DatasetState

        ds = DatasetState(
            id="test",
            name="Test",
            file_path=Path("/test.csv"),
            test_mode="oscillation",
        )

        ds.metadata["key"] = "value"
        assert ds.metadata["key"] == "value", "Should be able to modify metadata"


# =============================================================================
# Test JAX Utils (Home Page Dependencies)
# =============================================================================


class TestJaxUtils:
    """Test JAX utility functions used by HomePage.

    These tests verify the get_jax_info function returns the correct
    structure expected by HomePage._update_jax_status().
    """

    @pytest.mark.smoke
    def test_get_jax_info_import(self) -> None:
        """Test get_jax_info function can be imported."""
        from rheojax.gui.utils.jax_utils import get_jax_info

        assert callable(get_jax_info), "get_jax_info should be callable"

    @pytest.mark.smoke
    def test_get_jax_info_returns_expected_keys(self) -> None:
        """Test get_jax_info returns all keys expected by HomePage."""
        from rheojax.gui.utils.jax_utils import get_jax_info

        info = get_jax_info()

        expected_keys = [
            "devices",
            "default_device",
            "memory_used_mb",
            "memory_total_mb",
            "float64_enabled",
            "jit_cache_count",
        ]

        for key in expected_keys:
            assert key in info, f"get_jax_info should return '{key}'"

    @pytest.mark.smoke
    def test_get_jax_info_devices_is_list(self) -> None:
        """Test get_jax_info returns devices as list."""
        from rheojax.gui.utils.jax_utils import get_jax_info

        info = get_jax_info()

        assert isinstance(info["devices"], list), "devices should be a list"
        assert len(info["devices"]) > 0, "devices should not be empty"
        assert all(isinstance(d, str) for d in info["devices"]), (
            "devices should contain strings"
        )

    @pytest.mark.smoke
    def test_get_jax_info_default_device_is_string(self) -> None:
        """Test get_jax_info returns default_device as string."""
        from rheojax.gui.utils.jax_utils import get_jax_info

        info = get_jax_info()

        assert isinstance(info["default_device"], str), (
            "default_device should be a string"
        )

    @pytest.mark.smoke
    def test_get_jax_info_memory_values_are_numeric(self) -> None:
        """Test get_jax_info returns numeric memory values."""
        from rheojax.gui.utils.jax_utils import get_jax_info

        info = get_jax_info()

        assert isinstance(info["memory_used_mb"], (int, float)), (
            "memory_used_mb should be numeric"
        )
        assert isinstance(info["memory_total_mb"], (int, float)), (
            "memory_total_mb should be numeric"
        )

    @pytest.mark.smoke
    def test_get_jax_info_float64_is_bool(self) -> None:
        """Test get_jax_info returns float64_enabled as bool."""
        from rheojax.gui.utils.jax_utils import get_jax_info

        info = get_jax_info()

        assert isinstance(info["float64_enabled"], bool), (
            "float64_enabled should be bool"
        )

    @pytest.mark.smoke
    def test_get_jax_device_info_still_works(self) -> None:
        """Test original get_jax_device_info function still works."""
        from rheojax.gui.utils.jax_utils import get_jax_device_info

        info = get_jax_device_info()

        # Original function returns nested structure
        assert "memory" in info, "get_jax_device_info should return 'memory'"
        assert "backend" in info, "get_jax_device_info should return 'backend'"
        assert "float64_enabled" in info, (
            "get_jax_device_info should return 'float64_enabled'"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
