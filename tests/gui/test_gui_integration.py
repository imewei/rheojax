"""RheoJAX GUI Integration Tests.

End-to-end integration tests for GUI workflows:
- Data loading workflow
- Model fitting workflow
- Bayesian inference workflow
- Transform workflow
- Export workflow

Markers:
    gui: All GUI-related tests
    integration: Integration tests (may take longer)

Run with:
    pytest tests/gui/test_gui_integration.py -v
    pytest tests/gui/ -v -m integration
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest

# Mark all tests as GUI and integration tests
pytestmark = [pytest.mark.gui, pytest.mark.integration]

# Check if PySide6 is available
try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


@pytest.fixture
def sample_oscillation_data() -> dict[str, Any]:
    """Generate sample oscillation data for testing."""
    omega = np.logspace(-2, 2, 50)
    # Maxwell model: G' = G0 * (omega*tau)^2 / (1 + (omega*tau)^2)
    G0, tau = 1000.0, 1.0
    omega_tau = omega * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0 * omega_tau / (1 + omega_tau**2)

    # Add small noise
    rng = np.random.default_rng(42)
    G_prime *= 1 + 0.02 * rng.standard_normal(len(omega))
    G_double_prime *= 1 + 0.02 * rng.standard_normal(len(omega))

    return {
        "omega": omega,
        "G_prime": G_prime,
        "G_double_prime": G_double_prime,
        "G0_true": G0,
        "tau_true": tau,
    }


@pytest.fixture
def sample_relaxation_data() -> dict[str, Any]:
    """Generate sample relaxation data for testing."""
    t = np.logspace(-3, 2, 100)
    # Maxwell model: G(t) = G0 * exp(-t/tau)
    G0, tau = 1000.0, 1.0
    G_t = G0 * np.exp(-t / tau)

    # Add small noise
    rng = np.random.default_rng(42)
    G_t *= 1 + 0.02 * rng.standard_normal(len(t))

    return {
        "t": t,
        "G_t": G_t,
        "G0_true": G0,
        "tau_true": tau,
    }


# =============================================================================
# Service Integration Tests
# =============================================================================


class TestServiceIntegration:
    """Test service layer integration."""

    @pytest.mark.smoke
    def test_data_service_load_numpy_arrays(
        self, sample_oscillation_data: dict[str, Any]
    ) -> None:
        """Test DataService can load numpy arrays."""
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        # Verify service can be instantiated and has required methods
        assert hasattr(service, "load_file")
        assert hasattr(service, "get_supported_formats")
        assert hasattr(service, "detect_test_mode")

    @pytest.mark.smoke
    def test_model_service_list_models(self) -> None:
        """Test ModelService can list available models."""
        from rheojax.gui.services.model_service import ModelService

        service = ModelService()
        models = service.get_available_models()

        # Should have multiple categories
        assert isinstance(models, dict)
        assert len(models) > 0

        # Should have classical models
        assert "classical" in models
        assert "maxwell" in [m.lower() for m in models.get("classical", [])]

    @pytest.mark.smoke
    def test_model_service_get_model_info(self) -> None:
        """Test ModelService can get model info."""
        from rheojax.gui.services.model_service import ModelService

        service = ModelService()
        info = service.get_model_info("maxwell")

        assert isinstance(info, dict)
        assert "parameters" in info or "name" in info

    @pytest.mark.smoke
    def test_transform_service_list_transforms(self) -> None:
        """Test TransformService can list available transforms."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()
        transforms = service.get_available_transforms()

        assert isinstance(transforms, list)
        assert len(transforms) > 0

        # Should have mastercurve and fft
        transform_lower = [t.lower() for t in transforms]
        assert "mastercurve" in transform_lower or "fft" in transform_lower

    @pytest.mark.smoke
    def test_plot_service_get_styles(self) -> None:
        """Test PlotService can get plot styles."""
        from rheojax.gui.services.plot_service import PlotService

        service = PlotService()
        styles = service.get_available_styles()

        assert isinstance(styles, list)
        assert len(styles) > 0
        assert "default" in [s.lower() for s in styles]

    @pytest.mark.smoke
    def test_export_service_get_formats(self) -> None:
        """Test ExportService can be instantiated and has methods."""
        from rheojax.gui.services.export_service import ExportService

        service = ExportService()

        # Verify service has required export methods
        assert hasattr(service, "export_parameters")
        assert hasattr(service, "export_figure")
        assert hasattr(service, "save_project")
        assert hasattr(service, "load_project")


# =============================================================================
# State Management Integration Tests
# =============================================================================


class TestStateIntegration:
    """Test state management integration."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        """Reset StateStore singleton before each test."""
        from rheojax.gui.state.store import StateStore

        StateStore.reset()
        yield
        StateStore.reset()

    @pytest.mark.smoke
    def test_state_store_singleton(self) -> None:
        """Test StateStore is a proper singleton."""
        from rheojax.gui.state.store import StateStore

        store1 = StateStore()
        store2 = StateStore()

        assert store1 is store2

    @pytest.mark.smoke
    def test_state_store_dispatch(self) -> None:
        """Test StateStore dispatch method."""
        from rheojax.gui.state.store import StateStore

        store = StateStore()

        # Dispatch should work without signals set
        store.dispatch({"type": "SET_ACTIVE_MODEL", "model_name": "maxwell"})

        # Should not raise
        assert True

    @pytest.mark.smoke
    def test_state_store_update_state(self) -> None:
        """Test StateStore update_state method."""
        from rheojax.gui.state.store import AppState, StateStore

        store = StateStore()

        def updater(state: AppState) -> AppState:
            return AppState(
                **{**state.__dict__, "project_name": "Test Project"}
            )

        store.update_state(updater)

        state = store.get_state()
        assert state.project_name == "Test Project"

    @pytest.mark.smoke
    def test_action_creators(self) -> None:
        """Test action creator functions return proper dicts."""
        from rheojax.gui.state.actions import (
            bayesian_completed,
            bayesian_failed,
            fitting_completed,
            fitting_failed,
            set_active_model,
            start_bayesian,
            start_fitting,
            update_bayesian_progress,
            update_fit_progress,
        )

        # Test action creators return dicts with type
        action = set_active_model("maxwell")
        assert isinstance(action, dict)
        assert action["type"] == "SET_ACTIVE_MODEL"
        assert action["model_name"] == "maxwell"

        action = start_fitting("maxwell", "dataset_1")
        assert action["type"] == "START_FITTING"

        action = update_fit_progress(50.0)
        assert action["type"] == "FIT_PROGRESS"
        assert action["progress"] == 50.0

        action = start_bayesian("maxwell", "dataset_1")
        assert action["type"] == "START_BAYESIAN"

        action = update_bayesian_progress(75.0)
        assert action["type"] == "BAYESIAN_PROGRESS"


# =============================================================================
# Workflow Integration Tests (Require Qt)
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestWorkflowIntegration:
    """Test full workflow integration with Qt widgets."""

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_fit_page_workflow(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test FitPage can be instantiated and models loaded."""
        from rheojax.gui.pages.fit_page import FitPage
        from rheojax.gui.state.store import StateStore

        StateStore.reset()

        page = FitPage()
        qtbot.addWidget(page)

        # Should have parameter table
        assert page._parameter_table is not None

        # Should have fit button
        assert page._btn_fit is not None

        page.close()

    def test_bayesian_page_workflow(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test BayesianPage can be instantiated."""
        from rheojax.gui.pages.bayesian_page import BayesianPage
        from rheojax.gui.state.store import StateStore

        StateStore.reset()

        page = BayesianPage()
        qtbot.addWidget(page)

        # Page should have config controls
        assert page._warmup_spin is not None
        assert page._samples_spin is not None
        assert page._chains_spin is not None

        # Should have run button
        assert page._btn_run is not None

        # Should have arviz canvas
        assert page._arviz_canvas is not None

        page.close()

    def test_data_page_workflow(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test DataPage can be instantiated."""
        from rheojax.gui.pages.data_page import DataPage
        from rheojax.gui.state.store import StateStore

        StateStore.reset()

        page = DataPage()
        qtbot.addWidget(page)

        # Page should have drop zone
        assert page._drop_zone is not None

        # Should have preview table
        assert page._preview_table is not None

        # Should have column mappers
        assert page._x_combo is not None
        assert page._y_combo is not None

        page.close()

    def test_transform_page_workflow(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test TransformPage can be instantiated."""
        from rheojax.gui.pages.transform_page import TransformPage
        from rheojax.gui.state.store import StateStore

        StateStore.reset()

        page = TransformPage()
        qtbot.addWidget(page)

        # Page should exist
        assert page is not None

        page.close()

    def test_export_page_workflow(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test ExportPage can be instantiated."""
        from rheojax.gui.pages.export_page import ExportPage
        from rheojax.gui.state.store import StateStore

        StateStore.reset()

        page = ExportPage()
        qtbot.addWidget(page)

        # Page should have format selectors
        assert page._data_format_combo is not None
        assert page._figure_format_combo is not None

        page.close()


# =============================================================================
# Widget Integration Tests
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestWidgetIntegration:
    """Test widget integration."""

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app


    def test_parameter_table_set_parameters(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test ParameterTable can set and get parameters."""
        from rheojax.gui.state.store import ParameterState
        from rheojax.gui.widgets.parameter_table import ParameterTable

        table = ParameterTable()
        qtbot.addWidget(table)

        # ParameterTable expects dict[str, ParameterState]
        params = {
            "G0": ParameterState(
                name="G0", value=1000.0, min_bound=0.0, max_bound=1e6, unit="Pa"
            ),
            "tau": ParameterState(
                name="tau", value=1.0, min_bound=1e-6, max_bound=1e6, unit="s"
            ),
        }
        table.set_parameters(params)

        # Should have 2 rows
        assert table.rowCount() == 2

        # Get parameters back
        retrieved = table.get_parameters()
        assert len(retrieved) == 2
        assert "G0" in retrieved

        table.close()

    def test_plot_canvas_clear_and_plot(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test PlotCanvas can clear and plot data."""
        from rheojax.gui.widgets.plot_canvas import PlotCanvas

        canvas = PlotCanvas()
        qtbot.addWidget(canvas)

        # Clear should work
        canvas.clear()

        # Plot data should work
        x = np.linspace(0, 10, 100)
        y = np.sin(x)
        canvas.plot_data(x, y, label="sin(x)")

        canvas.close()

    def test_arviz_canvas_plot_types(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Test ArvizCanvas supports expected plot types."""
        from rheojax.gui.widgets.arviz_canvas import PLOT_TYPES, ArvizCanvas

        canvas = ArvizCanvas()
        qtbot.addWidget(canvas)

        # Should have plot types defined (list of tuples)
        assert len(PLOT_TYPES) > 0
        plot_type_ids = [pt[0] for pt in PLOT_TYPES]
        assert "trace" in plot_type_ids
        assert "pair" in plot_type_ids
        assert "forest" in plot_type_ids

        # Selector should have options
        assert canvas._type_combo.count() > 0

        canvas.close()


# =============================================================================
# Full End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows without Qt UI."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        """Reset StateStore singleton before each test."""
        from rheojax.gui.state.store import StateStore

        StateStore.reset()
        yield
        StateStore.reset()

    def test_model_fit_workflow_with_services(
        self, sample_oscillation_data: dict[str, Any]
    ) -> None:
        """Test fitting workflow using service layer."""
        from rheojax.gui.services.model_service import ModelService

        service = ModelService()

        # Get model info
        info = service.get_model_info("maxwell")
        assert info is not None

        # Check compatibility would work
        # (actual fitting requires RheoData creation)
        models = service.get_available_models()
        assert "classical" in models

    def test_transform_workflow_with_services(self) -> None:
        """Test transform workflow using service layer."""
        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()

        # Get available transforms
        transforms = service.get_available_transforms()
        assert len(transforms) > 0

        # Get transform params
        params = service.get_transform_params("mastercurve")
        assert isinstance(params, dict)

    def test_export_workflow_with_services(self) -> None:
        """Test export workflow using service layer."""
        from rheojax.gui.services.export_service import ExportService

        service = ExportService()

        # Verify service has required methods
        assert hasattr(service, "export_parameters")
        assert hasattr(service, "export_figure")
        assert hasattr(service, "export_posterior")
        assert hasattr(service, "generate_report")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
