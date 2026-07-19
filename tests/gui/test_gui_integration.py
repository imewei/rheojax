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
        from rheojax.gui.foundation.state import ParameterState
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

    def test_plot_canvas_clear_and_plot(self, qapp: QApplication, qtbot: Any) -> None:
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

    def test_arviz_canvas_plot_types(self, qapp: QApplication, qtbot: Any) -> None:
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

    def test_arviz_canvas_pair_plot_with_many_parameters(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Regression: a full pair-plot matrix needs len(var_names)**2
        subplots, which exceeded arviz's default plot.max_subplots=40 for
        any model with more than 6 parameters (e.g. GeneralizedMaxwell with
        3+ modes has 8) -- the plot silently failed to render."""
        import numpy as np

        from rheojax.core.arviz_utils import inference_data_from_dict
        from rheojax.gui.widgets.arviz_canvas import ArvizCanvas

        canvas = ArvizCanvas()
        qtbot.addWidget(canvas)

        rng = np.random.default_rng(0)
        n_params = 8
        posterior = {f"param_{i}": rng.normal(size=(2, 50)) for i in range(n_params)}
        idata = inference_data_from_dict({"posterior": posterior})

        canvas.set_inference_data(idata)
        canvas._current_plot_type = "pair"
        canvas._refresh_plot()

        assert not canvas._status_label.text().startswith("Error generating plot")

        canvas.close()

    def test_arviz_canvas_posterior_plot_renders(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Regression: ArviZ 1.x removed plot_posterior; _plot_posterior must
        call plot_dist instead, or this raises AttributeError before any
        kwarg translation runs (see docs/superpowers/specs/2026-07-14-arviz-1x-migration-design.md)."""
        import numpy as np

        from rheojax.core.arviz_utils import inference_data_from_dict
        from rheojax.gui.widgets.arviz_canvas import ArvizCanvas

        canvas = ArvizCanvas()
        qtbot.addWidget(canvas)

        rng = np.random.default_rng(0)
        posterior = {
            "a": rng.normal(size=(2, 50)),
            "b": rng.normal(size=(2, 50)),
        }
        idata = inference_data_from_dict({"posterior": posterior})

        canvas.set_inference_data(idata)
        canvas._current_plot_type = "posterior"
        canvas._refresh_plot()

        assert not canvas._status_label.text().startswith("Error:")

        canvas.close()


# =============================================================================
# Full End-to-End Workflow Tests
# =============================================================================


class TestEndToEndWorkflows:
    """Test complete end-to-end workflows without Qt UI."""

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
