"""RheoJAX GUI Performance Benchmarks.

Performance tests for GUI service operations and background workers.

Markers:
    gui: All GUI-related tests
    benchmark: Performance benchmark tests

Run with:
    pytest tests/gui/test_gui_performance.py -v --benchmark-only
    pytest tests/gui/test_gui_performance.py -v -m benchmark

Benchmark targets:
    - Service instantiation: <100ms
    - Model listing: <50ms
    - Parameter retrieval: <20ms
    - Plot generation: <500ms
"""

from typing import Any

import numpy as np
import pytest

# Mark all tests as GUI and benchmark tests
pytestmark = [pytest.mark.gui, pytest.mark.benchmark]

# Check if PySide6 is available
try:
    from PySide6.QtWidgets import QApplication

    HAS_PYSIDE6 = True
except ImportError:
    HAS_PYSIDE6 = False


# =============================================================================
# Service Performance Benchmarks
# =============================================================================


class TestServicePerformance:
    """Performance benchmarks for GUI service layer."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        """Reset StateStore singleton before each test."""
        from rheojax.gui.state.store import StateStore

        StateStore.reset()
        yield
        StateStore.reset()

    def test_model_service_instantiation_time(self) -> None:
        """Benchmark ModelService instantiation time.

        Target: <100ms for service creation with registry lookup
        """
        import time

        from rheojax.gui.services.model_service import ModelService

        start = time.perf_counter()
        for _ in range(10):
            service = ModelService()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 10 * 1000  # ms
        print(f"ModelService instantiation: {avg_time:.2f}ms avg")

        # Should be fast
        assert avg_time < 100, f"ModelService instantiation too slow: {avg_time:.2f}ms"

    def test_model_listing_performance(self) -> None:
        """Benchmark model listing performance.

        Target: <50ms to list all available models
        """
        import time

        from rheojax.gui.services.model_service import ModelService

        service = ModelService()

        # Warmup
        service.get_available_models()

        start = time.perf_counter()
        for _ in range(100):
            models = service.get_available_models()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100 * 1000  # ms
        print(f"Model listing: {avg_time:.2f}ms avg")

        # Should have models and be fast
        assert len(models) > 0
        assert avg_time < 50, f"Model listing too slow: {avg_time:.2f}ms"

    def test_model_info_retrieval_performance(self) -> None:
        """Benchmark model info retrieval performance.

        Target: <20ms per model info lookup
        """
        import time

        from rheojax.gui.services.model_service import ModelService

        service = ModelService()

        # Warmup
        service.get_model_info("maxwell")

        start = time.perf_counter()
        for _ in range(50):
            info = service.get_model_info("maxwell")
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 50 * 1000  # ms
        print(f"Model info retrieval: {avg_time:.2f}ms avg")

        # Should return info and be fast
        assert info is not None
        assert avg_time < 20, f"Model info retrieval too slow: {avg_time:.2f}ms"

    def test_transform_service_listing_performance(self) -> None:
        """Benchmark transform listing performance.

        Target: <30ms to list all transforms
        """
        import time

        from rheojax.gui.services.transform_service import TransformService

        service = TransformService()

        # Warmup
        service.get_available_transforms()

        start = time.perf_counter()
        for _ in range(100):
            transforms = service.get_available_transforms()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100 * 1000  # ms
        print(f"Transform listing: {avg_time:.2f}ms avg")

        assert len(transforms) > 0
        assert avg_time < 30, f"Transform listing too slow: {avg_time:.2f}ms"

    def test_data_service_supported_formats(self) -> None:
        """Benchmark data format lookup performance.

        Target: <10ms for format enumeration
        """
        import time

        from rheojax.gui.services.data_service import DataService

        service = DataService()

        start = time.perf_counter()
        for _ in range(100):
            formats = service.get_supported_formats()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100 * 1000  # ms
        print(f"Supported formats lookup: {avg_time:.2f}ms avg")

        assert len(formats) > 0
        assert avg_time < 10, f"Format lookup too slow: {avg_time:.2f}ms"


# =============================================================================
# State Management Performance Benchmarks
# =============================================================================


class TestStatePerformance:
    """Performance benchmarks for state management."""

    @pytest.fixture(autouse=True)
    def reset_store(self) -> None:
        """Reset StateStore singleton before each test."""
        from rheojax.gui.state.store import StateStore

        StateStore.reset()
        yield
        StateStore.reset()

    def test_state_dispatch_performance(self) -> None:
        """Benchmark state dispatch performance.

        Target: <1ms per action dispatch
        """
        import time

        from rheojax.gui.state.store import StateStore

        store = StateStore()

        start = time.perf_counter()
        for i in range(1000):
            store.dispatch({"type": "SET_ACTIVE_MODEL", "model_name": f"model_{i}"})
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 1000 * 1000  # ms
        print(f"State dispatch: {avg_time:.3f}ms avg")

        assert avg_time < 1, f"State dispatch too slow: {avg_time:.3f}ms"

    def test_state_update_performance(self) -> None:
        """Benchmark state update performance.

        Target: <5ms per state update (includes clone)
        """
        import time

        from rheojax.gui.state.store import AppState, StateStore

        store = StateStore()

        def simple_updater(state: AppState) -> AppState:
            return AppState(**{**state.__dict__, "project_name": "Updated"})

        start = time.perf_counter()
        for _ in range(100):
            store.update_state(simple_updater, track_undo=False, emit_signal=False)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100 * 1000  # ms
        print(f"State update: {avg_time:.2f}ms avg")

        assert avg_time < 5, f"State update too slow: {avg_time:.2f}ms"

    def test_state_clone_performance(self) -> None:
        """Benchmark state cloning performance.

        Target: <10ms to clone full state
        """
        import time

        from rheojax.gui.state.store import StateStore

        store = StateStore()
        state = store.get_state()

        start = time.perf_counter()
        for _ in range(100):
            cloned = state.clone()
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 100 * 1000  # ms
        print(f"State clone: {avg_time:.2f}ms avg")

        assert cloned is not state
        assert avg_time < 10, f"State clone too slow: {avg_time:.2f}ms"


# =============================================================================
# Widget Performance Benchmarks (Require Qt)
# =============================================================================


@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
class TestWidgetPerformance:
    """Performance benchmarks for GUI widgets."""

    @pytest.fixture
    def qapp(self, qtbot: Any) -> QApplication:
        """Get or create QApplication instance."""
        app = QApplication.instance()
        if app is None:
            app = QApplication([])
        return app

    def test_plot_canvas_render_performance(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Benchmark plot canvas rendering performance.

        Target: <500ms for line plot with 1000 points
        """
        import time

        from rheojax.gui.widgets.plot_canvas import PlotCanvas

        canvas = PlotCanvas()
        qtbot.addWidget(canvas)

        x = np.linspace(0, 100, 1000)
        y = np.sin(x) + np.random.randn(1000) * 0.1

        start = time.perf_counter()
        for _ in range(10):
            canvas.clear()
            canvas.plot_data(x, y, label="data")
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 10 * 1000  # ms
        print(f"Plot canvas render (1000 points): {avg_time:.2f}ms avg")

        assert avg_time < 500, f"Plot canvas render too slow: {avg_time:.2f}ms"

        canvas.close()

    def test_parameter_table_population_performance(
        self, qapp: QApplication, qtbot: Any
    ) -> None:
        """Benchmark parameter table population performance.

        Target: <100ms to populate 20 parameters
        """
        import time

        from rheojax.gui.state.store import ParameterState
        from rheojax.gui.widgets.parameter_table import ParameterTable

        table = ParameterTable()
        qtbot.addWidget(table)

        # Create 20 parameters
        params = {
            f"param_{i}": ParameterState(
                name=f"param_{i}",
                value=float(i),
                min_bound=0.0,
                max_bound=1e6,
            )
            for i in range(20)
        }

        start = time.perf_counter()
        for _ in range(10):
            table.set_parameters(params)
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 10 * 1000  # ms
        print(f"Parameter table population (20 params): {avg_time:.2f}ms avg")

        assert table.rowCount() == 20
        assert avg_time < 100, f"Parameter table population too slow: {avg_time:.2f}ms"

        table.close()


# =============================================================================
# NLSQ/MCMC Performance Baselines
# =============================================================================


class TestFittingPerformanceBaseline:
    """Baseline performance tests for NLSQ/MCMC fitting.

    These establish performance baselines for comparison.
    """

    @pytest.fixture
    def sample_maxwell_data(self) -> tuple[np.ndarray, np.ndarray]:
        """Generate sample Maxwell relaxation data."""
        t = np.logspace(-3, 2, 100)
        G0, tau = 1000.0, 1.0
        G_t = G0 * np.exp(-t / tau)

        # Add noise
        rng = np.random.default_rng(42)
        G_t *= 1 + 0.02 * rng.standard_normal(len(t))
        return t, G_t

    def test_nlsq_maxwell_fit_baseline(
        self, sample_maxwell_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Establish NLSQ fitting baseline for Maxwell model.

        Reference: ~50-200ms for simple Maxwell fit
        """
        import time

        from rheojax.models import Maxwell

        t, G_t = sample_maxwell_data
        model = Maxwell()

        # Warmup (JIT compilation)
        model.fit(t, G_t, test_mode="relaxation")

        start = time.perf_counter()
        for _ in range(5):
            model.fit(t, G_t, test_mode="relaxation")
        elapsed = time.perf_counter() - start

        avg_time = elapsed / 5 * 1000  # ms
        print(f"NLSQ Maxwell fit baseline: {avg_time:.2f}ms avg")

        # Log baseline (no hard assertion - this is for tracking)
        assert avg_time > 0

    @pytest.mark.slow
    def test_bayesian_maxwell_fit_baseline(
        self, sample_maxwell_data: tuple[np.ndarray, np.ndarray]
    ) -> None:
        """Establish Bayesian fitting baseline for Maxwell model.

        Reference: ~5-30s for MCMC with 500 warmup, 500 samples
        Note: Marked slow - runs only when explicitly included
        """
        import time

        from rheojax.models import Maxwell

        t, G_t = sample_maxwell_data
        model = Maxwell()

        # NLSQ first (warm start)
        model.fit(t, G_t, test_mode="relaxation")

        start = time.perf_counter()
        result = model.fit_bayesian(
            t,
            G_t,
            test_mode="relaxation",
            num_warmup=100,  # Reduced for test speed
            num_samples=100,
        )
        elapsed = time.perf_counter() - start

        print(f"Bayesian Maxwell fit baseline: {elapsed:.2f}s")

        # Verify result
        assert result is not None
        assert hasattr(result, "posterior_samples") or hasattr(result, "samples")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
