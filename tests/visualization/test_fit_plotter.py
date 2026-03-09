"""Tests for the FitPlotter class and Bayesian visualization primitives."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from rheojax.core.jax_config import safe_import_jax
from rheojax.visualization.fit_plotter import (
    FitPlotter,
    compute_credible_band,
    generate_diagnostic_suite,
)

jax, jnp = safe_import_jax()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def simple_data():
    """Simple exponential decay data for scalar fits."""
    np.random.seed(42)
    x = np.linspace(0.1, 10, 50)
    y_true = 2.5 * np.exp(-0.3 * x)
    y_data = y_true + np.random.normal(0, 0.1, len(x))
    return x, y_data, y_true


@pytest.fixture
def complex_data():
    """Complex modulus data for oscillation fits."""
    freq = np.logspace(-2, 2, 50)
    G0 = 1e5
    tau = 1.0
    omega_tau = freq * tau
    G_prime = G0 * omega_tau**2 / (1 + omega_tau**2)
    G_double_prime = G0 * omega_tau / (1 + omega_tau**2)
    G_star = G_prime + 1j * G_double_prime
    noise = np.random.RandomState(42).normal(0, 100, len(freq))
    G_star += noise + 1j * noise
    return freq, G_star


@pytest.fixture
def mock_model():
    """Mock model with model_function and parameters."""
    model = MagicMock()
    model.parameters = MagicMock()
    model.parameters.keys.return_value = ["G0", "tau"]
    model.__class__.__name__ = "Maxwell"

    def model_function(X, params, test_mode=None, **kwargs):
        G0, tau = params[0], params[1]
        return G0 * jnp.exp(-X / tau)

    model.model_function = model_function
    model._test_mode = "relaxation"
    return model


@pytest.fixture
def mock_fit_result():
    """Mock FitResult with covariance."""
    result = MagicMock()
    result.model_name = "maxwell"
    result.model_class_name = "Maxwell"
    result.protocol = "relaxation"
    result.params = {"G0": 2.5, "tau": 3.33}

    opt_result = MagicMock()
    opt_result.x = np.array([2.5, 3.33])
    opt_result.pcov = np.array([[0.01, 0.0], [0.0, 0.05]])
    result.optimization_result = opt_result
    result.fitted_curve = 2.5 * np.exp(-np.linspace(0.1, 10, 50) / 3.33)

    return result


@pytest.fixture
def mock_bayesian_result():
    """Mock BayesianResult with posterior samples."""
    result = MagicMock()
    rng = np.random.RandomState(42)
    result.posterior_samples = {
        "G0": rng.normal(2.5, 0.1, 1000),
        "tau": rng.normal(3.33, 0.2, 1000),
        "sigma": rng.exponential(0.1, 1000),
    }
    result.mcmc = MagicMock()
    result.num_chains = 4
    result.num_samples = 250

    # Mock to_inference_data for diagnostics
    result.to_inference_data = MagicMock()
    return result


@pytest.fixture
def plotter():
    """FitPlotter instance."""
    return FitPlotter()


# ---------------------------------------------------------------------------
# compute_credible_band tests
# ---------------------------------------------------------------------------

class TestComputeCredibleBand:
    """Tests for compute_credible_band()."""

    @pytest.mark.smoke
    def test_basic_scalar_output(self, simple_data, mock_model):
        """Credible band returns median, lower, upper with correct shapes."""
        x, _, _ = simple_data
        posterior = {
            "G0": np.random.normal(2.5, 0.1, 200),
            "tau": np.random.normal(3.33, 0.2, 200),
        }

        y_median, y_lower, y_upper = compute_credible_band(
            model_fn=mock_model.model_function,
            x_pred=x,
            posterior_samples=posterior,
            param_names=["G0", "tau"],
            credible_level=0.95,
            test_mode="relaxation",
        )

        assert y_median.shape == x.shape
        assert y_lower.shape == x.shape
        assert y_upper.shape == x.shape
        # Lower < median < upper
        assert np.all(y_lower <= y_median + 1e-10)
        assert np.all(y_median <= y_upper + 1e-10)

    @pytest.mark.smoke
    def test_noise_params_filtered(self, simple_data, mock_model):
        """Noise parameters (sigma) are automatically excluded."""
        x, _, _ = simple_data
        posterior = {
            "G0": np.random.normal(2.5, 0.1, 100),
            "tau": np.random.normal(3.33, 0.2, 100),
            "sigma": np.random.exponential(0.1, 100),
        }

        y_median, y_lower, y_upper = compute_credible_band(
            model_fn=mock_model.model_function,
            x_pred=x,
            posterior_samples=posterior,
            param_names=["G0", "tau", "sigma"],
            credible_level=0.95,
            test_mode="relaxation",
        )

        assert y_median.shape == x.shape
        assert np.all(np.isfinite(y_median))

    def test_subsampling(self, simple_data, mock_model):
        """Large posterior is subsampled to max_draws."""
        x, _, _ = simple_data
        n_total = 5000
        posterior = {
            "G0": np.random.normal(2.5, 0.1, n_total),
            "tau": np.random.normal(3.33, 0.2, n_total),
        }

        # Should not raise — internally subsamples to 100
        y_median, y_lower, y_upper = compute_credible_band(
            model_fn=mock_model.model_function,
            x_pred=x,
            posterior_samples=posterior,
            param_names=["G0", "tau"],
            max_draws=100,
            test_mode="relaxation",
        )

        assert y_median.shape == x.shape

    def test_complex_output(self, complex_data):
        """Credible band handles complex model output (G' + iG'')."""
        freq, _ = complex_data

        def complex_model(X, params, test_mode=None, **kwargs):
            G0, tau = params[0], params[1]
            omega_tau = X * tau
            Gp = G0 * omega_tau**2 / (1 + omega_tau**2)
            Gpp = G0 * omega_tau / (1 + omega_tau**2)
            return Gp + 1j * Gpp

        posterior = {
            "G0": np.random.normal(1e5, 1e3, 200),
            "tau": np.random.normal(1.0, 0.05, 200),
        }

        y_median, y_lower, y_upper = compute_credible_band(
            model_fn=complex_model,
            x_pred=freq,
            posterior_samples=posterior,
            param_names=["G0", "tau"],
            credible_level=0.95,
        )

        assert np.iscomplexobj(y_median)
        assert np.iscomplexobj(y_lower)
        assert np.iscomplexobj(y_upper)
        # G' bands: lower.real <= median.real <= upper.real
        assert np.all(np.real(y_lower) <= np.real(y_median) + 1e-6)

    def test_credible_level_affects_width(self, simple_data, mock_model):
        """Higher credible level produces wider bands."""
        x, _, _ = simple_data
        posterior = {
            "G0": np.random.normal(2.5, 0.3, 500),
            "tau": np.random.normal(3.33, 0.5, 500),
        }

        _, lo_90, hi_90 = compute_credible_band(
            mock_model.model_function, x, posterior, ["G0", "tau"],
            credible_level=0.90, test_mode="relaxation",
        )
        _, lo_99, hi_99 = compute_credible_band(
            mock_model.model_function, x, posterior, ["G0", "tau"],
            credible_level=0.99, test_mode="relaxation",
        )

        width_90 = np.mean(hi_90 - lo_90)
        width_99 = np.mean(hi_99 - lo_99)
        assert width_99 > width_90


# ---------------------------------------------------------------------------
# FitPlotter tests
# ---------------------------------------------------------------------------

class TestFitPlotterNLSQ:
    """Tests for FitPlotter.plot_nlsq()."""

    @pytest.mark.smoke
    def test_scalar_fit_no_residuals(
        self, plotter, simple_data, mock_fit_result, mock_model
    ):
        """Scalar NLSQ fit without residuals creates single axes."""
        x, y, _ = simple_data

        fig, axes = plotter.plot_nlsq(
            x, y, mock_fit_result, mock_model,
            show_residuals=False, show_uncertainty=False,
        )

        assert fig is not None
        assert axes is not None
        plt.close(fig)

    @pytest.mark.smoke
    def test_scalar_fit_with_residuals(
        self, plotter, simple_data, mock_fit_result, mock_model
    ):
        """Scalar NLSQ fit with residuals creates 2-row layout."""
        x, y, _ = simple_data

        fig, axes = plotter.plot_nlsq(
            x, y, mock_fit_result, mock_model,
            show_residuals=True, show_uncertainty=True,
        )

        assert fig is not None
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 2
        plt.close(fig)

    @pytest.mark.smoke
    def test_complex_fit(self, plotter, complex_data, mock_model):
        """Complex oscillation fit creates 2-column layout."""
        freq, G_star = complex_data

        # Mock fit result for complex data
        fit_result = MagicMock()
        fit_result.model_name = "maxwell"
        fit_result.model_class_name = "Maxwell"
        fit_result.protocol = "oscillation"
        fit_result.params = {"G0": 1e5, "tau": 1.0}

        opt_result = MagicMock()
        opt_result.x = np.array([1e5, 1.0])
        opt_result.pcov = np.eye(2) * 100
        fit_result.optimization_result = opt_result
        fit_result.fitted_curve = None

        # Override model function for complex
        def complex_model_fn(X, params, test_mode=None, **kwargs):
            G0, tau = params[0], params[1]
            omega_tau = X * tau
            Gp = G0 * omega_tau**2 / (1 + omega_tau**2)
            Gpp = G0 * omega_tau / (1 + omega_tau**2)
            return Gp + 1j * Gpp

        mock_model.model_function = complex_model_fn

        fig, axes = plotter.plot_nlsq(
            freq, G_star, fit_result, mock_model,
            show_residuals=True,
        )

        assert fig is not None
        assert isinstance(axes, np.ndarray)
        assert axes.shape == (2, 2)  # 2 rows (fit+resid) x 2 cols (G', G'')
        plt.close(fig)

    def test_deformation_mode_labels(
        self, plotter, simple_data, mock_fit_result, mock_model
    ):
        """Deformation mode affects axis labels."""
        x, y, _ = simple_data

        fig, _ = plotter.plot_nlsq(
            x, y, mock_fit_result, mock_model,
            show_residuals=False,
            deformation_mode="tension",
        )

        # Just verify it doesn't crash — label checking is visual
        assert fig is not None
        plt.close(fig)


class TestFitPlotterBayesian:
    """Tests for FitPlotter.plot_bayesian()."""

    @pytest.mark.smoke
    def test_scalar_bayesian(
        self, plotter, simple_data, mock_bayesian_result, mock_model
    ):
        """Bayesian posterior predictive for scalar data."""
        x, y, _ = simple_data

        fig, axes = plotter.plot_bayesian(
            x, y, mock_bayesian_result, mock_model,
            credible_level=0.95, max_draws=100,
        )

        assert fig is not None
        plt.close(fig)

    def test_bayesian_with_nlsq_overlay(
        self, plotter, simple_data, mock_bayesian_result,
        mock_fit_result, mock_model
    ):
        """Bayesian fit with NLSQ overlay."""
        x, y, _ = simple_data

        fig, axes = plotter.plot_bayesian(
            x, y, mock_bayesian_result, mock_model,
            show_nlsq_overlay=True, fit_result=mock_fit_result,
            max_draws=50,
        )

        assert fig is not None
        plt.close(fig)

    def test_bayesian_with_residuals(
        self, plotter, simple_data, mock_bayesian_result, mock_model
    ):
        """Bayesian fit with residuals subplot."""
        x, y, _ = simple_data

        fig, axes = plotter.plot_bayesian(
            x, y, mock_bayesian_result, mock_model,
            show_residuals=True, max_draws=50,
        )

        assert fig is not None
        assert isinstance(axes, np.ndarray)
        plt.close(fig)


class TestFitPlotterComparison:
    """Tests for FitPlotter.plot_comparison()."""

    @pytest.mark.smoke
    def test_comparison_plot(
        self, plotter, simple_data, mock_fit_result,
        mock_bayesian_result, mock_model
    ):
        """Side-by-side NLSQ vs Bayesian comparison."""
        x, y, _ = simple_data

        fig, axes = plotter.plot_comparison(
            x, y, mock_fit_result, mock_bayesian_result, mock_model,
            max_draws=50,
        )

        assert fig is not None
        assert isinstance(axes, np.ndarray)
        assert len(axes) == 2
        plt.close(fig)

    def test_comparison_complex_raises(
        self, plotter, complex_data, mock_fit_result,
        mock_bayesian_result, mock_model
    ):
        """Comparison plot raises NotImplementedError for complex data."""
        freq, G_star = complex_data

        with pytest.raises(NotImplementedError, match="complex"):
            plotter.plot_comparison(
                freq, G_star, mock_fit_result, mock_bayesian_result, mock_model,
            )


class TestFitPlotterParameterTable:
    """Tests for FitPlotter.plot_parameter_table()."""

    @pytest.mark.smoke
    def test_nlsq_only_table(self, plotter, mock_fit_result):
        """Parameter table with NLSQ result only."""
        fig, ax = plotter.plot_parameter_table(fit_result=mock_fit_result)

        assert fig is not None
        plt.close(fig)

    def test_bayesian_only_table(self, plotter, mock_bayesian_result):
        """Parameter table with Bayesian result only."""
        fig, ax = plotter.plot_parameter_table(bayesian_result=mock_bayesian_result)

        assert fig is not None
        plt.close(fig)

    def test_combined_table(self, plotter, mock_fit_result, mock_bayesian_result):
        """Parameter table with both NLSQ and Bayesian results."""
        fig, ax = plotter.plot_parameter_table(
            fit_result=mock_fit_result,
            bayesian_result=mock_bayesian_result,
        )

        assert fig is not None
        plt.close(fig)

    def test_no_results_raises(self, plotter):
        """Parameter table raises if no results provided."""
        with pytest.raises(ValueError, match="At least one"):
            plotter.plot_parameter_table()


# ---------------------------------------------------------------------------
# generate_diagnostic_suite tests
# ---------------------------------------------------------------------------

class TestDiagnosticSuite:
    """Tests for generate_diagnostic_suite()."""

    @pytest.mark.smoke
    def test_arviz_not_installed(self, mock_bayesian_result):
        """Raises ImportError when ArviZ is missing."""
        with patch.dict("sys.modules", {"arviz": None}):
            with pytest.raises(ImportError, match="ArviZ"):
                generate_diagnostic_suite(mock_bayesian_result)

    def test_mcmc_none_raises(self):
        """Raises ValueError when mcmc is None."""
        result = MagicMock()
        result.mcmc = None

        with pytest.raises(ValueError, match="mcmc is None"):
            generate_diagnostic_suite(result)


# ---------------------------------------------------------------------------
# Helper method tests
# ---------------------------------------------------------------------------

class TestFitPlotterHelpers:
    """Tests for FitPlotter private helpers."""

    def test_make_pred_grid_log(self, plotter):
        """Log-spaced grid for data spanning > 1.5 decades."""
        x = np.logspace(-2, 2, 50)
        grid = plotter._make_pred_grid(x, 100)

        assert len(grid) == 100
        assert grid[0] == pytest.approx(x.min(), rel=1e-3)
        assert grid[-1] == pytest.approx(x.max(), rel=1e-3)
        # Should be log-spaced (check ratio is roughly constant)
        ratios = grid[1:] / grid[:-1]
        assert np.std(ratios) / np.mean(ratios) < 0.01

    def test_make_pred_grid_linear(self, plotter):
        """Linear grid for data spanning < 1.5 decades."""
        x = np.linspace(1, 10, 50)
        grid = plotter._make_pred_grid(x, 100)

        assert len(grid) == 100
        # Should be linearly spaced (check spacing is roughly constant)
        diffs = np.diff(grid)
        assert np.std(diffs) / np.mean(diffs) < 0.01

    def test_infer_log_x(self, plotter, mock_fit_result):
        """Log x inferred for oscillation/flow protocols."""
        mock_fit_result.protocol = "oscillation"
        assert plotter._infer_log_x(mock_fit_result) is True

        mock_fit_result.protocol = "relaxation"
        assert plotter._infer_log_x(mock_fit_result) is False

    def test_infer_log_y(self, plotter, mock_fit_result):
        """Log y inferred for oscillation/relaxation protocols."""
        mock_fit_result.protocol = "oscillation"
        assert plotter._infer_log_y(mock_fit_result) is True

        mock_fit_result.protocol = "relaxation"
        assert plotter._infer_log_y(mock_fit_result) is True

        mock_fit_result.protocol = "creep"
        assert plotter._infer_log_y(mock_fit_result) is False


# ---------------------------------------------------------------------------
# Style tests
# ---------------------------------------------------------------------------

class TestFitPlotterStyles:
    """Verify style parameter propagation."""

    @pytest.mark.parametrize("style", ["default", "publication", "presentation"])
    def test_styles_dont_crash(
        self, plotter, simple_data, mock_fit_result, mock_model, style
    ):
        """All three style presets work without error."""
        x, y, _ = simple_data

        fig, _ = plotter.plot_nlsq(
            x, y, mock_fit_result, mock_model,
            style=style, show_residuals=False, show_uncertainty=False,
        )

        assert fig is not None
        plt.close(fig)
