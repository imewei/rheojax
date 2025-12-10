"""
Export Service
=============

Service for exporting results to various formats (HDF5, Excel, CSV, etc.).
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from matplotlib.figure import Figure

from rheojax.core.data import RheoData
from rheojax.io import save_excel, save_hdf5

logger = logging.getLogger(__name__)


class ExportService:
    """Service for result export operations.

    Features:
        - Multi-format export (HDF5, Excel, CSV, JSON)
        - Figure export (PNG, SVG, PDF)
        - Project file support (.rheo files)
        - Posterior sample export
        - Report generation

    Example
    -------
    >>> service = ExportService()
    >>> service.export_parameters(fit_result, 'params.csv', 'csv')
    >>> service.export_figure(fig, 'plot.png', dpi=300, format='png')
    """

    def __init__(self) -> None:
        """Initialize export service."""
        self._supported_formats = {
            "data": [".hdf5", ".h5", ".xlsx", ".csv", ".json"],
            "figure": [".png", ".pdf", ".svg", ".eps"],
            "project": [".rheo"],
        }

    def export_parameters(
        self,
        result: Any,
        path: Path | str,
        format: str | None = None,
    ) -> None:
        """Export fitted parameters to file.

        Parameters
        ----------
        result : FitResult or BayesianResult
            Fitting or Bayesian result
        path : Path or str
            Output file path
        format : str, optional
            Export format ('csv', 'json', 'xlsx', 'hdf5')
            Auto-detected from extension if None
        """
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        try:
            # Extract parameters
            if hasattr(result, "parameters"):
                params = result.parameters
            elif hasattr(result, "posterior_samples"):
                # For Bayesian results, use posterior means
                params = {k: float(np.mean(v)) for k, v in result.posterior_samples.items()}
            else:
                raise ValueError("Result does not contain parameters")

            if format == "csv":
                import pandas as pd

                df = pd.DataFrame([params])
                df.to_csv(path, index=False)

            elif format == "json":
                # Convert numpy types to Python types
                params_json = {k: float(v) if isinstance(v, np.ndarray) else v for k, v in params.items()}
                with open(path, "w") as f:
                    json.dump(params_json, f, indent=2)

            elif format in ["xlsx", "xls"]:
                import pandas as pd

                df = pd.DataFrame([params])
                df.to_excel(path, index=False)

            elif format in ["hdf5", "h5"]:
                import h5py

                with h5py.File(path, "w") as f:
                    params_group = f.create_group("parameters")
                    for key, value in params.items():
                        params_group.create_dataset(key, data=value)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported parameters to {path}")

        except Exception as e:
            logger.error(f"Parameter export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e

    def export_figure(
        self,
        fig: Figure,
        path: Path | str,
        dpi: int = 300,
        format: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Export figure to file.

        Parameters
        ----------
        fig : Figure
            Matplotlib figure
        path : Path or str
            Output file path
        dpi : int, default=300
            Resolution in dots per inch
        format : str, optional
            Figure format ('png', 'pdf', 'svg', 'eps')
            Auto-detected from extension if None
        **kwargs
            Additional savefig options
        """
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        try:
            fig.savefig(path, dpi=dpi, format=format, bbox_inches="tight", **kwargs)
            logger.info(f"Exported figure to {path}")

        except Exception as e:
            logger.error(f"Figure export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e

    def export_posterior(
        self,
        result: Any,
        path: Path | str,
        format: str = "hdf5",
    ) -> None:
        """Export posterior samples.

        Parameters
        ----------
        result : BayesianResult
            Bayesian inference result
        path : Path or str
            Output file path
        format : str, default='hdf5'
            Export format ('hdf5', 'xlsx', 'csv')
        """
        path = Path(path)

        try:
            if not hasattr(result, "posterior_samples"):
                raise ValueError("Result does not contain posterior samples")

            posterior_samples = result.posterior_samples

            if format in ["hdf5", "h5"]:
                import h5py

                with h5py.File(path, "w") as f:
                    posterior_group = f.create_group("posterior_samples")
                    for param, samples in posterior_samples.items():
                        posterior_group.create_dataset(param, data=samples)

                    # Add metadata
                    if hasattr(result, "metadata"):
                        meta_group = f.create_group("metadata")
                        for key, value in result.metadata.items():
                            if isinstance(value, (str, int, float, bool)):
                                meta_group.attrs[key] = value

            elif format in ["xlsx", "xls"]:
                import pandas as pd

                # Flatten samples to 2D
                data = {k: v.flatten() if v.ndim > 1 else v for k, v in posterior_samples.items()}
                df = pd.DataFrame(data)
                df.to_excel(path, index=False)

            elif format == "csv":
                import pandas as pd

                data = {k: v.flatten() if v.ndim > 1 else v for k, v in posterior_samples.items()}
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported posterior samples to {path}")

        except Exception as e:
            logger.error(f"Posterior export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e

    def save_project(
        self,
        state: dict[str, Any],
        path: Path | str,
    ) -> None:
        """Save project as .rheo file (ZIP with JSON + HDF5).

        Parameters
        ----------
        state : dict
            Application state (data, models, results, etc.)
        path : Path or str
            Output .rheo file path
        """
        path = Path(path)

        try:
            # Create temp files
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Save metadata as JSON
                metadata_path = tmpdir_path / "metadata.json"
                metadata = {
                    "version": "1.0",
                    "model_name": state.get("model_name"),
                    "test_mode": state.get("test_mode"),
                    "timestamp": str(np.datetime64("now")),
                }

                with open(metadata_path, "w") as f:
                    json.dump(metadata, f, indent=2)

                # Save data as HDF5
                if "data" in state and state["data"] is not None:
                    data_path = tmpdir_path / "data.hdf5"
                    save_hdf5(state["data"], str(data_path))

                # Save parameters as JSON
                if "parameters" in state and state["parameters"]:
                    params_path = tmpdir_path / "parameters.json"
                    params_json = {k: float(v) if isinstance(v, np.ndarray) else v for k, v in state["parameters"].items()}
                    with open(params_path, "w") as f:
                        json.dump(params_json, f, indent=2)

                # Create ZIP archive
                with ZipFile(path, "w") as zipf:
                    for file in tmpdir_path.glob("*"):
                        zipf.write(file, file.name)

            logger.info(f"Saved project to {path}")

        except Exception as e:
            logger.error(f"Project save failed: {e}")
            raise RuntimeError(f"Save failed: {e}") from e

    def load_project(self, path: Path | str) -> dict[str, Any]:
        """Load project from .rheo file.

        Parameters
        ----------
        path : Path or str
            .rheo file path

        Returns
        -------
        dict
            Application state
        """
        path = Path(path)

        try:
            import tempfile

            state = {}

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Extract ZIP
                with ZipFile(path, "r") as zipf:
                    zipf.extractall(tmpdir_path)

                # Load metadata
                metadata_path = tmpdir_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        state["metadata"] = json.load(f)

                # Load data
                data_path = tmpdir_path / "data.hdf5"
                if data_path.exists():
                    from rheojax.io import load_hdf5

                    state["data"] = load_hdf5(str(data_path))

                # Load parameters
                params_path = tmpdir_path / "parameters.json"
                if params_path.exists():
                    with open(params_path) as f:
                        state["parameters"] = json.load(f)

            logger.info(f"Loaded project from {path}")
            return state

        except Exception as e:
            logger.error(f"Project load failed: {e}")
            raise RuntimeError(f"Load failed: {e}") from e

    def generate_report(
        self,
        state: dict[str, Any],
        template: str,
        path: Path | str,
    ) -> None:
        """Generate Markdown or lightweight PDF report.

        If the output path ends with `.pdf`, a simple PDF is produced via
        matplotlib's PdfPages backend to avoid external dependencies.
        Otherwise a Markdown report is written.
        """
        path = Path(path)

        try:
            # Generate Markdown content first
            report_lines = []
            report_lines.append("# RheoJAX Analysis Report\n")
            report_lines.append(f"**Generated:** {np.datetime64('now')}\n\n")

            if "model_name" in state:
                report_lines.append(f"## Model: {state['model_name']}\n")
                report_lines.append(f"**Test Mode:** {state.get('test_mode', 'unknown')}\n\n")

            if "parameters" in state and state["parameters"]:
                report_lines.append("## Fitted Parameters\n")
                report_lines.append("| Parameter | Value |\n")
                report_lines.append("|-----------|-------|\n")
                for param, value in state["parameters"].items():
                    report_lines.append(f"| {param} | {float(value):.6g} |\n")
                report_lines.append("\n")

            if template == "bayesian" and "diagnostics" in state:
                report_lines.append("## MCMC Diagnostics\n")
                diag = state["diagnostics"]
                if "rhat" in diag:
                    report_lines.append("### R-hat Statistics\n")
                    report_lines.append("| Parameter | R-hat |\n")
                    report_lines.append("|-----------|-------|\n")
                    for param, rhat in diag["rhat"].items():
                        report_lines.append(f"| {param} | {rhat:.4f} |\n")
                    report_lines.append("\n")
                if "ess" in diag:
                    report_lines.append("### Effective Sample Size\n")
                    report_lines.append("| Parameter | ESS |\n")
                    report_lines.append("|-----------|-----|\n")
                    for param, ess in diag["ess"].items():
                        report_lines.append(f"| {param} | {ess:.0f} |\n")
                    report_lines.append("\n")

            if path.suffix.lower() == ".pdf":
                # Render a minimal PDF page with text content
                from matplotlib.backends.backend_pdf import PdfPages
                import matplotlib.pyplot as plt

                with PdfPages(path) as pdf:
                    fig, ax = plt.subplots(figsize=(8.5, 11))
                    ax.axis("off")
                    y = 1.0
                    for line in report_lines:
                        for subline in line.split("\n"):
                            if not subline:
                                y -= 0.03
                                continue
                            ax.text(0.02, y, subline, ha="left", va="top", fontsize=10)
                            y -= 0.03
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(report_lines)

            logger.info(f"Generated report at {path}")

        except Exception as e:
            logger.error(f"Report generation failed: {e}")
            raise RuntimeError(f"Report generation failed: {e}") from e

    def export_data(
        self,
        data: RheoData,
        path: Path | str,
        format: str | None = None,
    ) -> None:
        """Export rheological data.

        Parameters
        ----------
        data : RheoData
            Data to export
        path : Path or str
            Output file path
        format : str, optional
            Format (auto-detected from extension if None)
        """
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        try:
            x = np.asarray(data.x)
            y = np.asarray(data.y)

            if format == "csv":
                import pandas as pd

                if np.iscomplexobj(y):
                    df = pd.DataFrame({
                        "x": x,
                        "y_real": np.real(y),
                        "y_imag": np.imag(y),
                    })
                else:
                    df = pd.DataFrame({"x": x, "y": y})

                df.to_csv(path, index=False)

            elif format in ["xlsx", "xls"]:
                save_excel(data, str(path))

            elif format in ["hdf5", "h5"]:
                save_hdf5(data, str(path))

            elif format == "json":
                export_dict = {
                    "x": x.tolist(),
                    "y": y.tolist() if not np.iscomplexobj(y) else {
                        "real": np.real(y).tolist(),
                        "imag": np.imag(y).tolist(),
                    },
                    "metadata": data.metadata,
                }
                with open(path, "w") as f:
                    json.dump(export_dict, f, indent=2)

            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Exported data to {path}")

        except Exception as e:
            logger.error(f"Data export failed: {e}")
            raise RuntimeError(f"Export failed: {e}") from e
