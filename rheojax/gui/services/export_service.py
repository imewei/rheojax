"""
Export Service
=============

Service for exporting results to various formats (HDF5, Excel, CSV, etc.).
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import numpy as np
from matplotlib.figure import Figure

from rheojax.core.data import RheoData
from rheojax.io import save_excel, save_hdf5
from rheojax.logging import get_logger

logger = get_logger(__name__)


def _sanitize_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert metadata values to JSON-serializable types."""
    sanitized: dict[str, Any] = {}
    for key, value in metadata.items():
        if isinstance(value, np.ndarray):
            sanitized[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            sanitized[key] = value.item()
        elif isinstance(value, dict):
            sanitized[key] = _sanitize_metadata(value)
        elif isinstance(value, (str, int, float, bool, type(None), list)):
            sanitized[key] = value
        elif hasattr(value, "devices"):
            # VIS-P2-006: JAX DeviceArray → convert to Python scalar or list
            arr = np.asarray(value)
            sanitized[key] = arr.item() if arr.ndim == 0 else arr.tolist()
        else:
            # Fallback: convert to string (datetime, Path, etc.)
            sanitized[key] = str(value)
    return sanitized


class ExportService:
    """Service for result export operations.

    Features:
        - Multi-format export (HDF5, Excel, CSV, JSON)
        - Figure export (PNG, SVG, PDF)
        - Project file support (.rheojax files)
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
        logger.debug("Initializing ExportService")
        self._supported_formats = {
            "data": [".hdf5", ".h5", ".xlsx", ".csv", ".json"],
            "figure": [".png", ".pdf", ".svg", ".eps"],
            "project": [".rheojax"],
        }
        logger.debug(
            "ExportService initialized",
            supported_formats=self._supported_formats,
        )

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
        logger.debug(
            "Entering export_parameters",
            path=str(path),
            format=format,
        )
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        logger.info(
            "Starting export",
            format=format,
            filepath=str(path),
            export_type="parameters",
        )

        try:
            # Extract parameters (convert JAX/NumPy arrays to Python floats)
            if hasattr(result, "parameters"):
                params = {}
                for k, v in result.parameters.items():
                    if hasattr(v, "__array__"):
                        # VIS-P2-013: Handle both scalars and arrays (e.g. GMM mode weights)
                        arr = np.asarray(v)
                        params[k] = arr.item() if arr.ndim == 0 else arr.tolist()
                    else:
                        params[k] = v
                logger.debug("Extracted parameters from result.parameters")
            elif hasattr(result, "posterior_samples"):
                # For Bayesian results, use posterior means
                params = {
                    k: float(np.mean(np.asarray(v)))
                    for k, v in result.posterior_samples.items()
                }
                logger.debug(
                    "Extracted parameters from posterior_samples means",
                    num_params=len(params),
                )
            else:
                logger.error("Result does not contain parameters or posterior_samples")
                raise ValueError("Result does not contain parameters")

            if format == "csv":
                import pandas as pd

                df = pd.DataFrame([params])
                df.to_csv(path, index=False)
                logger.debug("Wrote parameters to CSV")

            elif format == "json":
                # Convert numpy/JAX types to Python types
                params_json = {
                    k: (
                        float(v)
                        if isinstance(v, (np.ndarray, np.floating, float))
                        or hasattr(v, "__jax_array__")
                        else v
                    )
                    for k, v in params.items()
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(params_json, f, indent=2)
                logger.debug("Wrote parameters to JSON")

            elif format in ["xlsx", "xls"]:
                import pandas as pd

                df = pd.DataFrame([params])
                df.to_excel(path, index=False)
                logger.debug("Wrote parameters to Excel")

            elif format in ["hdf5", "h5"]:
                import h5py

                with h5py.File(path, "w") as f:
                    params_group = f.create_group("parameters")
                    for key, value in params.items():
                        params_group.create_dataset(key, data=np.asarray(value))
                logger.debug("Wrote parameters to HDF5")

            else:
                logger.error("Unsupported export format", format=format)
                raise ValueError(f"Unsupported format: {format}")

            file_size = os.path.getsize(path)
            logger.info(
                "Export complete",
                format=format,
                filepath=str(path),
                file_size=file_size,
                export_type="parameters",
            )
            logger.debug("Exiting export_parameters successfully")

        except ImportError as e:
            from rheojax.gui.utils._dependency_guard import require_dependency

            pkg = "h5py" if "h5py" in str(e) else "pandas"
            require_dependency(pkg, f"exporting to {format}")
        except Exception as e:
            logger.error(
                "Parameter export failed",
                format=format,
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
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
        logger.debug(
            "Entering export_figure",
            path=str(path),
            dpi=dpi,
            format=format,
        )
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        logger.info(
            "Starting export",
            format=format,
            filepath=str(path),
            export_type="figure",
            dpi=dpi,
        )

        try:
            fig.savefig(path, dpi=dpi, format=format, bbox_inches="tight", **kwargs)
            file_size = os.path.getsize(path)
            logger.info(
                "Export complete",
                format=format,
                filepath=str(path),
                file_size=file_size,
                export_type="figure",
            )
            logger.debug("Exiting export_figure successfully")

        except Exception as e:
            logger.error(
                "Figure export failed",
                format=format,
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
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
        logger.debug(
            "Entering export_posterior",
            path=str(path),
            format=format,
        )
        path = Path(path)

        logger.info(
            "Starting export",
            format=format,
            filepath=str(path),
            export_type="posterior",
        )

        try:
            if not hasattr(result, "posterior_samples"):
                logger.error("Result does not contain posterior samples")
                raise ValueError("Result does not contain posterior samples")

            # Convert JAX arrays to NumPy at the I/O boundary
            posterior_samples = {
                k: np.asarray(v) for k, v in result.posterior_samples.items()
            }
            _first_samples = next(iter(posterior_samples.values()), None)
            if _first_samples is None:
                raise ValueError("posterior_samples is empty — nothing to export")
            num_samples = len(_first_samples)
            logger.debug(
                "Exporting posterior samples",
                num_params=len(posterior_samples),
                num_samples=num_samples,
            )

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
                logger.debug("Wrote posterior samples to HDF5")

            elif format in ["xlsx", "xls"]:
                import pandas as pd

                # Flatten samples to 2D
                data = {
                    k: v.flatten() if v.ndim > 1 else v
                    for k, v in posterior_samples.items()
                }
                df = pd.DataFrame(data)
                df.to_excel(path, index=False)
                logger.debug("Wrote posterior samples to Excel")

            elif format == "csv":
                import pandas as pd

                data = {
                    k: v.flatten() if v.ndim > 1 else v
                    for k, v in posterior_samples.items()
                }
                df = pd.DataFrame(data)
                df.to_csv(path, index=False)
                logger.debug("Wrote posterior samples to CSV")

            else:
                logger.error("Unsupported export format", format=format)
                raise ValueError(f"Unsupported format: {format}")

            file_size = os.path.getsize(path)
            logger.info(
                "Export complete",
                format=format,
                filepath=str(path),
                file_size=file_size,
                export_type="posterior",
            )
            logger.debug("Exiting export_posterior successfully")

        except Exception as e:
            logger.error(
                "Posterior export failed",
                format=format,
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Export failed: {e}") from e

    def save_project(
        self,
        state: dict[str, Any],
        path: Path | str,
    ) -> None:
        """Save project as .rheojax file (ZIP with JSON + HDF5).

        Parameters
        ----------
        state : dict
            Application state (data, models, results, etc.)
        path : Path or str
            Output .rheojax file path
        """
        logger.debug(
            "Entering save_project",
            path=str(path),
            state_keys=list(state.keys()),
        )
        path = Path(path)

        logger.info(
            "Starting export",
            format="rheojax",
            filepath=str(path),
            export_type="project",
        )

        try:
            # Create temp files
            import tempfile

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Save metadata as JSON
                metadata_path = tmpdir_path / "metadata.json"
                # GUI-IO-010: Standardise to UTC ISO-8601 timestamp so that
                # save_project and generate_report produce consistent formats.
                # Previously used str(np.datetime64("now")) which is naïve
                # local time; datetime.now(timezone.utc) is always UTC.
                metadata = {
                    "version": "1.0",
                    "model_name": state.get("model_name"),
                    "test_mode": state.get("test_mode"),
                    "deformation_mode": state.get("deformation_mode", "shear"),
                    "poisson_ratio": state.get("poisson_ratio", 0.5),
                    "timestamp": datetime.now(timezone.utc).strftime(
                        "%Y-%m-%dT%H:%M:%SZ"
                    ),
                }

                with open(metadata_path, "w", encoding="utf-8") as f:
                    json.dump(metadata, f, indent=2)
                logger.debug("Wrote project metadata")

                # Save data as HDF5
                if "data" in state and state["data"] is not None:
                    data_path = tmpdir_path / "data.hdf5"
                    save_hdf5(state["data"], str(data_path))
                    logger.debug("Wrote project data to HDF5")

                # Save parameters as JSON
                if "parameters" in state and state["parameters"]:
                    params_path = tmpdir_path / "parameters.json"
                    params_json = {}
                    for k, v in state["parameters"].items():
                        if hasattr(v, "__array__"):
                            arr = np.asarray(v)
                            params_json[k] = arr.item() if arr.ndim == 0 else arr.tolist()
                        else:
                            params_json[k] = v
                    with open(params_path, "w", encoding="utf-8") as f:
                        json.dump(params_json, f, indent=2)
                    logger.debug("Wrote project parameters")

                # VIS-P2-014: Save posterior samples if present
                if "posterior_samples" in state and state["posterior_samples"]:
                    try:
                        import h5py

                        posterior_path = tmpdir_path / "posterior.hdf5"
                        with h5py.File(posterior_path, "w") as hf:
                            for k, v in state["posterior_samples"].items():
                                hf.create_dataset(k, data=np.asarray(v))
                        logger.debug("Wrote posterior samples to project")
                    except ImportError:
                        logger.debug("h5py unavailable, skipping posterior export")

                # VIS-P2-014: Save fit result metadata if present
                if "fit_result" in state and state["fit_result"] is not None:
                    fit_meta_path = tmpdir_path / "fit_result.json"
                    fr = state["fit_result"]
                    fit_meta = {
                        "model_name": getattr(fr, "model_name", None),
                        "test_mode": getattr(fr, "test_mode", None),
                        "r_squared": float(getattr(fr, "r_squared", 0.0)),
                        "success": getattr(fr, "success", False),
                    }
                    with open(fit_meta_path, "w", encoding="utf-8") as f:
                        json.dump(fit_meta, f, indent=2)
                    logger.debug("Wrote fit result metadata to project")

                # Create ZIP archive
                with ZipFile(path, "w") as zipf:
                    for file in tmpdir_path.glob("*"):
                        zipf.write(file, file.name)
                logger.debug("Created ZIP archive")

            file_size = os.path.getsize(path)
            logger.info(
                "Export complete",
                format="rheojax",
                filepath=str(path),
                file_size=file_size,
                export_type="project",
            )
            logger.debug("Exiting save_project successfully")

        except Exception as e:
            logger.error(
                "Project save failed",
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Save failed: {e}") from e

    def load_project(self, path: Path | str) -> dict[str, Any]:
        """Load project from .rheojax file.

        Parameters
        ----------
        path : Path or str
            .rheojax file path

        Returns
        -------
        dict
            Application state
        """
        logger.debug(
            "Entering load_project",
            path=str(path),
        )
        path = Path(path)

        logger.info(
            "Starting load",
            format="rheojax",
            filepath=str(path),
            operation="load_project",
        )

        try:
            import tempfile

            state = {}

            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)

                # Extract ZIP (with path traversal guard — VIS-P2-007)
                with ZipFile(path, "r") as zipf:
                    for member in zipf.namelist():
                        dest = (tmpdir_path / member).resolve()
                        if not str(dest).startswith(str(tmpdir_path.resolve())):
                            raise ValueError(
                                f"Zip path traversal detected: {member}"
                            )
                    zipf.extractall(tmpdir_path)
                logger.debug("Extracted ZIP archive")

                # Load metadata
                metadata_path = tmpdir_path / "metadata.json"
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        state["metadata"] = json.load(f)
                    logger.debug("Loaded project metadata")

                # Load data
                data_path = tmpdir_path / "data.hdf5"
                if data_path.exists():
                    from rheojax.io import load_hdf5

                    state["data"] = load_hdf5(str(data_path))
                    logger.debug("Loaded project data from HDF5")

                # Load parameters
                params_path = tmpdir_path / "parameters.json"
                if params_path.exists():
                    with open(params_path) as f:
                        state["parameters"] = json.load(f)
                    logger.debug("Loaded project parameters")

                # VIS-P2-014: Load posterior samples if present
                posterior_path = tmpdir_path / "posterior.hdf5"
                if posterior_path.exists():
                    try:
                        import h5py

                        posterior = {}
                        with h5py.File(posterior_path, "r") as hf:
                            for k in hf.keys():
                                posterior[k] = np.array(hf[k])
                        state["posterior_samples"] = posterior
                        logger.debug("Loaded posterior samples from project")
                    except ImportError:
                        logger.debug("h5py unavailable, skipping posterior load")

                # VIS-P2-014: Load fit result metadata if present
                fit_meta_path = tmpdir_path / "fit_result.json"
                if fit_meta_path.exists():
                    with open(fit_meta_path) as f:
                        state["fit_result"] = json.load(f)
                    logger.debug("Loaded fit result metadata")

            file_size = os.path.getsize(path)
            logger.info(
                "Load complete",
                format="rheojax",
                filepath=str(path),
                file_size=file_size,
                operation="load_project",
            )
            logger.debug(
                "Exiting load_project successfully",
                state_keys=list(state.keys()),
            )
            return state

        except Exception as e:
            logger.error(
                "Project load failed",
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
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
        logger.debug(
            "Entering generate_report",
            path=str(path),
            template=template,
            state_keys=list(state.keys()),
        )
        path = Path(path)
        format = "pdf" if path.suffix.lower() == ".pdf" else "markdown"

        logger.info(
            "Starting export",
            format=format,
            filepath=str(path),
            export_type="report",
            template=template,
        )

        try:
            # Generate Markdown content first
            report_lines = []
            report_lines.append("# RheoJAX Analysis Report\n")
            _ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            report_lines.append(f"**Generated:** {_ts}\n\n")

            if "model_name" in state:
                report_lines.append(f"## Model: {state['model_name']}\n")
                report_lines.append(
                    f"**Test Mode:** {state.get('test_mode', 'unknown')}\n\n"
                )

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
                if "r_hat" in diag:
                    report_lines.append("### R-hat Statistics\n")
                    report_lines.append("| Parameter | R-hat |\n")
                    report_lines.append("|-----------|-------|\n")
                    for param, rhat in diag["r_hat"].items():
                        report_lines.append(f"| {param} | {rhat:.4f} |\n")
                    report_lines.append("\n")
                if "ess" in diag:
                    report_lines.append("### Effective Sample Size\n")
                    report_lines.append("| Parameter | ESS |\n")
                    report_lines.append("|-----------|-----|\n")
                    for param, ess in diag["ess"].items():
                        report_lines.append(f"| {param} | {ess:.0f} |\n")
                    report_lines.append("\n")

            logger.debug(
                "Generated report content",
                num_lines=len(report_lines),
            )

            if path.suffix.lower() == ".pdf":
                # Render a minimal PDF with automatic page-break support.
                # GUI-IO-019: Many-parameter models (e.g. GMM with 40+ params)
                # overflowed a single page when all lines were placed on one
                # axes.  Now a new page is started whenever y approaches the
                # bottom margin, preventing text from being clipped.
                import matplotlib.pyplot as plt
                from matplotlib.backends.backend_pdf import PdfPages

                def _new_page(pdf_handle, figures):
                    """Create a new blank axes page and append to figures list."""
                    _fig, _ax = plt.subplots(figsize=(8.5, 11))
                    _ax.axis("off")
                    figures.append((_fig, _ax))
                    return _fig, _ax, 0.97  # y starts near top

                with PdfPages(path) as pdf:
                    figures = []
                    fig, ax, y = _new_page(pdf, figures)
                    for line in report_lines:
                        for subline in line.split("\n"):
                            # GUI-IO-019: page-break when near bottom margin
                            if y < 0.05:
                                pdf.savefig(fig, bbox_inches="tight")
                                plt.close(fig)
                                figures.pop()
                                fig, ax, y = _new_page(pdf, figures)
                            if not subline:
                                y -= 0.03
                                continue
                            ax.text(0.02, y, subline, ha="left", va="top", fontsize=10)
                            y -= 0.03
                    # Save the last (or only) page
                    pdf.savefig(fig, bbox_inches="tight")
                    plt.close(fig)
                logger.debug("Wrote report to PDF")
            else:
                with open(path, "w", encoding="utf-8") as f:
                    f.writelines(report_lines)
                logger.debug("Wrote report to Markdown")

            file_size = os.path.getsize(path)
            logger.info(
                "Export complete",
                format=format,
                filepath=str(path),
                file_size=file_size,
                export_type="report",
            )
            logger.debug("Exiting generate_report successfully")

        except Exception as e:
            logger.error(
                "Report generation failed",
                template=template,
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
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
        logger.debug(
            "Entering export_data",
            path=str(path),
            format=format,
        )
        path = Path(path)

        if format is None:
            format = path.suffix.lstrip(".")

        logger.info(
            "Starting export",
            format=format,
            filepath=str(path),
            export_type="data",
        )

        try:
            x = np.asarray(data.x)
            y = np.asarray(data.y)
            logger.debug(
                "Prepared data for export",
                x_shape=x.shape,
                y_shape=y.shape,
                is_complex=np.iscomplexobj(y),
            )

            if format == "csv":
                import pandas as pd

                if np.iscomplexobj(y):
                    df = pd.DataFrame(
                        {
                            "x": x,
                            "y_real": np.real(y),
                            "y_imag": np.imag(y),
                        }
                    )
                else:
                    df = pd.DataFrame({"x": x, "y": y})

                df.to_csv(path, index=False)
                logger.debug("Wrote data to CSV")

            elif format in ["xlsx", "xls"]:
                save_excel(data, str(path))
                logger.debug("Wrote data to Excel")

            elif format in ["hdf5", "h5"]:
                save_hdf5(data, str(path))
                logger.debug("Wrote data to HDF5")

            elif format == "json":
                export_dict = {
                    "x": x.tolist(),
                    "y": (
                        y.tolist()
                        if not np.iscomplexobj(y)
                        else {
                            "real": np.real(y).tolist(),
                            "imag": np.imag(y).tolist(),
                        }
                    ),
                    "metadata": _sanitize_metadata(data.metadata),
                }
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(export_dict, f, indent=2, default=str)
                logger.debug("Wrote data to JSON")

            else:
                logger.error("Unsupported export format", format=format)
                raise ValueError(f"Unsupported format: {format}")

            file_size = os.path.getsize(path)
            logger.info(
                "Export complete",
                format=format,
                filepath=str(path),
                file_size=file_size,
                export_type="data",
            )
            logger.debug("Exiting export_data successfully")

        except Exception as e:
            logger.error(
                "Data export failed",
                format=format,
                filepath=str(path),
                error=str(e),
                exc_info=True,
            )
            raise RuntimeError(f"Export failed: {e}") from e
