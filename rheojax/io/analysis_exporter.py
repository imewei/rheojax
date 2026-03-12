"""Unified analysis exporter for Pipeline results.

Bundles data, model parameters, fit statistics, figures, transform results,
and Bayesian diagnostics into a structured output directory or single-file
archive (Excel with embedded plots, or HDF5).

The exporter is designed for reproducibility: every exported package contains
enough information to reconstruct the analysis.

Example:
    >>> from rheojax.io.analysis_exporter import AnalysisExporter
    >>> exporter = AnalysisExporter()
    >>> exporter.export_directory(pipeline, output_dir="./results/maxwell_fit")

    # Or from Pipeline fluent API:
    >>> pipeline.load("data.csv").fit("maxwell").plot_fit().export("./results")
"""

from __future__ import annotations

import datetime
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.io.json_encoder import NumpyJSONEncoder as _NumpyJSONEncoder
from rheojax.logging import get_logger

# Ensure float64 is configured before any JAX array handling
_jax, _jnp = safe_import_jax()

logger = get_logger(__name__)


class AnalysisExporter:
    """Export pipeline analysis results to structured output.

    Supports three export modes:
    - **Directory**: One folder with subfolders for data, figures, and reports.
    - **Excel**: Single .xlsx with sheets for parameters, data, and embedded plots.
    - **HDF5**: Compact archive for programmatic reload.

    The exporter collects state from a Pipeline instance (data, model,
    fit result, Bayesian result, transform cache, figures) and writes
    everything to the chosen format.
    """

    def __init__(
        self,
        figure_formats: tuple[str, ...] = ("pdf", "png"),
        figure_dpi: int = 300,
    ):
        """Initialize the exporter.

        Args:
            figure_formats: Formats for saved figures (default: ('pdf', 'png')).
            figure_dpi: DPI for raster figures (default: 300).
        """
        self.figure_formats = figure_formats
        self.figure_dpi = figure_dpi

    # ------------------------------------------------------------------
    # Directory export
    # ------------------------------------------------------------------

    def export_directory(
        self,
        pipeline: Any,
        output_dir: str | Path,
        *,
        include_data: bool = True,
        include_figures: bool = True,
        include_diagnostics: bool = True,
        include_summary: bool = True,
        data_format: str = "hdf5",
    ) -> Path:
        """Export full analysis to a structured directory.

        Creates the following structure:
            output_dir/
            ├── summary.json          # Analysis metadata + parameters
            ├── summary.txt           # Human-readable summary
            ├── data/
            │   ├── input_data.hdf5   # Raw input data
            │   └── fitted_data.hdf5  # Post-transform data (if different)
            ├── figures/
            │   ├── fit.pdf/.png      # NLSQ fit plot
            │   ├── bayesian.pdf/.png # Bayesian posterior predictive
            │   ├── transform_*.pdf   # Transform plots
            │   └── diagnostics/      # ArviZ diagnostic suite
            └── results/
                ├── fit_result.json    # FitResult serialization
                ├── fit_result.npz     # FitResult binary
                └── transforms/       # Transform result caches

        Args:
            pipeline: Pipeline instance with analysis state.
            output_dir: Root directory for export (created if needed).
            include_data: Save raw and transformed data.
            include_figures: Save all generated figures.
            include_diagnostics: Save MCMC diagnostic plots (if available).
            include_summary: Write summary.json and summary.txt.
            data_format: Format for data files ('hdf5' or 'npz').

        Returns:
            Path to the output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Exporting analysis", output_dir=str(output_dir))

        # Collect pipeline state
        state = self._collect_state(pipeline)

        # 1. Summary
        if include_summary:
            self._write_summary(state, output_dir)

        # 2. Data
        if include_data:
            self._write_data(state, output_dir, data_format)

        # 3. Fit result
        self._write_fit_results(state, output_dir)

        # 4. Transform results
        self._write_transform_results(state, output_dir)

        # 5. Figures
        if include_figures:
            self._write_figures(state, output_dir)

        # 6. Diagnostics
        if include_diagnostics:
            self._write_diagnostics(state, output_dir)

        logger.info(
            "Export complete",
            output_dir=str(output_dir),
            sections=self._list_sections(state),
        )

        return output_dir

    # ------------------------------------------------------------------
    # Excel export
    # ------------------------------------------------------------------

    def export_excel(
        self,
        pipeline: Any,
        filepath: str | Path,
        include_plots: bool = True,
    ) -> Path:
        """Export analysis to a single Excel workbook.

        Creates sheets:
        - Summary: Analysis metadata
        - Parameters: Model parameters with bounds and units
        - Fit Quality: R², RMSE, AIC, BIC, etc.
        - Data: Input x/y arrays
        - Predictions: Model predictions
        - Residuals: Fitting residuals
        - Transforms: Transform result summary
        - Plot_*: Embedded figures (if include_plots=True)

        Args:
            pipeline: Pipeline instance with analysis state.
            filepath: Output .xlsx path.
            include_plots: Embed matplotlib figures as images.

        Returns:
            Path to the output file.
        """
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "pandas is required for Excel export. Install with: pip install pandas openpyxl"
            ) from exc

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        state = self._collect_state(pipeline)

        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(filepath.parent), suffix=".tmp.xlsx"
        )
        os.close(tmp_fd)

        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                # Summary sheet
                self._write_excel_summary(writer, state)

                # Parameters sheet
                if state["fit_result"] is not None:
                    self._write_excel_parameters(writer, state)

                # Fit quality sheet
                if state["fit_result"] is not None:
                    self._write_excel_fit_quality(writer, state)

                # Data sheet
                if state["data"] is not None:
                    self._write_excel_data(writer, state)

                # Predictions sheet
                if state["fit_result"] is not None:
                    self._write_excel_predictions(writer, state)

                # Residuals sheet
                if (
                    state["fit_result"] is not None
                    and state["fit_result"].residuals is not None
                ):
                    self._write_excel_residuals(writer, state)

                # Transform summary sheet
                if state["transform_results"]:
                    self._write_excel_transforms(writer, state)

                # Bayesian summary sheet
                if state["bayesian_result"] is not None:
                    self._write_excel_bayesian(writer, state)

                # Embedded plots
                if include_plots:
                    self._write_excel_plots(writer, state)

            os.replace(tmp_path, str(filepath))
            tmp_path = None
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        logger.info("Exported analysis to Excel", filepath=str(filepath))
        return filepath

    # ------------------------------------------------------------------
    # State collection
    # ------------------------------------------------------------------

    def _collect_state(self, pipeline: Any) -> dict[str, Any]:
        """Collect all exportable state from a Pipeline."""
        data = getattr(pipeline, "data", None)
        last_model = getattr(pipeline, "_last_model", None)
        history = getattr(pipeline, "history", [])
        transform_results = getattr(pipeline, "_transform_results", {})
        bayesian_result = getattr(pipeline, "_last_bayesian_result", None)
        current_figure = getattr(pipeline, "_current_figure", None)
        diagnostic_results = getattr(pipeline, "_diagnostic_results", None)

        # Build fit result
        fit_result = None
        try:
            fit_result = pipeline.get_fit_result()
        except (ValueError, AttributeError):
            pass

        # Collect metadata
        metadata: dict[str, Any] = {}
        if data is not None:
            dm = getattr(data, "metadata", None) or {}
            metadata["test_mode"] = dm.get("test_mode")
            metadata["deformation_mode"] = dm.get("deformation_mode")
            metadata["domain"] = getattr(data, "domain", None)
            metadata["x_units"] = getattr(data, "x_units", None)
            metadata["y_units"] = getattr(data, "y_units", None)
            metadata["n_points"] = len(data.x)

        if last_model is not None:
            metadata["model_class"] = type(last_model).__name__
            metadata["n_params"] = len(list(last_model.parameters.keys()))

        return {
            "data": data,
            "model": last_model,
            "fit_result": fit_result,
            "bayesian_result": bayesian_result,
            "transform_results": transform_results,
            "history": history,
            "current_figure": current_figure,
            "diagnostic_results": diagnostic_results,
            "metadata": metadata,
            "timestamp": datetime.datetime.now(datetime.UTC).isoformat(),
        }

    def _list_sections(self, state: dict[str, Any]) -> list[str]:
        """List which sections were populated."""
        sections = []
        if state["data"] is not None:
            sections.append("data")
        if state["fit_result"] is not None:
            sections.append("fit")
        if state["bayesian_result"] is not None:
            sections.append("bayesian")
        if state["transform_results"]:
            sections.append("transforms")
        if state["current_figure"] is not None:
            sections.append("figures")
        return sections

    # ------------------------------------------------------------------
    # Directory export helpers
    # ------------------------------------------------------------------

    def _write_summary(self, state: dict[str, Any], output_dir: Path) -> None:
        """Write summary.json and summary.txt."""
        summary: dict[str, Any] = {
            "timestamp": state["timestamp"],
            "metadata": state["metadata"],
            "history": [
                {"operation": h[0], "details": list(h[1:]) if len(h) > 1 else []}
                for h in state["history"]
            ],
        }

        # Add fit statistics
        fr = state["fit_result"]
        if fr is not None:
            summary["fit"] = {
                "model_name": fr.model_name,
                "model_class": fr.model_class_name,
                "protocol": fr.protocol,
                "params": {
                    k: float(v) if np.isfinite(v) else str(float(v))
                    for k, v in fr.params.items()
                },
                "params_units": dict(fr.params_units),
                "statistics": {},
            }
            for attr in (
                "r_squared",
                "adj_r_squared",
                "rmse",
                "mae",
                "aic",
                "bic",
                "aicc",
            ):
                val = getattr(fr, attr, None)
                if val is not None and np.isfinite(val):
                    summary["fit"]["statistics"][attr] = float(val)

            # Confidence intervals
            ci = fr.confidence_intervals()
            if ci is not None:
                names = list(fr.params.keys())
                summary["fit"]["confidence_intervals"] = {
                    names[i]: [float(ci[i, 0]), float(ci[i, 1])]
                    for i in range(min(len(names), len(ci)))
                }

        # Add Bayesian summary
        br = state["bayesian_result"]
        if br is not None:
            bayes_summary: dict[str, Any] = {}
            for attr in ("num_samples", "num_warmup", "num_chains", "num_divergences"):
                val = getattr(br, attr, None)
                if val is not None:
                    bayes_summary[attr] = (
                        int(val) if isinstance(val, (int, np.integer)) else val
                    )
            if hasattr(br, "diagnostics") and br.diagnostics:
                diag = br.diagnostics
                bayes_summary["diagnostics"] = {
                    k: float(v) if isinstance(v, (float, np.floating)) else v
                    for k, v in diag.items()
                    if isinstance(v, (int, float, str, bool, np.integer, np.floating))
                }
            summary["bayesian"] = bayes_summary

        # Add transform list
        if state["transform_results"]:
            summary["transforms"] = list(state["transform_results"].keys())

        # Write JSON (atomic: write to temp file, then rename)
        json_path = output_dir / "summary.json"
        fd, tmp_json = tempfile.mkstemp(dir=output_dir, suffix=".json.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(summary, f, indent=2, cls=_NumpyJSONEncoder)
            os.replace(tmp_json, json_path)
        except BaseException:
            if os.path.exists(tmp_json):
                os.unlink(tmp_json)
            raise

        # Write human-readable text (atomic)
        txt_path = output_dir / "summary.txt"
        lines = self._format_text_summary(summary, state)
        fd, tmp_txt = tempfile.mkstemp(dir=output_dir, suffix=".txt.tmp")
        try:
            with os.fdopen(fd, "w") as f:
                f.write("\n".join(lines))
            os.replace(tmp_txt, txt_path)
        except BaseException:
            if os.path.exists(tmp_txt):
                os.unlink(tmp_txt)
            raise

    def _format_text_summary(
        self, summary: dict[str, Any], state: dict[str, Any]
    ) -> list[str]:
        """Format a human-readable text summary."""
        lines = [
            "=" * 60,
            "RheoJAX Analysis Report",
            "=" * 60,
            f"Timestamp: {summary['timestamp']}",
            "",
        ]

        meta = summary.get("metadata", {})
        if meta:
            lines.append("Data:")
            for key in (
                "domain",
                "test_mode",
                "deformation_mode",
                "n_points",
                "x_units",
                "y_units",
            ):
                val = meta.get(key)
                if val is not None:
                    lines.append(f"  {key}: {val}")
            lines.append("")

        fit = summary.get("fit")
        if fit:
            lines.append(f"Model: {fit['model_class']} ({fit['model_name']})")
            lines.append(f"Protocol: {fit.get('protocol', 'auto')}")
            lines.append(f"Parameters ({len(fit['params'])}):")
            ci_dict = fit.get("confidence_intervals", {})
            for name, value in fit["params"].items():
                unit = fit["params_units"].get(name, "")
                unit_str = f" {unit}" if unit else ""
                ci = ci_dict.get(name)
                ci_str = f"  CI: [{ci[0]:.4g}, {ci[1]:.4g}]" if ci else ""
                lines.append(f"  {name} = {value:.6g}{unit_str}{ci_str}")
            lines.append("")
            lines.append("Statistics:")
            for metric, value in fit.get("statistics", {}).items():
                lines.append(f"  {metric}: {value:.6g}")
            lines.append("")

        bayes = summary.get("bayesian")
        if bayes:
            lines.append("Bayesian Inference:")
            for key, val in bayes.items():
                if key != "diagnostics":
                    lines.append(f"  {key}: {val}")
            diag = bayes.get("diagnostics", {})
            if diag:
                lines.append("  Diagnostics:")
                for key, val in diag.items():
                    lines.append(f"    {key}: {val}")
            lines.append("")

        transforms = summary.get("transforms", [])
        if transforms:
            lines.append(f"Transforms applied ({len(transforms)}):")
            for t in transforms:
                lines.append(f"  - {t}")
            lines.append("")

        lines.append("History:")
        for h in summary.get("history", []):
            details = ", ".join(str(d) for d in h.get("details", []))
            lines.append(f"  {h['operation']}" + (f" ({details})" if details else ""))

        lines.extend(["", "=" * 60])
        return lines

    def _write_data(
        self, state: dict[str, Any], output_dir: Path, data_format: str
    ) -> None:
        """Write input data to data/ subdirectory."""
        data = state["data"]
        if data is None:
            return

        data_dir = output_dir / "data"
        data_dir.mkdir(exist_ok=True)

        if data_format == "hdf5":
            from rheojax.io.writers.hdf5_writer import save_hdf5

            save_hdf5(data, data_dir / "current_data.h5")
        elif data_format == "npz":
            from rheojax.io.writers.npz_writer import save_npz

            save_npz(data, data_dir / "current_data.npz")
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

    def _write_fit_results(self, state: dict[str, Any], output_dir: Path) -> None:
        """Write fit result to results/ subdirectory."""
        fr = state["fit_result"]
        if fr is None:
            return

        results_dir = output_dir / "results"
        results_dir.mkdir(exist_ok=True)

        # JSON (human-readable)
        fr.save(str(results_dir / "fit_result.json"))

        # NPZ (binary, fast reload)
        fr.save(str(results_dir / "fit_result.npz"))

    def _write_transform_results(self, state: dict[str, Any], output_dir: Path) -> None:
        """Write cached transform results to results/transforms/."""
        transforms = state["transform_results"]
        if not transforms:
            return

        transform_dir = output_dir / "results" / "transforms"
        transform_dir.mkdir(parents=True, exist_ok=True)

        entries: list[dict[str, Any]] = []
        for name, (result, pre_data) in transforms.items():
            safe_name = name.replace("/", "_").replace(" ", "_")
            entry: dict[str, Any] = {"transform_name": name}

            # Extract data from the transform result
            from rheojax.core.data import RheoData

            if isinstance(result, RheoData):
                result_data = result
                result_meta = None
            elif isinstance(result, tuple) and len(result) >= 2:
                result_data = result[0] if isinstance(result[0], RheoData) else None
                result_meta = result[1] if len(result) > 1 else None
            else:
                result_data = None
                result_meta = None

            # Save the output data
            if result_data is not None:
                from rheojax.io.writers.npz_writer import save_npz

                save_npz(result_data, transform_dir / f"{safe_name}_output.npz")
                entry["output_file"] = f"{safe_name}_output.npz"
                entry["output_n_points"] = len(result_data.x)

            # Save pre-transform data
            if pre_data is not None:
                from rheojax.io.writers.npz_writer import save_npz

                save_npz(pre_data, transform_dir / f"{safe_name}_input.npz")
                entry["input_file"] = f"{safe_name}_input.npz"

            # Save metadata if dict-serializable
            if result_meta is not None and isinstance(result_meta, dict):
                try:
                    meta_path = transform_dir / f"{safe_name}_meta.json"
                    with open(meta_path, "w") as f:
                        json.dump(result_meta, f, indent=2, cls=_NumpyJSONEncoder)
                    entry["meta_file"] = f"{safe_name}_meta.json"
                except (TypeError, ValueError):
                    logger.debug(
                        "Could not serialize transform metadata to JSON",
                        transform=name,
                    )

            entries.append(entry)

        # Write index with per-transform metadata
        index_path = transform_dir / "index.json"
        with open(index_path, "w") as f:
            json.dump(
                {"transforms": entries},
                f,
                indent=2,
                cls=_NumpyJSONEncoder,
            )

    def _write_figures(self, state: dict[str, Any], output_dir: Path) -> None:
        """Save all available figures to figures/ subdirectory."""
        fig_dir = output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        current_fig = state["current_figure"]
        if current_fig is not None:
            self._save_fig(current_fig, fig_dir / "last_plot", "last_plot")

    def _write_diagnostics(self, state: dict[str, Any], output_dir: Path) -> None:
        """Save MCMC diagnostic figures."""
        diag = state["diagnostic_results"]
        if diag is None:
            return

        diag_dir = output_dir / "figures" / "diagnostics"
        diag_dir.mkdir(parents=True, exist_ok=True)

        import matplotlib.pyplot as plt

        for name, fig_or_path in diag.items():
            if hasattr(fig_or_path, "savefig"):
                # It's a matplotlib Figure
                self._save_fig(fig_or_path, diag_dir / name, name)
                plt.close(fig_or_path)

    def _save_fig(self, fig: Any, base_path: Path, label: str) -> None:
        """Save a figure in all configured formats."""
        for fmt in self.figure_formats:
            out_path = base_path.with_suffix(f".{fmt}")
            try:
                fig.savefig(
                    out_path,
                    format=fmt,
                    dpi=self.figure_dpi,
                    bbox_inches="tight",
                )
                logger.debug("Figure saved", label=label, path=str(out_path))
            except Exception as e:
                logger.warning(
                    "Failed to save figure",
                    label=label,
                    format=fmt,
                    error=str(e),
                )

    # ------------------------------------------------------------------
    # Excel export helpers
    # ------------------------------------------------------------------

    def _write_excel_summary(self, writer: Any, state: dict[str, Any]) -> None:
        """Write summary sheet."""
        import pandas as pd

        rows = [
            {"Field": "Timestamp", "Value": state["timestamp"]},
        ]
        for key, val in state["metadata"].items():
            if val is not None:
                rows.append({"Field": key, "Value": str(val)})

        # History
        for i, h in enumerate(state["history"]):
            details = ", ".join(str(d) for d in h[1:]) if len(h) > 1 else ""
            rows.append(
                {
                    "Field": f"Step {i + 1}",
                    "Value": f"{h[0]}" + (f" ({details})" if details else ""),
                }
            )

        pd.DataFrame(rows).to_excel(writer, sheet_name="Summary", index=False)

    def _write_excel_parameters(self, writer: Any, state: dict[str, Any]) -> None:
        """Write parameters sheet with bounds, units, and confidence intervals."""
        import pandas as pd

        fr = state["fit_result"]
        ci = fr.confidence_intervals()
        rows = []
        for i, (name, value) in enumerate(fr.params.items()):
            row: dict[str, Any] = {
                "Parameter": name,
                "Value": float(value),
                "Units": fr.params_units.get(name, ""),
            }
            if ci is not None and i < len(ci):
                row["CI Lower (95%)"] = float(ci[i, 0])
                row["CI Upper (95%)"] = float(ci[i, 1])
            rows.append(row)

        pd.DataFrame(rows).to_excel(writer, sheet_name="Parameters", index=False)

    def _write_excel_fit_quality(self, writer: Any, state: dict[str, Any]) -> None:
        """Write fit quality metrics sheet."""
        import pandas as pd

        fr = state["fit_result"]
        rows = []
        for attr, label in [
            ("r_squared", "R²"),
            ("adj_r_squared", "Adjusted R²"),
            ("rmse", "RMSE"),
            ("mae", "MAE"),
            ("aic", "AIC"),
            ("bic", "BIC"),
            ("aicc", "AICc"),
            ("n_data", "Data points"),
            ("n_params", "Parameters"),
            ("success", "Converged"),
        ]:
            val = getattr(fr, attr, None)
            if val is not None:
                rows.append({"Metric": label, "Value": val})

        pd.DataFrame(rows).to_excel(writer, sheet_name="Fit Quality", index=False)

    def _write_excel_data(self, writer: Any, state: dict[str, Any]) -> None:
        """Write input data sheet."""
        import pandas as pd

        data = state["data"]
        x_arr = np.asarray(data.x)
        y_arr = np.asarray(data.y)

        dm = state["metadata"].get("deformation_mode")
        prefix = "E" if dm in ("tension", "bending", "compression") else "G"

        if np.iscomplexobj(y_arr):
            df = pd.DataFrame(
                {
                    "x": x_arr,
                    f"{prefix}' (Storage)": np.real(y_arr),
                    f"{prefix}'' (Loss)": np.imag(y_arr),
                }
            )
        elif y_arr.ndim == 2 and y_arr.shape[1] == 2:
            df = pd.DataFrame(
                {
                    "x": x_arr,
                    f"{prefix}' (Storage)": y_arr[:, 0],
                    f"{prefix}'' (Loss)": y_arr[:, 1],
                }
            )
        else:
            df = pd.DataFrame({"x": x_arr, "y": y_arr})

        df.to_excel(writer, sheet_name="Data", index=False)

    def _write_excel_predictions(self, writer: Any, state: dict[str, Any]) -> None:
        """Write model predictions sheet."""
        import pandas as pd

        fr = state["fit_result"]
        if fr.fitted_curve is None:
            return

        fc = np.asarray(fr.fitted_curve)
        X = np.asarray(fr.X) if fr.X is not None else np.arange(len(fc))

        dm = state["metadata"].get("deformation_mode")
        prefix = "E" if dm in ("tension", "bending", "compression") else "G"

        if np.iscomplexobj(fc):
            df = pd.DataFrame(
                {
                    "x": X,
                    f"{prefix}' (Fit)": np.real(fc),
                    f"{prefix}'' (Fit)": np.imag(fc),
                }
            )
        elif fc.ndim == 2 and fc.shape[1] == 2:
            df = pd.DataFrame(
                {
                    "x": X,
                    f"{prefix}' (Fit)": fc[:, 0],
                    f"{prefix}'' (Fit)": fc[:, 1],
                }
            )
        else:
            df = pd.DataFrame({"x": X, "Prediction": fc})

        df.to_excel(writer, sheet_name="Predictions", index=False)

    def _write_excel_residuals(self, writer: Any, state: dict[str, Any]) -> None:
        """Write residuals sheet."""
        import pandas as pd

        fr = state["fit_result"]
        residuals = np.asarray(fr.residuals)

        if np.iscomplexobj(residuals):
            df = pd.DataFrame(
                {
                    "Index": np.arange(len(residuals)),
                    "Residual (Real)": np.real(residuals),
                    "Residual (Imag)": np.imag(residuals),
                }
            )
        else:
            df = pd.DataFrame(
                {
                    "Index": np.arange(len(residuals)),
                    "Residual": residuals,
                }
            )

        df.to_excel(writer, sheet_name="Residuals", index=False)

    def _write_excel_transforms(self, writer: Any, state: dict[str, Any]) -> None:
        """Write transform summary sheet."""
        import pandas as pd

        rows = []
        for name, (result, pre_data) in state["transform_results"].items():
            from rheojax.core.data import RheoData

            row: dict[str, Any] = {"Transform": name}

            if pre_data is not None:
                row["Input N"] = len(pre_data.x)
                row["Input Domain"] = getattr(pre_data, "domain", "")

            if isinstance(result, RheoData):
                row["Output N"] = len(result.x)
                row["Output Domain"] = getattr(result, "domain", "")
            elif isinstance(result, tuple) and len(result) >= 1:
                r0 = result[0]
                if isinstance(r0, RheoData):
                    row["Output N"] = len(r0.x)
                    row["Output Domain"] = getattr(r0, "domain", "")

            rows.append(row)

        pd.DataFrame(rows).to_excel(writer, sheet_name="Transforms", index=False)

    def _write_excel_bayesian(self, writer: Any, state: dict[str, Any]) -> None:
        """Write Bayesian inference summary sheet."""
        import pandas as pd

        br = state["bayesian_result"]
        rows = []
        for attr in ("num_samples", "num_warmup", "num_chains", "num_divergences"):
            val = getattr(br, attr, None)
            if val is not None:
                rows.append({"Field": attr, "Value": val})

        # Posterior summary statistics
        posterior = getattr(br, "posterior_samples", None)
        if posterior and isinstance(posterior, dict):
            for name, samples in posterior.items():
                if name.startswith("sigma"):
                    continue
                arr = np.asarray(samples).ravel()
                rows.append({"Field": f"{name} (mean)", "Value": float(np.mean(arr))})
                rows.append({"Field": f"{name} (std)", "Value": float(np.std(arr))})
                rows.append(
                    {"Field": f"{name} (2.5%)", "Value": float(np.percentile(arr, 2.5))}
                )
                rows.append(
                    {
                        "Field": f"{name} (97.5%)",
                        "Value": float(np.percentile(arr, 97.5)),
                    }
                )

        pd.DataFrame(rows).to_excel(writer, sheet_name="Bayesian", index=False)

    def _write_excel_plots(self, writer: Any, state: dict[str, Any]) -> None:
        """Embed matplotlib figures into Excel worksheets."""
        import io

        try:
            from openpyxl.drawing.image import Image as XLImage
        except ImportError:
            logger.debug(
                "openpyxl.drawing.image not available — skipping plot embedding"
            )
            return

        fig = state["current_figure"]
        if fig is None:
            return

        workbook = writer.book
        sheet_name = "Plot_Analysis"
        if sheet_name not in workbook.sheetnames:
            workbook.create_sheet(sheet_name)
        sheet = workbook[sheet_name]

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        # openpyxl reads the buffer lazily at workbook save time,
        # so give it an independent copy to avoid early-close issues.
        img = XLImage(io.BytesIO(buf.getvalue()))
        img.anchor = "A1"
        sheet.add_image(img)
