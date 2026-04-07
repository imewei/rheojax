"""Excel writer for rheological data and results."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)


def _to_python_scalar(value: Any) -> Any:
    """Convert JAX/numpy scalars to native Python types for Excel compatibility."""
    if isinstance(value, (int, float, str, bool)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    # JAX arrays: check for .item() method (0-d arrays)
    if hasattr(value, "item") and hasattr(value, "shape"):
        try:
            if value.shape == () or value.size == 1:
                return value.item()
        except (TypeError, ValueError):
            pass
    return value


def save_excel(
    results: dict[str, Any], filepath: str | Path, include_plots: bool = False, **kwargs
) -> None:
    """Save results to Excel file for reporting.

    Creates an Excel workbook with multiple sheets for different result types:
    - Parameters sheet: Model parameters and values
    - Fit Quality sheet: R², RMSE, and other metrics
    - Predictions sheet: Model predictions
    - Residuals sheet: Fitting residuals

    Args:
        results: Dictionary containing results
            - 'parameters': dict of parameter names and values
            - 'fit_quality': dict of fit metrics (R2, RMSE, etc.)
            - 'predictions': array of model predictions (optional)
            - 'residuals': array of residuals (optional)
        filepath: Output file path (.xlsx)
        include_plots: Include embedded plots (requires matplotlib)
        **kwargs: Additional arguments

    Raises:
        ImportError: If pandas or openpyxl not installed
        ValueError: If results format is invalid
        IOError: If file cannot be written
    """
    try:
        import pandas as pd
    except ImportError as exc:
        logger.error(
            "pandas import failed",
            error_type="ImportError",
            suggestion="pip install pandas openpyxl",
            exc_info=True,
        )
        raise ImportError(
            "pandas is required for Excel writing. Install with: pip install pandas openpyxl"
        ) from exc

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with log_io(logger, "write", filepath=str(filepath)) as ctx:
        sheets_written = []

        if not results:
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(filepath.parent), suffix=".tmp.xlsx"
            )
            try:
                os.close(tmp_fd)
            except OSError:
                os.unlink(tmp_path)
                raise
            try:
                with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                    pd.DataFrame({"Info": ["No results to export"]}).to_excel(
                        writer, sheet_name="Empty", index=False
                    )
                os.replace(tmp_path, str(filepath))
                tmp_path = None  # type: ignore[assignment]  # prevent cleanup
            finally:
                if tmp_path is not None:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
            ctx["sheets_written"] = ["Empty"]
            ctx["num_sheets"] = 1
            ctx["include_plots"] = include_plots
            return

        # Create Excel writer with atomic write (temp file + replace)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(filepath.parent), suffix=".tmp.xlsx"
        )
        try:
            os.close(tmp_fd)
        except OSError:
            os.unlink(tmp_path)
            raise
        try:
            with pd.ExcelWriter(tmp_path, engine="openpyxl") as writer:
                # Write parameters sheet
                if "parameters" in results:
                    logger.debug(
                        "Creating parameters dataframe",
                        num_parameters=len(results["parameters"]),
                    )
                    params_df = _create_parameters_dataframe(results["parameters"])
                    params_df.to_excel(writer, sheet_name="Parameters", index=False)
                    sheets_written.append("Parameters")
                    logger.debug("Parameters sheet written", rows=len(params_df))

                # Write fit quality sheet
                if "fit_quality" in results:
                    logger.debug(
                        "Creating fit quality dataframe",
                        num_metrics=len(results["fit_quality"]),
                    )
                    quality_df = _create_quality_dataframe(results["fit_quality"])
                    quality_df.to_excel(writer, sheet_name="Fit Quality", index=False)
                    sheets_written.append("Fit Quality")
                    logger.debug("Fit Quality sheet written", rows=len(quality_df))

                # Write predictions sheet
                if "predictions" in results:
                    logger.debug(
                        "Creating predictions dataframe",
                        num_predictions=len(results["predictions"]),
                    )
                    pred_df = _create_predictions_dataframe(
                        results["predictions"],
                        deformation_mode=results.get("deformation_mode"),
                    )
                    pred_df.to_excel(writer, sheet_name="Predictions", index=False)
                    sheets_written.append("Predictions")
                    logger.debug("Predictions sheet written", rows=len(pred_df))

                # Write residuals sheet
                if "residuals" in results:
                    logger.debug(
                        "Creating residuals dataframe",
                        num_residuals=len(results["residuals"]),
                    )
                    resid_df = _create_residuals_dataframe(results["residuals"])
                    resid_df.to_excel(writer, sheet_name="Residuals", index=False)
                    sheets_written.append("Residuals")
                    logger.debug("Residuals sheet written", rows=len(resid_df))

                # Embed plots if requested
                if include_plots and "plots" in results:
                    logger.debug(
                        "Embedding plots",
                        num_plots=len(results["plots"]),
                    )
                    _embed_plots(writer, results["plots"])
                    sheets_written.extend(
                        [f"Plot_{name[:25]}" for name in results["plots"].keys()]
                    )
                    logger.debug(
                        "Plots embedded", plot_names=list(results["plots"].keys())
                    )

            os.replace(tmp_path, str(filepath))
            tmp_path = None  # type: ignore[assignment]  # prevent cleanup
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        ctx["sheets_written"] = sheets_written
        ctx["num_sheets"] = len(sheets_written)
        ctx["include_plots"] = include_plots


def _create_parameters_dataframe(parameters: dict[str, Any]) -> Any:
    """Create DataFrame for parameters.

    Args:
        parameters: Dictionary of parameter names and values

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    data = []
    for name, value in parameters.items():
        if isinstance(value, dict):
            # Parameter with additional info
            data.append(
                {
                    "Parameter": name,
                    "Value": _to_python_scalar(value.get("value", value)),
                    "Units": value.get("units", ""),
                    "Bounds": str(value.get("bounds", "")),
                }
            )
        else:
            # Simple parameter value (may be JAX/numpy scalar)
            data.append(
                {
                    "Parameter": name,
                    "Value": _to_python_scalar(value),
                    "Units": "",
                    "Bounds": "",
                }
            )

    return pd.DataFrame(data)


def _create_quality_dataframe(fit_quality: dict[str, Any]) -> Any:
    """Create DataFrame for fit quality metrics.

    Args:
        fit_quality: Dictionary of fit quality metrics

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    data = []
    for metric, value in fit_quality.items():
        data.append(
            {
                "Metric": metric,
                "Value": _to_python_scalar(value),
            }
        )

    return pd.DataFrame(data)


def _create_predictions_dataframe(
    predictions: np.ndarray,
    deformation_mode: str | None = None,
) -> Any:
    """Create DataFrame for predictions.

    Handles complex arrays (G*=G'+iG'' or E*=E'+iE'') by splitting into
    separate columns. Column labels use the appropriate modulus prefix
    based on the deformation mode.

    Args:
        predictions: Array of predictions (real or complex)
        deformation_mode: Deformation mode ('tension', 'bending',
            'compression', or None for shear). Controls column label prefix.

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    predictions = np.asarray(predictions)
    prefix = "E" if deformation_mode in {"tension", "bending", "compression"} else "G"
    if np.iscomplexobj(predictions):
        return pd.DataFrame(
            {
                "Index": np.arange(len(predictions)),
                f"{prefix}' (Storage)": np.real(predictions),
                f"{prefix}'' (Loss)": np.imag(predictions),
            }
        )
    if predictions.ndim == 2:
        # GMM output returns (N, 2) real arrays [G'/E', G''/E'']
        col_names = [f"Component_{i}" for i in range(predictions.shape[1])]
        if predictions.shape[1] == 2:
            col_names = [f"{prefix}' (Storage)", f"{prefix}'' (Loss)"]
        df_dict = {"Index": np.arange(len(predictions))}
        for i, name in enumerate(col_names):
            df_dict[name] = predictions[:, i]
        return pd.DataFrame(df_dict)
    return pd.DataFrame(
        {
            "Index": np.arange(len(predictions)),
            "Prediction": predictions,
        }
    )


def _create_residuals_dataframe(residuals: np.ndarray) -> Any:
    """Create DataFrame for residuals.

    Handles complex arrays by splitting into real/imaginary components.

    Args:
        residuals: Array of residuals (real or complex)

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    residuals = np.asarray(residuals)
    if np.iscomplexobj(residuals):
        return pd.DataFrame(
            {
                "Index": np.arange(len(residuals)),
                "Residual (Real)": np.real(residuals),
                "Residual (Imag)": np.imag(residuals),
            }
        )
    return pd.DataFrame(
        {
            "Index": np.arange(len(residuals)),
            "Residual": residuals,
        }
    )


def _embed_plots(writer: Any, plots: dict[str, Any]) -> None:
    """Embed plots in Excel workbook.

    Args:
        writer: ExcelWriter object
        plots: Dictionary of plot names and matplotlib figures

    Note:
        Requires openpyxl and matplotlib.
    """
    import io

    try:
        from openpyxl.drawing.image import Image as XLImage
    except ImportError:
        logger.warning(
            "openpyxl.drawing.image not available — plots will not be embedded. "
            "Install openpyxl with: pip install openpyxl",
            reason="ImportError",
        )
        return

    workbook = writer.book

    for plot_name, fig in plots.items():
        # Create sheet for this plot
        sheet_name = re.sub(
            r"[\\/*?\[\]:]", "_", f"Plot_{plot_name[:25]}"
        )  # Excel sheet name limit
        logger.debug(
            "Embedding plot",
            plot_name=plot_name,
            sheet_name=sheet_name,
        )
        if sheet_name not in workbook.sheetnames:
            workbook.create_sheet(sheet_name)
        sheet = workbook[sheet_name]

        # Save figure to bytes buffer
        buf = io.BytesIO()
        try:
            fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
            buf.seek(0)
            logger.debug(
                "Plot saved to buffer",
                plot_name=plot_name,
                buffer_size=buf.getbuffer().nbytes,
            )

            # Create and add image to sheet
            img = XLImage(buf)
            img.anchor = "A1"
            sheet.add_image(img)
        finally:
            buf.close()
