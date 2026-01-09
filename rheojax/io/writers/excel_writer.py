"""Excel writer for rheological data and results."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)


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

        # Create Excel writer
        with pd.ExcelWriter(filepath, engine="openpyxl") as writer:
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
                pred_df = _create_predictions_dataframe(results["predictions"])
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
                logger.debug("Plots embedded", plot_names=list(results["plots"].keys()))

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
                    "Value": value.get("value", value),
                    "Units": value.get("units", ""),
                    "Bounds": str(value.get("bounds", "")),
                }
            )
        else:
            # Simple parameter value
            data.append(
                {
                    "Parameter": name,
                    "Value": value,
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
                "Value": value,
            }
        )

    return pd.DataFrame(data)


def _create_predictions_dataframe(predictions: np.ndarray) -> Any:
    """Create DataFrame for predictions.

    Args:
        predictions: Array of predictions

    Returns:
        pandas DataFrame
    """
    import pandas as pd

    return pd.DataFrame(
        {
            "Index": np.arange(len(predictions)),
            "Prediction": predictions,
        }
    )


def _create_residuals_dataframe(residuals: np.ndarray) -> Any:
    """Create DataFrame for residuals.

    Args:
        residuals: Array of residuals

    Returns:
        pandas DataFrame
    """
    import pandas as pd

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
        logger.debug(
            "openpyxl.drawing.image not available, skipping plot embedding",
            reason="ImportError",
        )
        # openpyxl not available, skip plot embedding
        return

    workbook = writer.book

    for plot_name, fig in plots.items():
        # Create sheet for this plot
        sheet_name = f"Plot_{plot_name[:25]}"  # Excel sheet name limit
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
