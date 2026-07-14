"""TRIOS JSON DataSet dataclass.

This module provides the TRIOSDataSet dataclass for representing
column definitions and data values from a TRIOS JSON result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from rheojax.logging import get_logger

logger = get_logger(__name__)


@dataclass
class TRIOSDataSet:
    """Dataset containing column definitions and values.

    Represents a single dataset from a TRIOS JSON result, including
    column metadata (name, unit, type) and the data values.

    Attributes:
        properties: Dataset-level properties (metadata)
        columns: List of column definitions with name, unit, type
        values: 2D list of data values (rows x columns)
    """

    properties: dict[str, Any] = field(default_factory=dict)
    columns: list[dict[str, str]] = field(default_factory=list)
    values: list[list[Any]] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TRIOSDataSet:
        """Create TRIOSDataSet from dictionary.

        Args:
            data: Dictionary containing DataSet structure

        Returns:
            TRIOSDataSet instance
        """
        logger.debug("Parsing TRIOSDataSet from dict", data_keys=list(data.keys()))

        properties = data.get("Properties", {})
        logger.debug("Parsed dataset properties", property_count=len(properties))

        # Handle "Data" wrapper if present
        has_data_wrapper = "Data" in data
        data_section = data.get("Data", data)
        logger.debug("Data wrapper present", has_data_wrapper=has_data_wrapper)

        columns = data_section.get("columns", [])
        values = data_section.get("values", [])

        column_names = [col.get("name", f"col_{i}") for i, col in enumerate(columns)]
        logger.debug(
            "TRIOSDataSet parsed successfully",
            column_count=len(columns),
            row_count=len(values),
            column_names=column_names,
        )

        return cls(
            properties=properties,
            columns=columns,
            values=values,
        )

    def to_dataframe(self) -> pd.DataFrame:
        """Convert to pandas DataFrame with proper column names.

        Returns:
            DataFrame with column names from schema
        """
        logger.debug(
            "Converting TRIOSDataSet to DataFrame",
            column_count=len(self.columns),
            row_count=len(self.values),
        )

        if not self.columns or not self.values:
            logger.debug("Empty dataset, returning empty DataFrame")
            return pd.DataFrame()

        column_names = [
            col.get("name", f"col_{i}") for i, col in enumerate(self.columns)
        ]

        df = pd.DataFrame(self.values, columns=column_names)
        logger.debug("DataFrame created", shape=df.shape, columns=list(df.columns))

        # Convert numeric columns via per-column majority vote instead of a
        # single blanket errors="coerce"/"raise":
        # - A column that is genuinely non-numeric (e.g. a TRIOS "Point
        #   type"/status flag column, every value fails) is left untouched
        #   rather than silently replaced with all-NaN.
        # - A column that is mostly numeric with a few bad cells (e.g. a
        #   stray "n.a." placeholder) is still coerced to numeric -- so it
        #   stays usable downstream instead of becoming an all-or-nothing
        #   hard failure -- but an explicit warning names how many values
        #   were dropped, so the loss is never silent.
        converted_columns = []
        for col in df.columns:
            original = df[col]
            non_null = original.notna()
            n_non_null = int(non_null.sum())
            if n_non_null == 0:
                continue
            coerced = pd.to_numeric(original, errors="coerce")
            n_failed = int((coerced.isna() & non_null).sum())
            if n_failed == n_non_null:
                logger.debug(
                    "Column is entirely non-numeric, preserving original values",
                    column=col,
                )
                continue
            if n_failed > 0:
                logger.warning(
                    "Column has some non-numeric values; coercing those to NaN",
                    column=col,
                    n_failed=n_failed,
                    n_total=n_non_null,
                )
            df[col] = coerced
            converted_columns.append(col)

        logger.debug(
            "DataFrame conversion completed",
            shape=df.shape,
            numeric_columns=converted_columns,
        )
        return df

    def get_units(self) -> dict[str, str]:
        """Extract units mapping from column definitions.

        Returns:
            Dictionary mapping column names to units
        """
        logger.debug("Extracting units from column definitions")

        units: dict[str, str] = {}
        for col in self.columns:
            name = col.get("name", "")
            unit = col.get("unit", "")
            if name and unit:
                units[name] = unit

        logger.debug(
            "Units extraction completed",
            unit_count=len(units),
            units=units,
        )
        return units

    @property
    def column_names(self) -> list[str]:
        """Get list of column names."""
        return [col.get("name", "") for col in self.columns]

    @property
    def n_rows(self) -> int:
        """Get number of data rows."""
        return len(self.values)

    @property
    def n_columns(self) -> int:
        """Get number of columns."""
        return len(self.columns)
