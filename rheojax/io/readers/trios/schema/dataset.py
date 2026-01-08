"""TRIOS JSON DataSet dataclass.

This module provides the TRIOSDataSet dataclass for representing
column definitions and data values from a TRIOS JSON result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd


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
        properties = data.get("Properties", {})

        # Handle "Data" wrapper if present
        data_section = data.get("Data", data)
        columns = data_section.get("columns", [])
        values = data_section.get("values", [])

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
        if not self.columns or not self.values:
            return pd.DataFrame()

        column_names = [
            col.get("name", f"col_{i}") for i, col in enumerate(self.columns)
        ]

        df = pd.DataFrame(self.values, columns=column_names)

        # Convert numeric columns
        for col in df.columns:
            try:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            except (ValueError, TypeError):
                pass

        return df

    def get_units(self) -> dict[str, str]:
        """Extract units mapping from column definitions.

        Returns:
            Dictionary mapping column names to units
        """
        units: dict[str, str] = {}
        for col in self.columns:
            name = col.get("name", "")
            unit = col.get("unit", "")
            if name and unit:
                units[name] = unit
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
