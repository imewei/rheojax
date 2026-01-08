"""TRIOS JSON Experiment and Result dataclasses.

This module provides dataclasses for representing the complete structure
of a TRIOS JSON export, including experiment metadata, results, and datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from rheojax.io.readers.trios.schema.dataset import TRIOSDataSet


@dataclass
class TRIOSResult:
    """Single result set within a TRIOS experiment.

    Represents one test result containing properties (metadata)
    and one or more datasets.

    Attributes:
        properties: Result-level properties (e.g., step number, name)
        datasets: List of datasets (typically one)
    """

    properties: dict[str, Any] = field(default_factory=dict)
    datasets: list[TRIOSDataSet] = field(default_factory=list)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TRIOSResult:
        """Create TRIOSResult from dictionary.

        Args:
            data: Dictionary containing Result structure

        Returns:
            TRIOSResult instance
        """
        properties = data.get("Properties", {})

        datasets = []
        dataset_list = data.get("DataSet", [])
        if isinstance(dataset_list, dict):
            # Single dataset wrapped in object
            dataset_list = [dataset_list]

        for ds_data in dataset_list:
            datasets.append(TRIOSDataSet.from_dict(ds_data))

        return cls(properties=properties, datasets=datasets)

    def get_dataframe(self, dataset_index: int = 0) -> pd.DataFrame:
        """Get DataFrame from specified dataset.

        Args:
            dataset_index: Index of dataset to extract (default: 0)

        Returns:
            DataFrame from the dataset
        """
        if not self.datasets or dataset_index >= len(self.datasets):
            return pd.DataFrame()
        return self.datasets[dataset_index].to_dataframe()

    def get_units(self, dataset_index: int = 0) -> dict[str, str]:
        """Get units mapping from specified dataset.

        Args:
            dataset_index: Index of dataset (default: 0)

        Returns:
            Dictionary mapping column names to units
        """
        if not self.datasets or dataset_index >= len(self.datasets):
            return {}
        return self.datasets[dataset_index].get_units()

    @property
    def step(self) -> int | None:
        """Get step number from properties if present."""
        return self.properties.get("Step")

    @property
    def name(self) -> str | None:
        """Get result name from properties if present."""
        return self.properties.get("Name")


@dataclass
class TRIOSExperiment:
    """Root container for TRIOS JSON experiment data.

    Represents the complete experiment structure as defined in
    TRIOSJSONExportSchema.json.

    Attributes:
        properties: Experiment-level properties (name, date, operator, etc.)
        sample: Sample information (name, description, etc.)
        procedure: Test procedure definition
        results: List of result sets (one per test)
        schema_version: Schema version for validation
    """

    properties: dict[str, Any] = field(default_factory=dict)
    sample: dict[str, Any] = field(default_factory=dict)
    procedure: dict[str, Any] = field(default_factory=dict)
    results: list[TRIOSResult] = field(default_factory=list)
    schema_version: str | None = None

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> TRIOSExperiment:
        """Create TRIOSExperiment from parsed JSON dict.

        Args:
            data: Parsed JSON dictionary (should contain "Experiment" key)

        Returns:
            TRIOSExperiment instance

        Raises:
            ValueError: If JSON structure is invalid
        """
        # Handle both wrapped and unwrapped formats
        if "Experiment" in data:
            exp_data = data["Experiment"]
        else:
            exp_data = data

        properties = exp_data.get("Properties", {})
        sample = exp_data.get("Sample", {})
        procedure = exp_data.get("Procedure", {})

        # Parse results
        results = []
        results_list = exp_data.get("Results", [])
        if isinstance(results_list, dict):
            # Single result wrapped in object
            results_list = [results_list]

        for result_data in results_list:
            results.append(TRIOSResult.from_dict(result_data))

        # Get schema version if present
        schema_version = data.get("$schema") or data.get("schemaVersion")

        return cls(
            properties=properties,
            sample=sample,
            procedure=procedure,
            results=results,
            schema_version=schema_version,
        )

    def get_dataframe(
        self, result_index: int = 0, dataset_index: int = 0
    ) -> pd.DataFrame:
        """Extract DataFrame from specified result set.

        Args:
            result_index: Index of result to extract (default: 0)
            dataset_index: Index of dataset within result (default: 0)

        Returns:
            DataFrame from the specified result/dataset
        """
        if not self.results or result_index >= len(self.results):
            return pd.DataFrame()
        return self.results[result_index].get_dataframe(dataset_index)

    def get_dataframes_by_step(
        self,
        result_index: int = 0,
        step_col: str = "Step",
    ) -> dict[int, pd.DataFrame]:
        """Split result DataFrame by step column.

        Args:
            result_index: Index of result to split (default: 0)
            step_col: Name of step column (default: "Step")

        Returns:
            Dictionary mapping step number to DataFrame
        """
        df = self.get_dataframe(result_index)

        if df.empty or step_col not in df.columns:
            return {0: df} if not df.empty else {}

        step_dfs = {}
        for step_val, group in df.groupby(step_col, sort=False):
            step_dfs[int(step_val)] = group.reset_index(drop=True)

        return step_dfs

    def get_units(
        self, result_index: int = 0, dataset_index: int = 0
    ) -> dict[str, str]:
        """Get units mapping from specified result.

        Args:
            result_index: Index of result (default: 0)
            dataset_index: Index of dataset (default: 0)

        Returns:
            Dictionary mapping column names to units
        """
        if not self.results or result_index >= len(self.results):
            return {}
        return self.results[result_index].get_units(dataset_index)

    def get_all_dataframes(self) -> list[pd.DataFrame]:
        """Get DataFrames from all results.

        Returns:
            List of DataFrames, one per result
        """
        dfs = []
        for result in self.results:
            df = result.get_dataframe()
            if not df.empty:
                dfs.append(df)
        return dfs

    @property
    def experiment_name(self) -> str | None:
        """Get experiment name from properties."""
        return self.properties.get("Name")

    @property
    def date(self) -> str | None:
        """Get experiment date from properties."""
        return self.properties.get("Date")

    @property
    def operator(self) -> str | None:
        """Get operator from properties."""
        return self.properties.get("Operator")

    @property
    def instrument_name(self) -> str | None:
        """Get instrument name from properties."""
        return self.properties.get("InstrumentName")

    @property
    def sample_name(self) -> str | None:
        """Get sample name."""
        return self.sample.get("Name")

    @property
    def n_results(self) -> int:
        """Get number of results."""
        return len(self.results)

    def get_metadata(self) -> dict[str, Any]:
        """Get consolidated metadata from experiment.

        Returns:
            Dictionary with all experiment metadata
        """
        metadata = {}

        # Add properties
        for key, value in self.properties.items():
            metadata[_snake_case(key)] = value

        # Add sample info
        if self.sample:
            for key, value in self.sample.items():
                metadata[f"sample_{_snake_case(key)}"] = value

        # Add procedure info
        if self.procedure:
            metadata["procedure_name"] = self.procedure.get("Name")

        return metadata


def _snake_case(s: str) -> str:
    """Convert CamelCase to snake_case."""
    result = []
    for i, char in enumerate(s):
        if char.isupper() and i > 0:
            result.append("_")
        result.append(char.lower())
    return "".join(result)
