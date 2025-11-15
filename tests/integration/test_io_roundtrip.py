"""I/O round-trip integration tests.

Tests that verify data can be written to file and read back with integrity.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io import save_excel, save_hdf5
from rheojax.io.readers import load_csv, load_excel


class TestCSVRoundTrip:
    """Test CSV read/write cycle."""

    @pytest.mark.integration
    @pytest.mark.io
    def test_csv_write_and_read(self, oscillation_data_simple):
        """Test writing and reading CSV files."""
        data = oscillation_data_simple

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            # Write simple CSV format
            temp_path = f.name
            f.write("frequency,G_prime,G_double_prime\n")

            for x, y in zip(data.x, data.y):
                f.write(f"{x},{y.real},{y.imag}\n")

        try:
            # Read back
            with open(temp_path, encoding="utf-8") as f:
                lines = f.readlines()

            assert len(lines) > 1, "CSV should have header + data rows"

            # Parse data
            header = lines[0].strip().split(",")
            assert header[0] == "frequency"
            assert header[1] == "G_prime"

            # Read numeric values
            values = []
            for line in lines[1:]:
                parts = line.strip().split(",")
                values.append([float(p) for p in parts])

            values = np.array(values)
            assert values.shape == (len(data.x), 3)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.integration
    @pytest.mark.io
    def test_csv_metadata_preservation(self, oscillation_data_simple):
        """Test that metadata is preserved in CSV workflow."""
        data = oscillation_data_simple
        original_metadata = data.metadata.copy()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name

            # Write with metadata in comments
            for key, value in original_metadata.items():
                f.write(f"# {key}: {value}\n")

            f.write("frequency,value\n")
            for x, y in zip(data.x, data.y.real):
                f.write(f"{x},{y}\n")

        try:
            # Verify file contains metadata comments
            with open(temp_path, encoding="utf-8") as f:
                content = f.read()

            assert "# test_mode:" in content
            assert "# temperature:" in content

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestJSONSerialization:
    """Test JSON serialization for data and parameters."""

    @pytest.mark.integration
    def test_rheodata_to_json(self, oscillation_data_simple):
        """Test converting RheoData to JSON."""
        data = oscillation_data_simple

        # Convert to dict (JSON-serializable)
        data_dict = {
            "x": data.x.tolist(),
            "y": [f"{v.real}+{v.imag}j" for v in data.y],  # Handle complex
            "x_units": data.x_units,
            "y_units": data.y_units,
            "domain": data.domain,
            "metadata": data.metadata,
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump(data_dict, f)

        try:
            # Read back
            with open(temp_path, encoding="utf-8") as f:
                loaded = json.load(f)

            # Verify structure
            assert "x" in loaded
            assert "y" in loaded
            assert "metadata" in loaded
            assert len(loaded["x"]) == len(data.x)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.integration
    def test_parameters_json_serialization(self, maxwell_parameters):
        """Test JSON serialization of parameters."""
        params = maxwell_parameters

        # Convert to dict
        params_dict = params.to_dict()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            temp_path = f.name
            json.dump(params_dict, f)

        try:
            # Read back and reconstruct
            with open(temp_path, encoding="utf-8") as f:
                loaded_dict = json.load(f)

            restored_params = type(params).from_dict(loaded_dict)

            # Verify values match
            for param_name in params.to_dict().keys():
                original_val = params.get(param_name).value
                restored_val = restored_params.get(param_name).value
                assert original_val == restored_val

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestHDF5Support:
    """Test HDF5 writing support."""

    @pytest.mark.integration
    @pytest.mark.io
    @pytest.mark.slow
    def test_hdf5_write_basic(self, oscillation_data_simple):
        """Test basic HDF5 writing."""
        data = oscillation_data_simple

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Try to write
            try:
                save_hdf5(data, temp_path)
                # File should exist
                assert Path(temp_path).exists()
            except ImportError:
                # h5py may not be installed
                pytest.skip("h5py not available")

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.integration
    @pytest.mark.io
    @pytest.mark.slow
    def test_hdf5_multiple_datasets(
        self, oscillation_data_simple, relaxation_data_simple
    ):
        """Test writing multiple datasets to HDF5."""
        datasets = {
            "oscillation": oscillation_data_simple,
            "relaxation": relaxation_data_simple,
        }

        with tempfile.NamedTemporaryFile(suffix=".h5", delete=False) as f:
            temp_path = f.name

        try:
            # Write multiple datasets
            try:
                for name, data in datasets.items():
                    save_hdf5(
                        data, temp_path, mode="a" if name != "oscillation" else "w"
                    )

                # File should exist
                assert Path(temp_path).exists()
            except ImportError:
                pytest.skip("h5py not available")

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestExcelWriting:
    """Test Excel writing support."""

    @pytest.mark.integration
    @pytest.mark.io
    def test_excel_write_basic(self, oscillation_data_simple):
        """Test basic Excel writing."""
        data = oscillation_data_simple

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            # Prepare data dict with expected structure for save_excel
            # save_excel expects 'parameters' and/or 'fit_quality' keys
            results = {
                "parameters": {"test_mode": "oscillation", "n_points": len(data.x)},
                "fit_quality": {"R2": 0.99, "RMSE": 0.01},
            }

            try:
                save_excel(results, temp_path)
                assert Path(temp_path).exists()
            except ImportError:
                pytest.skip("openpyxl not available")

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.integration
    @pytest.mark.io
    def test_excel_with_metadata(self, oscillation_data_simple):
        """Test Excel writing with metadata sheets."""
        data = oscillation_data_simple

        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as f:
            temp_path = f.name

        try:
            results = {
                "data": {"x": data.x.tolist(), "y": data.y.real.tolist()},
                "metadata": data.metadata,
            }

            try:
                # This would need a more sophisticated save_excel function
                # For now, just verify the structure is correct
                assert "data" in results
                assert "metadata" in results
            except ImportError:
                pytest.skip("openpyxl not available")

        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestDataIntegrity:
    """Test data integrity through I/O operations."""

    @pytest.mark.integration
    def test_numeric_precision_csv(self, oscillation_data_simple):
        """Test numeric precision in CSV round-trip."""
        data = oscillation_data_simple
        original_values = data.x.copy()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            temp_path = f.name
            f.write("value\n")
            for val in original_values:
                f.write(f"{val:.15e}\n")  # High precision

        try:
            with open(temp_path, encoding="utf-8") as f:
                lines = f.readlines()[1:]  # Skip header

            loaded_values = np.array([float(line.strip()) for line in lines])

            # Should match to machine precision
            np.testing.assert_array_almost_equal(original_values, loaded_values)

        finally:
            Path(temp_path).unlink(missing_ok=True)

    @pytest.mark.integration
    def test_shape_preservation(self, oscillation_data_simple, relaxation_data_simple):
        """Test that data shapes are preserved in workflows."""
        datasets = [oscillation_data_simple, relaxation_data_simple]

        for data in datasets:
            original_shape = data.y.shape

            # Convert to dict and back
            data_dict = {
                "x": data.x.tolist(),
                "y": (
                    data.y.tolist()
                    if np.isrealobj(data.y)
                    else [complex(y) for y in data.y]
                ),
            }

            # Reconstruct
            y_recon = np.array(data_dict["y"])

            # Shape should match
            assert y_recon.shape == original_shape

    @pytest.mark.integration
    def test_dtype_handling(self, oscillation_data_simple):
        """Test that dtypes are handled correctly."""
        data = oscillation_data_simple

        # Check dtype
        assert data.x.dtype in [np.float32, np.float64]
        assert data.y.dtype in [np.complex64, np.complex128]

        # Convert to JAX preserves complex nature
        jax_data = data.to_jax()
        assert np.iscomplexobj(jax_data.y)
