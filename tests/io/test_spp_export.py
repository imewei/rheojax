"""Tests for SPP export functionality."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.io.spp_export import (
    export_spp_csv,
    export_spp_hdf5,
    export_spp_txt,
    to_matlab_dict,
)


def _make_mock_spp_results(n_points: int = 500) -> dict:
    """Create mock SPP results dict for testing."""
    time = np.linspace(0, 2 * np.pi, n_points)
    strain = 0.5 * np.sin(time)
    rate = 0.5 * np.cos(time)
    stress = 100.0 * strain + 30.0 * np.sin(3 * time)

    # Create moduli arrays
    Gp_t = 100.0 * np.ones(n_points) + 5.0 * np.sin(2 * time)
    Gpp_t = 30.0 * np.ones(n_points) + 3.0 * np.cos(2 * time)

    return {
        "time_new": time,
        "strain_recon": strain,
        "rate_recon": rate,
        "stress_recon": stress,
        "Gp_t": Gp_t,
        "Gpp_t": Gpp_t,
        "G_star_t": np.sqrt(Gp_t**2 + Gpp_t**2),
        "tan_delta_t": Gpp_t / Gp_t,
        "delta_t": np.arctan(Gpp_t / Gp_t),
        "disp_stress": np.zeros(n_points),
        "eq_strain_est": strain,
        "Gp_t_dot": np.gradient(Gp_t, time),
        "Gpp_t_dot": np.gradient(Gpp_t, time),
        "G_speed": np.abs(np.gradient(Gp_t, time)),
        "delta_t_dot": np.zeros(n_points),
        "Delta": 0.0,
        # Frenet-Serret frame
        "T_vec": np.column_stack(
            [np.ones(n_points), np.zeros(n_points), np.zeros(n_points)]
        ),
        "N_vec": np.column_stack(
            [np.zeros(n_points), np.ones(n_points), np.zeros(n_points)]
        ),
        "B_vec": np.column_stack(
            [np.zeros(n_points), np.zeros(n_points), np.ones(n_points)]
        ),
    }


# ============================================================================
# Text Export Tests
# ============================================================================


class TestExportSppTxt:
    """Tests for MATLAB-compatible text export."""

    def test_creates_main_file(self):
        """Test that main SPP data file is created."""
        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sample"
            export_spp_txt(filepath, results, omega=1.0, include_fsf=False)

            main_file = Path(tmpdir) / "test_sample_SPP_NUMERICAL.txt"
            assert main_file.exists()

    def test_creates_fsf_file(self):
        """Test that Frenet-Serret frame file is created when requested."""
        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sample"
            export_spp_txt(filepath, results, omega=1.0, include_fsf=True)

            fsf_file = Path(tmpdir) / "test_sample_SPP_NUMERICAL_FSFRAME.txt"
            assert fsf_file.exists()

    def test_main_file_has_correct_columns(self):
        """Test that main file has 15 columns matching MATLAB format."""
        results = _make_mock_spp_results(n_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sample"
            export_spp_txt(filepath, results, omega=1.0, include_fsf=False)

            main_file = Path(tmpdir) / "test_sample_SPP_NUMERICAL.txt"
            lines = main_file.read_text().split("\r\n")

            # Find first data line (after headers)
            data_lines = [
                ln
                for ln in lines
                if ln
                and not any(
                    x in ln
                    for x in [
                        "Data calculated",
                        "Frequency",
                        "Number",
                        "Step",
                        "Time",
                        "[s]",
                        "differentiation",
                    ]
                )
            ]
            if data_lines:
                first_data = data_lines[0].split("\t")
                assert len(first_data) == 15

    def test_header_matches_matlab_format(self):
        """Test that header row matches MATLAB SPPplus format."""
        results = _make_mock_spp_results(n_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sample"
            export_spp_txt(filepath, results, omega=1.0, step_size=8, num_mode=1)

            main_file = Path(tmpdir) / "test_sample_SPP_NUMERICAL.txt"
            content = main_file.read_text()

            # Check for expected headers
            assert "Time" in content
            assert "Strain" in content
            assert "G'_t" in content
            assert 'G"_t' in content
            assert "displacement stress" in content

    def test_parameters_written_correctly(self):
        """Test that analysis parameters are written to file."""
        results = _make_mock_spp_results(n_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sample"
            export_spp_txt(
                filepath,
                results,
                omega=1.5,
                n_harmonics=39,
                n_cycles=2,
                step_size=8,
                num_mode=1,
                analysis_type="NUMERICAL",
            )

            main_file = Path(tmpdir) / "test_sample_SPP_NUMERICAL.txt"
            content = main_file.read_text()

            assert "Frequency:" in content
            assert "Step size" in content
            assert "Standard differentiation" in content

    def test_fourier_analysis_type(self):
        """Test Fourier analysis type creates correct filename."""
        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_sample"
            export_spp_txt(filepath, results, omega=1.0, analysis_type="FOURIER")

            fourier_file = Path(tmpdir) / "test_sample_SPP_FOURIER.txt"
            assert fourier_file.exists()


# ============================================================================
# HDF5 Export Tests
# ============================================================================


class TestExportSppHdf5:
    """Tests for HDF5 export."""

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py not installed"),
        reason="h5py required",
    )
    def test_creates_hdf5_file(self):
        """Test that HDF5 file is created."""
        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            export_spp_hdf5(filepath, results, omega=1.0, gamma_0=0.5)
            assert filepath.exists()

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py not installed"),
        reason="h5py required",
    )
    def test_hdf5_contains_expected_groups(self):
        """Test that HDF5 file contains expected data groups."""
        import h5py

        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            export_spp_hdf5(filepath, results, omega=1.0, gamma_0=0.5)

            with h5py.File(filepath, "r") as f:
                assert "metadata" in f
                assert "spp_data" in f
                assert "waveforms" in f
                assert "frenet_serret" in f

    @pytest.mark.skipif(
        not pytest.importorskip("h5py", reason="h5py not installed"),
        reason="h5py required",
    )
    def test_hdf5_metadata_attributes(self):
        """Test that HDF5 metadata contains expected attributes."""
        import h5py

        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.h5"
            export_spp_hdf5(filepath, results, omega=1.5, gamma_0=0.3)

            with h5py.File(filepath, "r") as f:
                assert f["metadata"].attrs["omega"] == 1.5
                assert f["metadata"].attrs["gamma_0"] == 0.3


# ============================================================================
# CSV Export Tests
# ============================================================================


class TestExportSppCsv:
    """Tests for CSV export."""

    def test_creates_csv_file(self):
        """Test that CSV file is created."""
        results = _make_mock_spp_results()

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            export_spp_csv(filepath, results)
            assert filepath.exists()

    def test_csv_has_header_row(self):
        """Test that CSV has proper header row."""
        results = _make_mock_spp_results(n_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            export_spp_csv(filepath, results)

            lines = filepath.read_text().strip().split("\n")
            header = lines[0]

            assert "time" in header
            assert "Gp_t" in header
            assert "Gpp_t" in header

    def test_csv_includes_fsf_when_requested(self):
        """Test that CSV includes Frenet-Serret columns when requested."""
        results = _make_mock_spp_results(n_points=100)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            export_spp_csv(filepath, results, include_fsf=True)

            header = filepath.read_text().split("\n")[0]
            assert "T_x" in header
            assert "N_x" in header
            assert "B_x" in header

    def test_csv_data_rows_count(self):
        """Test that CSV has correct number of data rows."""
        n_points = 250
        results = _make_mock_spp_results(n_points=n_points)

        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.csv"
            export_spp_csv(filepath, results)

            lines = filepath.read_text().strip().split("\n")
            # Header + data rows
            assert len(lines) == n_points + 1


# ============================================================================
# MATLAB Dict Export Tests
# ============================================================================


class TestToMatlabDict:
    """Tests for MATLAB-compatible dict export."""

    def test_returns_dict_with_out_spp(self):
        """Test that dict contains out_spp key."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(results, omega=1.0)

        assert "out_spp" in mat_dict

    def test_out_spp_has_required_keys(self):
        """Test that out_spp has info, headers, and data."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(results, omega=1.0)

        out_spp = mat_dict["out_spp"]
        assert "info" in out_spp
        assert "headers" in out_spp
        assert "data" in out_spp

    def test_data_has_15_columns(self):
        """Test that data matrix has 15 columns matching MATLAB."""
        results = _make_mock_spp_results(n_points=100)
        mat_dict = to_matlab_dict(results, omega=1.0)

        data = mat_dict["out_spp"]["data"]
        assert data.shape[1] == 15

    def test_info_contains_frequency(self):
        """Test that info dict contains frequency."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(results, omega=2.5)

        info = mat_dict["out_spp"]["info"]
        assert info["frequency"] == 2.5

    def test_includes_out_fsf_when_frame_available(self):
        """Test that out_fsf is included when Frenet-Serret data available."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(results, omega=1.0)

        assert "out_fsf" in mat_dict
        assert mat_dict["out_fsf"]["data"].shape[1] == 9

    def test_numerical_analysis_type(self):
        """Test numerical analysis type sets correct data_calc."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(results, omega=1.0, analysis_type="numerical")

        info = mat_dict["out_spp"]["info"]
        assert "numerical" in info["data_calc"].lower()

    def test_fourier_analysis_type(self):
        """Test Fourier analysis type sets correct data_calc."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(results, omega=1.0, analysis_type="fourier")

        info = mat_dict["out_spp"]["info"]
        assert "fourier" in info["data_calc"].lower()

    def test_optional_parameters_included(self):
        """Test that optional parameters are included when provided."""
        results = _make_mock_spp_results()
        mat_dict = to_matlab_dict(
            results, omega=1.0, n_harmonics=39, n_cycles=2, step_size=8, num_mode=1
        )

        info = mat_dict["out_spp"]["info"]
        assert info["number_of_harmonics"] == 39
        assert info["number_of_cycles"] == 2
        assert info["diff_step_size"] == 8
        assert "Standard" in info["diff_type"]
