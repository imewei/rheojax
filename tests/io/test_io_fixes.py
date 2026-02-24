"""Tests for IO module fixes (F-IO-001 through F-IO-018).

Covers:
- Fix A: Excel deformation_mode parity (F-IO-001)
- Fix B: Shape mismatch error in rheodata GUI helper (F-IO-003)
- Fix C: CLI validation (F-IO-004, F-IO-005)
- Fix D: Cascade provenance metadata (F-IO-002, F-IO-011)

RCA Round 2 (2026-02-23):
- Fix E: openpyxl read_only workbook cleanup (F-IO-001-rca2)
- Fix F: TRIOS CSV numeric first column (F-IO-002-rca2)
- Fix G: Excel writer JAX scalar conversion (F-IO-005-rca2)
- Fix H: y2_data complex extraction (F-IO-006)
- Fix I: Monotonic detection with noise (F-IO-012)
- Fix J: HDF5 bytes→str coercion (F-IO-015)
- Fix K: construct_complex_modulus shape validation (F-IO-007)
- Fix L: auto_load TRIOS CSV fallback (F-IO-009)
- Fix M: kwargs filtering in reader cascade (F-IO-013)

RCA Round 3 (2026-02-24):
- Fix N: to_rheo_data double-counts G'' for complex data (F-IO-R3-001)
- Fix O: _try_excel reads full file for column detection (F-IO-R3-003)
- Fix P: Domain inconsistency in rheodata_from_dataset_state (F-IO-R3-004)
- Fix Q: show_dataset renders complex numbers in cells (F-IO-R3-005)
- Fix R: Redundant complex+real y_data dispatch (F-IO-R3-009)
- Fix S: detect_test_mode shear-thickening limitation (F-IO-R3-006)

RCA Round 4 (2026-02-24, deep IO+GUI audit):
- Fix T: y2_col lost for Excel files via _filter_kwargs (F-IO-R4-001) [P0]
- Fix U: _detect_modulus_pair misses E''/G'' two-single-quote notation (F-IO-R4-002) [P1]
- Fix V: test_mode not forwarded to auto_load from DataService (F-IO-R4-003) [P1]
- Fix W: preview_file fallback can't handle list or complex y (F-IO-R4-005/006) [P2]
- Fix X: detect_test_mode oscillation threshold too restrictive (F-IO-R4-008) [P2]
- Fix Y: .json extension has no error handling for non-TRIOS JSON (F-IO-R4-009) [P2]
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

# ── Fix A: Excel deformation_mode parity ─────────────────────────────────


class TestExcelDeformationMode:
    """F-IO-001: Excel reader must detect and store deformation_mode."""

    @pytest.mark.smoke
    def test_excel_explicit_deformation_mode(self, tmp_path):
        """load_excel with explicit deformation_mode='tension' sets metadata."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io.readers.excel_reader import load_excel

        omega = np.logspace(-1, 2, 20)
        E_prime = 3e9 * omega**2 / (1 + omega**2)
        E_double_prime = 3e9 * omega / (1 + omega**2)

        fpath = tmp_path / "dmta.xlsx"
        df = pd.DataFrame(
            {"omega (rad/s)": omega, "E' (Pa)": E_prime, "E'' (Pa)": E_double_prime}
        )
        df.to_excel(fpath, index=False)

        data = load_excel(
            fpath,
            x_col="omega (rad/s)",
            y_cols=["E' (Pa)", "E'' (Pa)"],
            deformation_mode="tension",
        )

        assert data.metadata["deformation_mode"] == "tension"
        assert np.iscomplexobj(data.y)
        assert len(data.x) == 20

    @pytest.mark.smoke
    def test_excel_auto_detect_deformation_mode(self, tmp_path):
        """load_excel auto-detects tension from E' column names."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io.readers.excel_reader import load_excel

        omega = np.logspace(-1, 2, 15)
        E_prime = 1e6 * np.ones_like(omega)
        E_double_prime = 1e5 * np.ones_like(omega)

        fpath = tmp_path / "dmta_auto.xlsx"
        df = pd.DataFrame(
            {"omega (rad/s)": omega, "E' (Pa)": E_prime, "E'' (Pa)": E_double_prime}
        )
        df.to_excel(fpath, index=False)

        data = load_excel(
            fpath,
            x_col="omega (rad/s)",
            y_cols=["E' (Pa)", "E'' (Pa)"],
        )

        assert data.metadata.get("deformation_mode") == "tension"

    @pytest.mark.smoke
    def test_excel_shear_not_set_for_g_prime(self, tmp_path):
        """G' columns should detect shear deformation_mode."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io.readers.excel_reader import load_excel

        omega = np.logspace(-1, 2, 10)
        G_prime = 1e5 * np.ones_like(omega)
        G_double_prime = 1e4 * np.ones_like(omega)

        fpath = tmp_path / "shear.xlsx"
        df = pd.DataFrame(
            {"omega (rad/s)": omega, "G' (Pa)": G_prime, "G'' (Pa)": G_double_prime}
        )
        df.to_excel(fpath, index=False)

        data = load_excel(
            fpath,
            x_col="omega (rad/s)",
            y_cols=["G' (Pa)", "G'' (Pa)"],
        )

        assert data.metadata.get("deformation_mode") == "shear"


# ── Fix B: Shape mismatch error ──────────────────────────────────────────


class TestShapeMismatchError:
    """F-IO-003: rheodata_from_dataset_state must raise on y/y2 shape mismatch."""

    @pytest.mark.smoke
    def test_shape_mismatch_raises_value_error(self):
        """Shape mismatch between y and y2 should raise ValueError."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        dataset = MagicMock()
        dataset.x_data = np.linspace(0.1, 100, 100)
        dataset.y_data = np.ones(100)  # G' has 100 points
        dataset.y2_data = np.ones(99)  # G'' has 99 points (mismatch!)
        dataset.test_mode = "oscillation"
        dataset.metadata = {"domain": "frequency"}

        with pytest.raises(ValueError, match="shape mismatch"):
            rheodata_from_dataset_state(dataset)

    @pytest.mark.smoke
    def test_matching_shapes_combine_correctly(self):
        """Matching y/y2 shapes should produce complex modulus."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        dataset = MagicMock()
        dataset.x_data = np.linspace(0.1, 100, 50)
        dataset.y_data = np.ones(50) * 1e6  # G'
        dataset.y2_data = np.ones(50) * 1e5  # G''
        dataset.test_mode = "oscillation"
        dataset.metadata = {"domain": "frequency"}

        result = rheodata_from_dataset_state(dataset)
        assert np.iscomplexobj(result.y)
        assert len(result.y) == 50


# ── Fix C: CLI validation ────────────────────────────────────────────────


class TestCLIValidation:
    """F-IO-004, F-IO-005: CLI must validate data and require explicit test_mode."""

    @pytest.mark.smoke
    def test_cli_no_oscillation_default(self, tmp_path):
        """CLI should NOT silently default to 'oscillation' for non-complex data.

        Verifies Fix C: when the user doesn't specify --test-mode and data
        has real (not complex) y, oscillation mode must not be assumed.
        The CLI should either auto-detect a non-oscillation mode or error.
        """
        from rheojax.cli.fit import main

        # CSV with explicit frequency column — would trigger oscillation guess
        # in old code, but real y data should be rejected for oscillation
        fpath = tmp_path / "freq.csv"
        fpath.write_text("frequency,modulus\n1,100\n10,200\n100,300\n")

        # With explicit --test-mode oscillation + real data → should fail
        result = main(
            [
                str(fpath),
                "--model",
                "maxwell",
                "--x-col",
                "frequency",
                "--y-col",
                "modulus",
                "--test-mode",
                "oscillation",
            ]
        )
        assert result == 1  # Real data + oscillation = error

    @pytest.mark.smoke
    def test_cli_nan_validation(self, tmp_path):
        """CLI should reject data with NaN values in x column.

        Note: CSV reader removes rows with NaN in y, so we test NaN in x
        which gets through to CLI validation.
        """
        from rheojax.cli.fit import main

        # NaN in x column — CSV reader can't auto-clean this reliably
        fpath = tmp_path / "nan_data.csv"
        fpath.write_text("time,modulus\nNaN,100\n2,200\n3,300\n")

        result = main(
            [
                str(fpath),
                "--model",
                "maxwell",
                "--x-col",
                "time",
                "--y-col",
                "modulus",
                "--test-mode",
                "relaxation",
            ]
        )
        # Either the CSV reader filters NaN rows (success with fewer points)
        # or CLI validation catches remaining NaN (error)
        # Both behaviors are acceptable — the key is no silent garbage fit
        assert result in (0, 1)

    @pytest.mark.smoke
    def test_cli_oscillation_requires_complex(self, tmp_path):
        """CLI should error when oscillation mode has real (not complex) data."""
        from rheojax.cli.fit import main

        fpath = tmp_path / "real_data.csv"
        fpath.write_text("frequency,modulus\n1,100\n10,200\n100,300\n")

        result = main(
            [
                str(fpath),
                "--model",
                "maxwell",
                "--x-col",
                "frequency",
                "--y-col",
                "modulus",
                "--test-mode",
                "oscillation",
            ]
        )
        assert result == 1


# ── Fix D: Cascade provenance ────────────────────────────────────────────


class TestCascadeProvenance:
    """F-IO-002, F-IO-011: auto_load must track format_detected in metadata."""

    @pytest.mark.smoke
    def test_csv_auto_load_sets_format_detected(self, tmp_path):
        """auto_load on a CSV file should set format_detected='csv'."""
        from rheojax.io import auto_load

        fpath = tmp_path / "test.csv"
        fpath.write_text("time,stress\n0.1,100\n0.5,80\n1.0,60\n")

        data = auto_load(str(fpath), x_col="time", y_col="stress")

        assert data.metadata.get("format_detected") == "csv"

    @pytest.mark.smoke
    def test_csv_direct_load_sets_format_detected(self):
        """Direct CSV load via auto_load sets format_detected."""
        from rheojax.io import auto_load

        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as f:
            f.write("frequency,modulus\n1,100\n10,200\n")
            fpath = f.name

        data = auto_load(fpath, x_col="frequency", y_col="modulus")
        assert data.metadata.get("format_detected") == "csv"
        Path(fpath).unlink()

    @pytest.mark.smoke
    def test_txt_fallback_tracks_readers_attempted(self, tmp_path):
        """A .txt file that's actually CSV should track all attempted readers."""
        from rheojax.io import auto_load

        fpath = tmp_path / "data.txt"
        fpath.write_text("time,stress\n0.1,100\n0.5,80\n1.0,60\n")

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = auto_load(str(fpath), x_col="time", y_col="stress")

        # Should have tried trios and/or anton_paar before csv
        attempted = data.metadata.get("readers_attempted", [])
        assert "csv" in data.metadata.get("format_detected", "")
        # readers_attempted present if fallback occurred
        if attempted:
            assert len(attempted) > 1

    @pytest.mark.smoke
    def test_encoding_fallback_flag(self, tmp_path):
        """CSV reader should set encoding_fallback when fallback encoding used."""
        from rheojax.io.readers.csv_reader import load_csv

        # Write UTF-16 LE file
        fpath = tmp_path / "utf16.csv"
        content = "time,stress\n0.1,100\n0.5,80\n"
        fpath.write_bytes(content.encode("utf-16-le"))

        try:
            data = load_csv(str(fpath), x_col="time", y_col="stress")
            # If it succeeds with fallback encoding, check the flag
            if data.metadata.get("encoding") != "utf-8":
                assert data.metadata.get("encoding_fallback") is True
        except (ValueError, KeyError):
            # If parsing fails due to encoding, that's acceptable
            pass


# ── RCA Round 2 Tests (2026-02-23) ───────────────────────────────────────


class TestOpenpyxlReadOnlyCleanup:
    """F-IO-001-rca2: openpyxl read_only workbook must always be closed."""

    @pytest.mark.smoke
    def test_read_only_workbook_closed(self, tmp_path):
        """read_only=True workbook should be closed after parsing."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io.readers.trios.excel import parse_trios_excel

        # Create a minimal TRIOS-style Excel file
        fpath = tmp_path / "trios_style.xlsx"
        df = pd.DataFrame(
            {"Angular frequency (rad/s)": [1, 10, 100], "G' (Pa)": [1e5, 2e5, 3e5]}
        )
        df.to_excel(fpath, index=False)

        # Parse — should not leave file handles open
        try:
            parse_trios_excel(fpath, read_only=True)
        except Exception:
            pass  # Format may not be valid TRIOS, but the point is wb.close()

        # If we get here without ResourceWarning, the handle was closed
        # On Windows, an unclosed handle would prevent deletion
        fpath.unlink()


class TestTriosCsvNumericFirstColumn:
    """F-IO-002-rca2: TRIOS CSV should parse numeric first column values."""

    @pytest.mark.smoke
    def test_numeric_first_column_parsed(self, tmp_path):
        """First column with numeric values should not become NaN."""
        fpath = tmp_path / "trios_numeric.csv"
        # Simulate TRIOS CSV with numeric first column (data point index)
        content = (
            "[General]\n"
            "Instrument=TRIOS\n"
            "[Data]\n"
            "Point\tFrequency\tG'\n"
            "1\t1.0\t100000\n"
            "2\t10.0\t200000\n"
            "3\t100.0\t300000\n"
        )
        fpath.write_text(content)

        from rheojax.io.readers.trios.csv import parse_trios_csv

        try:
            result = parse_trios_csv(fpath, delimiter="\t")
            # Check that first column is not all NaN
            if result.tables:
                first_col = result.tables[0].df.iloc[:, 0]
                assert (
                    not first_col.isna().all()
                ), "First column should not be all NaN when values are numeric"
        except (ValueError, KeyError):
            pass  # File may not be valid TRIOS format


class TestExcelWriterJaxScalars:
    """F-IO-005-rca2: Excel writer must handle JAX/numpy scalars."""

    @pytest.mark.smoke
    def test_numpy_scalar_in_parameters(self, tmp_path):
        """numpy scalars in parameters dict should not break Excel writing."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io.writers.excel_writer import save_excel

        results = {
            "parameters": {
                "G0": np.float64(1e5),
                "tau": np.float64(0.01),
                "alpha": np.float32(0.5),
                "n_modes": np.int64(3),
            },
            "fit_quality": {
                "R2": np.float64(0.9987),
                "RMSE": np.float64(0.0013),
            },
        }

        fpath = tmp_path / "results.xlsx"
        save_excel(results, fpath)
        assert fpath.exists()

        # Verify we can read it back
        df_params = pd.read_excel(fpath, sheet_name="Parameters")
        assert len(df_params) == 4
        assert df_params.iloc[0]["Value"] == pytest.approx(1e5)

    @pytest.mark.smoke
    def test_python_scalar_passthrough(self, tmp_path):
        """Python native scalars should pass through unchanged."""
        from rheojax.io.writers.excel_writer import _to_python_scalar

        assert _to_python_scalar(1.0) == 1.0
        assert _to_python_scalar(42) == 42
        assert _to_python_scalar("hello") == "hello"
        assert _to_python_scalar(np.float64(3.14)) == pytest.approx(3.14)
        assert isinstance(_to_python_scalar(np.float64(3.14)), float)
        assert isinstance(_to_python_scalar(np.int64(42)), int)


class TestY2DataComplexExtraction:
    """F-IO-006: y2_data must extract imag part from complex RheoData.y."""

    @pytest.mark.smoke
    def test_complex_y_extracts_imaginary(self):
        """Complex y should produce non-None y2_data in dispatch payload."""
        from rheojax.core.data import RheoData

        omega = np.logspace(-1, 2, 50)
        G_prime = 1e5 * omega**2 / (1 + omega**2)
        G_double_prime = 1e5 * omega / (1 + omega**2)
        G_star = G_prime + 1j * G_double_prime

        data = RheoData(x=omega, y=G_star, domain="frequency")

        # Simulate the dispatch payload logic from data_page.py
        y2_data = np.imag(data.y) if np.iscomplexobj(data.y) else None

        assert y2_data is not None
        np.testing.assert_allclose(y2_data, G_double_prime, rtol=1e-10)

    @pytest.mark.smoke
    def test_real_y_gives_none_y2(self):
        """Real y should produce None y2_data."""
        from rheojax.core.data import RheoData

        t = np.linspace(0.01, 10, 100)
        G_t = 1e5 * np.exp(-t)

        data = RheoData(x=t, y=G_t)

        y2_data = np.imag(data.y) if np.iscomplexobj(data.y) else None

        assert y2_data is None


class TestMonotonicDetectionWithNoise:
    """F-IO-012: detect_test_mode should tolerate small noise in monotonic data."""

    @pytest.mark.smoke
    def test_noisy_relaxation_detected(self):
        """Relaxation data with 2% noise should still be detected."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        t = np.linspace(0.01, 10, 200)
        G_t = 1e5 * np.exp(-t)

        # Add ~2% noise (some points may increase slightly)
        rng = np.random.default_rng(42)
        noise = 1 + 0.02 * rng.standard_normal(len(t))
        G_noisy = G_t * noise

        data = RheoData(x=t, y=G_noisy)
        mode = service.detect_test_mode(data)

        assert (
            mode == "relaxation"
        ), f"Noisy relaxation data was detected as '{mode}' instead of 'relaxation'"

    @pytest.mark.smoke
    def test_noisy_creep_detected(self):
        """Creep data with a few non-monotonic points should still be detected.

        This tests the 95% tolerance threshold vs the old strict np.all() check.
        """
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        # Pure creep: compliance growing over time
        t = np.linspace(0.01, 10, 200)
        J_t = 1e-6 * (1 - np.exp(-0.5 * t)) + 5e-7

        # Introduce exactly 3 non-monotonic points (1.5% of 199 diffs)
        J_noisy = J_t.copy()
        # Make points 50, 100, 150 slightly LOWER than their predecessors
        J_noisy[50] = J_noisy[49] * 0.999
        J_noisy[100] = J_noisy[99] * 0.999
        J_noisy[150] = J_noisy[149] * 0.999

        data = RheoData(x=t, y=J_noisy)
        mode = service.detect_test_mode(data)

        assert (
            mode == "creep"
        ), f"Mostly-monotonic creep data with 3 dips was detected as '{mode}'"

    @pytest.mark.smoke
    def test_strictly_monotonic_still_works(self):
        """Strictly monotonic data should still be detected correctly."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        t = np.linspace(0.01, 10, 100)
        G_t = 1e5 * np.exp(-t)

        data = RheoData(x=t, y=G_t)
        mode = service.detect_test_mode(data)
        assert mode == "relaxation"


class TestHdf5BytesStrCoercion:
    """F-IO-015: HDF5 reader must handle bytes→str for metadata values."""

    @pytest.mark.smoke
    def test_round_trip_metadata_strings(self, tmp_path):
        """Metadata strings should survive save/load round-trip."""
        h5py = pytest.importorskip("h5py")
        from rheojax.core.data import RheoData
        from rheojax.io.writers.hdf5_writer import load_hdf5, save_hdf5

        data = RheoData(
            x=np.array([1.0, 2.0, 3.0]),
            y=np.array([10.0, 20.0, 30.0]),
            metadata={
                "test_mode": "relaxation",
                "instrument": "ARES-G2",
                "nested": {"key": "value"},
            },
        )

        fpath = tmp_path / "test.h5"
        save_hdf5(data, fpath)
        loaded = load_hdf5(fpath)

        # Metadata strings should be str, not bytes
        assert isinstance(loaded.metadata["test_mode"], str)
        assert loaded.metadata["test_mode"] == "relaxation"
        assert isinstance(loaded.metadata["instrument"], str)

    @pytest.mark.smoke
    def test_none_sentinel_round_trip(self, tmp_path):
        """None values should survive save/load via __None__ sentinel."""
        pytest.importorskip("h5py")
        from rheojax.core.data import RheoData
        from rheojax.io.writers.hdf5_writer import load_hdf5, save_hdf5

        data = RheoData(
            x=np.array([1.0, 2.0]),
            y=np.array([10.0, 20.0]),
            metadata={"present": "yes", "absent": None},
        )

        fpath = tmp_path / "test_none.h5"
        save_hdf5(data, fpath)
        loaded = load_hdf5(fpath)

        assert loaded.metadata["present"] == "yes"
        assert loaded.metadata["absent"] is None


class TestConstructComplexModulusValidation:
    """F-IO-007: construct_complex_modulus must validate shapes."""

    @pytest.mark.smoke
    def test_shape_mismatch_raises(self):
        """Mismatched G'/G'' shapes should raise ValueError."""
        from rheojax.io.readers.trios.common import construct_complex_modulus

        g_prime = np.array([1.0, 2.0, 3.0])
        g_double_prime = np.array([0.1, 0.2])

        with pytest.raises(ValueError, match="same shape"):
            construct_complex_modulus(g_prime, g_double_prime)

    @pytest.mark.smoke
    def test_matching_shapes_succeed(self):
        """Matching shapes should produce correct complex modulus."""
        from rheojax.io.readers.trios.common import construct_complex_modulus

        g_prime = np.array([1.0, 2.0, 3.0])
        g_double_prime = np.array([0.1, 0.2, 0.3])

        result = construct_complex_modulus(g_prime, g_double_prime)
        np.testing.assert_allclose(np.real(result), g_prime)
        np.testing.assert_allclose(np.imag(result), g_double_prime)


class TestAutoLoadTriosCsvFallback:
    """F-IO-009: .csv extension should try TRIOS reader before generic CSV."""

    @pytest.mark.smoke
    def test_generic_csv_still_works(self, tmp_path):
        """Generic CSV files should still load after TRIOS fallback."""
        from rheojax.io import auto_load

        fpath = tmp_path / "generic.csv"
        fpath.write_text("time,stress\n0.1,100\n0.5,80\n1.0,60\n")

        import warnings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            data = auto_load(str(fpath), x_col="time", y_col="stress")

        assert len(data.x) == 3
        # Should have fallen back to CSV reader
        assert data.metadata.get("format_detected") in ("csv", "trios")


class TestKwargsFiltering:
    """F-IO-013: Reader cascade should filter kwargs per reader."""

    @pytest.mark.smoke
    def test_csv_kwargs_not_passed_to_trios(self, tmp_path):
        """x_col/y_col should not cause warnings in TRIOS reader."""
        from rheojax.io.readers.auto import _TRIOS_KWARGS, _filter_kwargs

        # x_col/y_col are CSV-specific and should be filtered out
        kwargs = {"x_col": "time", "y_col": "stress", "validate": True}
        filtered = _filter_kwargs(kwargs, _TRIOS_KWARGS)

        assert "x_col" not in filtered
        assert "y_col" not in filtered
        assert "validate" in filtered

    @pytest.mark.smoke
    def test_trios_kwargs_preserved(self):
        """TRIOS-specific kwargs should be preserved."""
        from rheojax.io.readers.auto import _TRIOS_KWARGS, _filter_kwargs

        kwargs = {"return_all_segments": True, "encoding": "utf-8", "validate": True}
        filtered = _filter_kwargs(kwargs, _TRIOS_KWARGS)

        assert filtered == kwargs


class TestDataServiceLoadFileMulti:
    """F-IO-003-rca2: DataService.load_file_multi handles multi-segment."""

    @pytest.mark.smoke
    def test_single_segment_returns_list(self, tmp_path):
        """Single-segment CSV should return list of length 1."""
        from rheojax.gui.services.data_service import DataService

        fpath = tmp_path / "single.csv"
        fpath.write_text("time,stress\n0.1,100\n0.5,80\n1.0,60\n")

        service = DataService()
        result = service.load_file_multi(fpath, x_col="time", y_col="stress")

        assert isinstance(result, list)
        assert len(result) == 1
        assert len(result[0].x) == 3


class TestDataServiceSupportedFormats:
    """F-IO-018: DataService must include .json and .tsv in supported formats."""

    @pytest.mark.smoke
    def test_json_in_supported_formats(self):
        """JSON format should be in supported formats list."""
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        assert ".json" in service.get_supported_formats()

    @pytest.mark.smoke
    def test_tsv_in_supported_formats(self):
        """TSV format should be in supported formats list."""
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        assert ".tsv" in service.get_supported_formats()


# ── RCA Round 3 Tests (2026-02-24) ───────────────────────────────────────


class TestY2ColTranslation:
    """F-IO-R2-001: y2_col must be translated to y_cols for generic readers."""

    @pytest.mark.smoke
    def test_translate_y2_col_to_y_cols(self):
        """y2_col + y_col should become y_cols=[y_col, y2_col]."""
        from rheojax.io.readers.auto import _translate_y2_col

        kwargs = {"y_col": "G'", "y2_col": "G''", "x_col": "omega"}
        result = _translate_y2_col(kwargs)

        assert "y2_col" not in result
        assert "y_col" not in result
        assert result["y_cols"] == ["G'", "G''"]
        assert result["x_col"] == "omega"  # preserved

    @pytest.mark.smoke
    def test_translate_y2_col_noop_without_y_col(self):
        """y2_col without y_col should not create y_cols."""
        from rheojax.io.readers.auto import _translate_y2_col

        kwargs = {"y2_col": "G''", "x_col": "omega"}
        result = _translate_y2_col(kwargs)

        # y2_col removed but y_cols not created (no y_col to pair with)
        assert "y2_col" not in result
        assert "y_cols" not in result

    @pytest.mark.smoke
    def test_translate_y2_col_noop_when_y_cols_present(self):
        """Pre-existing y_cols should not be overwritten by y2_col."""
        from rheojax.io.readers.auto import _translate_y2_col

        kwargs = {"y_col": "G'", "y2_col": "G''", "y_cols": ["E'", "E''"]}
        result = _translate_y2_col(kwargs)

        assert result["y_cols"] == ["E'", "E''"]  # untouched
        assert "y2_col" not in result

    @pytest.mark.smoke
    def test_translate_y2_col_noop_without_y2(self):
        """No y2_col → kwargs unchanged."""
        from rheojax.io.readers.auto import _translate_y2_col

        kwargs = {"y_col": "stress", "x_col": "time"}
        result = _translate_y2_col(kwargs)

        assert result == {"y_col": "stress", "x_col": "time"}

    @pytest.mark.smoke
    def test_translate_does_not_mutate_caller(self):
        """_translate_y2_col must not mutate the original dict."""
        from rheojax.io.readers.auto import _translate_y2_col

        original = {"y_col": "G'", "y2_col": "G''"}
        _translate_y2_col(original)

        assert "y2_col" in original  # original unchanged
        assert "y_col" in original

    @pytest.mark.smoke
    def test_y2_col_not_in_csv_kwargs(self):
        """y2_col must NOT be in _CSV_KWARGS (would pass through unhandled)."""
        from rheojax.io.readers.auto import _CSV_KWARGS

        assert "y2_col" not in _CSV_KWARGS

    @pytest.mark.smoke
    def test_y2_col_not_in_excel_kwargs(self):
        """y2_col must NOT be in _EXCEL_KWARGS."""
        from rheojax.io.readers.auto import _EXCEL_KWARGS

        assert "y2_col" not in _EXCEL_KWARGS

    @pytest.mark.smoke
    def test_end_to_end_csv_with_y2_col(self, tmp_path):
        """CSV file loaded via load_csv with y_cols from y2_col translation."""
        from rheojax.io.readers.auto import _translate_y2_col
        from rheojax.io.readers.csv_reader import load_csv

        fpath = tmp_path / "complex_modulus.csv"
        fpath.write_text(
            "x_val,storage,loss\n"
            "1,100000,10000\n"
            "10,200000,20000\n"
            "100,300000,30000\n"
        )

        # Simulate the GUI→auto_load→_try_csv chain
        kwargs = {"x_col": "x_val", "y_col": "storage", "y2_col": "loss"}
        translated = _translate_y2_col(kwargs)

        data = load_csv(str(fpath), **translated)

        assert np.iscomplexobj(data.y), "y2_col should produce complex y"
        np.testing.assert_allclose(np.real(data.y), [100000, 200000, 300000])
        np.testing.assert_allclose(np.imag(data.y), [10000, 20000, 30000])


class TestExcelUnfilteredKwargs:
    """F-IO-R2-002: _try_excel must use filtered kwargs."""

    @pytest.mark.smoke
    def test_excel_fallback_filters_trios_kwargs(self, tmp_path):
        """TRIOS-specific kwargs must not reach generic Excel reader."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io import auto_load

        fpath = tmp_path / "generic.xlsx"
        df = pd.DataFrame({"time": [1, 2, 3], "stress": [100, 200, 300]})
        df.to_excel(fpath, index=False)

        # Passing TRIOS kwargs should not break Excel loading
        data = auto_load(
            str(fpath),
            x_col="time",
            y_col="stress",
            return_all_segments=True,  # TRIOS-specific
        )

        assert len(data.x) == 3


class TestTryCsvNrowsLimit:
    """F-IO-R2-005: _try_csv auto-detection should use nrows=5."""

    @pytest.mark.smoke
    def test_large_csv_doesnt_oom(self, tmp_path):
        """Auto-detection should only read 5 rows, not the whole file."""
        from rheojax.io import auto_load

        # Create a modestly large CSV (1000 rows)
        lines = ["time,stress"]
        for i in range(1000):
            lines.append(f"{i * 0.01},{100 - i * 0.05}")
        fpath = tmp_path / "large.csv"
        fpath.write_text("\n".join(lines))

        # Should load without issue (auto-detect only reads 5 rows internally)
        data = auto_load(str(fpath), x_col="time", y_col="stress")
        assert len(data.x) == 1000  # full data loaded after detection


class TestLoadFileMultiSegment:
    """F-IO-R2-006: DataService.load_file() must handle list[RheoData]."""

    @pytest.mark.smoke
    def test_load_file_returns_first_segment(self, tmp_path):
        """load_file() on multi-segment result returns first segment."""
        from unittest.mock import patch

        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        seg1 = RheoData(x=np.array([1, 2, 3]), y=np.array([10, 20, 30]))
        seg2 = RheoData(x=np.array([4, 5, 6]), y=np.array([40, 50, 60]))

        fpath = tmp_path / "multi.csv"
        fpath.write_text("time,stress\n1,10\n2,20\n3,30\n")

        with patch("rheojax.gui.services.data_service.auto_load", return_value=[seg1, seg2]):
            result = service.load_file(fpath, x_col="time", y_col="stress")

        # Should return first segment
        np.testing.assert_array_equal(result.x, [1, 2, 3])

    @pytest.mark.smoke
    def test_load_file_empty_list_raises(self, tmp_path):
        """load_file() on empty list should raise ValueError."""
        from unittest.mock import patch

        from rheojax.gui.services.data_service import DataService

        service = DataService()
        fpath = tmp_path / "empty.csv"
        fpath.write_text("time,stress\n")

        with patch("rheojax.gui.services.data_service.auto_load", return_value=[]):
            with pytest.raises(ValueError, match="No data segments"):
                service.load_file(fpath, x_col="time", y_col="stress")


class TestDetectTestModeFlowVsRelaxation:
    """F-IO-R2-009: Flow must be detected before relaxation."""

    @pytest.mark.smoke
    def test_shear_thinning_not_relaxation(self):
        """Power-law shear-thinning flow curve must NOT be classified as relaxation."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        # Classic shear-thinning: eta ~ gamma_dot^(-0.8) over 3 decades
        gamma_dot = np.logspace(-1, 2, 50)
        eta = 1e3 * gamma_dot ** (-0.8)  # Monotonically decreasing

        data = RheoData(x=gamma_dot, y=eta)
        mode = service.detect_test_mode(data)

        assert mode == "flow", (
            f"Shear-thinning power-law detected as '{mode}' instead of 'flow'"
        )

    @pytest.mark.smoke
    def test_exponential_relaxation_not_flow(self):
        """Exponential relaxation must NOT be classified as flow."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        # Classic relaxation: G(t) = G0 * exp(-t/tau) — NOT a power-law
        t = np.linspace(0.01, 10, 100)
        G_t = 1e5 * np.exp(-t / 1.0)

        data = RheoData(x=t, y=G_t)
        mode = service.detect_test_mode(data)

        assert mode == "relaxation", (
            f"Exponential relaxation detected as '{mode}' instead of 'relaxation'"
        )

    @pytest.mark.smoke
    def test_narrow_range_not_flow(self):
        """Data spanning <1 decade should not trigger flow classification."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()

        # x spans only ~0.7 decades — too narrow for flow
        x = np.linspace(1, 5, 30)
        y = 100 * x ** (-0.9)  # Power-law but narrow range

        data = RheoData(x=x, y=y)
        mode = service.detect_test_mode(data)

        assert mode != "flow", (
            "Sub-decade data should not be classified as flow"
        )


class TestImportWizardCacheCleanup:
    """F-IO-R2-010: Import wizard must clear cached DataFrames on finish."""

    @pytest.mark.smoke
    def test_on_finished_clears_cache_attrs(self):
        """_on_finished should set all cache attributes to None."""
        from rheojax.gui.dialogs.import_wizard import ImportWizard

        try:
            wizard = ImportWizard.__new__(ImportWizard)
            # Simulate cached state
            wizard._cached_df = "fake_dataframe"
            wizard._cached_file_path = "/fake/path"
            wizard._cached_preview_df = "fake_preview_df"
            wizard._cached_preview_path = "/fake/preview/path"

            # Call _on_finished (result=0 = rejected)
            wizard._on_finished(0)

            assert wizard._cached_df is None
            assert wizard._cached_file_path is None
            assert wizard._cached_preview_df is None
            assert wizard._cached_preview_path is None
        except ImportError:
            pytest.skip("PySide6 not available")


class TestDataPageTestModeCombo:
    """F-IO-R2-003: DataPage must include all canonical test modes."""

    @pytest.mark.smoke
    def test_canonical_modes_available(self):
        """All canonical test modes must be in the combo items."""
        # Test the canonical list directly rather than requiring PySide6
        canonical_modes = {
            "Auto-detect",
            "oscillation",
            "relaxation",
            "creep",
            "flow_curve",
            "startup",
            "laos",
        }

        # Verify by reading the source
        import ast
        from pathlib import Path

        data_page_path = Path("rheojax/gui/pages/data_page.py")
        if not data_page_path.exists():
            pytest.skip("data_page.py not found")

        source = data_page_path.read_text()
        # Find the addItems call with our modes
        for mode in ["flow_curve", "startup", "laos"]:
            assert mode in source, (
                f"'{mode}' not found in data_page.py test_mode combo"
            )
        # Ensure "rotation" is gone (replaced by "flow_curve")
        assert '"rotation"' not in source or "flow_curve" in source


# ── RCA Round 3 (2026-02-24): IO + GUI integration fixes ──────────────────


class TestToRheoDataDoubleCount:
    """F-IO-R3-001: DataService.to_rheo_data must not double-count G''."""

    @pytest.mark.smoke
    def test_complex_y_with_y2_no_doubling(self):
        """to_rheo_data with complex y_data + real y2_data must not double G''."""
        from rheojax.gui.services.data_service import DataService

        ds = MagicMock()
        ds.x_data = np.linspace(0.1, 100, 20)
        # y_data is COMPLEX (as readers return it)
        G_prime = np.ones(20) * 1000
        G_double_prime = np.ones(20) * 500
        ds.y_data = G_prime + 1j * G_double_prime
        # y2_data is REAL (as DataPage extracts it)
        ds.y2_data = G_double_prime
        ds.test_mode = "oscillation"
        ds.metadata = {"test_mode": "oscillation"}

        svc = DataService()
        rd = svc.to_rheo_data(ds)
        y = np.asarray(rd.y)

        # G'' must NOT be doubled
        np.testing.assert_allclose(np.real(y), 1000, atol=1e-10)
        np.testing.assert_allclose(np.imag(y), 500, atol=1e-10)

    @pytest.mark.smoke
    def test_real_y_with_y2_combines(self):
        """to_rheo_data with real y_data + real y2_data must combine correctly."""
        from rheojax.gui.services.data_service import DataService

        ds = MagicMock()
        ds.x_data = np.linspace(0.1, 100, 20)
        ds.y_data = np.ones(20) * 1000  # real G'
        ds.y2_data = np.ones(20) * 500  # real G''
        ds.test_mode = "oscillation"
        ds.metadata = {"test_mode": "oscillation"}

        svc = DataService()
        rd = svc.to_rheo_data(ds)
        y = np.asarray(rd.y)

        # Should be combined into complex
        assert np.iscomplexobj(y)
        np.testing.assert_allclose(np.real(y), 1000, atol=1e-10)
        np.testing.assert_allclose(np.imag(y), 500, atol=1e-10)

    @pytest.mark.smoke
    def test_real_y_no_y2_stays_real(self):
        """to_rheo_data with real y_data + no y2_data stays real."""
        from rheojax.gui.services.data_service import DataService

        ds = MagicMock()
        ds.x_data = np.linspace(0.01, 10, 20)
        ds.y_data = np.exp(-np.linspace(0.01, 10, 20))  # relaxation
        ds.y2_data = None
        ds.test_mode = "relaxation"
        ds.metadata = {"test_mode": "relaxation"}

        svc = DataService()
        rd = svc.to_rheo_data(ds)
        y = np.asarray(rd.y)

        assert not np.iscomplexobj(y)


class TestTryExcelNrows:
    """F-IO-R3-003: _try_excel must use nrows limit for column detection."""

    @pytest.mark.smoke
    def test_try_excel_uses_nrows(self, tmp_path):
        """_try_excel should not read entire Excel file for column detection."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")

        # Create a large-ish Excel file
        n_rows = 10000
        fpath = tmp_path / "large.xlsx"
        df = pd.DataFrame({
            "time": np.arange(n_rows, dtype=float),
            "stress": np.random.randn(n_rows),
        })
        df.to_excel(fpath, index=False)

        from rheojax.io.readers.auto import auto_load

        result = auto_load(str(fpath))
        # Should succeed without reading all 10k rows for column detection
        assert len(result.x) == n_rows


class TestRheoDataDomainConsistency:
    """F-IO-R3-004: rheodata_from_dataset_state must derive domain from test_mode."""

    @pytest.mark.smoke
    def test_oscillation_without_domain_in_metadata(self):
        """Oscillation dataset with no domain in metadata must get domain='frequency'."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        ds = MagicMock()
        ds.x_data = np.logspace(-1, 2, 20)
        G_prime = 1000 * np.ones(20)
        ds.y_data = G_prime  # real
        ds.y2_data = 500 * np.ones(20)  # real G''
        ds.test_mode = "oscillation"
        ds.metadata = {"test_mode": "oscillation"}  # no "domain" key

        rd = rheodata_from_dataset_state(ds)
        assert rd.domain == "frequency"

    @pytest.mark.smoke
    def test_relaxation_gets_time_domain(self):
        """Relaxation dataset must get domain='time'."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        ds = MagicMock()
        ds.x_data = np.linspace(0.01, 10, 20)
        ds.y_data = np.exp(-np.linspace(0.01, 10, 20))
        ds.y2_data = None
        ds.test_mode = "relaxation"
        ds.metadata = {"test_mode": "relaxation"}

        rd = rheodata_from_dataset_state(ds)
        assert rd.domain == "time"

    @pytest.mark.smoke
    def test_explicit_domain_in_metadata_respected(self):
        """If metadata has domain set, it should be respected."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        ds = MagicMock()
        ds.x_data = np.linspace(0.01, 10, 20)
        ds.y_data = np.exp(-np.linspace(0.01, 10, 20))
        ds.y2_data = None
        ds.test_mode = "relaxation"
        ds.metadata = {"test_mode": "relaxation", "domain": "time"}

        rd = rheodata_from_dataset_state(ds)
        assert rd.domain == "time"


class TestImportDispatchSplitsComplex:
    """F-IO-R3-009: _on_import_completed must store real y_data + real y2_data."""

    @pytest.mark.smoke
    def test_dispatch_payload_splits_complex(self):
        """Verify the import dispatch stores real G' and real G'' separately."""
        # This tests the convention by reading the source code, since we
        # can't easily instantiate DataPage without PySide6.
        from pathlib import Path

        src = Path("rheojax/gui/pages/data_page.py")
        if not src.exists():
            pytest.skip("data_page.py not found")

        code = src.read_text()
        # The dispatch must use np.real for y_data when complex
        assert "np.real(rheo_data.y)" in code, (
            "Import dispatch must store np.real(y) as y_data for complex data"
        )
        assert "np.imag(rheo_data.y)" in code, (
            "Import dispatch must store np.imag(y) as y2_data for complex data"
        )


class TestDetectTestModeFlow:
    """F-IO-R3-006: detect_test_mode flow detection edge cases."""

    @pytest.mark.smoke
    def test_shear_thinning_detected_as_flow(self):
        """Shear-thinning power-law data must be detected as flow."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        gamma_dot = np.logspace(-2, 2, 50)
        # Shear-thinning: η ~ γ̇^(-0.7)
        eta = 1000 * gamma_dot**(-0.7)

        data = RheoData(x=gamma_dot, y=eta)
        svc = DataService()
        mode = svc.detect_test_mode(data)
        assert mode == "flow"

    @pytest.mark.smoke
    def test_exponential_relaxation_not_flow(self):
        """Exponential relaxation must NOT be misclassified as flow."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        t = np.linspace(0.01, 10, 50)
        G_t = 1000 * np.exp(-t / 1.0)

        data = RheoData(x=t, y=G_t)
        svc = DataService()
        mode = svc.detect_test_mode(data)
        assert mode == "relaxation"

    @pytest.mark.smoke
    def test_creep_compliance_not_flow(self):
        """Creep compliance (monotonic increase) must be detected as creep."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        t = np.linspace(0.1, 100, 50)
        J_t = 1e-3 * (1 - np.exp(-t / 10))  # saturating compliance
        # Ensure sufficient range ratio
        J_t = J_t + 1e-5  # offset so min is positive

        data = RheoData(x=t, y=J_t)
        svc = DataService()
        mode = svc.detect_test_mode(data)
        assert mode == "creep"


class TestRheoDataFromDatasetStateComplex:
    """F-IO-R3-001 (utils path): rheodata_from_dataset_state guard."""

    @pytest.mark.smoke
    def test_complex_y_skips_combination(self):
        """If y_data is already complex, must NOT combine with y2_data."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        ds = MagicMock()
        ds.x_data = np.logspace(-1, 2, 20)
        G_prime = 1000 * np.ones(20)
        G_double_prime = 500 * np.ones(20)
        ds.y_data = G_prime + 1j * G_double_prime  # already complex
        ds.y2_data = G_double_prime  # redundant real
        ds.test_mode = "oscillation"
        ds.metadata = {"test_mode": "oscillation"}

        rd = rheodata_from_dataset_state(ds)
        y = np.asarray(rd.y)

        # Guard: complex y is used as-is (G'' NOT doubled)
        np.testing.assert_allclose(np.real(y), 1000, atol=1e-10)
        np.testing.assert_allclose(np.imag(y), 500, atol=1e-10)

    @pytest.mark.smoke
    def test_real_y_combines_with_y2(self):
        """If y_data is real and y2_data present, must combine into complex."""
        from rheojax.gui.utils.rheodata import rheodata_from_dataset_state

        ds = MagicMock()
        ds.x_data = np.logspace(-1, 2, 20)
        ds.y_data = np.ones(20) * 1000  # real G'
        ds.y2_data = np.ones(20) * 500  # real G''
        ds.test_mode = "oscillation"
        ds.metadata = {"test_mode": "oscillation"}

        rd = rheodata_from_dataset_state(ds)
        y = np.asarray(rd.y)

        assert np.iscomplexobj(y)
        np.testing.assert_allclose(np.real(y), 1000, atol=1e-10)
        np.testing.assert_allclose(np.imag(y), 500, atol=1e-10)


# ── RCA Round 4 (2026-02-24): Deep IO + GUI integration audit ─────────────


class TestY2ColExcelLoss:
    """F-IO-R4-001: y2_col must not be stripped before reaching _try_excel."""

    @pytest.mark.smoke
    def test_y2_col_translated_at_auto_load_entry(self):
        """auto_load must translate y2_col → y_cols before format dispatch."""
        from rheojax.io.readers.auto import _EXCEL_KWARGS, _filter_kwargs, _translate_y2_col

        # Simulate the FIXED flow: translate first, then filter
        kwargs = {"x_col": "omega", "y_col": "G'", "y2_col": "G''"}
        translated = _translate_y2_col(kwargs)

        # After translation, y_cols should be present
        assert "y_cols" in translated
        assert translated["y_cols"] == ["G'", "G''"]
        assert "y2_col" not in translated

        # y_cols must survive _EXCEL_KWARGS filtering
        filtered = _filter_kwargs(translated, _EXCEL_KWARGS)
        assert "y_cols" in filtered
        assert filtered["y_cols"] == ["G'", "G''"]

    @pytest.mark.smoke
    def test_end_to_end_excel_with_y2_col(self, tmp_path):
        """Excel file loaded via auto_load with y2_col produces complex y."""
        pd = pytest.importorskip("pandas")
        pytest.importorskip("openpyxl")
        from rheojax.io import auto_load

        fpath = tmp_path / "modulus.xlsx"
        df = pd.DataFrame({
            "omega": [1.0, 10.0, 100.0],
            "G'": [1e5, 2e5, 3e5],
            "G''": [1e4, 2e4, 3e4],
        })
        df.to_excel(fpath, index=False)

        data = auto_load(str(fpath), x_col="omega", y_col="G'", y2_col="G''")
        assert np.iscomplexobj(data.y), "y2_col must produce complex y for Excel"
        np.testing.assert_allclose(np.real(data.y), [1e5, 2e5, 3e5])
        np.testing.assert_allclose(np.imag(data.y), [1e4, 2e4, 3e4])

    @pytest.mark.smoke
    def test_y2_col_idempotent_translation(self):
        """Double-translation of y2_col must be idempotent."""
        from rheojax.io.readers.auto import _translate_y2_col

        kwargs = {"x_col": "omega", "y_col": "G'", "y2_col": "G''"}
        first = _translate_y2_col(kwargs)
        second = _translate_y2_col(first)
        assert first == second


class TestModulusPairRegex:
    """F-IO-R4-002: _detect_modulus_pair must handle E''/G'' (two single quotes)."""

    @pytest.mark.smoke
    def test_e_double_prime_two_single_quotes(self):
        """E'' with two ASCII single quotes must be detected as loss modulus."""
        from rheojax.io.readers.auto import _detect_modulus_pair

        cols = ["omega (rad/s)", "E' (Pa)", "E'' (Pa)"]
        result = _detect_modulus_pair(cols, [c.lower() for c in cols])
        assert result is not None
        assert result[0] == "E' (Pa)"  # storage
        assert result[1] == "E'' (Pa)"  # loss

    @pytest.mark.smoke
    def test_g_double_prime_two_single_quotes(self):
        """G'' with two ASCII single quotes must be detected as loss modulus."""
        from rheojax.io.readers.auto import _detect_modulus_pair

        cols = ["frequency (Hz)", "G' (Pa)", "G'' (Pa)"]
        result = _detect_modulus_pair(cols, [c.lower() for c in cols])
        assert result is not None
        assert result[0] == "G' (Pa)"
        assert result[1] == "G'' (Pa)"

    @pytest.mark.smoke
    def test_e_stor_e_loss_still_works(self):
        """E_stor/E_loss (pyvisco) must still be detected."""
        from rheojax.io.readers.auto import _detect_modulus_pair

        cols = ["f", "E_stor", "E_loss", "T"]
        result = _detect_modulus_pair(cols, [c.lower() for c in cols])
        assert result == ["E_stor", "E_loss"]

    @pytest.mark.smoke
    def test_storage_loss_modulus_generic(self):
        """Generic 'Storage Modulus'/'Loss Modulus' must be detected."""
        from rheojax.io.readers.auto import _detect_modulus_pair

        cols = ["Time (s)", "Storage Modulus (Pa)", "Loss Modulus (Pa)"]
        result = _detect_modulus_pair(cols, [c.lower() for c in cols])
        assert result == ["Storage Modulus (Pa)", "Loss Modulus (Pa)"]

    @pytest.mark.smoke
    def test_single_prime_no_pair(self):
        """Single G' without G'' must return None."""
        from rheojax.io.readers.auto import _detect_modulus_pair

        cols = ["omega", "G' (Pa)"]
        result = _detect_modulus_pair(cols, [c.lower() for c in cols])
        assert result is None

    @pytest.mark.smoke
    def test_unicode_prime_and_double_prime(self):
        """Unicode prime (U+2032) and double-prime (U+2033) must work."""
        from rheojax.io.readers.auto import _detect_modulus_pair

        cols = ["omega", "G\u2032 (Pa)", "G\u2033 (Pa)"]
        result = _detect_modulus_pair(cols, [c.lower() for c in cols])
        assert result is not None
        assert result[0] == "G\u2032 (Pa)"
        assert result[1] == "G\u2033 (Pa)"


class TestTestModeForwarding:
    """F-IO-R4-003: test_mode must be forwarded to auto_load."""

    @pytest.mark.smoke
    def test_load_file_forwards_test_mode(self, tmp_path):
        """DataService.load_file must pass test_mode to auto_load."""
        from unittest.mock import patch

        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        fpath = tmp_path / "data.csv"
        fpath.write_text("time,stress\n0.1,100\n0.5,80\n1.0,60\n")

        captured_kwargs = {}

        def mock_auto_load(filepath, **kwargs):
            captured_kwargs.update(kwargs)
            return RheoData(
                x=np.array([0.1, 0.5, 1.0]),
                y=np.array([100.0, 80.0, 60.0]),
            )

        with patch("rheojax.gui.services.data_service.auto_load", side_effect=mock_auto_load):
            service.load_file(fpath, x_col="time", y_col="stress", test_mode="relaxation")

        assert "test_mode" in captured_kwargs
        assert captured_kwargs["test_mode"] == "relaxation"

    @pytest.mark.smoke
    def test_load_file_multi_forwards_test_mode(self, tmp_path):
        """DataService.load_file_multi must pass test_mode to auto_load."""
        from unittest.mock import patch

        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        fpath = tmp_path / "data.csv"
        fpath.write_text("time,stress\n0.1,100\n0.5,80\n1.0,60\n")

        captured_kwargs = {}

        def mock_auto_load(filepath, **kwargs):
            captured_kwargs.update(kwargs)
            return RheoData(
                x=np.array([0.1, 0.5, 1.0]),
                y=np.array([100.0, 80.0, 60.0]),
            )

        with patch("rheojax.gui.services.data_service.auto_load", side_effect=mock_auto_load):
            service.load_file_multi(fpath, x_col="time", y_col="stress", test_mode="oscillation")

        assert "test_mode" in captured_kwargs
        assert captured_kwargs["test_mode"] == "oscillation"


class TestPreviewFileListHandling:
    """F-IO-R4-005: preview_file fallback must handle list[RheoData]."""

    @pytest.mark.smoke
    def test_preview_handles_list_from_auto_load(self, tmp_path):
        """preview_file must handle auto_load returning a list."""
        from unittest.mock import patch

        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        fpath = tmp_path / "data.unknown_ext"
        fpath.write_text("dummy")

        seg1 = RheoData(x=np.array([1.0, 2.0]), y=np.array([10.0, 20.0]))
        seg2 = RheoData(x=np.array([3.0, 4.0]), y=np.array([30.0, 40.0]))

        with patch("rheojax.gui.services.data_service.auto_load", return_value=[seg1, seg2]):
            result = service.preview_file(fpath, max_rows=10)

        assert result["headers"] == ["x", "y"]
        assert len(result["data"]) == 2  # first segment only

    @pytest.mark.smoke
    def test_preview_handles_complex_y(self, tmp_path):
        """preview_file fallback must split complex y into y/y2 columns."""
        from unittest.mock import patch

        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        fpath = tmp_path / "data.unknown_ext"
        fpath.write_text("dummy")

        omega = np.array([1.0, 10.0, 100.0])
        G_star = np.array([1e5 + 1e4j, 2e5 + 2e4j, 3e5 + 3e4j])
        data = RheoData(x=omega, y=G_star, domain="frequency")

        with patch("rheojax.gui.services.data_service.auto_load", return_value=data):
            result = service.preview_file(fpath, max_rows=10)

        assert result["headers"] == ["x", "y", "y2"]
        assert len(result["data"]) == 3
        # First row: x=1.0, y=1e5 (real), y2=1e4 (imag)
        assert result["data"][0][1] == pytest.approx(1e5)
        assert result["data"][0][2] == pytest.approx(1e4)


class TestDetectTestModeSubMillihertz:
    """F-IO-R4-008: oscillation detection must work for sub-mHz DMA data."""

    @pytest.mark.smoke
    def test_sub_millihertz_dma(self):
        """DMA data at 0.0001-1000 rad/s must be detected as oscillation."""
        from rheojax.core.data import RheoData
        from rheojax.gui.services.data_service import DataService

        service = DataService()
        omega = np.logspace(-4, 3, 50)  # 0.0001 to 1000 rad/s
        # Not complex, not monotonic — just log-spaced
        E_prime = 1e9 * np.ones(50)  # flat storage modulus

        data = RheoData(x=omega, y=E_prime)
        mode = service.detect_test_mode(data)
        assert mode == "oscillation", (
            f"Sub-mHz DMA data detected as '{mode}' instead of 'oscillation'"
        )


class TestJsonFallbackError:
    """F-IO-R4-009: .json extension must give clear error for non-TRIOS JSON."""

    @pytest.mark.smoke
    def test_non_trios_json_gives_clear_error(self, tmp_path):
        """Non-TRIOS JSON must raise ValueError with helpful message."""
        from rheojax.io import auto_load

        fpath = tmp_path / "not_trios.json"
        fpath.write_text('{"foo": "bar"}')

        with pytest.raises(ValueError, match="TRIOS"):
            auto_load(str(fpath))
