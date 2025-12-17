"""Tests for RheoCompass CSV parser (Anton Paar).

This module tests the full RheoCompass parser implementation including:
- Interval-based parsing
- Test type auto-detection
- Metadata extraction
- Derived quantity computation
- Multi-interval handling
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io.readers import (
    IntervalBlock,
    load_anton_paar,
    parse_rheocompass_intervals,
    save_intervals_to_excel,
)

# Fixture directory
FIXTURES_DIR = Path(__file__).parent.parent / "fixtures" / "rheocompass"


# =============================================================================
# Phase 2: Foundational Tests
# =============================================================================


class TestParseRheocompassIntervals:
    """Tests for parse_rheocompass_intervals function (T017)."""

    def test_parse_creep_fixture(self):
        """Smoke test: parse creep test fixture."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        global_meta, blocks = parse_rheocompass_intervals(filepath)

        assert len(blocks) == 1
        assert blocks[0].interval_index == 1
        assert len(blocks[0].df) == 10
        assert "Time" in blocks[0].df.columns
        assert "Project" in global_meta

    def test_parse_multi_interval(self):
        """Test parsing file with multiple intervals."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        global_meta, blocks = parse_rheocompass_intervals(filepath)

        assert len(blocks) == 3
        assert blocks[0].interval_index == 1
        assert blocks[1].interval_index == 2
        assert blocks[2].interval_index == 3

    def test_missing_file_raises(self):
        """Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            parse_rheocompass_intervals("nonexistent.csv")

    def test_non_rheocompass_file_raises(self, tmp_path):
        """Test ValueError for file without interval markers."""
        bad_file = tmp_path / "bad.csv"
        bad_file.write_text("a,b,c\n1,2,3\n")

        with pytest.raises(ValueError, match="No interval blocks"):
            parse_rheocompass_intervals(bad_file)


class TestEncodingDetection:
    """Tests for encoding detection (T011)."""

    def test_utf8_encoding(self):
        """Test UTF-8 encoded file is parsed correctly."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        _, blocks = parse_rheocompass_intervals(filepath, encoding="utf-8")
        assert len(blocks) == 1


# =============================================================================
# Phase 3: Creep and Relaxation Tests (US1 + US2)
# =============================================================================


class TestCreepLoading:
    """Tests for creep test loading (US1: T018-T019)."""

    def test_creep_loading_basic(self):
        """T018: Test creep loading returns correct structure."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        assert isinstance(data, RheoData)
        assert data.test_mode == "creep"
        assert data.x_units == "s"
        assert data.y_units == "1/Pa"
        assert len(data.x) == 10
        assert data.domain == "time"

    def test_creep_time_values(self):
        """Test time values are correctly extracted."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        # First time point should be 0.1s
        assert np.isclose(data.x[0], 0.1)
        # Last time point should be 100s
        assert np.isclose(data.x[-1], 100.0)

    def test_creep_compliance_values(self):
        """Test compliance values are correctly extracted."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        # First compliance value from fixture is 5.0e-6 1/Pa
        # But strain is in %, so computed J(t) = (0.05/100) / 100 = 5e-6
        assert np.isclose(data.y[0], 5.0e-6, rtol=1e-3)

    def test_creep_compliance_calculation(self, tmp_path):
        """T019: Test compliance calculated when J(t) column missing."""
        # Create fixture without compliance column
        content = """Project:\tTest
Interval and data points:\t1\t3
Interval data:\tTime\tShear Stress\tShear Strain
\t[s]\t[Pa]\t[%]
1.0\t100\t1.0
2.0\t100\t2.0
3.0\t100\t3.0
"""
        filepath = tmp_path / "no_compliance.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        # J(t) = strain / stress
        # Strain is 1.0% = 0.01 (after % conversion)
        # Stress is 100 Pa
        # J(t) = 0.01 / 100 = 1e-4 1/Pa
        assert data.test_mode == "creep"
        # Actually strain % -> dimensionless conversion applies
        # First strain in data = 1.0% -> 0.01 dimensionless
        # J(t) = 0.01 / 100 = 1e-4
        assert np.isclose(data.y[0], 0.01 / 100, rtol=1e-3)


class TestRelaxationLoading:
    """Tests for relaxation test loading (US2: T020-T021)."""

    def test_relaxation_loading_basic(self):
        """T020: Test relaxation loading returns correct structure."""
        filepath = FIXTURES_DIR / "relaxation_test.csv"
        data = load_anton_paar(filepath)

        assert isinstance(data, RheoData)
        assert data.test_mode == "relaxation"
        assert data.x_units == "s"
        assert data.y_units == "Pa"
        assert len(data.x) == 10
        assert data.domain == "time"

    def test_relaxation_modulus_values(self):
        """Test relaxation modulus values are correctly extracted."""
        filepath = FIXTURES_DIR / "relaxation_test.csv"
        data = load_anton_paar(filepath)

        # First G(t) value should be 1.0e6 Pa
        assert np.isclose(data.y[0], 1.0e6)

    def test_relaxation_modulus_calculation(self, tmp_path):
        """T021: Test G(t) calculated when column missing."""
        # Create fixture without relaxation modulus column
        content = """Project:\tTest
Interval and data points:\t1\t3
Interval data:\tTime\tShear Strain\tShear Stress
\t[s]\t[%]\t[Pa]
0.1\t1.0\t10000
0.2\t1.0\t8000
0.3\t1.0\t6000
"""
        filepath = tmp_path / "no_modulus.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        # G(t) = stress / strain = 10000 / 0.01 = 1e6
        # (strain converted from % to dimensionless)
        assert data.test_mode == "relaxation"
        # First stress = 10000 Pa, strain = 1% = 0.01
        # G(t) = 10000 / 0.01 = 1e6 Pa
        assert np.isclose(data.y[0], 10000 / 0.01, rtol=1e-3)


# =============================================================================
# Phase 4: Oscillatory Tests (US3)
# =============================================================================


class TestOscillatoryLoading:
    """Tests for oscillatory test loading (US3: T028-T030)."""

    def test_oscillatory_loading_basic(self):
        """T028: Test oscillatory loading returns complex G*."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        data = load_anton_paar(filepath)

        assert isinstance(data, RheoData)
        assert data.test_mode == "oscillation"
        assert data.x_units == "rad/s"
        assert data.y_units == "Pa"
        assert len(data.x) == 10
        assert data.domain == "frequency"

    def test_oscillatory_complex_modulus(self):
        """Test complex modulus G* = G' + i*G'' is computed."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        data = load_anton_paar(filepath)

        # y should be complex
        assert data.is_complex
        # Real part should be G' (storage modulus)
        assert np.isclose(data.y_real[0], 1000)  # First G' = 1000 Pa
        # Imaginary part should be G'' (loss modulus)
        assert np.isclose(data.y_imag[0], 100)  # First G'' = 100 Pa

    def test_oscillatory_storage_loss_modulus(self):
        """T030: Test G'/G'' accessible via properties."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        data = load_anton_paar(filepath)

        assert data.storage_modulus is not None
        assert data.loss_modulus is not None
        assert np.isclose(data.storage_modulus[0], 1000)
        assert np.isclose(data.loss_modulus[0], 100)

    def test_hz_to_rad_s_conversion(self, tmp_path):
        """T029: Test Hz to rad/s conversion."""
        content = """Project:\tTest
Interval and data points:\t1\t2
Interval data:\tFrequency\tG'\tG''
\t[Hz]\t[Pa]\t[Pa]
1.0\t1000\t100
10.0\t2000\t200
"""
        filepath = tmp_path / "hz_freq.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        # 1 Hz = 2π rad/s
        assert np.isclose(data.x[0], 2 * np.pi)
        assert np.isclose(data.x[1], 20 * np.pi)


# =============================================================================
# Phase 5: Auto-Detection Tests (US5)
# =============================================================================


class TestAutoDetection:
    """Tests for test type auto-detection (US5: T035-T040)."""

    def test_detect_creep(self):
        """T035: Test auto-detection identifies creep."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)
        assert data.test_mode == "creep"

    def test_detect_relaxation(self):
        """T036: Test auto-detection identifies relaxation."""
        filepath = FIXTURES_DIR / "relaxation_test.csv"
        data = load_anton_paar(filepath)
        assert data.test_mode == "relaxation"

    def test_detect_oscillation(self):
        """T037: Test auto-detection identifies oscillation."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        data = load_anton_paar(filepath)
        assert data.test_mode == "oscillation"

    def test_detect_rotation(self):
        """T038: Test auto-detection identifies rotation."""
        filepath = FIXTURES_DIR / "flow_curve.csv"
        data = load_anton_paar(filepath)
        assert data.test_mode == "rotation"

    def test_explicit_mode_overrides(self):
        """T039: Test explicit test_mode overrides auto-detection."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath, test_mode="relaxation")
        assert data.test_mode == "relaxation"

    def test_ambiguous_warning(self, tmp_path):
        """T040: Test warning for ambiguous test type."""
        # Create ambiguous file (no clear test type indicators)
        # Need columns that don't match any test type patterns
        content = """Project:\tTest
Interval and data points:\t1\t2
Interval data:\tParam_A\tParam_B
1.0\t100
2.0\t200
"""
        filepath = tmp_path / "ambiguous.csv"
        filepath.write_text(content)

        with pytest.warns(UserWarning, match="Could not auto-detect"):
            data = load_anton_paar(filepath)
        # Should default to relaxation when ambiguous
        assert data is not None


# =============================================================================
# Phase 6: Metadata Tests (US6)
# =============================================================================


class TestMetadataExtraction:
    """Tests for metadata extraction (US6: T046-T049)."""

    def test_geometry_extraction(self):
        """T046: Test geometry extraction."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        assert "geometry" in data.metadata
        assert data.metadata["geometry"] == "Plate-Plate"

    def test_gap_diameter_extraction(self):
        """T047: Test gap and diameter extraction."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        assert "gap" in data.metadata
        assert "diameter" in data.metadata
        assert data.metadata["gap"] == "1.0 mm"
        assert data.metadata["diameter"] == "25 mm"

    def test_temperature_extraction(self):
        """T048: Test temperature extraction."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        assert "temperature" in data.metadata
        assert "25.0" in data.metadata["temperature"]

    def test_normal_force_preserved(self, tmp_path):
        """T049: Test normal force column preserved in metadata."""
        content = """Project:\tTest
Interval and data points:\t1\t2
Interval data:\tTime\tShear Stress\tShear Strain\tNormal Force
\t[s]\t[Pa]\t[%]\t[N]
1.0\t100\t0.5\t0.1
2.0\t100\t1.0\t0.2
"""
        filepath = tmp_path / "with_normal_force.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        assert "normal_force" in data.metadata
        assert len(data.metadata["normal_force"]) == 2


# =============================================================================
# Phase 7: Rotational Tests (US4)
# =============================================================================


class TestRotationalLoading:
    """Tests for rotational/flow test loading (US4: T055-T057)."""

    def test_rotational_loading_basic(self):
        """T055: Test rotational loading returns correct structure."""
        filepath = FIXTURES_DIR / "flow_curve.csv"
        data = load_anton_paar(filepath)

        assert isinstance(data, RheoData)
        assert data.test_mode == "rotation"
        assert data.x_units == "1/s"
        # Viscosity unit may vary (Pa·s vs Pa.s) - check contains Pa
        assert "Pa" in data.y_units
        assert len(data.x) == 10

    def test_shear_rate_ordering_preserved(self):
        """T056: Test shear rate ordering preserved."""
        filepath = FIXTURES_DIR / "flow_curve.csv"
        data = load_anton_paar(filepath)

        # First shear rate should be 0.01 1/s
        assert np.isclose(data.x[0], 0.01)
        # Data should be in ascending order
        assert np.all(np.diff(data.x) > 0)

    def test_y_col_parameter(self):
        """T057: Test y_col allows selecting alternative column."""
        filepath = FIXTURES_DIR / "flow_curve.csv"
        # Use shear_stress as y instead of viscosity
        data = load_anton_paar(filepath, y_col="shear_stress")

        assert data.y_units == "Pa"
        # First shear stress value should be 0.50 Pa
        assert np.isclose(data.y[0], 0.50)


# =============================================================================
# Phase 8: Multi-Interval Tests
# =============================================================================


class TestMultiInterval:
    """Tests for multi-interval handling (T061-T064)."""

    def test_single_interval_returns_rheodata(self):
        """T061: Single-interval file returns RheoData (not list)."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        assert isinstance(data, RheoData)
        assert not isinstance(data, list)

    def test_multi_interval_returns_list(self):
        """T062: Multi-interval file returns list when return_all=True."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        data_list = load_anton_paar(filepath, return_all=True)

        assert isinstance(data_list, list)
        assert len(data_list) == 3

    def test_interval_selection(self):
        """T063: Test interval parameter selects specific interval."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        data = load_anton_paar(filepath, interval=2)

        assert isinstance(data, RheoData)
        assert data.metadata["interval_index"] == 2

    def test_progress_callback(self):
        """T064: Test progress_callback invoked."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        progress_calls = []

        def callback(current, total):
            progress_calls.append((current, total))

        load_anton_paar(filepath, return_all=True, progress_callback=callback)

        assert len(progress_calls) == 3
        assert progress_calls[-1] == (3, 3)


# =============================================================================
# Phase 9: Edge Cases and Error Handling
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases (T070-T074)."""

    def test_bracket_unit_notation(self):
        """T071: Test bracket [unit] notation."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)
        # If parsing works, units were correctly extracted
        assert data.x_units == "s"

    def test_parentheses_unit_notation(self, tmp_path):
        """T071: Test parentheses (unit) notation."""
        content = """Project:\tTest
Interval and data points:\t1\t2
Interval data:\tTime (s)\tCompliance (1/Pa)
1.0\t1e-5
2.0\t2e-5
"""
        filepath = tmp_path / "paren_units.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)
        assert data.x_units == "s"

    def test_unicode_headers(self):
        """T072: Test Unicode characters in headers."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        # The fixture has η* and δ columns
        _, blocks = parse_rheocompass_intervals(filepath)
        # Should parse without error
        assert len(blocks) == 1


class TestErrorHandling:
    """Tests for error handling (T075-T078)."""

    def test_missing_file_error(self):
        """T075: Test FileNotFoundError for missing file."""
        with pytest.raises(FileNotFoundError):
            load_anton_paar("nonexistent_file.csv")

    def test_no_intervals_error(self, tmp_path):
        """T076: Test ValueError for no interval blocks."""
        bad_file = tmp_path / "not_rheocompass.csv"
        bad_file.write_text("a,b,c\n1,2,3\n")

        with pytest.raises(ValueError, match="No interval blocks"):
            load_anton_paar(bad_file)

    def test_invalid_interval_error(self):
        """T078: Test error for interval index out of range."""
        filepath = FIXTURES_DIR / "creep_test.csv"

        with pytest.raises(ValueError, match="Interval 99 not found"):
            load_anton_paar(filepath, interval=99)


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete workflows."""

    def test_creep_to_model_workflow(self):
        """Test loading creep data for model fitting."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        # Data should be ready for model fitting
        assert len(data.x) > 0
        assert len(data.y) > 0
        assert data.test_mode == "creep"
        assert data.domain == "time"

    def test_frequency_sweep_complex_modulus(self):
        """Test frequency sweep returns complex G* for model fitting."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        data = load_anton_paar(filepath)

        # Complex modulus should be accessible
        assert data.is_complex
        assert data.storage_modulus is not None
        assert data.loss_modulus is not None
        assert data.tan_delta is not None

    def test_multi_temp_mastercurve_data(self):
        """Test multi-interval data for mastercurve analysis."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        data_list = load_anton_paar(filepath, return_all=True)

        # Should have 3 intervals
        assert len(data_list) == 3
        # Each interval should be oscillatory
        for data in data_list:
            assert data.test_mode == "oscillation"
            assert data.is_complex


# =============================================================================
# Excel Export Tests
# =============================================================================


class TestExcelExport:
    """Tests for save_intervals_to_excel function."""

    def test_export_single_interval(self, tmp_path):
        """Test exporting single RheoData to Excel."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        output_path = tmp_path / "output.xlsx"
        save_intervals_to_excel(data, output_path)

        assert output_path.exists()

        # Verify sheet names
        import pandas as pd
        xlsx = pd.ExcelFile(output_path)
        assert "Metadata" in xlsx.sheet_names
        assert "Interval_1" in xlsx.sheet_names

    def test_export_multi_interval(self, tmp_path):
        """Test exporting multi-interval data creates one sheet per interval."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        data_list = load_anton_paar(filepath, return_all=True)

        output_path = tmp_path / "multi_output.xlsx"
        save_intervals_to_excel(data_list, output_path)

        assert output_path.exists()

        # Verify sheet names
        import pandas as pd
        xlsx = pd.ExcelFile(output_path)
        assert "Metadata" in xlsx.sheet_names
        assert "Interval_1" in xlsx.sheet_names
        assert "Interval_2" in xlsx.sheet_names
        assert "Interval_3" in xlsx.sheet_names
        assert len(xlsx.sheet_names) == 4  # Metadata + 3 intervals

    def test_export_metadata_content(self, tmp_path):
        """Test metadata sheet contains expected content."""
        filepath = FIXTURES_DIR / "multi_interval.csv"
        data_list = load_anton_paar(filepath, return_all=True)

        output_path = tmp_path / "meta_test.xlsx"
        save_intervals_to_excel(data_list, output_path)

        import pandas as pd
        meta_df = pd.read_excel(output_path, sheet_name="Metadata")

        # Check global metadata present
        properties = meta_df["Property"].tolist()
        assert any("Test Mode" in p for p in properties)
        assert any("Points" in p for p in properties)

    def test_export_interval_data_columns(self, tmp_path):
        """Test interval sheets have correct columns for oscillatory data."""
        filepath = FIXTURES_DIR / "frequency_sweep.csv"
        data = load_anton_paar(filepath)

        output_path = tmp_path / "osc_output.xlsx"
        save_intervals_to_excel(data, output_path)

        import pandas as pd
        df = pd.read_excel(output_path, sheet_name="Interval_1")

        # Oscillatory data should have G' and G'' columns
        columns = df.columns.tolist()
        assert any("G'" in col or "Storage" in col for col in columns)
        assert any("G''" in col or "Loss" in col for col in columns)
        assert any("Frequency" in col for col in columns)

    def test_export_without_metadata_sheet(self, tmp_path):
        """Test exporting without metadata sheet."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        output_path = tmp_path / "no_meta.xlsx"
        save_intervals_to_excel(data, output_path, include_metadata_sheet=False)

        import pandas as pd
        xlsx = pd.ExcelFile(output_path)
        assert "Metadata" not in xlsx.sheet_names
        assert "Interval_1" in xlsx.sheet_names

    def test_export_custom_sheet_prefix(self, tmp_path):
        """Test custom sheet prefix."""
        filepath = FIXTURES_DIR / "creep_test.csv"
        data = load_anton_paar(filepath)

        output_path = tmp_path / "custom.xlsx"
        save_intervals_to_excel(data, output_path, sheet_prefix="Data")

        import pandas as pd
        xlsx = pd.ExcelFile(output_path)
        assert "Data_1" in xlsx.sheet_names

    def test_export_empty_list_raises(self, tmp_path):
        """Test ValueError for empty list."""
        output_path = tmp_path / "empty.xlsx"

        with pytest.raises(ValueError, match="cannot be empty"):
            save_intervals_to_excel([], output_path)


# =============================================================================
# European Decimal Separator Tests
# =============================================================================


class TestEuropeanDecimalSeparator:
    """Tests for European locale decimal separator handling (T074)."""

    def test_comma_decimal_separator(self, tmp_path):
        """Test parsing with comma as decimal separator (European format)."""
        # Create file with European decimal format (comma as decimal)
        content = """Project:\tEuropean Test
Interval and data points:\t1\t5
Interval data:\tTime\tShear Stress\tShear Strain
\t[s]\t[Pa]\t[%]
0,1\t100,5\t0,05
0,5\t100,8\t0,25
1,0\t101,2\t0,50
2,5\t102,0\t1,25
5,0\t103,5\t2,50
"""
        filepath = tmp_path / "european.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        # Values should be correctly parsed
        assert np.isclose(data.x[0], 0.1, rtol=1e-2)
        assert np.isclose(data.x[2], 1.0, rtol=1e-2)
        assert np.isclose(data.x[4], 5.0, rtol=1e-2)

    def test_european_relaxation_data(self, tmp_path):
        """Test parsing relaxation data with European comma decimal format."""
        content = """Project:\tEuropean Relaxation
Interval and data points:\t1\t5
Interval data:\tTime\tShear Strain\tShear Stress
\t[s]\t[%]\t[Pa]
0,01\t1,0\t50000
0,1\t1,0\t45000
1,0\t1,0\t35000
10,0\t1,0\t20000
100,0\t1,0\t10000
"""
        filepath = tmp_path / "euro_relax.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        # Values should be correctly parsed
        assert data is not None
        assert len(data.x) == 5
        assert np.isclose(data.x[0], 0.01, rtol=1e-2)
        assert np.isclose(data.x[4], 100.0, rtol=1e-2)


# =============================================================================
# Performance Tests (for CI regression testing)
# =============================================================================


class TestPerformance:
    """Performance regression tests for CI.

    Run with: pytest tests/io/test_anton_paar.py -m benchmark
    """

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_parse_10k_points_under_2_seconds(self, tmp_path):
        """SC-004: Load time <2s for 10,000 points."""
        import time

        # Create file with 10,000 data points
        content = """Project:\tPerformance Test
Interval and data points:\t1\t10000
Interval data:\tTime\tShear Stress\tShear Strain
\t[s]\t[Pa]\t[%]
"""
        for i in range(10000):
            t = i * 0.01
            stress = 100.0
            strain = 0.05 * i * 0.001
            content += f"{t:.4f}\t{stress:.1f}\t{strain:.6f}\n"

        filepath = tmp_path / "large_test.csv"
        filepath.write_text(content)

        start = time.perf_counter()
        data = load_anton_paar(filepath)
        elapsed = time.perf_counter() - start

        assert len(data.x) == 10000
        assert elapsed < 2.0, f"Loading took {elapsed:.2f}s, exceeds 2s limit"

    @pytest.mark.slow
    @pytest.mark.benchmark
    def test_export_10k_points_under_5_seconds(self, tmp_path):
        """Export 10k points should complete in <5s."""
        import time

        # Create file with 10,000 data points
        content = """Project:\tExport Performance Test
Interval and data points:\t1\t10000
Interval data:\tTime\tShear Stress\tShear Strain
\t[s]\t[Pa]\t[%]
"""
        for i in range(10000):
            t = i * 0.01
            stress = 100.0
            strain = 0.05 * i * 0.001
            content += f"{t:.4f}\t{stress:.1f}\t{strain:.6f}\n"

        filepath = tmp_path / "large_export_test.csv"
        filepath.write_text(content)

        data = load_anton_paar(filepath)

        output_path = tmp_path / "large_output.xlsx"
        start = time.perf_counter()
        save_intervals_to_excel(data, output_path)
        elapsed = time.perf_counter() - start

        assert output_path.exists()
        assert elapsed < 5.0, f"Export took {elapsed:.2f}s, exceeds 5s limit"
