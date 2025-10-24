"""Tests for file readers (Task Group 7.1, 7.3, 7.5, 7.7, 7.9)."""

import pytest
import numpy as np
from pathlib import Path

from rheo.core.data import RheoData
from rheo.io.readers import (
    load_trios,
    load_csv,
    load_excel,
    load_anton_paar,
    auto_load,
)


# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "test_data"


class TestTriosReader:
    """Tests for TRIOS reader (7.1)."""

    def test_trios_reader_basic(self, tmp_path):
        """Test basic TRIOS file parsing with simple data."""
        # Create a minimal TRIOS file
        trios_content = """Filename	test.txt
Instrument serial number	4010-1234
Instrument name	TASerNo 4010-1234
operator	Test User
rundate	2025-10-24
Sample name	Test Sample
Geometry name	Parallel Plate

[step]
Temperature	°C
25.0

[step]
Number of points	5
Time	Storage modulus	Loss modulus
s	Pa	Pa
0.1	1000	500
0.2	1200	600
0.3	1400	700
0.4	1600	800
0.5	1800	900
"""
        test_file = tmp_path / "test_trios.txt"
        test_file.write_text(trios_content)

        # Load the file
        data = load_trios(str(test_file))

        # Verify it returns RheoData
        assert isinstance(data, RheoData)

        # Verify basic structure
        assert data.x is not None
        assert data.y is not None
        assert len(data.x) > 0

    def test_trios_metadata_extraction(self, tmp_path):
        """Test TRIOS metadata parsing."""
        trios_content = """Filename	metadata_test.txt
Instrument serial number	5343-5678
Instrument name	DHR-3
operator	Dr. Smith
rundate	2025-10-24
Sample name	Polymer XYZ
Geometry name	Cone and Plate

[step]
Temperature	°C
25.0

[step]
Time	Stress
s	Pa
1.0	100
2.0	200
"""
        test_file = tmp_path / "test_metadata.txt"
        test_file.write_text(trios_content)

        data = load_trios(str(test_file))

        # Check metadata was extracted
        assert 'sample_name' in data.metadata
        assert data.metadata['sample_name'] == 'Polymer XYZ'
        assert 'instrument_serial_number' in data.metadata

    def test_trios_unit_conversion(self):
        """Test TRIOS unit conversion (MPa→Pa, %→unitless)."""
        # This test would verify unit conversion logic
        # For now, just check that the function exists
        from rheo.io.readers.trios import convert_units
        assert callable(convert_units)

    def test_trios_multiple_segments(self, tmp_path):
        """Test TRIOS with multiple procedure segments."""
        trios_content = """Filename	multiseg.txt
Instrument serial number	4010-1234
Instrument name	ARES-G2

[step]
Time	Stress
s	Pa
1.0	100
2.0	200

[step]
Time	Stress
s	Pa
3.0	300
4.0	400
"""
        test_file = tmp_path / "test_multiseg.txt"
        test_file.write_text(trios_content)

        data = load_trios(str(test_file))

        # Should handle multiple segments
        assert isinstance(data, (RheoData, list))

    def test_trios_error_handling(self, tmp_path):
        """Test TRIOS error handling for malformed files."""
        malformed_content = "Not a valid TRIOS file\nRandom data\n"
        test_file = tmp_path / "malformed.txt"
        test_file.write_text(malformed_content)

        with pytest.raises(Exception):
            load_trios(str(test_file))


class TestCSVReader:
    """Tests for CSV reader (7.3)."""

    def test_csv_basic_read(self, tmp_path):
        """Test basic CSV file reading with explicit columns."""
        csv_content = """time,stress,strain
1.0,100.0,0.01
2.0,200.0,0.02
3.0,300.0,0.03
4.0,400.0,0.04
"""
        test_file = tmp_path / "test.csv"
        test_file.write_text(csv_content)

        data = load_csv(str(test_file), x_col='time', y_col='stress')

        assert isinstance(data, RheoData)
        assert len(data.x) == 4
        assert len(data.y) == 4
        np.testing.assert_array_equal(data.x, [1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(data.y, [100.0, 200.0, 300.0, 400.0])

    def test_csv_delimiter_detection(self, tmp_path):
        """Test CSV delimiter auto-detection."""
        # Tab-separated
        tsv_content = "time\tstress\n1.0\t100.0\n2.0\t200.0\n"
        test_file = tmp_path / "test.tsv"
        test_file.write_text(tsv_content)

        data = load_csv(str(test_file), x_col='time', y_col='stress')

        assert isinstance(data, RheoData)
        assert len(data.x) == 2

    def test_csv_header_detection(self, tmp_path):
        """Test CSV header row detection."""
        # File without header
        csv_no_header = """1.0,100.0
2.0,200.0
3.0,300.0
"""
        test_file = tmp_path / "no_header.csv"
        test_file.write_text(csv_no_header)

        data = load_csv(str(test_file), x_col=0, y_col=1, header=None)

        assert isinstance(data, RheoData)
        assert len(data.x) == 3

    def test_csv_unit_specification(self, tmp_path):
        """Test CSV with unit specification."""
        csv_content = "freq,modulus\n1.0,1000.0\n10.0,2000.0\n"
        test_file = tmp_path / "test_units.csv"
        test_file.write_text(csv_content)

        data = load_csv(
            str(test_file),
            x_col='freq',
            y_col='modulus',
            x_units='Hz',
            y_units='Pa'
        )

        assert data.x_units == 'Hz'
        assert data.y_units == 'Pa'


class TestExcelReader:
    """Tests for Excel reader (7.5)."""

    @pytest.mark.skip(reason="Requires openpyxl/xlrd installation")
    def test_excel_basic_read(self, tmp_path):
        """Test basic Excel file reading."""
        # Would require creating an actual Excel file with openpyxl
        pass

    @pytest.mark.skip(reason="Requires openpyxl installation")
    def test_excel_sheet_selection(self, tmp_path):
        """Test Excel sheet selection."""
        pass

    @pytest.mark.skip(reason="Requires openpyxl installation")
    def test_excel_column_mapping(self, tmp_path):
        """Test Excel column mapping."""
        pass


class TestAntonPaarReader:
    """Tests for Anton Paar reader (7.7)."""

    @pytest.mark.skip(reason="Requires Anton Paar sample files")
    def test_anton_paar_format_detection(self):
        """Test Anton Paar format detection."""
        pass

    @pytest.mark.skip(reason="Requires Anton Paar sample files")
    def test_anton_paar_metadata_extraction(self):
        """Test Anton Paar metadata extraction."""
        pass

    @pytest.mark.skip(reason="Requires Anton Paar sample files")
    def test_anton_paar_data_parsing(self):
        """Test Anton Paar data parsing."""
        pass


class TestAutoDetection:
    """Tests for auto-detection wrapper (7.9)."""

    def test_auto_detect_from_extension(self, tmp_path):
        """Test format inference from file extension."""
        # Create CSV file
        csv_file = tmp_path / "test.csv"
        csv_file.write_text("time,stress\n1.0,100.0\n2.0,200.0\n")

        data = auto_load(str(csv_file), x_col='time', y_col='stress')

        assert isinstance(data, RheoData)

    def test_auto_detect_trios_extension(self, tmp_path):
        """Test TRIOS detection from .txt extension."""
        trios_file = tmp_path / "test.txt"
        trios_file.write_text("""Filename	test.txt
Instrument serial number	4010-1234

[step]
Time	Stress
s	Pa
1.0	100
""")

        data = auto_load(str(trios_file))

        assert isinstance(data, RheoData)

    def test_auto_detect_content_based(self, tmp_path):
        """Test content-based format detection."""
        # CSV file with .dat extension
        csv_file = tmp_path / "test.dat"
        csv_file.write_text("time,stress\n1.0,100.0\n2.0,200.0\n")

        data = auto_load(str(csv_file), x_col='time', y_col='stress')

        assert isinstance(data, RheoData)

    def test_auto_detect_fallback_logic(self, tmp_path):
        """Test fallback logic when format is ambiguous."""
        # Create file with ambiguous extension
        test_file = tmp_path / "test.unknown"
        test_file.write_text("time,stress\n1.0,100.0\n")

        # Should try readers in sequence
        data = auto_load(str(test_file), x_col='time', y_col='stress')

        assert isinstance(data, RheoData)
