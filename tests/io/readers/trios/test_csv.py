"""Unit and integration tests for rheojax.io.readers.trios.csv.

Targets the TRIOS CSV parser: encoding/delimiter detection, metadata and
header discovery, unit-row handling, EU decimal parsing, multi-table
(multi-step) exports, complex-modulus construction, NaN filtering, and the
data-loss guards required by the "no silent data loss" project rule.

All fixtures are small synthetic TRIOS-format files written via ``tmp_path`` —
no dependence on proprietary instrument data.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io.readers.trios.csv import (
    _default_x_units,
    _default_y_units,
    _is_numeric,
    detect_delimiter,
    detect_encoding,
    detect_header_row,
    detect_repeated_headers,
    extract_units_from_header,
    load_trios_csv,
    parse_metadata_header,
    parse_trios_csv,
)

# ---------------------------------------------------------------------------
# Pure helper functions
# ---------------------------------------------------------------------------


class TestDefaultUnits:
    def test_default_x_units_oscillation(self):
        assert _default_x_units("oscillation") == "rad/s"

    def test_default_x_units_rotation(self):
        assert _default_x_units("rotation") == "1/s"

    def test_default_x_units_time_fallback(self):
        assert _default_x_units("relaxation") == "s"
        assert _default_x_units("creep") == "s"

    def test_default_y_units_creep(self):
        assert _default_y_units("creep") == "1/Pa"

    def test_default_y_units_rotation(self):
        assert _default_y_units("rotation") == "Pa*s"

    def test_default_y_units_pa_fallback(self):
        assert _default_y_units("oscillation") == "Pa"
        assert _default_y_units("relaxation") == "Pa"


class TestIsNumeric:
    def test_plain_number(self):
        assert _is_numeric("1.5")

    def test_eu_decimal_comma(self):
        assert _is_numeric("1,5")

    def test_scientific(self):
        assert _is_numeric("1.23E+04")

    def test_non_numeric(self):
        assert not _is_numeric("abc")
        assert not _is_numeric("")


class TestDetectDelimiter:
    def test_tab_preferred_for_trios(self):
        content = "Variables\tTime\tStress\n1\t2\t3\n4\t5\t6\n"
        assert detect_delimiter(content) == "\t"

    def test_comma_delimited(self):
        content = "a,b,c\n1,2,3\n4,5,6\n7,8,9\n"
        assert detect_delimiter(content) == ","

    def test_eu_decimal_prefers_tab(self):
        # decimal comma inflates comma count; any tab is a stronger signal.
        content = "x\ty\n1,5\t2,7\n3,1\t4,9\n"
        assert detect_delimiter(content, decimal_separator=",") == "\t"

    def test_all_metadata_then_extended_data(self):
        # First 20 lines are metadata-prefixed; data only appears later, forcing
        # the extended [20:60] search branch.
        lines = [f"step {i}" for i in range(22)]
        lines.append("10\t20\t30")
        content = "\n".join(lines) + "\n"
        assert detect_delimiter(content) == "\t"

    def test_no_data_defaults_to_tab(self):
        content = "step 1\nprocedure foo\n"
        assert detect_delimiter(content) == "\t"


class TestParseMetadataHeader:
    def test_number_of_points_sets_header_next_line(self):
        lines = [
            "Filename\ttest.csv",
            "Sample name\tGel A",
            "Number of points\t5",
            "Variables\tTime\tStress",
        ]
        metadata, header_row = parse_metadata_header(lines, "\t")
        assert metadata["filename"] == "test.csv"
        assert metadata["sample_name"] == "Gel A"
        assert metadata["number_of_points"] == 5
        # Header is the line after "Number of points".
        assert header_row == 3

    def test_variables_marker_is_header(self):
        lines = ["Filename\tx.csv", "Variables\tTime\tStress\tStrain"]
        _, header_row = parse_metadata_header(lines, "\t")
        assert header_row == 1

    def test_non_integer_number_of_points_ignored(self):
        lines = ["Number of points\tN/A", "Variables\tTime\tStress"]
        metadata, header_row = parse_metadata_header(lines, "\t")
        assert "number_of_points" not in metadata
        assert header_row == 1


class TestDetectHeaderRow:
    def test_variables_marker(self):
        lines = ["junk", "Variables\tTime\tStress"]
        assert detect_header_row(lines, "\t") == 1

    def test_number_of_points_marker(self):
        lines = ["Number of points\t7", "Time\tStress"]
        assert detect_header_row(lines, "\t") == 1

    def test_multiple_non_numeric_columns(self):
        lines = ["label\tTime\tStress\tStrain"]
        assert detect_header_row(lines, "\t") == 0

    def test_fallback_to_start_index(self):
        lines = ["1\t2\t3", "4\t5\t6"]
        # No header-like row; returns the start index.
        assert detect_header_row(lines, "\t", start_index=0) == 0


class TestExtractUnits:
    def test_units_in_parentheses(self):
        header = ["Angular Frequency (rad/s)", "Storage Modulus (Pa)"]
        units = extract_units_from_header(header)
        assert units["Angular Frequency (rad/s)"] == "rad/s"
        # Also stored under the name without the unit suffix.
        assert units["Angular Frequency"] == "rad/s"
        assert units["Storage Modulus"] == "Pa"

    def test_separate_unit_row(self):
        header = ["Time", "Stress"]
        unit_row = ["s", "Pa"]
        units = extract_units_from_header(header, unit_row)
        assert units["Time"] == "s"
        assert units["Stress"] == "Pa"


class TestDetectRepeatedHeaders:
    def test_finds_repeated_header_lines(self):
        first_header = ["Variables", "Time", "Stress"]
        lines = [
            "Variables\tTime\tStress",
            "1\t0.1\t100",
            "Variables\tTime\tStress",
            "2\t0.2\t200",
        ]
        starts = detect_repeated_headers(lines, "\t", first_header, start_index=1)
        assert starts == [2]

    def test_no_repeats(self):
        first_header = ["Variables", "Time", "Stress"]
        lines = ["1\t0.1\t100", "2\t0.2\t200"]
        assert detect_repeated_headers(lines, "\t", first_header, 0) == []


class TestDetectEncoding:
    def test_utf8(self, tmp_path):
        f = tmp_path / "u.csv"
        f.write_text("Time\tStress\n1\t2\n", encoding="utf-8")
        assert detect_encoding(f) == "utf-8"

    def test_latin1_fallback(self, tmp_path):
        f = tmp_path / "l.csv"
        # 0xE9 ("é") is invalid as standalone UTF-8 but valid Latin-1.
        f.write_bytes(b"Temperature\t25 \xb0C\n")
        assert detect_encoding(f) in ("latin-1", "cp1252")


# ---------------------------------------------------------------------------
# parse_trios_csv — structure-level parsing
# ---------------------------------------------------------------------------


def _write(tmp_path, name, content):
    p = tmp_path / name
    p.write_text(content)
    return str(p)


OSC_CSV = """Filename\tfreq.csv
Sample name\tPolymer Melt

[step]
Number of points\t3
Variables\tAngular frequency\tStorage modulus\tLoss modulus
\trad/s\tPa\tPa
data point 1\t1.0\t1000\t500
data point 2\t10.0\t1200\t600
data point 3\t100.0\t1400\t700
"""


class TestParseTriosCsvStructure:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_trios_csv(str(tmp_path / "missing.csv"))

    def test_no_data_tables(self, tmp_path):
        # Header markers push header_row past the end of the file.
        path = _write(tmp_path, "empty.csv", "Number of points\t5\n")
        with pytest.raises(ValueError, match="No data"):
            parse_trios_csv(path)

    def test_basic_oscillation_table(self, tmp_path):
        path = _write(tmp_path, "osc.csv", OSC_CSV)
        trios = parse_trios_csv(path)
        assert trios.format == "csv"
        assert len(trios.tables) == 1
        table = trios.primary_table
        assert list(table.df.columns) == [
            "Angular frequency",
            "Storage modulus",
            "Loss modulus",
        ]
        assert table.df.shape == (3, 3)
        # Unit row consumed, not treated as data (no silent first-row loss).
        np.testing.assert_allclose(
            table.df["Angular frequency"].values, [1.0, 10.0, 100.0]
        )
        assert table.units["Storage modulus"] == "Pa"

    def test_metadata_extracted(self, tmp_path):
        path = _write(tmp_path, "osc.csv", OSC_CSV)
        trios = parse_trios_csv(path)
        assert trios.metadata["filename"] == "freq.csv"
        assert trios.metadata["sample_name"] == "Polymer Melt"


class TestEuDecimal:
    def test_eu_decimal_values_parsed(self, tmp_path):
        content = (
            "Number of points\t3\n"
            "Variables\tTime\tStress\n"
            "\ts\tPa\n"
            "data point 1\t0,1\t1000,5\n"
            "data point 2\t0,2\t2000,5\n"
            "data point 3\t0,3\t3000,5\n"
        )
        path = _write(tmp_path, "eu.csv", content)
        trios = parse_trios_csv(path, decimal_separator=",", delimiter="\t")
        np.testing.assert_allclose(trios.primary_table.df["Time"].values, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(
            trios.primary_table.df["Stress"].values, [1000.5, 2000.5, 3000.5]
        )


class TestMalformedRows:
    def test_malformed_rows_warn_not_silently_dropped(self, tmp_path):
        content = (
            "Number of points\t3\n"
            "Variables\tTime\tStress\n"
            "\ts\tPa\n"
            "data point 1\t0.1\t100\n"
            "data point 2\t0.2\n"  # too few columns
            "data point 3\t0.3\t300\n"
        )
        path = _write(tmp_path, "malformed.csv", content)
        with pytest.warns(UserWarning, match="Skipped .* malformed rows"):
            trios = parse_trios_csv(path)
        # The two well-formed rows survive.
        assert trios.primary_table.df.shape[0] == 2

    def test_blank_separator_lines_not_truncating(self, tmp_path):
        content = (
            "Number of points\t4\n"
            "Variables\tTime\tStress\n"
            "\ts\tPa\n"
            "data point 1\t0.1\t100\n"
            "\n"
            "data point 2\t0.2\t200\n"
            "\n"
            "data point 3\t0.3\t300\n"
        )
        path = _write(tmp_path, "blanks.csv", content)
        trios = parse_trios_csv(path)
        # Blank lines between data rows must not truncate the table.
        assert trios.primary_table.df.shape[0] == 3


class TestNanCells:
    def test_empty_cell_becomes_nan(self, tmp_path):
        # Empty cell must be interior (a trailing empty cell is stripped off the
        # line and would make the row malformed rather than NaN-bearing).
        content = (
            "Number of points\t3\n"
            "Variables\tTime\tStress\tTemperature\n"
            "\ts\tPa\t°C\n"
            "data point 1\t0.1\t100\t25\n"
            "data point 2\t0.2\t\t25\n"
            "data point 3\t0.3\t300\t25\n"
        )
        path = _write(tmp_path, "nan.csv", content)
        trios = parse_trios_csv(path)
        stress = trios.primary_table.df["Stress"].values
        assert np.isnan(stress[1])
        # Neighbouring rows are intact — no silent shift/loss.
        np.testing.assert_allclose(stress[[0, 2]], [100.0, 300.0])


class TestMultiTable:
    MULTI = (
        "Filename\tmulti.csv\n"
        "\n"
        "[step]\n"
        "Number of points\t3\n"
        "Variables\tTime\tStress\n"
        "\ts\tPa\n"
        "data point 1\t0.1\t100\n"
        "data point 2\t0.2\t200\n"
        "data point 3\t0.3\t300\n"
        "\n"
        "[step]\n"
        "Number of points\t3\n"
        "Variables\tTime\tStress\n"
        "\ts\tPa\n"
        "data point 1\t0.4\t400\n"
        "data point 2\t0.5\t500\n"
        "data point 3\t0.6\t600\n"
    )

    def test_second_table_not_discarded(self, tmp_path):
        path = _write(tmp_path, "multi.csv", self.MULTI)
        trios = parse_trios_csv(path)
        # Both step sections parsed — no silent loss after the first [step].
        assert len(trios.tables) == 2
        np.testing.assert_allclose(
            trios.tables[1].df["Time"].values, [0.4, 0.5, 0.6]
        )

    def test_load_returns_all_segments(self, tmp_path):
        path = _write(tmp_path, "multi.csv", self.MULTI)
        result = load_trios_csv(path)
        assert isinstance(result, list)
        assert len(result) == 2
        for d in result:
            assert isinstance(d, RheoData)
            assert len(d.x) == 3


# ---------------------------------------------------------------------------
# load_trios_csv — RheoData conversion
# ---------------------------------------------------------------------------


class TestLoadComplexModulus:
    def test_complex_modulus_constructed(self, tmp_path):
        path = _write(tmp_path, "osc.csv", OSC_CSV)
        data = load_trios_csv(path)
        assert isinstance(data, RheoData)
        assert np.iscomplexobj(data.y)
        np.testing.assert_allclose(
            data.y, [1000 + 500j, 1200 + 600j, 1400 + 700j]
        )
        assert data.y_units == "Pa"

    def test_hz_converted_to_rad_per_s(self, tmp_path):
        content = (
            "Number of points\t2\n"
            "Variables\tFrequency\tStorage modulus\tLoss modulus\n"
            "\tHz\tPa\tPa\n"
            "data point 1\t1.0\t100\t50\n"
            "data point 2\t2.0\t200\t100\n"
        )
        path = _write(tmp_path, "hz.csv", content)
        data = load_trios_csv(path)
        # Hz → rad/s multiplies by 2*pi.
        np.testing.assert_allclose(data.x, np.array([1.0, 2.0]) * 2 * np.pi)
        assert data.x_units == "rad/s"


class TestNanFilteringGuards:
    """Segment-level NaN filtering and the partial-data-loss guard."""

    def _stepped(self, empty_steps):
        # Stress sits in an interior column (Temperature trails) so an empty
        # Stress cell yields NaN instead of a stripped/malformed row.
        rows = []
        n = 1
        for step in (1, 2, 3):
            for _ in range(2):
                stress = "" if step in empty_steps else str(step * 100)
                rows.append(
                    f"data point {n}\t{step}\t{n * 0.1:.1f}\t{stress}\t25"
                )
                n += 1
        return (
            "Number of points\t6\n"
            "Variables\tStep\tTime\tStress\tTemperature\n"
            "\t \ts\tPa\t°C\n" + "\n".join(rows) + "\n"
        )

    def test_one_empty_segment_warns_and_survives(self, tmp_path):
        # Step 3 all-NaN → 1 of 3 dropped (< 50%): warn, return the rest.
        path = _write(tmp_path, "one_empty.csv", self._stepped(empty_steps={3}))
        with pytest.warns(Warning, match="no valid data after NaN filtering"):
            result = load_trios_csv(path, return_all_segments=True)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_excessive_segment_loss_raises(self, tmp_path):
        # Steps 2 and 3 all-NaN → 2 of 3 dropped (> 50%): refuse partial dataset.
        path = _write(tmp_path, "most_empty.csv", self._stepped(empty_steps={2, 3}))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="refusing to return a partially"):
                load_trios_csv(path, return_all_segments=True)


class TestNoValidSegments:
    def test_all_nan_raises(self, tmp_path):
        content = (
            "Number of points\t2\n"
            "Variables\tTime\tStress\tTemperature\n"
            "\ts\tPa\t°C\n"
            "data point 1\t0.1\t\t25\n"
            "data point 2\t0.2\t\t25\n"
        )
        path = _write(tmp_path, "allnan.csv", content)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(ValueError, match="No valid data segments"):
                load_trios_csv(path)


class TestTestModeOverride:
    def test_explicit_test_mode_is_honored(self, tmp_path):
        # A non-None test_mode overrides auto-detection (IO-FIX-002 else-branch).
        path = _write(tmp_path, "osc.csv", OSC_CSV)
        data = load_trios_csv(path, test_mode="oscillation")
        assert data.metadata["test_mode"] == "oscillation"
