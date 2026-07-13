"""Unit and integration tests for rheojax.io.readers.trios.json.

Targets the TRIOS JSON parser: schema loading/validation, encoding cascade,
JSON-decode error surfacing, structural error handling, result selection
(single / all / out-of-range), multi-step splitting, complex-modulus
construction, unit conversion, NaN filtering, and CamelCase→snake metadata.

Fixtures are small synthetic TRIOS-JSON documents written via ``tmp_path``.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.io.readers.trios.json import (
    _load_schema,
    _snake_case,
    load_trios_json,
    parse_trios_json,
    validate_schema,
)
from rheojax.io.readers.trios.schema.dataset import TRIOSDataSet


def _dataset(columns, values):
    return {"Properties": {}, "columns": columns, "values": values}


def _experiment(results, *, properties=None, sample=None, procedure=None):
    exp = {
        "Properties": properties or {"Name": "Exp1", "Operator": "Alice"},
        "Sample": sample or {"Name": "Gel"},
        "Procedure": procedure or {"Name": "Relaxation Test"},
        "Results": results,
    }
    return {"Experiment": exp}


def _result(columns, values, *, props=None):
    return {"Properties": props or {}, "DataSet": [_dataset(columns, values)]}


RELAX_COLUMNS = [{"name": "Time", "unit": "s"}, {"name": "Relaxation modulus", "unit": "Pa"}]
RELAX_VALUES = [[0.1, 1000.0], [0.2, 800.0], [0.3, 600.0]]

OSC_COLUMNS = [
    {"name": "Angular frequency", "unit": "rad/s"},
    {"name": "Storage modulus", "unit": "Pa"},
    {"name": "Loss modulus", "unit": "Pa"},
]
OSC_VALUES = [[1.0, 1000.0, 500.0], [10.0, 1200.0, 600.0], [100.0, 1400.0, 700.0]]


def _write_json(tmp_path, name, obj):
    p = tmp_path / name
    p.write_text(json.dumps(obj), encoding="utf-8")
    return str(p)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestSnakeCase:
    def test_camel_to_snake(self):
        assert _snake_case("InstrumentName") == "instrument_name"
        assert _snake_case("Name") == "name"
        assert _snake_case("XValue") == "x_value"


class TestSchemaLoading:
    def test_load_schema_returns_dict_or_none(self):
        schema = _load_schema()
        # The bundled schema either loads as a dict or is absent (None).
        assert schema is None or isinstance(schema, dict)

    def test_load_schema_missing_returns_none(self, monkeypatch, tmp_path):
        import rheojax.io.readers.trios.json as jmod

        monkeypatch.setattr(jmod, "SCHEMA_PATH", tmp_path / "nope.json")
        assert jmod._load_schema() is None


class TestValidateSchema:
    def test_missing_schema_skips_validation(self, monkeypatch, tmp_path):
        import rheojax.io.readers.trios.json as jmod

        monkeypatch.setattr(jmod, "SCHEMA_PATH", tmp_path / "nope.json")
        is_valid, errors = jmod.validate_schema({"anything": 1})
        assert is_valid is True
        assert errors == []

    def test_valid_data_against_trivial_schema(self, monkeypatch):
        pytest.importorskip("jsonschema")
        import rheojax.io.readers.trios.json as jmod

        monkeypatch.setattr(jmod, "_load_schema", lambda: {"type": "object"})
        is_valid, errors = jmod.validate_schema({"a": 1})
        assert is_valid is True
        assert errors == []

    def test_invalid_data_reports_error(self, monkeypatch):
        pytest.importorskip("jsonschema")
        import rheojax.io.readers.trios.json as jmod

        monkeypatch.setattr(jmod, "_load_schema", lambda: {"type": "array"})
        is_valid, errors = jmod.validate_schema({"a": 1})
        assert is_valid is False
        assert errors and "validation error" in errors[0].lower()

    def test_invalid_data_raises_when_requested(self, monkeypatch):
        pytest.importorskip("jsonschema")
        import rheojax.io.readers.trios.json as jmod

        monkeypatch.setattr(jmod, "_load_schema", lambda: {"type": "array"})
        with pytest.raises(ValueError, match="validation error"):
            jmod.validate_schema({"a": 1}, raise_on_error=True)


# ---------------------------------------------------------------------------
# parse_trios_json
# ---------------------------------------------------------------------------


class TestParseTriosJson:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            parse_trios_json(str(tmp_path / "missing.json"))

    def test_invalid_json_syntax(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(json.JSONDecodeError):
            parse_trios_json(str(p))

    def test_utf8_sig_bom_decoded(self, tmp_path):
        obj = _experiment([_result(RELAX_COLUMNS, RELAX_VALUES)])
        p = tmp_path / "bom.json"
        # UTF-8 BOM must be handled by the encoding cascade.
        p.write_bytes(b"\xef\xbb\xbf" + json.dumps(obj).encode("utf-8"))
        experiment, metadata = parse_trios_json(str(p), validate=False)
        assert experiment.n_results == 1
        assert metadata["source_format"] == "json"
        assert metadata["source_file"] == "bom.json"

    def test_metadata_camel_to_snake(self, tmp_path):
        obj = _experiment(
            [_result(RELAX_COLUMNS, RELAX_VALUES)],
            properties={"Name": "Exp1", "InstrumentName": "DHR-3"},
        )
        path = _write_json(tmp_path, "meta.json", obj)
        _, metadata = parse_trios_json(path, validate=False)
        assert metadata["instrument_name"] == "DHR-3"
        assert metadata["sample_name"] == "Gel"

    def test_invalid_structure_raises_valueerror(self, tmp_path):
        # Non-iterable Results → enumerate() raises TypeError inside from_json,
        # which parse_trios_json wraps as a ValueError.
        obj = {"Experiment": {"Results": 42}}
        path = _write_json(tmp_path, "struct.json", obj)
        with pytest.raises(ValueError, match="Invalid TRIOS JSON structure"):
            parse_trios_json(path, validate=False)


# ---------------------------------------------------------------------------
# load_trios_json — RheoData conversion
# ---------------------------------------------------------------------------


class TestLoadRelaxation:
    def test_basic_relaxation(self, tmp_path):
        obj = _experiment([_result(RELAX_COLUMNS, RELAX_VALUES)])
        path = _write_json(tmp_path, "relax.json", obj)
        data = load_trios_json(path)
        assert isinstance(data, RheoData)
        assert data.metadata["test_mode"] == "relaxation"
        assert data.domain == "time"
        np.testing.assert_allclose(data.x, [0.1, 0.2, 0.3])
        np.testing.assert_allclose(data.y, [1000.0, 800.0, 600.0])
        assert data.x_units == "s"

    def test_no_results_raises(self, tmp_path):
        obj = _experiment([])
        path = _write_json(tmp_path, "noresults.json", obj)
        with pytest.raises(ValueError, match="No results found"):
            load_trios_json(path)

    def test_empty_dataframe_result_skipped_raises(self, tmp_path):
        obj = _experiment([_result([], [])])
        path = _write_json(tmp_path, "emptydf.json", obj)
        with pytest.raises(ValueError, match="No valid data segments"):
            load_trios_json(path)


class TestLoadComplexModulus:
    def test_oscillation_complex(self, tmp_path):
        obj = _experiment([_result(OSC_COLUMNS, OSC_VALUES)])
        path = _write_json(tmp_path, "osc.json", obj)
        data = load_trios_json(path)
        assert np.iscomplexobj(data.y)
        np.testing.assert_allclose(
            data.y, [1000 + 500j, 1200 + 600j, 1400 + 700j]
        )
        assert data.metadata["test_mode"] == "oscillation"
        assert data.x_units == "rad/s"

    def test_hz_converted(self, tmp_path):
        cols = [
            {"name": "Frequency", "unit": "Hz"},
            {"name": "Storage modulus", "unit": "Pa"},
            {"name": "Loss modulus", "unit": "Pa"},
        ]
        obj = _experiment([_result(cols, [[1.0, 100.0, 50.0], [2.0, 200.0, 100.0]])])
        path = _write_json(tmp_path, "hz.json", obj)
        data = load_trios_json(path)
        np.testing.assert_allclose(data.x, np.array([1.0, 2.0]) * 2 * np.pi)


class TestResultSelection:
    def _multi(self):
        return _experiment(
            [
                _result(RELAX_COLUMNS, RELAX_VALUES, props={"Step": 1, "Name": "A"}),
                _result(
                    RELAX_COLUMNS,
                    [[0.4, 500.0], [0.5, 400.0]],
                    props={"Step": 2, "Name": "B"},
                ),
            ]
        )

    def test_default_selects_first_result(self, tmp_path):
        path = _write_json(tmp_path, "multi.json", self._multi())
        data = load_trios_json(path)
        assert isinstance(data, RheoData)
        assert len(data.x) == 3

    def test_result_index_selects_second(self, tmp_path):
        path = _write_json(tmp_path, "multi.json", self._multi())
        data = load_trios_json(path, result_index=1)
        assert isinstance(data, RheoData)
        np.testing.assert_allclose(data.x, [0.4, 0.5])
        assert data.metadata["result_index"] == 1

    def test_result_index_all(self, tmp_path):
        path = _write_json(tmp_path, "multi.json", self._multi())
        result = load_trios_json(path, result_index=-1)
        assert isinstance(result, list)
        assert len(result) == 2

    def test_result_index_out_of_range(self, tmp_path):
        path = _write_json(tmp_path, "multi.json", self._multi())
        with pytest.raises(ValueError, match="out of range"):
            load_trios_json(path, result_index=5)

    def test_result_properties_in_metadata(self, tmp_path):
        path = _write_json(tmp_path, "multi.json", self._multi())
        data = load_trios_json(path)
        # Result-level properties are prefixed and snake-cased.
        assert data.metadata.get("result_step") == 1
        assert data.metadata.get("result_name") == "A"


class TestMultiStepSplit:
    def test_split_by_step_column(self, tmp_path):
        cols = [
            {"name": "Step", "unit": ""},
            {"name": "Time", "unit": "s"},
            {"name": "Relaxation modulus", "unit": "Pa"},
        ]
        values = [
            [1, 0.1, 1000.0],
            [1, 0.2, 900.0],
            [2, 0.3, 800.0],
            [2, 0.4, 700.0],
        ]
        obj = _experiment([_result(cols, values)])
        path = _write_json(tmp_path, "stepped.json", obj)
        result = load_trios_json(path, return_all_segments=True)
        assert isinstance(result, list)
        assert len(result) == 2
        # No data lost across the split.
        total = sum(len(d.x) for d in result)
        assert total == 4


class TestNanFiltering:
    def test_nan_rows_removed(self, tmp_path):
        values = [[0.1, 1000.0], [0.2, None], [0.3, 600.0]]
        obj = _experiment([_result(RELAX_COLUMNS, values)])
        path = _write_json(tmp_path, "nan.json", obj)
        data = load_trios_json(path)
        # The NaN row is dropped; two finite points remain.
        assert len(data.x) == 2
        assert np.all(np.isfinite(data.y))


class TestToDataframeNumericCoercion:
    """Regression tests (PR #67) for TRIOSDataSet.to_dataframe()'s per-column
    numeric coercion, which must neither silently NaN-fill a genuinely
    non-numeric column nor hard-crash a mostly-numeric column over one bad
    cell."""

    def test_fully_non_numeric_column_preserved(self):
        ds = TRIOSDataSet(
            columns=[{"name": "Time"}, {"name": "Point type"}],
            values=[[0.1, "Normal"], [0.2, "Rejected"], [0.3, "Interpolated"]],
        )
        df = ds.to_dataframe()
        assert df["Point type"].tolist() == ["Normal", "Rejected", "Interpolated"]
        assert np.issubdtype(df["Time"].dtype, np.number)

    def test_mostly_numeric_column_with_one_bad_cell_still_coerced(self):
        ds = TRIOSDataSet(
            columns=[{"name": "Time"}, {"name": "Normal Force"}],
            values=[[0.1, 100.0], [0.2, "n.a."], [0.3, 105.0]],
        )
        df = ds.to_dataframe()
        # Coerced to numeric (usable downstream), not left as strings and
        # not raising -- the bad cell becomes NaN, not a hard failure.
        assert np.issubdtype(df["Normal Force"].dtype, np.number)
        assert np.isnan(df["Normal Force"].iloc[1])
        np.testing.assert_allclose(df["Normal Force"].dropna(), [100.0, 105.0])
