import pytest
from rheojax.gui.foundation.pipeline_bridge import pipeline_inputs_from_library

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef


def _ref(id: str) -> DatasetRef:
    return DatasetRef(
        id=id, name=id, protocol_type="oscillation", origin="imported",
        units={}, row_count=1, hash="h", provenance={}, lineage=[],
    )


def _lib(*ids: str) -> DatasetLibrary:
    lib = DatasetLibrary()
    for i in ids:
        lib.add(_ref(i))
    return lib


def test_resolves_ids_in_order():
    lib = _lib("a", "b")
    out = pipeline_inputs_from_library(lib, ["b", "a"])
    assert [r.id for r in out] == ["b", "a"]


def test_empty_ids_returns_empty():
    lib = _lib("a")
    assert pipeline_inputs_from_library(lib, []) == []


def test_unknown_id_raises_key_error():
    lib = _lib("a")
    with pytest.raises(KeyError):
        pipeline_inputs_from_library(lib, ["missing"])


def test_returns_dataset_ref_objects():
    lib = _lib("x")
    out = pipeline_inputs_from_library(lib, ["x"])
    assert len(out) == 1
    assert isinstance(out[0], DatasetRef)
    assert out[0].id == "x"
