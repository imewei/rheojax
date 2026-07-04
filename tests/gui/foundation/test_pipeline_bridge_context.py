from __future__ import annotations

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.foundation.pipeline_bridge import pipeline_context_from_library


class _RheoData:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def _ref(i, protocol):
    return DatasetRef(
        id=i,
        name=i,
        protocol_type=protocol,
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_pipeline_context_from_library_seeds_data():
    lib = DatasetLibrary()
    lib.add(_ref("a", "flow_curve"))
    payload = _RheoData([1.0], [2.0])
    lib.store_payload("a", payload)
    ctx = pipeline_context_from_library(lib, ["a"])
    assert ctx["data"] is payload


def test_pipeline_context_from_library_empty_ids():
    lib = DatasetLibrary()
    assert pipeline_context_from_library(lib, []) == {}


def test_context_includes_dataset_id():
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    lib.store_payload("d1", object())
    ctx = pipeline_context_from_library(lib, ["d1"])
    assert ctx["dataset_id"] == "d1"
    assert "data" in ctx


def test_pipeline_context_from_library_works_for_derived_datasets_too():
    """The whole point: a derived dataset (no backing file) must still be
    usable as pipeline input via context-seeding, unlike the file-path-only
    load step."""
    lib = DatasetLibrary()
    lib.add(
        DatasetRef(
            id="derived1",
            name="derived1",
            protocol_type="oscillation",
            origin="derived",
            units={},
            row_count=1,
            hash="",
            provenance={},
            lineage=["a"],
        )
    )
    payload = _RheoData([3.0], [4.0])
    lib.store_payload("derived1", payload)
    ctx = pipeline_context_from_library(lib, ["derived1"])
    assert ctx["data"] is payload
