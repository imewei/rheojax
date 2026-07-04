import pytest

from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef


def _ref(id, ptype, origin="imported"):
    return DatasetRef(
        id=id,
        name=id,
        protocol_type=ptype,
        origin=origin,
        units={},
        row_count=10,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_typed_query_includes_derived():
    lib = DatasetLibrary()
    lib.add(_ref("a", "oscillation"))
    lib.add(_ref("b", "oscillation", origin="derived"))  # derived still matches by type
    lib.add(_ref("c", "flow_curve"))
    ids = {r.id for r in lib.datasets_of_type("oscillation")}
    assert ids == {"a", "b"}


def test_remove_and_get():
    lib = DatasetLibrary()
    lib.add(_ref("a", "creep"))
    assert lib.get("a").protocol_type == "creep"
    lib.remove("a")
    assert lib.all() == []


def test_payload_roundtrip():
    lib = DatasetLibrary()
    lib.add(_ref("a", "oscillation"))
    payload = object()  # stand-in for a real RheoData
    lib.store_payload("a", payload)
    assert lib.load_payload("a") is payload


def test_add_rejects_collision_by_default():
    lib = DatasetLibrary()
    lib.add(_ref("a", "oscillation"))
    with pytest.raises(ValueError):
        lib.add(_ref("a", "creep"))
    assert lib.get("a").protocol_type == "oscillation"  # unchanged


def test_add_overwrite_true_replaces():
    lib = DatasetLibrary()
    lib.add(_ref("a", "oscillation"))
    lib.add(_ref("a", "creep"), overwrite=True)
    assert lib.get("a").protocol_type == "creep"


def test_add_overwrite_clears_stale_payload_when_new_payload_absent():
    lib = DatasetLibrary()
    lib.add(_ref("a", "oscillation"))
    lib.store_payload("a", object())
    lib.add(_ref("a", "creep"), overwrite=True)  # no new store_payload call
    with pytest.raises(KeyError):
        lib.load_payload("a")
