from rheojax.gui.foundation.library import DatasetLibrary, DatasetRef
from rheojax.gui.workspace.transform.transform_controller import infer_output_protocol


def _ref(id, ptype):
    return DatasetRef(
        id=id,
        name=id,
        protocol_type=ptype,
        origin="imported",
        units={},
        row_count=1,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_infer_output_protocol_same_domain_transform():
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    # smooth_derivative is same-domain (processing category) -- keeps source protocol
    result = infer_output_protocol(lib, "smooth_derivative", {"input": "d1"})
    assert result == "oscillation"


def test_infer_output_protocol_domain_changing_transform_returns_empty():
    lib = DatasetLibrary()
    lib.add(_ref("d1", "oscillation"))
    # fft_analysis is domain-changing (spectral category) -- returns ""
    result = infer_output_protocol(lib, "fft_analysis", {"input": "d1"})
    assert result == ""
