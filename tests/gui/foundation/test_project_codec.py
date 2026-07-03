import json
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.project_codec import (
    read_result_arrays,
    save_project_v2,
    write_result_arrays,
)
from rheojax.gui.foundation.state import (
    AppState,
    FitState,
    PipelineState,
    PipelineStepConfig,
    TransformState,
)


def test_flat_scalar_and_array_roundtrip(tmp_path):
    path = tmp_path / "result.hdf5"
    result = {"r_squared": 0.98, "success": True, "message": "ok", "note": None,
              "x_fit": np.array([1.0, 2.0, 3.0])}
    json_shape = write_result_arrays(path, result)
    assert json_shape["r_squared"] == 0.98
    assert json_shape["x_fit"] == {"$hdf5_ref": "x_fit"}
    restored = read_result_arrays(path, json_shape)
    assert restored["r_squared"] == 0.98
    assert restored["success"] is True
    assert restored["note"] is None
    np.testing.assert_array_equal(restored["x_fit"], result["x_fit"])


def test_nested_dict_of_arrays_roundtrip(tmp_path):
    # posterior_samples / sample_stats shape from subprocess_bayesian.py
    path = tmp_path / "nuts_result.hdf5"
    result = {
        "posterior_samples": {"G_p": np.array([1.0, 2.0]), "tau": np.array([0.5, 0.6])},
        "sample_stats": {"energy": np.array([10.0, 11.0])},
        "r_hat": {"G_p": 1.01, "tau": 1.02},
    }
    json_shape = write_result_arrays(path, result)
    assert json_shape["posterior_samples"]["G_p"] == {"$hdf5_ref": "posterior_samples/G_p"}
    assert json_shape["r_hat"] == {"G_p": 1.01, "tau": 1.02}   # no arrays -- stays inline
    restored = read_result_arrays(path, json_shape)
    np.testing.assert_array_equal(restored["posterior_samples"]["G_p"], result["posterior_samples"]["G_p"])
    np.testing.assert_array_equal(restored["sample_stats"]["energy"], result["sample_stats"]["energy"])
    assert restored["r_hat"] == {"G_p": 1.01, "tau": 1.02}


def test_tuple_values_roundtrip_as_tuples(tmp_path):
    # credible_intervals shape from subprocess_bayesian.py: dict[str, tuple[float, float, float]]
    path = tmp_path / "ci_result.hdf5"
    result = {"credible_intervals": {"G_p": (0.9, 1.0, 1.1)}}
    json_shape = write_result_arrays(path, result)
    restored = read_result_arrays(path, json_shape)
    assert restored["credible_intervals"]["G_p"] == (0.9, 1.0, 1.1)
    assert isinstance(restored["credible_intervals"]["G_p"], tuple)


def test_list_of_scalars_stays_a_list(tmp_path):
    path = tmp_path / "list_result.hdf5"
    result = {"iterations": [1, 2, 3]}
    json_shape = write_result_arrays(path, result)
    restored = read_result_arrays(path, json_shape)
    assert restored["iterations"] == [1, 2, 3]
    assert isinstance(restored["iterations"], list)


def test_numpy_scalar_normalized_to_python_scalar(tmp_path):
    path = tmp_path / "npscalar_result.hdf5"
    result = {"chi_squared": np.float64(3.14)}
    json_shape = write_result_arrays(path, result)
    assert isinstance(json_shape["chi_squared"], float)
    assert json_shape["chi_squared"] == pytest.approx(3.14)


def test_unsupported_leaf_type_raises_type_error(tmp_path):
    path = tmp_path / "bad_result.hdf5"

    class Unsupported:
        pass

    with pytest.raises(TypeError):
        write_result_arrays(path, {"bad": Unsupported()})


def _dataset_ref(id_):
    return DatasetRef(id=id_, name=id_, protocol_type="oscillation", origin="imported",
                      units={"x": "rad/s", "y": "Pa"}, row_count=3, hash="h", provenance={},
                      lineage=[])


def test_save_project_v2_writes_expected_archive_members(tmp_path):
    state = AppState()
    ref = _dataset_ref("d1")
    payload = RheoData(x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], initial_test_mode="oscillation")
    state.library.add(ref)
    state.library.store_payload("d1", payload)
    state.fit = FitState(protocol="oscillation", model_key="maxwell")

    out_path = tmp_path / "project.rheojax"
    save_project_v2(state, out_path)

    assert out_path.exists()
    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())
    assert "metadata.json" in names
    assert "manifest.json" in names
    assert "library/manifest.json" in names
    assert "library/d1.hdf5" in names
    assert "fit.json" in names
    assert "transform.json" in names
    assert "pipeline.json" in names
    assert "job_history.json" in names
    assert "project.json" in names
    assert "ui.json" in names


def test_save_project_v2_metadata_version(tmp_path):
    save_project_v2(AppState(), tmp_path / "p.rheojax")
    with zipfile.ZipFile(tmp_path / "p.rheojax") as zf:
        meta = json.loads(zf.read("metadata.json"))
    assert meta["version"] == "2.0"


def test_save_project_v2_multi_dataset_multi_slice_archive(tmp_path):
    """Exercises every non-trivial branch: 2 library datasets (one without a stored
    payload), a raw NLSQ result dict on FitState, real RheoData under both sides of
    TransformState.result plus an extra non-RheoData key, and a job_history record with
    both a 'fit' step (nlsq phase result dict) and a transform-type step ('output' RheoData)."""
    state = AppState()
    state.library.add(_dataset_ref("d1"))
    state.library.add(_dataset_ref("d2"))
    state.library.store_payload(
        "d1", RheoData(x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], initial_test_mode="oscillation")
    )
    # d2 intentionally left without a stored payload.

    state.fit = FitState(
        protocol="oscillation",
        model_key="maxwell",
        nlsq_result={"r_squared": 0.97, "params": np.array([1.0, 2.0])},
    )

    state.transform = TransformState(
        transform_key="smoothing",
        result={
            "input": RheoData(x=[1.0, 2.0], y=[3.0, 4.0], initial_test_mode="oscillation"),
            "output": RheoData(x=[1.0, 2.0], y=[5.0, 6.0], initial_test_mode="oscillation"),
            "warnings": ["clipped 2 points"],
        },
    )

    state.pipeline = PipelineState(
        steps=[PipelineStepConfig(id="s1", step_type="transform"),
               PipelineStepConfig(id="s2", step_type="fit")],
        selected_dataset_ids=["d1", "d2"],
        name="batch-a",
    )

    state.job_history.by_id["job1"] = {
        "status": "completed",
        "step_results": {
            "step-fit-1": {
                "step_type": "fit",
                "nlsq": {"result": {"r_squared": 0.9, "params": np.array([1.0, 2.0])},
                         "duration_s": 1.2},
                "nuts": None,
            },
            "step-transform-1": {
                "step_type": "other",
                "output": RheoData(x=[1.0, 2.0], y=[7.0, 8.0], initial_test_mode="oscillation"),
            },
        },
    }

    out_path = tmp_path / "project.rheojax"
    save_project_v2(state, out_path)

    with zipfile.ZipFile(out_path) as zf:
        names = set(zf.namelist())

        library_manifest = json.loads(zf.read("library/manifest.json"))
        assert {e["id"] for e in library_manifest} == {"d1", "d2"}
        assert "library/d1.hdf5" in names
        assert "library/d2.hdf5" not in names  # no stored payload -> not written

        fit = json.loads(zf.read("fit.json"))
        assert fit["nlsq_result_ref"] is not None
        assert f"fit_results/{fit['nlsq_result_ref']}.hdf5" in names
        assert fit["nuts_result_ref"] is None

        transform = json.loads(zf.read("transform.json"))
        input_ref = transform["result_refs"]["input"]
        output_ref = transform["result_refs"]["output"]
        assert input_ref is not None and output_ref is not None
        assert f"transform_results/{input_ref}.hdf5" in names
        assert f"transform_results/{output_ref}.hdf5" in names
        assert transform["result_extras"] == {"warnings": ["clipped 2 points"]}

        pipeline = json.loads(zf.read("pipeline.json"))
        assert pipeline["name"] == "batch-a"
        assert [s["id"] for s in pipeline["steps"]] == ["s1", "s2"]

        job_history = json.loads(zf.read("job_history.json"))
        fit_step = job_history["job1"]["step_results"]["step-fit-1"]
        nlsq_ref = fit_step["nlsq"]["result_ref"]
        assert nlsq_ref is not None
        assert f"job_results/{nlsq_ref}.hdf5" in names
        assert "result" not in fit_step["nlsq"]

        transform_step = job_history["job1"]["step_results"]["step-transform-1"]
        assert "output" not in transform_step
        assert f"transform_results/{transform_step['output_ref']}.hdf5" in names

        # manifest.json hashes every archive member and matches the actual bytes.
        manifest = json.loads(zf.read("manifest.json"))
        for rel_path in ("metadata.json", "library/manifest.json", "library/d1.hdf5",
                          f"fit_results/{fit['nlsq_result_ref']}.hdf5"):
            assert rel_path in manifest["members"]
            import hashlib
            assert manifest["members"][rel_path]["sha256"] == hashlib.sha256(zf.read(rel_path)).hexdigest()

    # atomic write: no leftover temp files
    assert not list(tmp_path.glob(f"{out_path.name}.tmp-*"))
