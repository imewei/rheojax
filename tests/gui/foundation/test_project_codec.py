import hashlib
import json
import uuid
import zipfile
from pathlib import Path

import numpy as np
import pytest

from rheojax.core.data import RheoData
from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.foundation.project_codec import (
    _validate_ref_id,
    load_project_v2,
    read_result_arrays,
    save_project_v2,
    write_result_arrays,
)
from rheojax.gui.foundation.state import (
    AppState,
    FitState,
    NlsqConfig,
    NutsConfig,
    ParameterConfig,
    PipelineState,
    PipelineStepConfig,
    TransformState,
)


def test_flat_scalar_and_array_roundtrip(tmp_path):
    path = tmp_path / "result.hdf5"
    result = {
        "r_squared": 0.98,
        "success": True,
        "message": "ok",
        "note": None,
        "x_fit": np.array([1.0, 2.0, 3.0]),
    }
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
    assert json_shape["posterior_samples"]["G_p"] == {
        "$hdf5_ref": "posterior_samples/G_p"
    }
    assert json_shape["r_hat"] == {
        "G_p": 1.01,
        "tau": 1.02,
    }  # no arrays -- stays inline
    restored = read_result_arrays(path, json_shape)
    np.testing.assert_array_equal(
        restored["posterior_samples"]["G_p"], result["posterior_samples"]["G_p"]
    )
    np.testing.assert_array_equal(
        restored["sample_stats"]["energy"], result["sample_stats"]["energy"]
    )
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
    return DatasetRef(
        id=id_,
        name=id_,
        protocol_type="oscillation",
        origin="imported",
        units={"x": "rad/s", "y": "Pa"},
        row_count=3,
        hash="h",
        provenance={},
        lineage=[],
    )


def test_save_project_v2_writes_expected_archive_members(tmp_path):
    state = AppState()
    ref = _dataset_ref("d1")
    payload = RheoData(
        x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], initial_test_mode="oscillation"
    )
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
        "d1",
        RheoData(
            x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], initial_test_mode="oscillation"
        ),
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
            "input": RheoData(
                x=[1.0, 2.0], y=[3.0, 4.0], initial_test_mode="oscillation"
            ),
            "output": RheoData(
                x=[1.0, 2.0], y=[5.0, 6.0], initial_test_mode="oscillation"
            ),
            "warnings": ["clipped 2 points"],
        },
    )

    state.pipeline = PipelineState(
        steps=[
            PipelineStepConfig(id="s1", step_type="transform"),
            PipelineStepConfig(id="s2", step_type="fit"),
        ],
        selected_dataset_ids=["d1", "d2"],
        name="batch-a",
    )

    state.job_history.by_id["job1"] = {
        "status": "completed",
        "step_results": {
            "step-fit-1": {
                "step_type": "fit",
                "nlsq": {
                    "result": {"r_squared": 0.9, "params": np.array([1.0, 2.0])},
                    "duration_s": 1.2,
                },
                "nuts": None,
            },
            "step-transform-1": {
                "step_type": "other",
                "output": RheoData(
                    x=[1.0, 2.0], y=[7.0, 8.0], initial_test_mode="oscillation"
                ),
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
        for rel_path in (
            "metadata.json",
            "library/manifest.json",
            "library/d1.hdf5",
            f"fit_results/{fit['nlsq_result_ref']}.hdf5",
        ):
            assert rel_path in manifest["members"]
            import hashlib

            assert (
                manifest["members"][rel_path]["sha256"]
                == hashlib.sha256(zf.read(rel_path)).hexdigest()
            )

    # atomic write: no leftover temp files
    assert not list(tmp_path.glob(f"{out_path.name}.tmp-*"))


def test_round_trip_full_state(tmp_path):
    state = AppState()
    ref = _dataset_ref("d1")
    payload = RheoData(
        x=[1.0, 2.0, 3.0], y=[10.0, 20.0, 30.0], initial_test_mode="oscillation"
    )
    state.library.add(ref)
    state.library.store_payload("d1", payload)
    state.fit = FitState(
        protocol="oscillation",
        model_key="maxwell",
        nlsq_config=NlsqConfig(
            n_starts=5,
            parameters=[
                ParameterConfig(
                    name="G0", value=1.0, lower=0.0, upper=10.0, fixed=False
                )
            ],
        ),
        nuts_config=NutsConfig(run_nuts=False, max_tree_depth=7),
        nlsq_result={"parameters": {"G0": 1.0}, "x_fit": np.array([1.0, 2.0])},
    )
    # transform.result holds real RheoData under "input"/"output" (transform_controller.py's
    # actual shape) plus non-array extras -- this must survive the round trip too (§6.1 fix).
    state.transform.result = {
        "input": payload,
        "output": payload,
        "protocol_type": "oscillation",
    }

    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)
    loaded = load_project_v2(path)

    assert loaded.fit.protocol == "oscillation"
    assert loaded.fit.model_key == "maxwell"
    assert loaded.fit.nlsq_config.n_starts == 5
    assert loaded.fit.nlsq_config.parameters == [
        ParameterConfig(name="G0", value=1.0, lower=0.0, upper=10.0, fixed=False)
    ]
    assert loaded.fit.nuts_config.run_nuts is False
    # Regression: max_tree_depth was missing from NutsConfig entirely, so it
    # was silently dropped on every project save/reload even though the live
    # single-run flow forwarded it to the sampler correctly.
    assert loaded.fit.nuts_config.max_tree_depth == 7
    assert loaded.fit.nlsq_result["parameters"] == {"G0": 1.0}
    np.testing.assert_array_equal(loaded.fit.nlsq_result["x_fit"], np.array([1.0, 2.0]))
    assert loaded.library.get("d1").protocol_type == "oscillation"
    restored_payload = loaded.library.load_payload("d1")
    np.testing.assert_array_equal(restored_payload.x, payload.x)
    assert loaded.transform.result["protocol_type"] == "oscillation"
    np.testing.assert_array_equal(loaded.transform.result["input"].x, payload.x)
    np.testing.assert_array_equal(loaded.transform.result["output"].x, payload.x)


def test_round_trip_job_history_with_fit_phase_results(tmp_path):
    state = AppState()
    state.job_history.by_id["job1"] = {
        "dataset_id": "d1",
        "status": "completed",
        "error": None,
        "step_results": {
            "s1": {
                "step_type": "fit",
                "nlsq": {
                    "status": "completed",
                    "error": None,
                    "result": {
                        "parameters": {"G0": 2.0},
                        "x_fit": np.array([3.0, 4.0]),
                    },
                },
                "nuts": None,
            },
        },
    }
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)
    loaded = load_project_v2(path)

    phase = loaded.job_history.by_id["job1"]["step_results"]["s1"]["nlsq"]
    assert phase["status"] == "completed"
    assert phase["result"]["parameters"] == {"G0": 2.0}
    np.testing.assert_array_equal(phase["result"]["x_fit"], np.array([3.0, 4.0]))
    assert (
        "result_ref" not in phase
    )  # rehydrated back to a plain "result" dict, not left as a reference


def test_round_trip_job_history_with_transform_step_output(tmp_path):
    # A Pipeline "transform" step's job_history record embeds a real RheoData under "output"
    # (Plan 2's PipelineBatchRunner._prepare_job_record(), step_type == "other") -- this must
    # round-trip via save_hdf5/load_hdf5 (transform_results/), not raise TypeError from
    # json.dumps() trying to serialize the RheoData directly.
    state = AppState()
    output_payload = RheoData(
        x=[0.1, 1.0], y=[5.0, 6.0], initial_test_mode="oscillation"
    )
    state.job_history.by_id["job1"] = {
        "dataset_id": "d1",
        "status": "completed",
        "error": None,
        "step_results": {
            "s1": {
                "step_type": "other",
                "output": output_payload,
                "protocol_type": "oscillation",
                "dataset_id": None,
            },
        },
    }
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)  # must not raise TypeError
    loaded = load_project_v2(path)

    step_record = loaded.job_history.by_id["job1"]["step_results"]["s1"]
    assert "output_ref" not in step_record
    np.testing.assert_array_equal(step_record["output"].x, output_payload.x)
    assert step_record["protocol_type"] == "oscillation"


def test_job_history_transform_step_output_none_round_trips(tmp_path):
    # A failed transform pipeline step's record (step_type == "other") may carry
    # "output": None -- save_project_v2 must not crash trying to save_hdf5(None, ...),
    # and load_project_v2 must round-trip it back to "output": None (the same "no ref
    # means no output" convention _restore_result_dict uses for fit/NUTS results).
    state = AppState()
    state.job_history.by_id["job1"] = {
        "dataset_id": "d1",
        "status": "failed",
        "error": "transform raised",
        "step_results": {
            "s1": {
                "step_type": "other",
                "output": None,
                "protocol_type": "oscillation",
                "dataset_id": None,
            },
        },
    }
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)  # must not raise
    loaded = load_project_v2(path)

    step_record = loaded.job_history.by_id["job1"]["step_results"]["s1"]
    assert step_record["output"] is None
    assert "output_ref" not in step_record


def test_load_rejects_wrong_version(tmp_path):
    path = tmp_path / "v1_style.rheojax"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", json.dumps({"version": "1.0"}))
    with pytest.raises(ValueError, match="version"):
        load_project_v2(path)


def test_load_rejects_duplicate_zip_entries(tmp_path):
    path = tmp_path / "dup.rheojax"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", json.dumps({"version": "2.0", "timestamp": "x"}))
        zf.writestr("metadata.json", json.dumps({"version": "2.0", "timestamp": "y"}))
    with pytest.raises(ValueError, match="duplicate"):
        load_project_v2(path)


def test_load_rejects_disallowed_member(tmp_path):
    path = tmp_path / "extra.rheojax"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", json.dumps({"version": "2.0", "timestamp": "x"}))
        zf.writestr("not_in_schema.txt", "hi")
    with pytest.raises(ValueError, match="not_in_schema"):
        load_project_v2(path)


def test_load_rejects_checksum_mismatch(tmp_path):
    state = AppState()
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)

    import shutil

    corrupted = tmp_path / "corrupted.rheojax"
    shutil.copy(path, corrupted)
    # Rewrite the zip with a tampered project.json (zipfile can't edit in place -- rebuild).
    with zipfile.ZipFile(path) as src, zipfile.ZipFile(corrupted, "w") as dst:
        for item in src.infolist():
            data = src.read(item.filename)
            if item.filename == "project.json":
                data = b'{"path": "TAMPERED", "name": null}'
            dst.writestr(item, data)
    with pytest.raises(ValueError, match="checksum"):
        load_project_v2(corrupted)


def test_load_rejects_zip_slip_member_name(tmp_path):
    """Vector 1 (CWE-22): a zip member name like 'library/../../../evil.hdf5' satisfies
    the pre-fix allowlist (startswith('library/') and endswith('.hdf5')) but resolves
    outside tmp_root on extraction. _is_allowed_member must reject any '..' path segment
    before the prefix/suffix check runs, so this never reaches extraction at all."""
    path = tmp_path / "evil.rheojax"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("metadata.json", json.dumps({"version": "2.0", "timestamp": "x"}))
        zf.writestr("library/../../../evil.hdf5", b"payload")
    with pytest.raises(ValueError, match="disallowed member"):
        load_project_v2(path)


def test_load_rejects_ref_id_path_traversal_in_manifest(tmp_path):
    """Vector 2 (CWE-22): a ref id read from JSON *content* (library/manifest.json here)
    never passes through the zip member allowlist -- it's just a string field -- so a
    '../'-containing id must be caught separately by _validate_ref_id when building the
    HDF5 path from it."""
    metadata_bytes = json.dumps({"version": "2.0", "timestamp": "x"}).encode()
    members = {
        "metadata.json": metadata_bytes,
        "library/manifest.json": json.dumps(
            [
                {
                    "id": "../../../evil",
                    "name": "d1",
                    "protocol_type": "oscillation",
                    "origin": "imported",
                    "units": {},
                    "row_count": 1,
                    "hash": "h",
                    "provenance": {},
                    "lineage": [],
                }
            ]
        ).encode(),
        "fit.json": b"{}",
        "transform.json": json.dumps({"result_refs": {}, "result_extras": {}}).encode(),
        "pipeline.json": json.dumps({"steps": []}).encode(),
        "job_history.json": b"{}",
        "project.json": json.dumps({"path": None, "name": None}).encode(),
        "ui.json": b"{}",
    }
    # manifest.json must declare every OTHER member exactly (load_project_v2 now
    # verifies membership, not just per-file hashes -- see test_load_rejects_
    # manifest_membership_mismatch below).
    manifest = {
        "members": {
            name: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
            for name, data in members.items()
        }
    }
    path = tmp_path / "evil_ref_id.rheojax"
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
        zf.writestr("manifest.json", json.dumps(manifest))
    with pytest.raises(ValueError, match="invalid archive reference id"):
        load_project_v2(path)


def test_round_trip_job_history_empty_fit_result_dict_stays_empty_dict(tmp_path):
    # _persist_result_dict used `if not raw` (truthiness), so a legitimate empty dict
    # `{}` was silently indistinguishable from "no result" and restored as None.
    state = AppState()
    state.job_history.by_id["job1"] = {
        "dataset_id": "d1",
        "status": "completed",
        "error": None,
        "step_results": {
            "s1": {
                "step_type": "fit",
                "nlsq": {"status": "completed", "error": None, "result": {}},
                "nuts": None,
            },
        },
    }
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)
    loaded = load_project_v2(path)
    phase = loaded.job_history.by_id["job1"]["step_results"]["s1"]["nlsq"]
    assert phase["result"] == {}
    assert phase["result"] is not None


def test_load_rejects_manifest_membership_mismatch(tmp_path):
    # A manifest declaring an entry for a member that isn't actually in the archive
    # (or vice versa) must be rejected -- not silently skipped -- even if every entry
    # that IS present in both happens to hash-match.
    state = AppState()
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)

    with zipfile.ZipFile(path) as src:
        manifest = json.loads(src.read("manifest.json"))
        manifest["members"]["phantom/not_real.hdf5"] = {"sha256": "0" * 64, "size": 0}
        tampered = tmp_path / "tampered.rheojax"
        with zipfile.ZipFile(tampered, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "manifest.json":
                    data = json.dumps(manifest).encode()
                dst.writestr(item, data)

    with pytest.raises(ValueError, match="does not exactly match"):
        load_project_v2(tampered)


def test_load_rejects_metadata_checksum_tampering(tmp_path):
    # metadata.json IS hashed at save time (registered via _write_json like every other
    # JSON member) -- the loader must actually verify it, not exempt it the way it
    # legitimately exempts manifest.json's own unhashable self-reference.
    state = AppState()
    path = tmp_path / "project.rheojax"
    save_project_v2(state, path)

    with zipfile.ZipFile(path) as src:
        tampered = tmp_path / "tampered_meta.rheojax"
        with zipfile.ZipFile(tampered, "w") as dst:
            for item in src.infolist():
                data = src.read(item.filename)
                if item.filename == "metadata.json":
                    data = json.dumps(
                        {"version": "2.0", "timestamp": "TAMPERED"}
                    ).encode()
                dst.writestr(item, data)

    with pytest.raises(ValueError, match="checksum"):
        load_project_v2(tampered)


def test_validate_ref_id_rejects_traversal_and_accepts_uuid():
    good = uuid.uuid4().hex
    assert _validate_ref_id(good) == good
    for bad in ("../../etc/passwd", "foo/bar", "foo\\bar", "", None):
        with pytest.raises(ValueError):
            _validate_ref_id(bad)
