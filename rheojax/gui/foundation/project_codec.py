"""Versioned .rheojax v2 project persistence.

write_result_arrays()/read_result_arrays() recursively split an NLSQ/NUTS result dict (or any
value tree over the same universe rheojax.gui.jobs.subprocess_fit's _is_serializable() already
enumerates: str/int/float/bool/None/np.ndarray/dict/list/tuple, at any nesting depth) between an
HDF5 file (array leaves) and a JSON-safe structure (everything else). This is the SAME raw-h5py
pattern ExportService.save_project() v1 already uses for posterior_samples -- not the
RheoData-typed save_hdf5()/load_hdf5() helpers, which don't apply to this shape.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import tempfile
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

import h5py
import numpy as np

from rheojax.io import save_hdf5


def _write_walk(value: Any, key_path: str, hf: h5py.File, out: dict | list) -> Any:
    if isinstance(value, np.ndarray):
        hf.create_dataset(key_path, data=value)
        return {"$hdf5_ref": key_path}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {
            k: _write_walk(v, f"{key_path}/{k}" if key_path else str(k), hf, out)
            for k, v in value.items()
        }
    if isinstance(value, tuple):
        return {"$tuple": [
            _write_walk(v, f"{key_path}/{i}", hf, out) for i, v in enumerate(value)
        ]}
    if isinstance(value, list):
        return [_write_walk(v, f"{key_path}/{i}", hf, out) for i, v in enumerate(value)]
    raise TypeError(
        f"write_result_arrays: unsupported value type {type(value).__name__!r} at "
        f"{key_path or '<root>'} -- not one of str/int/float/bool/None/np.ndarray/dict/list/tuple"
    )


def write_result_arrays(path: Path, result: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "w") as hf:
        return {k: _write_walk(v, str(k), hf, {}) for k, v in result.items()}


def _read_walk(node: Any, hf: h5py.File) -> Any:
    if isinstance(node, dict):
        if set(node.keys()) == {"$hdf5_ref"}:
            return np.asarray(hf[node["$hdf5_ref"]])
        if set(node.keys()) == {"$tuple"}:
            return tuple(_read_walk(v, hf) for v in node["$tuple"])
        return {k: _read_walk(v, hf) for k, v in node.items()}
    if isinstance(node, list):
        return [_read_walk(v, hf) for v in node]
    return node


def read_result_arrays(path: Path, json_shape: dict[str, Any]) -> dict[str, Any]:
    path = Path(path)
    with h5py.File(path, "r") as hf:
        return {k: _read_walk(v, hf) for k, v in json_shape.items()}


_ARCHIVE_VERSION = "2.0"


def save_project_v2(state, path: Path) -> None:
    """Encode an AppState into a versioned .rheojax v2 ZIP archive (spec §6.1), written
    atomically via a temp-file + fsync + os.replace (spec §6.2)."""
    path = Path(path)
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        member_hashes: dict[str, dict[str, int | str]] = {}

        def _write_json(rel_path: str, obj: Any) -> None:
            full = tmp_root / rel_path
            full.parent.mkdir(parents=True, exist_ok=True)
            data = json.dumps(obj, indent=2).encode("utf-8")
            full.write_bytes(data)
            member_hashes[rel_path] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }

        def _register_binary(rel_path: str) -> None:
            full = tmp_root / rel_path
            data = full.read_bytes()
            member_hashes[rel_path] = {
                "sha256": hashlib.sha256(data).hexdigest(),
                "size": len(data),
            }

        _write_json("metadata.json", {
            "version": _ARCHIVE_VERSION,
            "timestamp": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        })

        library_manifest = [dataclasses.asdict(ref) for ref in state.library.all()]
        _write_json("library/manifest.json", library_manifest)
        for ref in state.library.all():
            rel = f"library/{ref.id}.hdf5"
            full = tmp_root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            try:
                payload = state.library.load_payload(ref.id)
            except KeyError:
                continue
            save_hdf5(payload, str(full))
            _register_binary(rel)

        def _persist_result_dict(raw: dict | None, result_dir: str) -> tuple[str | None, dict | None]:
            """Splits a raw NLSQ/NUTS/phase result dict into an HDF5 array payload
            (result_dir/<uuid>.hdf5, via write_result_arrays) plus a JSON-safe metadata
            shape, returning (result_id, json_shape). Returns (None, None) for a falsy
            `raw`. This is the ONE place any raw fit/NUTS-shaped result dict gets
            persisted -- fit.json's nlsq_result/nuts_result AND every job_history.json
            fit-step phase result both route through this helper, so there is exactly one
            persistence path for this value shape, not two independently-maintained ones."""
            if not raw:
                return None, None
            result_id = uuid.uuid4().hex
            rel = f"{result_dir}/{result_id}.hdf5"
            full = tmp_root / rel
            full.parent.mkdir(parents=True, exist_ok=True)
            json_shape = write_result_arrays(full, raw)
            _register_binary(rel)
            return result_id, json_shape

        # nlsq_result/nuts_result are "Any"-typed fields that may hold nested dataclass
        # instances (e.g. RheoData); dataclasses.asdict() recurses into dict/list values
        # regardless of what they hold, so asdict-ing state.fit directly would silently
        # flatten any such nested dataclass before we get a chance to persist it specially.
        # Source the raw values from the live object and strip the fields via replace()
        # before asdict, instead of asdict-then-pop.
        raw_nlsq_result = state.fit.nlsq_result
        raw_nuts_result = state.fit.nuts_result
        fit_dict = dataclasses.asdict(
            dataclasses.replace(state.fit, nlsq_result=None, nuts_result=None)
        )
        for result_key, raw in (("nlsq_result", raw_nlsq_result), ("nuts_result", raw_nuts_result)):
            fit_dict.pop(result_key, None)
            result_id, json_shape = _persist_result_dict(raw, "fit_results")
            fit_dict[f"{result_key}_ref"] = result_id
            fit_dict[f"{result_key}_meta"] = json_shape
        _write_json("fit.json", fit_dict)

        # TransformState.result holds real RheoData objects under "input"/"output" (set by
        # transform_controller.py's _run()) -- these are not JSON-serializable, so they're
        # extracted to transform_results/<id>.hdf5 via save_hdf5 (the RheoData-typed helper,
        # NOT write_result_arrays -- these are real RheoData, unlike fit/NUTS result dicts),
        # exactly mirroring how library/<dataset_id>.hdf5 payloads above are handled.
        # Same asdict-recursion hazard as above: source `result` from the live object and
        # strip it via replace() before asdict, rather than asdict-then-pop.
        raw_transform_result = state.transform.result
        transform_dict = dataclasses.asdict(dataclasses.replace(state.transform, result=None))
        transform_dict.pop("result", None)
        transform_result_refs: dict[str, str | None] = {"input": None, "output": None}
        if raw_transform_result:
            for side in ("input", "output"):
                payload = raw_transform_result.get(side)
                if payload is None:
                    continue
                result_id = uuid.uuid4().hex
                rel = f"transform_results/{result_id}.hdf5"
                full = tmp_root / rel
                full.parent.mkdir(parents=True, exist_ok=True)
                save_hdf5(payload, str(full))
                _register_binary(rel)
                transform_result_refs[side] = result_id
        transform_dict["result_refs"] = transform_result_refs
        transform_dict["result_extras"] = {
            k: v for k, v in (raw_transform_result or {}).items() if k not in ("input", "output")
        }
        _write_json("transform.json", transform_dict)

        _write_json("pipeline.json", dataclasses.asdict(state.pipeline))

        # job_history.by_id records hold RAW fit-phase result dicts in memory (Plan 2's
        # PipelineBatchRunner._prepare_job_record() intentionally does NOT persist them itself
        # -- see Plan 2 Task 7 -- so there is exactly one place, here, that ever calls
        # write_result_arrays() for a pipeline job's results, avoiding the "external results_dir
        # never actually gets copied into the archive" gap).
        job_history_out: dict[str, dict] = {}
        for job_id, record in state.job_history.by_id.items():
            record = dict(record)
            step_results_out = {}
            for step_id, step_record in record.get("step_results", {}).items():
                step_record = dict(step_record)
                if step_record.get("step_type") == "fit":
                    for phase_key in ("nlsq", "nuts"):
                        phase = step_record.get(phase_key)
                        if phase is None:
                            continue
                        phase = dict(phase)
                        result_id, json_shape = _persist_result_dict(phase.pop("result", None), "job_results")
                        phase["result_ref"] = result_id
                        phase["result_meta"] = json_shape
                        step_record[phase_key] = phase
                elif "output" in step_record:
                    # A "transform" pipeline step's record (Plan 2's
                    # PipelineBatchRunner._prepare_job_record(), step_type == "other") embeds
                    # the transformed RheoData directly under "output" -- this is NOT
                    # JSON-serializable, so it's persisted the same way state.transform.result's
                    # RheoData is handled above (save_hdf5, not write_result_arrays -- this is
                    # a real RheoData, not a result dict), into transform_results/ (the same
                    # prefix, since it's the same kind of payload).
                    output = step_record.pop("output")
                    result_id = uuid.uuid4().hex
                    rel = f"transform_results/{result_id}.hdf5"
                    full = tmp_root / rel
                    full.parent.mkdir(parents=True, exist_ok=True)
                    save_hdf5(output, str(full))
                    _register_binary(rel)
                    step_record["output_ref"] = result_id
                step_results_out[step_id] = step_record
            record["step_results"] = step_results_out
            job_history_out[job_id] = record
        _write_json("job_history.json", job_history_out)

        _write_json("project.json", {"path": state.project.path, "name": state.project.name})
        _write_json("ui.json", dataclasses.asdict(state.ui))

        _write_json("manifest.json", {"members": member_hashes})

        tmp_zip = path.with_name(f"{path.name}.tmp-{uuid.uuid4().hex}")
        try:
            with ZipFile(tmp_zip, "w", compression=ZIP_DEFLATED) as zf:
                for rel_path in tmp_root.rglob("*"):
                    if rel_path.is_file():
                        zf.write(rel_path, rel_path.relative_to(tmp_root).as_posix())
            with open(tmp_zip, "rb") as f:
                os.fsync(f.fileno())
            os.replace(tmp_zip, path)
        except Exception:
            if tmp_zip.exists():
                tmp_zip.unlink()
            raise
