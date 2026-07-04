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

from rheojax.io import load_hdf5, save_hdf5


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
            if raw is None:
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
                    # A failed transform step may carry "output": None -- mirror the "no ref
                    # means no output" convention _restore_result_dict already uses on load,
                    # rather than crashing inside save_hdf5.
                    output = step_record.pop("output")
                    result_id = None
                    if output is not None:
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


_ALLOWED_TOP_LEVEL = {
    "metadata.json", "manifest.json", "library/manifest.json", "fit.json",
    "transform.json", "pipeline.json", "job_history.json", "project.json", "ui.json",
}
_ALLOWED_PREFIXES = ("library/", "fit_results/", "transform_results/", "job_results/")

_MAX_MEMBERS = 10_000
_MAX_TOTAL_BYTES = 2 * 1024**3
_MAX_MEMBER_BYTES = 500 * 1024**2


def _is_allowed_member(name: str) -> bool:
    if "\\" in name or name.startswith("/") or any(part == ".." for part in name.split("/")):
        return False
    if name in _ALLOWED_TOP_LEVEL:
        return True
    return any(name.startswith(p) and name.endswith((".hdf5", "/manifest.json")) for p in _ALLOWED_PREFIXES)


def _validate_ref_id(ref_id: str) -> str:
    """Rejects a ref id that could escape tmp_root when interpolated into a path
    (spec §6.2) -- legitimate ids are always uuid4().hex, which never contains
    '/', '\\', or '..'."""
    if not ref_id or "/" in ref_id or "\\" in ref_id or ".." in ref_id:
        raise ValueError(f"invalid archive reference id: {ref_id!r}")
    return ref_id


def load_project_v2(path: Path):
    """Decode a versioned .rheojax v2 ZIP archive back into an AppState (spec §6.2):
    archive-member allowlist, size/count limits, duplicate-entry rejection, encrypted/
    unsupported-compression rejection, SHA-256 checksum verification, all-or-nothing decode."""
    from rheojax.gui.foundation.library import DatasetRef
    from rheojax.gui.foundation.state import (
        AppState,
        FitState,
        JobHistoryState,
        NlsqConfig,
        NutsConfig,
        ParameterConfig,
        PipelineState,
        PipelineStepConfig,
        ProjectState,
        TransformState,
        UiState,
    )

    path = Path(path)
    with ZipFile(path) as zf:
        infos = zf.infolist()

        if len(infos) > _MAX_MEMBERS:
            raise ValueError(f"Archive has {len(infos)} members, exceeds limit of {_MAX_MEMBERS}")

        seen_names: set[str] = set()
        total_bytes = 0
        for info in infos:
            if info.filename in seen_names:
                raise ValueError(f"Archive contains duplicate member: {info.filename}")
            seen_names.add(info.filename)
            if info.flag_bits & 0x1:
                raise ValueError(f"Archive member {info.filename} is encrypted; rejected")
            if info.compress_type not in (0, 8):  # ZIP_STORED, ZIP_DEFLATED
                raise ValueError(f"Archive member {info.filename} uses an unsupported compression type")
            if info.file_size > _MAX_MEMBER_BYTES:
                raise ValueError(f"Archive member {info.filename} exceeds the {_MAX_MEMBER_BYTES}-byte limit")
            total_bytes += info.file_size
            if not _is_allowed_member(info.filename):
                raise ValueError(f"Archive contains a disallowed member: {info.filename}")
        if total_bytes > _MAX_TOTAL_BYTES:
            raise ValueError(f"Archive's total uncompressed size exceeds {_MAX_TOTAL_BYTES} bytes")

        metadata = json.loads(zf.read("metadata.json"))
        if metadata.get("version") != _ARCHIVE_VERSION:
            raise ValueError(
                f"Unsupported project version {metadata.get('version')!r}, expected {_ARCHIVE_VERSION!r}"
            )

        manifest = json.loads(zf.read("manifest.json"))
        checksums = manifest["members"]
        # manifest.json can't hash itself (chicken-and-egg -- its own content includes
        # every OTHER member's hash), but metadata.json IS written and hashed before
        # manifest.json at save time, so it belongs in this set, not skipped alongside it.
        hashable_names = seen_names - {"manifest.json"}
        if set(checksums) != hashable_names:
            raise ValueError(
                "Archive manifest does not exactly match its members "
                f"(manifest-only: {sorted(set(checksums) - hashable_names)}, "
                f"archive-only: {sorted(hashable_names - set(checksums))})"
            )
        for name in hashable_names:
            data = zf.read(name)
            expected = checksums[name].get("sha256")
            actual = hashlib.sha256(data).hexdigest()
            if expected != actual:
                raise ValueError(f"checksum mismatch for archive member: {name}")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_root = Path(tmpdir)
            resolved_root = tmp_root.resolve()
            for name in seen_names:
                dest = tmp_root / name
                if resolved_root not in dest.resolve().parents:
                    raise ValueError(f"archive member resolves outside extraction root: {name}")
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(zf.read(name))

            state = AppState()

            library_manifest = json.loads((tmp_root / "library" / "manifest.json").read_text())
            for ref_dict in library_manifest:
                ref = DatasetRef(**ref_dict)
                _validate_ref_id(ref.id)
                state.library.add(ref)
                hdf5_path = tmp_root / "library" / f"{ref.id}.hdf5"
                if hdf5_path.exists():
                    state.library.store_payload(ref.id, load_hdf5(str(hdf5_path)))

            def _restore_result_dict(ref_id: str | None, meta: dict | None, result_dir: str) -> dict | None:
                """Inverse of save_project_v2's _persist_result_dict -- the ONE restoration
                path for any persisted fit/NUTS/phase result dict, used below for both
                fit.json's nlsq_result/nuts_result and every job_history.json fit-step phase
                result, mirroring how they share one persistence path on save."""
                if ref_id is None:
                    return None
                _validate_ref_id(ref_id)
                result_path = tmp_root / result_dir / f"{ref_id}.hdf5"
                return read_result_arrays(result_path, meta)

            fit_dict = json.loads((tmp_root / "fit.json").read_text())
            for result_key in ("nlsq_result", "nuts_result"):
                ref_id = fit_dict.pop(f"{result_key}_ref", None)
                meta = fit_dict.pop(f"{result_key}_meta", None)
                fit_dict[result_key] = _restore_result_dict(ref_id, meta, "fit_results")
            nlsq_cfg = fit_dict.pop("nlsq_config", {}) or {}
            nuts_cfg = fit_dict.pop("nuts_config", {}) or {}
            raw_parameters = nlsq_cfg.pop("parameters", []) or []
            state.fit = FitState(
                **{k: v for k, v in fit_dict.items() if k in FitState.__dataclass_fields__},
                nlsq_config=NlsqConfig(
                    **nlsq_cfg, parameters=[ParameterConfig(**p) for p in raw_parameters]
                ),
                nuts_config=NutsConfig(**nuts_cfg),
            )

            transform_dict = json.loads((tmp_root / "transform.json").read_text())
            result_refs = transform_dict.pop("result_refs", {}) or {}
            result_extras = transform_dict.pop("result_extras", {}) or {}
            restored_result: dict | None = None
            if any(result_refs.values()) or result_extras:
                restored_result = dict(result_extras)
                for side in ("input", "output"):
                    ref_id = result_refs.get(side)
                    if ref_id is not None:
                        _validate_ref_id(ref_id)
                        restored_result[side] = load_hdf5(str(tmp_root / "transform_results" / f"{ref_id}.hdf5"))
            transform_dict["result"] = restored_result
            state.transform = TransformState(**transform_dict)

            pipeline_dict = json.loads((tmp_root / "pipeline.json").read_text())
            pipeline_dict["steps"] = [PipelineStepConfig(**s) for s in pipeline_dict.get("steps", [])]
            state.pipeline = PipelineState(**pipeline_dict)

            job_history_dict = json.loads((tmp_root / "job_history.json").read_text())
            for record in job_history_dict.values():
                for step_record in record.get("step_results", {}).values():
                    if step_record.get("step_type") == "fit":
                        for phase_key in ("nlsq", "nuts"):
                            phase = step_record.get(phase_key)
                            if phase is None:
                                continue
                            ref_id = phase.pop("result_ref", None)
                            meta = phase.pop("result_meta", None)
                            phase["result"] = _restore_result_dict(ref_id, meta, "job_results")
                    elif "output_ref" in step_record:
                        # Inverse of save_project_v2's "output" -> "output_ref" persistence
                        # for a "transform" pipeline step's record (see the matching comment
                        # there). A None ref_id means the step had no output (e.g. a failed
                        # transform) -- mirror _restore_result_dict's "no ref means no output"
                        # convention rather than validating a None id.
                        ref_id = step_record.pop("output_ref")
                        if ref_id is None:
                            step_record["output"] = None
                        else:
                            _validate_ref_id(ref_id)
                            step_record["output"] = load_hdf5(
                                str(tmp_root / "transform_results" / f"{ref_id}.hdf5")
                            )
            state.job_history = JobHistoryState(by_id=job_history_dict)

            project_dict = json.loads((tmp_root / "project.json").read_text())
            state.project = ProjectState(path=project_dict.get("path"), name=project_dict.get("name"))

            ui_dict = json.loads((tmp_root / "ui.json").read_text())
            state.ui = UiState(**ui_dict)

            return state
