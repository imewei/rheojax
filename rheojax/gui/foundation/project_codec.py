"""Versioned .rheojax v2 project persistence.

write_result_arrays()/read_result_arrays() recursively split an NLSQ/NUTS result dict (or any
value tree over the same universe rheojax.gui.jobs.subprocess_fit's _is_serializable() already
enumerates: str/int/float/bool/None/np.ndarray/dict/list/tuple, at any nesting depth) between an
HDF5 file (array leaves) and a JSON-safe structure (everything else). This is the SAME raw-h5py
pattern ExportService.save_project() v1 already uses for posterior_samples -- not the
RheoData-typed save_hdf5()/load_hdf5() helpers, which don't apply to this shape.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


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
