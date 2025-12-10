"""Anton Paar file reader for rheological data.

Note: This is a skeleton implementation. Full implementation requires
access to Anton Paar sample files and format specifications.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path
from typing import Iterable, Tuple

import numpy as np
import pandas as pd

from rheojax.core.data import RheoData


def _detect_delimiter(sample_line: str) -> str:
    """Pick a reasonable delimiter for Anton Paar exports."""

    if ";" in sample_line:
        return ";"
    if "\t" in sample_line:
        return "\t"
    return ","


def _extract_unit(name: str) -> Tuple[str | None, str]:
    """Return (clean_name, unit) from a column header like "Time [s]"."""

    match = re.search(r"^(.*?)[\[(](.*?)[\])]", name)
    if match:
        base = match.group(1).strip().lower().replace(" ", "_")
        unit = match.group(2).strip().lower()
        return base or name.lower(), unit
    return name.lower(), None


def _to_rheo_units(x: np.ndarray, unit: str | None) -> tuple[np.ndarray, str | None]:
    """Normalize common Anton Paar units to RheoJAX canonical choices."""

    if not unit:
        return x, unit

    unit_l = unit.lower()
    if unit_l in {"hz", "1/hz"}:
        return x * 2 * np.pi, "rad/s"
    if unit_l in {"ms"}:
        return x / 1000.0, "s"
    if unit_l in {"min", "mins", "minutes"}:
        return x * 60.0, "s"
    return x, unit


def _normalize_y_units(y: np.ndarray, unit: str | None) -> tuple[np.ndarray, str | None]:
    if not unit:
        return y, unit

    unit_l = unit.lower()
    if unit_l == "kpa":
        return y * 1e3, "pa"
    if unit_l == "mpa":
        return y * 1e6, "pa"
    return y, unit


def load_anton_paar(filepath: str | Path, **kwargs) -> RheoData:
    """Load Anton Paar rheometer exports (.txt / .csv style).

    The parser is deliberately lightweight: it ignores comment lines starting
    with ``#`` or ``;``, detects a simple delimiter (``;`` → tab → comma), and
    extracts units from column headers of the form ``Name [unit]``. We map
    common headers to ``x`` (time/frequency) and ``y``/``y2`` (moduli or
    viscosity). A handful of unit normalizations are applied:

    - Hz → rad/s
    - ms/min → s
    - kPa/MPa → Pa
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    # Discover delimiter from the first non-comment line
    with filepath.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip() and not line.lstrip().startswith(("#", ";")):
                delimiter = _detect_delimiter(line)
                break
        else:
            raise ValueError("File is empty or only contains comments")

    df = pd.read_csv(
        filepath,
        sep=delimiter,
        engine="python",
        comment="#",
        skip_blank_lines=True,
    )

    names_units: list[tuple[str, str | None]] = [_extract_unit(str(col)) for col in df.columns]

    def _pick(candidates: Iterable[str]) -> int | None:
        lowered = [n for n, _ in names_units]
        for cand in candidates:
            if cand in lowered:
                return lowered.index(cand)
        return None

    x_idx = _pick(["angular_frequency", "frequency", "omega", "time", "t"])
    if x_idx is None:
        raise ValueError("Could not locate time/frequency column in Anton Paar file")

    y_idx = _pick(["g'", "g_prime", "storage_modulus", "storage modulus", "modulus", "viscosity", "eta", "g1"])
    if y_idx is None:
        raise ValueError("Could not locate modulus/viscosity column in Anton Paar file")

    y2_idx = _pick(["g''", "g_second", "loss_modulus", "loss modulus", "g2"])

    x_raw = df.iloc[:, x_idx].to_numpy()
    y_raw = df.iloc[:, y_idx].to_numpy()
    y2_raw = df.iloc[:, y2_idx].to_numpy() if y2_idx is not None else None

    x_name, x_unit = names_units[x_idx]
    y_name, y_unit = names_units[y_idx]
    y2_name, y2_unit = (names_units[y2_idx] if y2_idx is not None else (None, None))

    x, x_unit_norm = _to_rheo_units(np.asarray(x_raw, dtype=float), x_unit)
    y, y_unit_norm = _normalize_y_units(np.asarray(y_raw, dtype=float), y_unit)
    y2 = None
    if y2_raw is not None:
        y2, y2_unit_norm = _normalize_y_units(np.asarray(y2_raw, dtype=float), y2_unit)
    else:
        y2_unit_norm = None

    domain = "frequency" if "freq" in x_name or "omega" in x_name else "time"
    metadata = {"source": "anton_paar", "x_label": x_name, "y_label": y_name}
    if y2_name:
        metadata["y2_label"] = y2_name
    if y2 is not None:
        metadata["y2"] = y2
        if y2_unit_norm or y2_unit:
            metadata["y2_units"] = y2_unit_norm or y2_unit

    return RheoData(
        x=x,
        y=y,
        x_units=x_unit_norm or x_unit,
        y_units=y_unit_norm or y_unit,
        domain=domain,
        metadata=metadata,
        validate=False,
    )
