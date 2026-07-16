"""CLI/non-interactive dataset import, backing `rheojax-gui --import FILE --protocol PROTOCOL`.

Deliberately does NOT reuse the legacy ImportWizard/DataService.load_file path (per the design
spec, that path is bug-prone); this uses rheojax.io's format auto-detection directly and
validates against the same shape/NaN/monotonicity checks the workspace's Data step already
enforces (rheojax.gui.workspace.fit.step2_data._validate_shape_and_values), plus a
contract-driven column-count check.
"""

from __future__ import annotations

import hashlib
import uuid
from pathlib import Path

import numpy as np

from rheojax.core.data import RheoData
from rheojax.core.inventory import Protocol
from rheojax.gui.foundation.contract import input_contract
from rheojax.gui.foundation.library import DatasetRef
from rheojax.gui.workspace.fit.step2_data import _validate_shape_and_values
from rheojax.io import auto_load


def import_dataset(path: Path, protocol: str) -> tuple[DatasetRef, RheoData]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Import file not found: {path}")

    valid_protocols = {p.value for p in Protocol}
    if protocol not in valid_protocols:
        raise ValueError(
            f"Unknown protocol {protocol!r}; expected one of {sorted(valid_protocols)}"
        )

    data = auto_load(str(path))
    if isinstance(data, list):
        raise ValueError(
            f"Import file contains {len(data)} segments; "
            "CLI import supports single-segment files only"
        )

    contract = input_contract(protocol)
    errors = _validate_shape_and_values(data)
    # A contract with 3 columns (x + 2 y-roles, e.g. omega/G_prime/G_double_prime)
    # expects a complex-valued y (G' + i*G''), matching how rheojax.io's CSV
    # reader packs a detected modulus pair; anything else expects real 1-D y.
    y_arr = np.asarray(data.y)
    expects_complex = len(contract.columns) == 3
    if expects_complex and not np.iscomplexobj(y_arr):
        errors.append(
            f"expected complex modulus data (G', G'') for protocol {protocol!r}, "
            "got real-valued y"
        )
    elif not expects_complex and (y_arr.ndim != 1 or np.iscomplexobj(y_arr)):
        errors.append(
            f"expected a single real-valued y column for protocol {protocol!r}, "
            f"got shape {y_arr.shape} dtype {y_arr.dtype}"
        )
    if errors:
        raise ValueError(
            f"Import file failed contract validation for protocol {protocol!r}: {errors}"
        )

    data.update_metadata({"test_mode": protocol})

    # Mirror step2_data.py's needs_hz_conversion/apply_unit_conversion: a contract
    # declaring "x": "Hz->rad/s" means fits assume rad/s, but the reader only
    # *detects* Hz for classification -- it never converts (data_formats_by_protocol.md
    # §2, §6.1). Skipping this here would silently feed Hz-labeled x into a fit,
    # off by a factor of 2*pi with no error.
    units: dict[str, str] = {}
    if data.x_units:
        if contract.unit_conversions.get(
            "x"
        ) == "Hz->rad/s" and data.x_units.strip().lower() in (
            "hz",
            "hertz",
        ):
            data.x = np.asarray(data.x) * (2 * np.pi)
            data.x_units = "rad/s"
        units["x"] = data.x_units
    # Same reasoning as the x block above, for the flow_curve stress<->
    # viscosity guard (contract.py's "y": "stress<->viscosity" entry,
    # step2_data.py's needs_quantity_conversion()): the reader auto-detects
    # y_units (including inferring it from a bare "Viscosity"/"Stress"
    # header via infer_y_unit_from_name()), but this CLI import path
    # previously never copied it into DatasetRef.units, so the guard could
    # never see it -- unlike the interactive GUI import (window.py), which
    # does copy both x and y units. Without this, a CLI-imported viscosity
    # column silently gets fit as stress with no warning.
    if data.y_units:
        units["y"] = data.y_units

    ref = DatasetRef(
        id=uuid.uuid4().hex,
        name=path.stem,
        protocol_type=protocol,
        origin="imported",
        units=units,
        row_count=len(data.x),
        hash=hashlib.sha256(path.read_bytes()).hexdigest(),
        provenance={"source": "cli_import", "path": str(path)},
        lineage=[],
    )
    return ref, data
