"""SPP (Sequence of Physical Processes) data export module.

This module provides export functions for SPP analysis results in formats
compatible with MATLAB SPPplus and R oreo packages.

Supported Formats
-----------------
- Text (.txt): Tab-delimited with headers matching MATLAB SPPplus format
- HDF5 (.h5): Hierarchical format with datasets and metadata
- CSV (.csv): Standard comma-separated values
- MAT-compatible dict: For scipy.io.savemat

Output Structure
----------------
The MATLAB-compatible output follows SPPplus_print_v2.m format:
- spp_data_out: 15-column matrix (time, strain, rate, stress, G'_t, G''_t,
  |G*_t|, tan(delta_t), delta_t, disp_stress, eq_strain, Gp_t_dot, Gpp_t_dot,
  G_speed, delta_t_dot)
- fsf_data_out: 9-column matrix (T_x, T_y, T_z, N_x, N_y, N_z, B_x, B_y, B_z)

References
----------
- MATLAB SPPplus_print_v2.m: Standard SPP output format
- R oreo package: Rpp_num() output structure
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rheojax.logging import get_logger, log_io

logger = get_logger(__name__)

if TYPE_CHECKING:
    pass


# ============================================================================
# Shared Data-Extraction Helper
# ============================================================================


def _extract_spp_arrays(
    spp_results: dict, n_points: int | None = None
) -> dict[str, np.ndarray]:
    """Extract and validate the SPP data arrays from a results dict.

    Parameters
    ----------
    spp_results : dict
        Output from spp_numerical_analysis() or spp_fourier_analysis()
    n_points : int, optional
        Expected number of data points. Inferred from the first available
        array when not provided.

    Returns
    -------
    dict[str, np.ndarray]
        Mapping of canonical key names to 1-D NumPy arrays, all of length
        ``n_points``. Missing keys are filled with zeros (or derived values
        where applicable).
    """
    if n_points is None:
        for key in ("Gp_t", "Gpp_t", "time_new", "strain_recon"):
            if key in spp_results:
                n_points = len(np.asarray(spp_results[key]))
                break
        else:
            # IO-R6-005: Raise instead of silently producing 15 empty arrays
            raise ValueError(
                "Cannot normalize SPP results: no length-bearing key found "
                "(expected at least one of Gp_t, Gpp_t, time_new, strain_recon). "
                f"Available keys: {sorted(spp_results.keys())}"
            )

    def _get(key: str, default_val: float = 0.0) -> np.ndarray:
        return np.asarray(spp_results.get(key, np.full(n_points, default_val)))

    Gp_t = _get("Gp_t")
    Gpp_t = _get("Gpp_t")

    return {
        "time": _get("time_new"),
        "strain": _get("strain_recon"),
        "rate": _get("rate_recon"),
        "stress": _get("stress_recon"),
        "Gp_t": Gp_t,
        "Gpp_t": Gpp_t,
        "G_star_t": np.asarray(
            spp_results.get("G_star_t", np.sqrt(Gp_t**2 + Gpp_t**2))
        ),
        "tan_delta_t": np.asarray(
            spp_results.get("tan_delta_t", Gpp_t / np.maximum(np.abs(Gp_t), 1e-12))
        ),
        "delta_t": np.asarray(
            spp_results.get(
                "delta_t",
                np.arctan(
                    spp_results.get(
                        "tan_delta_t", Gpp_t / np.maximum(np.abs(Gp_t), 1e-12)
                    )
                ),
            )
        ),
        "disp_stress": _get("disp_stress"),
        "eq_strain_est": _get("eq_strain_est"),
        "Gp_t_dot": _get("Gp_t_dot"),
        "Gpp_t_dot": _get("Gpp_t_dot"),
        "G_speed": _get("G_speed"),
        "delta_t_dot": _get("delta_t_dot"),
    }


# ============================================================================
# MATLAB-Compatible Text Export
# ============================================================================


def export_spp_txt(
    filepath: str | Path,
    spp_results: dict,
    omega: float,
    *,
    analysis_type: str = "NUMERICAL",
    n_harmonics: int | None = None,
    n_cycles: int | None = None,
    step_size: int | None = None,
    num_mode: int | None = None,
    include_fsf: bool = True,
    precision: int = 7,
) -> None:
    """
    Export SPP results to MATLAB-compatible text format.

    Creates output files matching SPPplus_print_v2.m format:
    - {base}_SPP_{analysis_type}.txt: Main SPP data (15 columns)
    - {base}_SPP_{analysis_type}_FSFRAME.txt: Frenet-Serret frame (9 columns)

    Parameters
    ----------
    filepath : str or Path
        Output file path (without extension)
    spp_results : dict
        Output from spp_numerical_analysis() or spp_fourier_analysis()
    omega : float
        Angular frequency (rad/s)
    analysis_type : str
        Analysis type identifier ("NUMERICAL" or "FOURIER")
    n_harmonics : int, optional
        Number of harmonics used (Fourier mode only)
    n_cycles : int, optional
        Number of cycles in input
    step_size : int, optional
        Step size for numerical differentiation
    num_mode : int, optional
        Numerical differentiation mode (1=standard, 2=looped)
    include_fsf : bool
        If True, also export Frenet-Serret frame data
    precision : int
        Decimal precision for floating-point output

    Examples
    --------
    >>> from rheojax.utils.spp_kernels import spp_numerical_analysis
    >>> from rheojax.io.spp_export import export_spp_txt
    >>>
    >>> # Perform SPP analysis
    >>> results = spp_numerical_analysis(time, strain, rate, stress, omega, gamma_0)
    >>>
    >>> # Export to MATLAB-compatible format
    >>> export_spp_txt("my_sample", results, omega, step_size=8, num_mode=1)
    >>> # Creates: my_sample_SPP_NUMERICAL.txt, my_sample_SPP_NUMERICAL_FSFRAME.txt
    """
    filepath = Path(filepath)
    base_name = filepath.stem

    logger.debug(
        "Extracting SPP data arrays",
        base_name=base_name,
        analysis_type=analysis_type,
        omega=omega,
    )

    # Extract data arrays via shared helper
    arrays = _extract_spp_arrays(spp_results)
    time = arrays["time"]
    strain = arrays["strain"]
    rate = arrays["rate"]
    stress = arrays["stress"]
    Gp_t = arrays["Gp_t"]
    Gpp_t = arrays["Gpp_t"]
    G_star_t = arrays["G_star_t"]
    tan_delta_t = arrays["tan_delta_t"]
    delta_t = arrays["delta_t"]
    disp_stress = arrays["disp_stress"]
    eq_strain_est = arrays["eq_strain_est"]
    Gp_t_dot = arrays["Gp_t_dot"]
    Gpp_t_dot = arrays["Gpp_t_dot"]
    G_speed = arrays["G_speed"]
    delta_t_dot = arrays["delta_t_dot"]

    logger.debug(
        "Building 15-column SPP data matrix",
        data_points=len(time),
        num_columns=15,
    )

    # Build 15-column data matrix matching MATLAB format
    spp_data_out = np.column_stack(
        [
            time,  # 1 - time [s]
            strain,  # 2 - strain [-]
            rate,  # 3 - rate [1/s]
            stress,  # 4 - stress [Pa]
            Gp_t,  # 5 - G'_t [Pa]
            Gpp_t,  # 6 - G''_t [Pa]
            G_star_t,  # 7 - |G*_t| [Pa]
            tan_delta_t,  # 8 - tan(delta_t) []
            delta_t,  # 9 - delta_t [rad]
            disp_stress,  # 10 - displacement stress [Pa]
            eq_strain_est,  # 11 - est. equilibrium strain [-]
            Gp_t_dot,  # 12 - dG'_t/dt [Pa/s]
            Gpp_t_dot,  # 13 - dG''_t/dt [Pa/s]
            G_speed,  # 14 - Speed of G*_t [Pa/s]
            delta_t_dot,  # 15 - norm. PAV []
        ]
    )

    # Write main SPP data file
    main_filename = filepath.parent / f"{base_name}_SPP_{analysis_type}.txt"
    with log_io(logger, "write", filepath=str(main_filename)) as ctx:
        _write_spp_main_txt(
            main_filename,
            spp_data_out,
            omega=omega,
            n_harmonics=n_harmonics,
            n_cycles=n_cycles,
            step_size=step_size,
            num_mode=num_mode,
            analysis_type=analysis_type,
            precision=precision,
        )
        ctx["data_rows"] = len(spp_data_out)
        ctx["columns"] = 15
        ctx["analysis_type"] = analysis_type

    # Write Frenet-Serret frame file if requested
    # IO-R6-009: Guard all three FSF vectors to prevent KeyError
    _fsf_keys_txt = ("T_vec", "N_vec", "B_vec")
    if include_fsf and all(k in spp_results for k in _fsf_keys_txt):
        logger.debug(
            "Building Frenet-Serret frame data matrix",
            data_points=len(time),
            num_columns=9,
        )
        T_vec = np.asarray(spp_results["T_vec"])
        N_vec = np.asarray(spp_results["N_vec"])
        B_vec = np.asarray(spp_results["B_vec"])

        fsf_data_out = np.column_stack(
            [
                T_vec[:, 0],
                T_vec[:, 1],
                T_vec[:, 2],  # T_x, T_y, T_z
                N_vec[:, 0],
                N_vec[:, 1],
                N_vec[:, 2],  # N_x, N_y, N_z
                B_vec[:, 0],
                B_vec[:, 1],
                B_vec[:, 2],  # B_x, B_y, B_z
            ]
        )

        fsf_filename = filepath.parent / f"{base_name}_SPP_{analysis_type}_FSFRAME.txt"
        with log_io(logger, "write", filepath=str(fsf_filename)) as ctx:
            _write_spp_fsf_txt(
                fsf_filename,
                fsf_data_out,
                omega=omega,
                n_harmonics=n_harmonics,
                n_cycles=n_cycles,
                step_size=step_size,
                num_mode=num_mode,
                analysis_type=analysis_type,
                precision=precision,
            )
            ctx["data_rows"] = len(fsf_data_out)
            ctx["columns"] = 9
            ctx["analysis_type"] = analysis_type


def _write_spp_main_txt(
    filepath: Path,
    data: np.ndarray,
    *,
    omega: float,
    n_harmonics: int | None,
    n_cycles: int | None,
    step_size: int | None,
    num_mode: int | None,
    analysis_type: str,
    precision: int,
) -> None:
    """Write main SPP data file in MATLAB format."""
    tmp_path = filepath.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", newline="") as f:
            # Write analysis type header
            if analysis_type == "NUMERICAL":
                f.write("Data calculated via numerical differentiation\r\n")
            else:
                f.write("Data calculated via Fourier domain filtering\r\n")

            # Write parameters
            f.write(f"Angular Frequency (rad/s):\t{omega:.{precision}f}\r\n")
            if n_harmonics is not None:
                f.write(f"Number of harmonics used:\t{n_harmonics}\r\n")
            if n_cycles is not None:
                f.write(f"Number of cycles in input:\t{n_cycles}\r\n")
            if step_size is not None:
                f.write(f"Step size for numerical diff.:\t{step_size}\r\n")
            if num_mode is not None:
                if num_mode == 1:
                    f.write("Standard differentiation\r\n")
                elif num_mode == 2:
                    f.write("Looped differentiation\r\n")

            # Write column headers (matching MATLAB exactly)
            headers1 = [
                "Time",
                "Strain",
                "Rate",
                "Stress",
                "G'_t",
                'G"_t',
                "|G*_t|",
                "tan(delta_t)",
                "delta_t",
                "displacement stress",
                "est. elastic stress",
                "dG'_{t}/dt",
                'dG"_{t}/dt',
                "Speed",
                "norm. PAV",
            ]
            headers2 = [
                "[s]",
                "[-]",
                "[1/s]",
                "[Pa]",
                "[Pa]",
                "[Pa]",
                "[Pa]",
                "[]",
                "[rad]",
                "[Pa]",
                "[Pa]",
                "[Pa/s]",
                "[Pa/s]",
                "[Pa/s]",
                "[]",
            ]
            f.write("\t".join(headers1) + "\r\n")
            f.write("\t".join(headers2) + "\r\n")

            # Write data rows
            fmt = f"%.{precision}f"
            for row in data:
                f.write("\t".join(fmt % val for val in row) + "\r\n")
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _write_spp_fsf_txt(
    filepath: Path,
    data: np.ndarray,
    *,
    omega: float,
    n_harmonics: int | None,
    n_cycles: int | None,
    step_size: int | None,
    num_mode: int | None,
    analysis_type: str,
    precision: int,
) -> None:
    """Write Frenet-Serret frame data file in MATLAB format."""
    tmp_path = filepath.with_suffix(".tmp")
    try:
        with open(tmp_path, "w", newline="") as f:
            # Write analysis type header
            if analysis_type == "NUMERICAL":
                f.write("Data calculated via numerical differentiation\r\n")
            else:
                f.write("Data calculated via Fourier domain filtering\r\n")

            # Write parameters
            f.write(f"Angular Frequency (rad/s):\t{omega:.{precision}f}\r\n")
            if n_harmonics is not None:
                f.write(f"Number of harmonics used:\t{n_harmonics}\r\n")
            if n_cycles is not None:
                f.write(f"Number of cycles in input:\t{n_cycles}\r\n")
            if step_size is not None:
                f.write(f"Step size for numerical diff.:\t{step_size}\r\n")
            if num_mode is not None:
                if num_mode == 1:
                    f.write("Standard differentiation\r\n")
                elif num_mode == 2:
                    f.write("Looped differentiation\r\n")

            # Write column headers (matching MATLAB exactly)
            headers1 = [
                "Tangent(x)",
                "Tangent(y)",
                "Tangent(z)",
                "Normal(x)",
                "Normal(y)",
                "Normal(z)",
                "Binormal(x)",
                "Binormal(y)",
                "Binormal(z)",
            ]
            headers2 = [
                "[-]",
                "[1/s]",
                "[Pa]",
                "[-]",
                "[1/s]",
                "[Pa]",
                "[-]",
                "[1/s]",
                "[Pa]",
            ]
            f.write("\t".join(headers1) + "\r\n")
            f.write("\t".join(headers2) + "\r\n")

            # Write data rows
            fmt = f"%.{precision}f"
            for row in data:
                f.write("\t".join(fmt % val for val in row) + "\r\n")
        os.replace(tmp_path, filepath)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


# ============================================================================
# HDF5 Export
# ============================================================================


def export_spp_hdf5(
    filepath: str | Path,
    spp_results: dict,
    omega: float,
    gamma_0: float,
    *,
    analysis_type: str = "numerical",
    metadata: dict | None = None,
) -> None:
    """
    Export SPP results to HDF5 format.

    Creates a hierarchical structure with:
    - /spp_data: Main SPP arrays (Gp_t, Gpp_t, etc.)
    - /waveforms: Reconstructed signals (strain, rate, stress)
    - /frenet_serret: T, N, B frame vectors
    - /metadata: Analysis parameters

    Parameters
    ----------
    filepath : str or Path
        Output file path (.h5 extension)
    spp_results : dict
        Output from spp_numerical_analysis() or spp_fourier_analysis()
    omega : float
        Angular frequency (rad/s)
    gamma_0 : float
        Strain amplitude
    analysis_type : str
        Analysis type identifier ("numerical" or "fourier")
    metadata : dict, optional
        Additional metadata to include

    Examples
    --------
    >>> from rheojax.io.spp_export import export_spp_hdf5
    >>> export_spp_hdf5("results.h5", spp_results, omega=1.0, gamma_0=0.5)
    """
    try:
        import h5py
    except ImportError as e:
        logger.error(
            "h5py import failed",
            error_type="ImportError",
            suggestion="pip install h5py",
            exc_info=True,
        )
        raise ImportError(
            "h5py is required for HDF5 export. Install with: pip install h5py"
        ) from e

    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".h5")

    with log_io(logger, "write", filepath=str(filepath)) as ctx:
        datasets_written = []

        filepath.parent.mkdir(parents=True, exist_ok=True)
        tmp_fd, tmp_path = tempfile.mkstemp(
            dir=str(filepath.parent), suffix=".hdf5.tmp"
        )
        try:
            os.close(tmp_fd)
        except OSError:
            os.unlink(tmp_path)
            raise
        try:
            with h5py.File(tmp_path, "w") as f:
                # Metadata group
                logger.debug(
                    "Writing metadata group",
                    omega=omega,
                    gamma_0=gamma_0,
                    analysis_type=analysis_type,
                )
                meta = f.create_group("metadata")
                meta.attrs["omega"] = omega
                meta.attrs["gamma_0"] = gamma_0
                meta.attrs["analysis_type"] = analysis_type
                if "Delta" in spp_results:
                    meta.attrs["phase_offset_Delta"] = float(spp_results["Delta"])
                if metadata:
                    for key, value in metadata.items():
                        if isinstance(value, dict):
                            meta.attrs[key] = json.dumps(value)
                        elif isinstance(
                            value, (str, int, float, bool, np.integer, np.floating)
                        ):
                            meta.attrs[key] = value
                        else:
                            meta.attrs[key] = str(value)
                    logger.debug(
                        "Custom metadata stored",
                        metadata_keys=list(metadata.keys()),
                    )

                # Main SPP data group
                logger.debug("Writing spp_data group")
                spp_data = f.create_group("spp_data")
                spp_keys_written = []
                for key in [
                    "Gp_t",
                    "Gpp_t",
                    "G_star_t",
                    "tan_delta_t",
                    "delta_t",
                    "disp_stress",
                    "eq_strain_est",
                    "Gp_t_dot",
                    "Gpp_t_dot",
                    "G_speed",
                    "delta_t_dot",
                ]:
                    if key in spp_results:
                        spp_data.create_dataset(key, data=np.asarray(spp_results[key]))
                        spp_keys_written.append(key)
                datasets_written.extend(spp_keys_written)
                logger.debug("SPP data datasets written", datasets=spp_keys_written)

                # Waveforms group
                logger.debug("Writing waveforms group")
                waveforms = f.create_group("waveforms")
                waveform_keys_written = []
                for key in ["time_new", "strain_recon", "rate_recon", "stress_recon"]:
                    if key in spp_results:
                        waveforms.create_dataset(key, data=np.asarray(spp_results[key]))
                        waveform_keys_written.append(key)
                datasets_written.extend(waveform_keys_written)
                logger.debug(
                    "Waveform datasets written", datasets=waveform_keys_written
                )

                # Frenet-Serret frame group
                # IO-R6-009: Guard all three FSF vectors — if T_vec is present
                # but N_vec/B_vec are missing, KeyError would destroy the entire
                # HDF5 write (including valid spp_data already written).
                _fsf_keys = ("T_vec", "N_vec", "B_vec")
                if all(k in spp_results for k in _fsf_keys):
                    logger.debug("Writing frenet_serret group")
                    fsf = f.create_group("frenet_serret")
                    for k in _fsf_keys:
                        fsf.create_dataset(k, data=np.asarray(spp_results[k]))
                    datasets_written.extend(["T_vec", "N_vec", "B_vec"])
                    logger.debug("Frenet-Serret frame datasets written")

            os.replace(tmp_path, str(filepath))
            tmp_path = None  # type: ignore[assignment]  # prevent cleanup
        finally:
            if tmp_path is not None:
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

        ctx["datasets_written"] = len(datasets_written)
        ctx["has_fsf"] = "T_vec" in spp_results
        ctx["analysis_type"] = analysis_type


# ============================================================================
# CSV Export
# ============================================================================


def export_spp_csv(
    filepath: str | Path,
    spp_results: dict,
    *,
    include_fsf: bool = False,
) -> None:
    """
    Export SPP results to CSV format.

    Parameters
    ----------
    filepath : str or Path
        Output file path (.csv extension)
    spp_results : dict
        Output from spp_numerical_analysis() or spp_fourier_analysis()
    include_fsf : bool
        If True, include Frenet-Serret frame columns

    Examples
    --------
    >>> from rheojax.io.spp_export import export_spp_csv
    >>> export_spp_csv("results.csv", spp_results)
    """
    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".csv")

    logger.debug(
        "Building CSV column structure",
        include_fsf=include_fsf,
    )

    # Build column dict
    columns = {}
    if "time_new" in spp_results:
        time = np.asarray(spp_results["time_new"])
    elif "Gp_t" in spp_results:
        time = np.arange(len(spp_results["Gp_t"]))
    else:
        raise ValueError(
            "spp_results must contain 'time_new' or 'Gp_t' to determine array length"
        )
    columns["time"] = time

    for key in [
        "strain_recon",
        "rate_recon",
        "stress_recon",
        "Gp_t",
        "Gpp_t",
        "G_star_t",
        "tan_delta_t",
        "delta_t",
        "disp_stress",
        "eq_strain_est",
        "Gp_t_dot",
        "Gpp_t_dot",
        "G_speed",
        "delta_t_dot",
    ]:
        if key in spp_results:
            columns[key] = np.asarray(spp_results[key])

    # IO-R6-009: Guard all three FSF vectors to prevent KeyError
    _fsf_keys_csv = ("T_vec", "N_vec", "B_vec")
    if include_fsf and all(k in spp_results for k in _fsf_keys_csv):
        logger.debug("Including Frenet-Serret frame columns")
        T_vec = np.asarray(spp_results["T_vec"])
        N_vec = np.asarray(spp_results["N_vec"])
        B_vec = np.asarray(spp_results["B_vec"])
        columns["T_x"] = T_vec[:, 0]
        columns["T_y"] = T_vec[:, 1]
        columns["T_z"] = T_vec[:, 2]
        columns["N_x"] = N_vec[:, 0]
        columns["N_y"] = N_vec[:, 1]
        columns["N_z"] = N_vec[:, 2]
        columns["B_x"] = B_vec[:, 0]
        columns["B_y"] = B_vec[:, 1]
        columns["B_z"] = B_vec[:, 2]

    with log_io(logger, "write", filepath=str(filepath)) as ctx:
        # Atomic write: write to temp file, then os.replace()
        tmp_path = filepath.with_suffix(".csv.tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8", newline="") as f:
                # Header
                logger.debug(
                    "Writing CSV header",
                    num_columns=len(columns),
                    column_names=list(columns.keys()),
                )
                # R11-SPP-IO-001: Use CRLF to match MATLAB convention
                # (consistent with export_spp_txt)
                f.write(",".join(columns.keys()) + "\r\n")
                # Data
                data = np.column_stack(list(columns.values()))
                logger.debug(
                    "Writing CSV data rows",
                    data_shape=data.shape,
                )
                for row in data:
                    f.write(",".join(f"{val:.7g}" for val in row) + "\r\n")
            os.replace(tmp_path, filepath)
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            raise

        ctx["data_rows"] = len(data)
        ctx["columns"] = len(columns)
        ctx["include_fsf"] = include_fsf


# ============================================================================
# MAT-Compatible Dict (for scipy.io.savemat)
# ============================================================================


def to_matlab_dict(
    spp_results: dict,
    omega: float,
    *,
    analysis_type: str = "numerical",
    n_harmonics: int | None = None,
    n_cycles: int | None = None,
    step_size: int | None = None,
    num_mode: int | None = None,
) -> dict:
    """
    Convert SPP results to a dict compatible with scipy.io.savemat.

    Returns a structure matching MATLAB SPPplus .mat file format:
    - out_spp.info: Analysis parameters
    - out_spp.headers: Column headers
    - out_spp.data: 15-column data matrix

    Parameters
    ----------
    spp_results : dict
        Output from spp_numerical_analysis() or spp_fourier_analysis()
    omega : float
        Angular frequency (rad/s)
    analysis_type : str
        Analysis type identifier ("numerical" or "fourier")
    n_harmonics : int, optional
        Number of harmonics used (Fourier mode only)
    n_cycles : int, optional
        Number of cycles in input
    step_size : int, optional
        Step size for numerical differentiation
    num_mode : int, optional
        Numerical differentiation mode (1=standard, 2=looped)

    Returns
    -------
    dict
        Dictionary ready for scipy.io.savemat()

    Examples
    --------
    >>> from scipy.io import savemat
    >>> from rheojax.io.spp_export import to_matlab_dict
    >>>
    >>> mat_data = to_matlab_dict(spp_results, omega=1.0, step_size=8)
    >>> savemat("results.mat", mat_data)
    """
    logger.debug(
        "Converting SPP results to MATLAB dict",
        analysis_type=analysis_type,
        omega=omega,
    )

    # Extract data arrays via shared helper
    arrays = _extract_spp_arrays(spp_results)
    time = arrays["time"]
    strain = arrays["strain"]
    rate = arrays["rate"]
    stress = arrays["stress"]
    Gp_t = arrays["Gp_t"]
    Gpp_t = arrays["Gpp_t"]
    G_star_t = arrays["G_star_t"]
    tan_delta_t = arrays["tan_delta_t"]
    delta_t = arrays["delta_t"]
    disp_stress = arrays["disp_stress"]
    eq_strain_est = arrays["eq_strain_est"]
    Gp_t_dot = arrays["Gp_t_dot"]
    Gpp_t_dot = arrays["Gpp_t_dot"]
    G_speed = arrays["G_speed"]
    delta_t_dot = arrays["delta_t_dot"]

    logger.debug(
        "Building MATLAB 15-column data matrix",
        data_points=len(time),
    )

    # Build 15-column data matrix
    spp_data_out = np.column_stack(
        [
            time,
            strain,
            rate,
            stress,
            Gp_t,
            Gpp_t,
            G_star_t,
            tan_delta_t,
            delta_t,
            disp_stress,
            eq_strain_est,
            Gp_t_dot,
            Gpp_t_dot,
            G_speed,
            delta_t_dot,
        ]
    )

    # Build info struct
    info: dict[str, str | float | int] = {"frequency": omega}
    if analysis_type == "numerical":
        info["data_calc"] = "Data calculated via numerical differentiation"
    else:
        info["data_calc"] = "Data calculated via Fourier domain filtering"
    if n_harmonics is not None:
        info["number_of_harmonics"] = n_harmonics
    if n_cycles is not None:
        info["number_of_cycles"] = n_cycles
    if step_size is not None:
        info["diff_step_size"] = step_size
    if num_mode is not None:
        info["diff_type"] = (
            "Standard differentiation" if num_mode == 1 else "Looped differentiation"
        )

    # Headers matching MATLAB format
    headers = np.array(
        [
            [
                "Time",
                "Strain",
                "Rate",
                "Stress",
                "G'_t",
                'G"_t',
                "|G*_t|",
                "tan(delta_t)",
                "delta_t",
                "displacement stress",
                "est. elastic stress",
                "dG'_{t}/dt",
                'dG"_{t}/dt',
                "Speed",
                "norm. PAV",
            ],
            [
                "[s]",
                "[-]",
                "[1/s]",
                "[Pa]",
                "[Pa]",
                "[Pa]",
                "[Pa]",
                "[]",
                "[rad]",
                "[Pa]",
                "[Pa]",
                "[Pa/s]",
                "[Pa/s]",
                "[Pa/s]",
                "[]",
            ],
        ],
        dtype=object,
    )

    # Main output structure
    out_spp = {
        "info": info,
        "headers": headers,
        "data": spp_data_out,
    }
    logger.debug("SPP structure built", data_shape=spp_data_out.shape)

    # Build FSF structure if available
    result = {"out_spp": out_spp}

    # IO-R6-009: Guard all three FSF vectors to prevent KeyError
    _fsf_keys_mat = ("T_vec", "N_vec", "B_vec")
    if all(k in spp_results for k in _fsf_keys_mat):
        logger.debug("Building Frenet-Serret frame structure")
        T_vec = np.asarray(spp_results["T_vec"])
        N_vec = np.asarray(spp_results["N_vec"])
        B_vec = np.asarray(spp_results["B_vec"])

        fsf_data_out = np.column_stack(
            [
                T_vec[:, 0],
                T_vec[:, 1],
                T_vec[:, 2],
                N_vec[:, 0],
                N_vec[:, 1],
                N_vec[:, 2],
                B_vec[:, 0],
                B_vec[:, 1],
                B_vec[:, 2],
            ]
        )

        fsf_headers = np.array(
            [
                [
                    "Tangent(x)",
                    "Tangent(y)",
                    "Tangent(z)",
                    "Normal(x)",
                    "Normal(y)",
                    "Normal(z)",
                    "Binormal(x)",
                    "Binormal(y)",
                    "Binormal(z)",
                ],
                [
                    "[-]",
                    "[1/s]",
                    "[Pa]",
                    "[-]",
                    "[1/s]",
                    "[Pa]",
                    "[-]",
                    "[1/s]",
                    "[Pa]",
                ],
            ],
            dtype=object,
        )

        out_fsf = {
            "info": info.copy(),
            "headers": fsf_headers,
            "data": fsf_data_out,
        }
        result["out_fsf"] = out_fsf
        logger.debug("FSF structure built", data_shape=fsf_data_out.shape)

    logger.debug(
        "MATLAB dict conversion completed",
        output_keys=list(result.keys()),
        has_fsf="out_fsf" in result,
    )
    return result


# ============================================================================
# Convenience Exports
# ============================================================================

__all__ = [
    "export_spp_txt",
    "export_spp_hdf5",
    "export_spp_csv",
    "to_matlab_dict",
]
