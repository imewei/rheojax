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

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    pass


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

    # Extract data arrays
    time = np.asarray(spp_results.get("time_new", np.arange(len(spp_results["Gp_t"]))))
    strain = np.asarray(spp_results.get("strain_recon", np.zeros(len(time))))
    rate = np.asarray(spp_results.get("rate_recon", np.zeros(len(time))))
    stress = np.asarray(spp_results.get("stress_recon", np.zeros(len(time))))
    Gp_t = np.asarray(spp_results["Gp_t"])
    Gpp_t = np.asarray(spp_results["Gpp_t"])
    G_star_t = np.asarray(spp_results.get("G_star_t", np.sqrt(Gp_t**2 + Gpp_t**2)))
    tan_delta_t = np.asarray(
        spp_results.get("tan_delta_t", Gpp_t / np.maximum(np.abs(Gp_t), 1e-12))
    )
    delta_t = np.asarray(spp_results.get("delta_t", np.arctan(tan_delta_t)))
    disp_stress = np.asarray(spp_results.get("disp_stress", np.zeros(len(time))))
    eq_strain_est = np.asarray(spp_results.get("eq_strain_est", np.zeros(len(time))))
    Gp_t_dot = np.asarray(spp_results.get("Gp_t_dot", np.zeros(len(time))))
    Gpp_t_dot = np.asarray(spp_results.get("Gpp_t_dot", np.zeros(len(time))))
    G_speed = np.asarray(spp_results.get("G_speed", np.zeros(len(time))))
    delta_t_dot = np.asarray(spp_results.get("delta_t_dot", np.zeros(len(time))))

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

    # Write Frenet-Serret frame file if requested
    if include_fsf and "T_vec" in spp_results:
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
    with open(filepath, "w", newline="\r\n") as f:
        # Write analysis type header
        if analysis_type == "NUMERICAL":
            f.write("Data calculated via numerical differentiation\r\n")
        else:
            f.write("Data calculated via Fourier domain filtering\r\n")

        # Write parameters
        f.write(f"Frequency:\t{omega:.{precision}f}\r\n")
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
    with open(filepath, "w", newline="\r\n") as f:
        # Write analysis type header
        if analysis_type == "NUMERICAL":
            f.write("Data calculated via numerical differentiation\r\n")
        else:
            f.write("Data calculated via Fourier domain filtering\r\n")

        # Write parameters
        f.write(f"Frequency:\t{omega:.{precision}f}\r\n")
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
        raise ImportError(
            "h5py is required for HDF5 export. Install with: pip install h5py"
        ) from e

    filepath = Path(filepath)
    if not filepath.suffix:
        filepath = filepath.with_suffix(".h5")

    with h5py.File(filepath, "w") as f:
        # Metadata group
        meta = f.create_group("metadata")
        meta.attrs["omega"] = omega
        meta.attrs["gamma_0"] = gamma_0
        meta.attrs["analysis_type"] = analysis_type
        if "Delta" in spp_results:
            meta.attrs["phase_offset_Delta"] = float(spp_results["Delta"])
        if metadata:
            for key, value in metadata.items():
                meta.attrs[key] = value

        # Main SPP data group
        spp_data = f.create_group("spp_data")
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

        # Waveforms group
        waveforms = f.create_group("waveforms")
        for key in ["time_new", "strain_recon", "rate_recon", "stress_recon"]:
            if key in spp_results:
                waveforms.create_dataset(key, data=np.asarray(spp_results[key]))

        # Frenet-Serret frame group
        if "T_vec" in spp_results:
            fsf = f.create_group("frenet_serret")
            fsf.create_dataset("T_vec", data=np.asarray(spp_results["T_vec"]))
            fsf.create_dataset("N_vec", data=np.asarray(spp_results["N_vec"]))
            fsf.create_dataset("B_vec", data=np.asarray(spp_results["B_vec"]))


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

    # Build column dict
    columns = {}
    time = np.asarray(spp_results.get("time_new", np.arange(len(spp_results["Gp_t"]))))
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

    if include_fsf and "T_vec" in spp_results:
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

    # Write to CSV
    with open(filepath, "w") as f:
        # Header
        f.write(",".join(columns.keys()) + "\n")
        # Data
        data = np.column_stack(list(columns.values()))
        for row in data:
            f.write(",".join(f"{val:.7g}" for val in row) + "\n")


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
    # Extract data arrays
    time = np.asarray(spp_results.get("time_new", np.arange(len(spp_results["Gp_t"]))))
    strain = np.asarray(spp_results.get("strain_recon", np.zeros(len(time))))
    rate = np.asarray(spp_results.get("rate_recon", np.zeros(len(time))))
    stress = np.asarray(spp_results.get("stress_recon", np.zeros(len(time))))
    Gp_t = np.asarray(spp_results["Gp_t"])
    Gpp_t = np.asarray(spp_results["Gpp_t"])
    G_star_t = np.asarray(spp_results.get("G_star_t", np.sqrt(Gp_t**2 + Gpp_t**2)))
    tan_delta_t = np.asarray(
        spp_results.get("tan_delta_t", Gpp_t / np.maximum(np.abs(Gp_t), 1e-12))
    )
    delta_t = np.asarray(spp_results.get("delta_t", np.arctan(tan_delta_t)))
    disp_stress = np.asarray(spp_results.get("disp_stress", np.zeros(len(time))))
    eq_strain_est = np.asarray(spp_results.get("eq_strain_est", np.zeros(len(time))))
    Gp_t_dot = np.asarray(spp_results.get("Gp_t_dot", np.zeros(len(time))))
    Gpp_t_dot = np.asarray(spp_results.get("Gpp_t_dot", np.zeros(len(time))))
    G_speed = np.asarray(spp_results.get("G_speed", np.zeros(len(time))))
    delta_t_dot = np.asarray(spp_results.get("delta_t_dot", np.zeros(len(time))))

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

    # Build FSF structure if available
    result = {"out_spp": out_spp}

    if "T_vec" in spp_results:
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
            "info": info,
            "headers": fsf_headers,
            "data": fsf_data_out,
        }
        result["out_fsf"] = out_fsf

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
