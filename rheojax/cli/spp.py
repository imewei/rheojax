"""SPP (Sequence of Physical Processes) CLI commands.

This module provides command-line tools for SPP analysis of LAOS data.

Example usage:
    rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5 --output results.csv
    rheojax spp batch data_dir/ --omega 1.0 --output-dir results/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from rheojax.core.jax_config import safe_import_jax
from rheojax.io import auto_load as load_data
from rheojax.logging import get_logger
from rheojax.transforms.spp_decomposer import SPPDecomposer

if TYPE_CHECKING:
    from argparse import Namespace

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for SPP CLI commands."""
    parser = argparse.ArgumentParser(
        prog="rheojax spp",
        description="SPP (Sequence of Physical Processes) analysis for LAOS data",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Analyze single file
  rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5

  # Analyze with Bayesian inference
  rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5 --bayesian

  # Batch processing
  rheojax spp batch data_dir/ --omega 1.0 --output-dir results/

  # Export MATLAB-compatible format
  rheojax spp analyze data.csv --omega 1.0 --gamma-0 0.5 --export-matlab
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="SPP commands")

    # Analyze command
    analyze_parser = subparsers.add_parser(
        "analyze",
        help="Analyze single LAOS dataset with SPP",
    )
    _add_analyze_args(analyze_parser)

    # Batch command
    batch_parser = subparsers.add_parser(
        "batch",
        help="Batch process multiple LAOS datasets",
    )
    _add_batch_args(batch_parser)

    return parser


def _add_analyze_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for analyze command."""
    parser.add_argument(
        "input_file",
        type=Path,
        help="Input data file (CSV, Excel, or TRIOS format)",
    )
    parser.add_argument(
        "--omega",
        type=float,
        required=True,
        help="Angular frequency (rad/s)",
    )
    parser.add_argument(
        "--gamma-0",
        type=float,
        required=True,
        help="Strain amplitude (dimensionless)",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=39,
        help="Number of harmonics for analysis (default: 39)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=8,
        help="Differentiation step size (default: 8)",
    )
    parser.add_argument(
        "--numerical",
        action="store_true",
        help="Use numerical differentiation method",
    )
    parser.add_argument(
        "--bayesian",
        action="store_true",
        help="Run Bayesian inference on SPP parameters",
    )
    parser.add_argument(
        "--num-warmup",
        type=int,
        default=1000,
        help="NUTS warmup samples (default: 1000)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=2000,
        help="NUTS posterior samples (default: 2000)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Output file path (default: input_spp.csv)",
    )
    parser.add_argument(
        "--export-matlab",
        action="store_true",
        help="Export MATLAB-compatible spp_data_out format",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default="time",
        help="Column name for time/x data (default: time)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default="stress",
        help="Column name for stress/y data (default: stress)",
    )
    parser.add_argument(
        "--strain-col",
        type=str,
        help="Column name for strain data (optional)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )


def _add_batch_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for batch command."""
    parser.add_argument(
        "input_dir",
        type=Path,
        help="Directory containing input data files",
    )
    parser.add_argument(
        "--omega",
        type=float,
        required=True,
        help="Angular frequency (rad/s)",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        help="Output directory (default: input_dir/spp_results)",
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.csv",
        help="File pattern to match (default: *.csv)",
    )
    parser.add_argument(
        "--n-harmonics",
        type=int,
        default=39,
        help="Number of harmonics for analysis (default: 39)",
    )
    parser.add_argument(
        "--step-size",
        type=int,
        default=8,
        help="Differentiation step size (default: 8)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output",
    )


def run_analyze(args: Namespace) -> int:
    """Run SPP analysis on a single file."""
    logger.info(
        "Running SPP analyze command",
        command="spp analyze",
        input_file=str(args.input_file),
    )
    logger.debug(
        "SPP analyze parameters",
        omega=args.omega,
        gamma_0=args.gamma_0,
        n_harmonics=args.n_harmonics,
        step_size=args.step_size,
        numerical=args.numerical,
        bayesian=args.bayesian,
        x_col=args.x_col,
        y_col=args.y_col,
    )

    if args.verbose:
        print(f"Loading data from {args.input_file}...")

    # Load data
    try:
        logger.debug("Loading data file", file=str(args.input_file))
        loaded = load_data(
            str(args.input_file),
            x_col=args.x_col,
            y_col=args.y_col,
        )
        # Handle potential list return (take first dataset if list)
        if isinstance(loaded, list):
            rheo_data = loaded[0]
        else:
            rheo_data = loaded
        logger.debug("Data loaded successfully", data_points=len(rheo_data.x))
    except Exception as e:
        logger.error("Failed to load data", file=str(args.input_file), exc_info=True)
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    # Add metadata
    rheo_data.metadata["omega"] = args.omega
    rheo_data.metadata["gamma_0"] = args.gamma_0
    rheo_data.metadata["test_mode"] = "oscillation"

    if args.verbose:
        print(f"  Data points: {len(rheo_data.x)}")
        print(f"  Omega: {args.omega} rad/s")
        print(f"  Gamma_0: {args.gamma_0}")

    # Create SPP decomposer
    spp = SPPDecomposer(
        omega=args.omega,
        gamma_0=args.gamma_0,
        n_harmonics=args.n_harmonics,
        step_size=args.step_size,
        use_numerical_method=args.numerical,
    )

    # Run analysis
    if args.verbose:
        print("Running SPP analysis...")

    try:
        logger.debug("Starting SPP transform")
        spp.transform(rheo_data)
        results = spp.get_results()
        logger.debug(
            "SPP analysis completed",
            sigma_sy=results.get("sigma_sy"),
            sigma_dy=results.get("sigma_dy"),
            G_cage_mean=results.get("G_cage_mean"),
        )
    except Exception as e:
        logger.error("SPP analysis failed", exc_info=True)
        print(f"Error in SPP analysis: {e}", file=sys.stderr)
        return 1

    # Print summary
    print("\n=== SPP Analysis Results ===")
    print(f"Static yield stress (σ_sy):  {results['sigma_sy']:.2f} Pa")
    print(f"Dynamic yield stress (σ_dy): {results['sigma_dy']:.2f} Pa")
    print(f"Cage modulus (G_cage):       {results['G_cage_mean']:.2f} Pa")
    if "I3_I1_ratio" in results:
        print(f"I3/I1 ratio:                 {results['I3_I1_ratio']:.4f}")

    # Output file
    if args.output:
        output_path = args.output
    else:
        output_path = args.input_file.with_suffix("").with_name(
            f"{args.input_file.stem}_spp.csv"
        )

    # Save results
    if args.verbose:
        print(f"\nSaving results to {output_path}...")

    try:
        logger.debug(
            "Saving results",
            output_path=str(output_path),
            matlab_format=args.export_matlab,
        )
        _save_results(results, output_path, args.export_matlab, omega=args.omega)
        logger.info("Results saved successfully", output_path=str(output_path))
    except Exception as e:
        logger.error(
            "Failed to save results", output_path=str(output_path), exc_info=True
        )
        print(f"Error saving results: {e}", file=sys.stderr)
        return 1

    print(f"\nResults saved to: {output_path}")

    # Bayesian inference (optional)
    if args.bayesian:
        if args.verbose:
            print("\nRunning Bayesian inference...")

        logger.info(
            "Starting Bayesian inference",
            num_warmup=args.num_warmup,
            num_samples=args.num_samples,
        )

        try:
            from rheojax.models.spp_yield_stress import SPPYieldStress

            model = SPPYieldStress()
            gamma_0_array = np.array([args.gamma_0])
            sigma_sy_array = np.array([results["sigma_sy"]])

            model.fit(gamma_0_array, sigma_sy_array, test_mode="oscillation")
            bayes_result = model.fit_bayesian(
                gamma_0_array,
                sigma_sy_array,
                test_mode="oscillation",
                num_warmup=args.num_warmup,
                num_samples=args.num_samples,
            )

            logger.debug("Bayesian inference completed successfully")

            print("\n=== Bayesian Inference Results ===")
            print(f"Posterior samples: {args.num_samples}")
            if hasattr(bayes_result, "summary"):
                print(bayes_result.summary)

        except Exception as e:
            logger.error("Bayesian inference failed", exc_info=True)
            print(f"Bayesian inference failed: {e}", file=sys.stderr)
            # Don't fail the whole command if Bayesian fails

    return 0


def run_batch(args: Namespace) -> int:
    """Run SPP analysis on multiple files."""
    input_dir = args.input_dir
    if not input_dir.is_dir():
        print(f"Error: {input_dir} is not a directory", file=sys.stderr)
        return 1

    # Find files
    files = list(input_dir.glob(args.pattern))
    if not files:
        print(f"No files matching {args.pattern} in {input_dir}", file=sys.stderr)
        return 1

    print(f"Found {len(files)} files to process")

    # Output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = input_dir / "spp_results"

    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each file
    success_count = 0
    for i, file_path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] Processing {file_path.name}...", end=" ")

        try:
            loaded = load_data(str(file_path))
            # Handle potential list return (take first dataset if list)
            if isinstance(loaded, list):
                rheo_data = loaded[0]
            else:
                rheo_data = loaded
            rheo_data.metadata["omega"] = args.omega
            rheo_data.metadata["test_mode"] = "oscillation"

            # Extract gamma_0 from filename or use default
            gamma_0 = rheo_data.metadata.get("gamma_0", 1.0)

            spp = SPPDecomposer(
                omega=args.omega,
                gamma_0=gamma_0,
                n_harmonics=args.n_harmonics,
                step_size=args.step_size,
            )
            spp.transform(rheo_data)
            results = spp.get_results()

            output_path = output_dir / f"{file_path.stem}_spp.csv"
            _save_results(results, output_path, matlab_format=False, omega=args.omega)

            print(f"σ_sy={results['sigma_sy']:.1f} Pa")
            success_count += 1

        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\nCompleted: {success_count}/{len(files)} files processed successfully")
    print(f"Results saved to: {output_dir}")

    return 0 if success_count == len(files) else 1


def _save_results(
    results: dict,
    output_path: Path,
    matlab_format: bool = False,
    omega: float = 1.0,
) -> None:
    """Save SPP results to file."""
    import pandas as pd

    if matlab_format:
        # MATLAB-compatible spp_data_out format
        from rheojax.io.spp_export import export_spp_txt

        export_spp_txt(str(output_path), results, omega)
    else:
        # Standard CSV format
        # Extract scalar metrics
        metrics = {
            "sigma_sy": results.get("sigma_sy", np.nan),
            "sigma_dy": results.get("sigma_dy", np.nan),
            "G_cage_mean": results.get("G_cage_mean", np.nan),
            "Gp_t_mean": results.get("Gp_t_mean", np.nan),
            "Gpp_t_mean": results.get("Gpp_t_mean", np.nan),
            "I3_I1_ratio": results.get("I3_I1_ratio", np.nan),
            "S_factor": results.get("S_factor", np.nan),
            "T_factor": results.get("T_factor", np.nan),
        }

        df = pd.DataFrame([metrics])
        df.to_csv(output_path, index=False)


def main(args: list[str] | None = None) -> int:
    """Main entry point for SPP CLI."""
    parser = create_parser()
    parsed_args = parser.parse_args(args)

    if parsed_args.command is None:
        parser.print_help()
        return 0

    if parsed_args.command == "analyze":
        return run_analyze(parsed_args)
    elif parsed_args.command == "batch":
        return run_batch(parsed_args)
    else:
        print(f"Unknown command: {parsed_args.command}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
