"""CLI subcommand for Bayesian inference with NUTS sampling.

Usage:
    rheojax bayesian data.csv --model maxwell --x-col time --y-col modulus
    rheojax bayesian data.csv --model maxwell --warmup 1000 --samples 2000 --chains 4
    rheojax bayesian data.trios --model springpot --warm-start
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

from rheojax.core.jax_config import safe_import_jax
from rheojax.logging import get_logger

if TYPE_CHECKING:
    pass

jax, jnp = safe_import_jax()

logger = get_logger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """Create argument parser for bayesian subcommand."""
    parser = argparse.ArgumentParser(
        prog="rheojax bayesian",
        description="Bayesian inference (NUTS) for rheological models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Bayesian inference with NLSQ warm-start (recommended)
  rheojax bayesian data.csv --model maxwell --x-col time --y-col G_t --warm-start

  # Quick demo (1 chain, fewer samples)
  rheojax bayesian data.csv --model maxwell --x-col time --y-col G_t --chains 1 --samples 500

  # Full production run
  rheojax bayesian data.csv --model springpot --x-col freq --y-cols "G_prime,G_double_prime" \\
      --test-mode oscillation --warmup 1000 --samples 2000 --chains 4 --warm-start

  # Output as JSON
  rheojax bayesian data.csv --model maxwell --x-col time --y-col G_t --json

  # DMTA data with deformation mode
  rheojax bayesian dmta.csv --model maxwell --x-col freq --y-cols "E_prime,E_double_prime" \\
      --deformation-mode tension --poisson-ratio 0.5 --warm-start
        """,
    )

    parser.add_argument(
        "input_file",
        type=Path,
        help="Input data file (CSV, Excel, or TRIOS format)",
    )
    parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Model name (e.g., maxwell, springpot, fractional_zener_solid_solid)",
    )
    parser.add_argument(
        "--test-mode",
        "-t",
        type=str,
        default=None,
        help="Test mode (oscillation, relaxation, creep, flow_curve, startup, laos)",
    )
    parser.add_argument(
        "--x-col",
        type=str,
        default=None,
        help="X column name or index (for CSV/Excel)",
    )
    parser.add_argument(
        "--y-col",
        type=str,
        default=None,
        help="Y column name or index (for CSV/Excel, single-column data)",
    )
    parser.add_argument(
        "--y-cols",
        type=str,
        default=None,
        help="Comma-separated Y column names (for complex data like G',G'')",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=1000,
        help="Number of warmup iterations per chain (default: 1000)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=2000,
        help="Number of posterior samples per chain (default: 2000)",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=4,
        help="Number of MCMC chains (default: 4)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducibility (default: 0)",
    )
    parser.add_argument(
        "--warm-start",
        action="store_true",
        help="Run NLSQ first and use results to warm-start NUTS",
    )
    parser.add_argument(
        "--deformation-mode",
        type=str,
        default=None,
        help="Deformation mode for DMTA (tension, shear, bending, compression)",
    )
    parser.add_argument(
        "--poisson-ratio",
        type=float,
        default=None,
        help="Poisson ratio for E*-G* conversion (default: 0.5 for rubber)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output results as JSON",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output file for results (default: stdout)",
    )

    return parser


def main(args: list[str] | None = None) -> int:
    """Run Bayesian inference from CLI."""
    parser = create_parser()
    parsed = parser.parse_args(args)

    logger.info(
        "CLI bayesian command",
        model=parsed.model,
        input_file=str(parsed.input_file),
    )

    # Validate input file
    if not parsed.input_file.exists():
        print(f"Error: File not found: {parsed.input_file}", file=sys.stderr)
        return 1

    # Load data
    try:
        from rheojax.io import auto_load

        load_kwargs: dict = {}
        if parsed.x_col is not None:
            load_kwargs["x_col"] = parsed.x_col
        if parsed.y_col is not None:
            load_kwargs["y_col"] = parsed.y_col
        if parsed.y_cols is not None:
            load_kwargs["y_cols"] = [c.strip() for c in parsed.y_cols.split(",")]

        data = auto_load(str(parsed.input_file), **load_kwargs)

        # Handle multi-segment data (use first segment)
        if isinstance(data, list):
            if not data:
                print("Error: No data segments found in file", file=sys.stderr)
                return 1
            print(f"Note: File contains {len(data)} segments, using first segment")
            data = data[0]

        logger.debug("Data loaded", shape=str(data.x.shape))

    except Exception as e:
        print(f"Error loading data: {e}", file=sys.stderr)
        return 1

    # Create model
    try:
        import rheojax.models  # noqa: F401 — trigger registration
        from rheojax.core.registry import ModelRegistry

        model = ModelRegistry.create(parsed.model)
        logger.debug("Model created", model=parsed.model)

    except KeyError:
        print(f"Error: Unknown model '{parsed.model}'", file=sys.stderr)
        print("Use 'rheojax inventory' to see available models", file=sys.stderr)
        return 1

    # Determine test mode
    test_mode = parsed.test_mode
    if test_mode is None:
        test_mode = getattr(data, "test_mode", None)
        if test_mode is None and hasattr(data, "metadata"):
            test_mode = data.metadata.get("test_mode")
    if test_mode is None:
        test_mode = "oscillation"
        print(f"Note: No test mode detected, defaulting to '{test_mode}'")

    # Build shared kwargs for deformation mode
    mode_kwargs: dict = {}
    if parsed.deformation_mode is not None:
        mode_kwargs["deformation_mode"] = parsed.deformation_mode
    if parsed.poisson_ratio is not None:
        mode_kwargs["poisson_ratio"] = parsed.poisson_ratio

    # Optional NLSQ warm-start
    if parsed.warm_start:
        print("Running NLSQ warm-start...")
        try:
            model.fit(data.x, data.y, test_mode=test_mode, **mode_kwargs)
            print("  NLSQ warm-start complete")
            logger.info("NLSQ warm-start complete", model=parsed.model)
        except Exception as e:
            # Reset partially-mutated state from failed fit to avoid
            # leaking stale kwargs/test_mode into subsequent Bayesian inference
            model._last_fit_kwargs = {}
            model._test_mode = None
            model.fitted_ = False
            print(f"  Warning: NLSQ warm-start failed ({e}), continuing with defaults")
            logger.warning("NLSQ warm-start failed", error=str(e))

    # Run Bayesian inference
    try:
        start_time = time.perf_counter()

        print(
            f"Running NUTS ({parsed.warmup} warmup, "
            f"{parsed.samples} samples, {parsed.chains} chains)..."
        )

        result = model.fit_bayesian(
            data.x,
            data.y,
            test_mode=test_mode,
            num_warmup=parsed.warmup,
            num_samples=parsed.samples,
            num_chains=parsed.chains,
            seed=parsed.seed,
            **mode_kwargs,
        )

        sampling_time = time.perf_counter() - start_time

        logger.info(
            "Bayesian inference complete",
            model=parsed.model,
            time=f"{sampling_time:.2f}s",
        )

    except Exception as e:
        print(f"Error during Bayesian inference: {e}", file=sys.stderr)
        logger.error(
            "Bayesian inference failed",
            model=parsed.model,
            error=str(e),
            exc_info=True,
        )
        return 1

    # Extract diagnostics
    diagnostics = result.diagnostics

    # Build output dict
    output = {
        "model": parsed.model,
        "test_mode": test_mode,
        "num_warmup": parsed.warmup,
        "num_samples": parsed.samples,
        "num_chains": parsed.chains,
        "seed": parsed.seed,
        "sampling_time_seconds": round(sampling_time, 3),
        "diagnostics": {
            "divergences": diagnostics.get("divergences", 0),
            "r_hat": {},
            "ess": {},
        },
        "summary": {},
    }

    rhat_dict = diagnostics.get("r_hat") or diagnostics.get("rhat") or {}
    ess_dict = diagnostics.get("ess", {})

    for param_name in result.posterior_samples:
        samples = result.posterior_samples[param_name]
        output["summary"][param_name] = {
            "mean": float(jnp.mean(samples)),
            "std": float(jnp.std(samples)),
            "q2.5": float(jnp.percentile(samples, 2.5)),
            "q50": float(jnp.percentile(samples, 50.0)),
            "q97.5": float(jnp.percentile(samples, 97.5)),
        }
        if param_name in rhat_dict:
            output["diagnostics"]["r_hat"][param_name] = float(
                rhat_dict[param_name]
            )
        if param_name in ess_dict:
            output["diagnostics"]["ess"][param_name] = float(ess_dict[param_name])

    # Format output
    if parsed.json_output:
        output_text = json.dumps(output, indent=2)
    else:
        lines = [
            f"Model: {parsed.model}",
            f"Test mode: {test_mode}",
            f"Sampling: {parsed.warmup} warmup, {parsed.samples} samples, "
            f"{parsed.chains} chains",
            f"Time: {sampling_time:.3f}s",
            f"Divergences: {diagnostics.get('divergences', 0)}",
            "",
            "Posterior Summary:",
            f"{'Parameter':20s} {'Mean':>12s} {'Std':>12s} "
            f"{'2.5%':>12s} {'97.5%':>12s} {'R-hat':>8s} {'ESS':>8s}",
            "-" * 88,
        ]

        for param_name, stats in output["summary"].items():
            r_hat = output["diagnostics"]["r_hat"].get(param_name, float("nan"))
            ess = output["diagnostics"]["ess"].get(param_name, float("nan"))
            lines.append(
                f"  {param_name:18s} {stats['mean']:12.6g} {stats['std']:12.6g} "
                f"{stats['q2.5']:12.6g} {stats['q97.5']:12.6g} "
                f"{r_hat:8.4f} {ess:8.0f}"
            )

        lines.append("-" * 88)
        output_text = "\n".join(lines)

    # Write output
    if parsed.output:
        parsed.output.write_text(output_text)
        print(f"Results written to {parsed.output}")
    else:
        print(output_text)

    return 0
