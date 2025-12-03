"""Generate synthetic LAOS datasets for SPP golden-data harness.

Datasets:
- sin_fundamental: fundamental + small 3rd harmonic
- sin_noisy: same with additive noise

Outputs written to golden_data/input/<dataset>.csv with columns: t, gamma, sigma.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def make_sin_fundamental(n_cycles: int = 3, pts_per_cycle: int = 256):
    omega = 2 * np.pi  # rad/s
    gamma_0 = 1.0
    h3 = 0.15
    t = np.linspace(0, n_cycles * 2 * np.pi / omega, n_cycles * pts_per_cycle, endpoint=False)
    gamma = gamma_0 * np.sin(omega * t) + h3 * np.sin(3 * omega * t)
    sigma = 50.0 * gamma  # simple linear-ish stress
    return t, gamma, sigma


def make_sin_noisy(n_cycles: int = 3, pts_per_cycle: int = 256, seed: int = 0):
    rng = np.random.default_rng(seed)
    t, gamma, sigma = make_sin_fundamental(n_cycles, pts_per_cycle)
    gamma += 0.01 * rng.standard_normal(gamma.shape)
    sigma += 0.5 * rng.standard_normal(sigma.shape)
    return t, gamma, sigma


def write_csv(path: Path, t: np.ndarray, gamma: np.ndarray, sigma: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.column_stack([t, gamma, sigma])
    header = "t,gamma,sigma"
    np.savetxt(path, data, delimiter=",", header=header, comments="")


def main(argv: list[str] | None = None):
    """Generate golden data inputs.

    Args:
        argv: Command line arguments. If None, uses sys.argv.
              Pass [] to use defaults when calling programmatically.
    """
    parser = argparse.ArgumentParser()
    # Default path is relative to script location for portability
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--outdir", type=Path, default=script_dir / "golden_data" / "input"
    )
    args = parser.parse_args(argv)

    outdir: Path = args.outdir
    outdir.mkdir(parents=True, exist_ok=True)

    datasets = {
        "sin_fundamental": make_sin_fundamental(),
        "sin_noisy": make_sin_noisy(),
    }

    for name, (t, gamma, sigma) in datasets.items():
        write_csv(outdir / f"{name}.csv", t, gamma, sigma)
        print(f"Wrote {name} -> {outdir / f'{name}.csv'}")


if __name__ == "__main__":
    main()
