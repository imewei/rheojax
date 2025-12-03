"""Run RheoJAX SPP analysis for golden-data harness.

Reads golden_data/input/<dataset>.csv and writes:
- golden_data/outputs/rheojax/<dataset>_spp_data_out.csv (15 columns)
- golden_data/outputs/rheojax/<dataset>_fsf_data_out.csv (9 columns)
- golden_data/outputs/rheojax/<dataset>_ft_out.csv (if available)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from rheojax.transforms.spp_decomposer import SPPDecomposer
from rheojax.core.data import RheoData


def load_input(path: Path):
    arr = np.loadtxt(path, delimiter=",", skiprows=1)
    t = arr[:, 0]
    gamma = arr[:, 1]
    sigma = arr[:, 2]
    return t, gamma, sigma


def run_dataset(dataset: str, input_dir: Path, output_dir: Path):
    t, gamma, sigma = load_input(input_dir / f"{dataset}.csv")

    omega = 2 * np.pi  # matches generator
    gamma_0 = float(np.max(np.abs(gamma)))

    data = RheoData(
        x=t,
        y=sigma,
        domain="time",
        metadata={"omega": omega, "gamma_0": gamma_0, "strain": gamma, "test_mode": "oscillation"},
    )

    decomposer = SPPDecomposer(omega=omega, gamma_0=gamma_0, n_harmonics=39)
    decomposer.transform(data)

    out_base = output_dir / dataset
    out_base.parent.mkdir(parents=True, exist_ok=True)

    spp = decomposer.results_["spp_data_out"]
    np.savetxt(out_base.with_name(out_base.name + "_spp_data_out.csv"), spp, delimiter=",")

    fsf = decomposer.results_.get("fsf_data_out")
    if fsf is not None:
        np.savetxt(out_base.with_name(out_base.name + "_fsf_data_out.csv"), fsf, delimiter=",")

    ft = decomposer.results_.get("ft_out")
    if ft is not None:
        np.savetxt(out_base.with_name(out_base.name + "_ft_out.csv"), ft, delimiter=",")


def main():
    parser = argparse.ArgumentParser()
    # Default paths are relative to script location for portability
    script_dir = Path(__file__).parent
    parser.add_argument(
        "--input_dir", type=Path, default=script_dir / "golden_data" / "input"
    )
    parser.add_argument(
        "--output_dir", type=Path, default=script_dir / "golden_data" / "outputs" / "rheojax"
    )
    parser.add_argument("datasets", nargs="*", default=["sin_fundamental", "sin_noisy"])
    args = parser.parse_args()

    for ds in args.datasets:
        run_dataset(ds, args.input_dir, args.output_dir)
        print(f"Wrote RheoJAX outputs for {ds} -> {args.output_dir}")


if __name__ == "__main__":
    main()
