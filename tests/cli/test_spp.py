"""Tests for rheojax.cli.spp — SPP analyze/batch subcommands.

Covers the command-execution body (data loading, SPP analysis, output/report
writing) and the argument-validation / error paths, using small synthetic LAOS
CSV fixtures so the suite stays fast and self-contained.
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd
import pytest

from rheojax.cli.spp import create_parser, main, run_analyze, run_batch


def _write_laos_csv(path, omega: float = 1.0, gamma_0: float = 0.5, n: int = 500):
    """Write a synthetic single-cycle LAOS dataset (time, stress) to ``path``."""
    t = np.linspace(0.0, 2.0 * np.pi / omega, n)
    strain = gamma_0 * np.sin(omega * t)
    # Linear elastic term + a third-harmonic contribution so SPP yields metrics.
    stress = 80.0 * strain + 5.0 * np.sin(3.0 * omega * t)
    pd.DataFrame({"time": t, "stress": stress}).to_csv(path, index=False)
    return path


def _write_laos_csv_with_strain(
    path, omega: float = 1.0, gamma_0: float = 0.5, n: int = 500
):
    """Write a synthetic LAOS dataset with an explicit, non-sinusoidal strain column."""
    t = np.linspace(0.0, 2.0 * np.pi / omega, n)
    strain = gamma_0 * np.sin(omega * t) + 0.05 * gamma_0 * np.sin(3.0 * omega * t)
    stress = 80.0 * strain + 5.0 * np.sin(3.0 * omega * t)
    pd.DataFrame({"time": t, "stress": stress, "strain_measured": strain}).to_csv(
        path, index=False
    )
    return path


def _analyze_ns(**overrides) -> argparse.Namespace:
    """Build a fully-populated Namespace for run_analyze with sane defaults."""
    defaults = dict(
        input_file=None,
        omega=1.0,
        gamma_0=0.5,
        n_harmonics=5,
        step_size=8,
        numerical=False,
        bayesian=False,
        num_warmup=2,
        num_samples=2,
        output=None,
        export_matlab=False,
        x_col=None,
        y_col=None,
        strain_col=None,
        verbose=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        assert isinstance(create_parser(), argparse.ArgumentParser)

    @pytest.mark.unit
    def test_analyze_required_args(self):
        parser = create_parser()
        ns = parser.parse_args(
            ["analyze", "data.csv", "--omega", "1.5", "--gamma-0", "0.4"]
        )
        assert ns.command == "analyze"
        assert str(ns.input_file) == "data.csv"
        assert ns.omega == 1.5
        assert ns.gamma_0 == 0.4
        # Defaults from _add_analyze_args
        assert ns.n_harmonics == 39
        assert ns.step_size == 8
        assert ns.numerical is False
        assert ns.bayesian is False

    @pytest.mark.unit
    def test_analyze_flags(self):
        parser = create_parser()
        # Global flags (-v) go before the subcommand name; see create_parser().
        ns = parser.parse_args(
            [
                "-v",
                "analyze",
                "d.csv",
                "--omega",
                "1.0",
                "--gamma-0",
                "0.5",
                "--numerical",
                "--bayesian",
                "--export-matlab",
                "--n-harmonics",
                "7",
                "-o",
                "out.csv",
            ]
        )
        assert ns.numerical is True
        assert ns.bayesian is True
        assert ns.export_matlab is True
        assert ns.n_harmonics == 7
        assert str(ns.output) == "out.csv"
        assert ns.verbose == 1

    @pytest.mark.unit
    def test_analyze_missing_required_omega_exits(self):
        parser = create_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["analyze", "d.csv", "--gamma-0", "0.5"])

    @pytest.mark.unit
    def test_batch_args(self):
        parser = create_parser()
        ns = parser.parse_args(
            ["batch", "some_dir", "--omega", "2.0", "--pattern", "*.txt"]
        )
        assert ns.command == "batch"
        assert str(ns.input_dir) == "some_dir"
        assert ns.omega == 2.0
        assert ns.pattern == "*.txt"


# ---------------------------------------------------------------------------
# main() entry point
# ---------------------------------------------------------------------------


class TestMain:
    @pytest.mark.smoke
    def test_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0

    @pytest.mark.smoke
    def test_no_command_prints_help_returns_0(self):
        assert main([]) == 0

    @pytest.mark.smoke
    def test_main_analyze_success(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        out = tmp_path / "out.csv"
        rc = main(
            [
                "analyze",
                str(csv),
                "--omega",
                "1.0",
                "--gamma-0",
                "0.5",
                "--n-harmonics",
                "5",
                "-o",
                str(out),
            ]
        )
        assert rc == 0
        assert out.exists()


# ---------------------------------------------------------------------------
# run_analyze — success + output
# ---------------------------------------------------------------------------


class TestRunAnalyze:
    @pytest.mark.smoke
    def test_success_writes_metrics_csv(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        out = tmp_path / "result.csv"
        rc = run_analyze(_analyze_ns(input_file=csv, output=out))
        assert rc == 0
        df = pd.read_csv(out)
        # Standard CSV metric columns produced by _save_results.
        for col in ("sigma_sy", "sigma_dy", "I3_I1_ratio", "S_factor", "T_factor"):
            assert col in df.columns
        assert np.isfinite(df["sigma_sy"].iloc[0])
        assert df["sigma_sy"].iloc[0] > 0

    @pytest.mark.unit
    def test_default_output_path(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "sample.csv")
        rc = run_analyze(_analyze_ns(input_file=csv, output=None))
        assert rc == 0
        assert (tmp_path / "sample_spp.csv").exists()

    @pytest.mark.unit
    def test_verbose_success(self, tmp_path, capsys):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        rc = run_analyze(
            _analyze_ns(input_file=csv, output=tmp_path / "o.csv", verbose=True)
        )
        assert rc == 0
        out = capsys.readouterr().out
        assert "Loading data from" in out
        assert "Running SPP analysis" in out

    @pytest.mark.unit
    def test_strain_col_is_wired_into_analysis(self, tmp_path):
        """--strain-col must actually feed the named column into the SPP
        transform (previously accepted by argparse but never read)."""
        csv = _write_laos_csv_with_strain(tmp_path / "laos_strain.csv")
        out_with = tmp_path / "with_strain.csv"
        out_without = tmp_path / "without_strain.csv"

        rc_with = run_analyze(
            _analyze_ns(
                input_file=csv, output=out_with, strain_col="strain_measured"
            )
        )
        rc_without = run_analyze(_analyze_ns(input_file=csv, output=out_without))

        assert rc_with == 0
        assert rc_without == 0
        sigma_sy_with = pd.read_csv(out_with)["sigma_sy"].iloc[0]
        sigma_sy_without = pd.read_csv(out_without)["sigma_sy"].iloc[0]
        assert sigma_sy_with != pytest.approx(sigma_sy_without)

    @pytest.mark.unit
    def test_numerical_method(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        out = tmp_path / "o.csv"
        rc = run_analyze(
            _analyze_ns(input_file=csv, output=out, numerical=True, step_size=1)
        )
        assert rc == 0
        df = pd.read_csv(out)
        # Numerical method fills the time-resolved moduli means.
        assert np.isfinite(df["Gp_t_mean"].iloc[0])

    @pytest.mark.unit
    def test_explicit_columns(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        out = tmp_path / "o.csv"
        rc = run_analyze(
            _analyze_ns(input_file=csv, output=out, x_col="time", y_col="stress")
        )
        assert rc == 0

    # --- validation / error paths ---

    @pytest.mark.unit
    def test_nonpositive_omega_returns_1(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        rc = run_analyze(_analyze_ns(input_file=csv, omega=0.0))
        assert rc == 1

    @pytest.mark.unit
    def test_nonpositive_gamma_0_returns_1(self, tmp_path):
        csv = _write_laos_csv(tmp_path / "laos.csv")
        rc = run_analyze(_analyze_ns(input_file=csv, gamma_0=-1.0))
        assert rc == 1

    @pytest.mark.smoke
    def test_nonexistent_file_returns_1(self, tmp_path):
        rc = run_analyze(_analyze_ns(input_file=tmp_path / "nope.csv"))
        assert rc == 1

    @pytest.mark.unit
    def test_unparseable_file_returns_1(self, tmp_path):
        bad = tmp_path / "junk.csv"
        bad.write_text("this is not tabular data at all\n")
        rc = run_analyze(_analyze_ns(input_file=bad))
        assert rc == 1

    @pytest.mark.unit
    def test_export_matlab_reports_error(self, tmp_path):
        # BUG: --export-matlab passes SPPDecomposer results straight to
        # export_spp_txt, which needs length-bearing keys (Gp_t/time_new/...)
        # that the decomposer nests under results["numerical"]. So MATLAB
        # export currently always fails; the CLI reports it and returns 1.
        csv = _write_laos_csv(tmp_path / "laos.csv")
        rc = run_analyze(
            _analyze_ns(input_file=csv, output=tmp_path / "o.txt", export_matlab=True)
        )
        assert rc == 1

    @pytest.mark.unit
    def test_bayesian_reports_error(self, tmp_path):
        # BUG: the Bayesian block builds length-1 arrays from a single analyze
        # run and calls model.fit(), which requires >=2 data points. So
        # --bayesian on a single file always fails and returns 1.
        csv = _write_laos_csv(tmp_path / "laos.csv", n=400)
        rc = run_analyze(
            _analyze_ns(
                input_file=csv,
                output=tmp_path / "o.csv",
                bayesian=True,
                num_warmup=2,
                num_samples=2,
            )
        )
        assert rc == 1


# ---------------------------------------------------------------------------
# run_batch
# ---------------------------------------------------------------------------


def _batch_ns(**overrides) -> argparse.Namespace:
    defaults = dict(
        input_dir=None,
        omega=1.0,
        output_dir=None,
        pattern="*.csv",
        n_harmonics=5,
        step_size=8,
        verbose=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestRunBatch:
    @pytest.mark.smoke
    def test_success_processes_all_files(self, tmp_path):
        in_dir = tmp_path / "data"
        in_dir.mkdir()
        _write_laos_csv(in_dir / "a.csv")
        _write_laos_csv(in_dir / "b.csv")
        out_dir = tmp_path / "results"
        rc = run_batch(_batch_ns(input_dir=in_dir, output_dir=out_dir))
        assert rc == 0
        assert (out_dir / "a_spp.csv").exists()
        assert (out_dir / "b_spp.csv").exists()

    @pytest.mark.unit
    def test_default_output_dir(self, tmp_path):
        in_dir = tmp_path / "data"
        in_dir.mkdir()
        _write_laos_csv(in_dir / "a.csv")
        rc = run_batch(_batch_ns(input_dir=in_dir, output_dir=None))
        assert rc == 0
        assert (in_dir / "spp_results" / "a_spp.csv").exists()

    @pytest.mark.unit
    def test_nonpositive_omega_returns_1(self, tmp_path):
        in_dir = tmp_path / "data"
        in_dir.mkdir()
        _write_laos_csv(in_dir / "a.csv")
        rc = run_batch(_batch_ns(input_dir=in_dir, omega=0.0))
        assert rc == 1

    @pytest.mark.unit
    def test_not_a_directory_returns_1(self, tmp_path):
        rc = run_batch(_batch_ns(input_dir=tmp_path / "missing_dir"))
        assert rc == 1

    @pytest.mark.unit
    def test_no_matching_files_returns_1(self, tmp_path):
        in_dir = tmp_path / "data"
        in_dir.mkdir()
        _write_laos_csv(in_dir / "a.csv")
        rc = run_batch(_batch_ns(input_dir=in_dir, pattern="*.txt"))
        assert rc == 1

    @pytest.mark.unit
    def test_partial_failure_returns_1(self, tmp_path):
        in_dir = tmp_path / "data"
        in_dir.mkdir()
        _write_laos_csv(in_dir / "good.csv")
        (in_dir / "bad.csv").write_text("not tabular data\n")
        rc = run_batch(_batch_ns(input_dir=in_dir, output_dir=tmp_path / "out"))
        assert rc == 1
        # The good file still produced output despite the failure.
        assert (tmp_path / "out" / "good_spp.csv").exists()

    @pytest.mark.unit
    def test_verbose_flag(self, tmp_path, capsys):
        in_dir = tmp_path / "data"
        in_dir.mkdir()
        _write_laos_csv(in_dir / "a.csv")
        rc = run_batch(
            _batch_ns(input_dir=in_dir, output_dir=tmp_path / "out", verbose=True)
        )
        assert rc == 0
        assert "Found 1 files to process" in capsys.readouterr().out
