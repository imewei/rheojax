"""Tests for rheojax.cli.cmd_batch — batch subcommand."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pytest

from rheojax.cli.cmd_batch import create_parser, main
from rheojax.core.data import RheoData


class TestCreateParser:
    @pytest.mark.unit
    def test_returns_argument_parser(self):
        parser = create_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_pattern_argument(self):
        parser = create_parser()
        ns = parser.parse_args(["data/*.csv", "--model", "maxwell"])
        assert ns.pattern == "data/*.csv"

    @pytest.mark.unit
    def test_has_model_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "springpot"])
        assert ns.model == "springpot"

    @pytest.mark.unit
    def test_has_test_mode_flag(self):
        parser = create_parser()
        ns = parser.parse_args(
            ["*.csv", "--model", "maxwell", "--test-mode", "relaxation"]
        )
        assert ns.test_mode == "relaxation"

    @pytest.mark.unit
    def test_has_output_dir_flag(self):
        parser = create_parser()
        ns = parser.parse_args(
            ["*.csv", "--model", "maxwell", "--output-dir", "results"]
        )
        assert str(ns.output_dir) == "results"

    @pytest.mark.unit
    def test_has_json_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "maxwell", "--json"])
        assert ns.json_output is True

    @pytest.mark.unit
    def test_has_max_iter_flag(self):
        parser = create_parser()
        ns = parser.parse_args(["*.csv", "--model", "maxwell", "--max-iter", "500"])
        assert ns.max_iter == 500


class TestMainHelp:
    @pytest.mark.smoke
    def test_main_help_exits_cleanly(self):
        with pytest.raises(SystemExit) as exc_info:
            main(["--help"])
        assert exc_info.value.code == 0


class TestMainErrors:
    @pytest.mark.smoke
    def test_main_no_matching_files_returns_1(self, tmp_path):
        # Pattern that matches nothing in tmp_path
        result = main(
            [
                str(tmp_path / "*.totally_nonexistent_extension_xyz"),
                "--model",
                "maxwell",
                "--test-mode",
                "relaxation",
            ]
        )
        assert result == 1

    @pytest.mark.unit
    def test_main_invalid_max_iter_returns_1(self, tmp_path):
        result = main(
            [
                str(tmp_path / "*.csv"),
                "--model",
                "maxwell",
                "--max-iter",
                "0",
            ]
        )
        assert result == 1


class TestMainJsonOutput:
    @pytest.mark.unit
    def test_json_stdout_is_valid_json_with_no_extra_text(self, tmp_path, capsys):
        # Regression: progress lines ("Found N file(s)...", "[i/n] ... OK") used
        # to print to stdout unconditionally, so `rheojax batch --json` interleaved
        # them with the final json.dumps(all_results), making stdout unparseable
        # for a downstream `| jq` or `| rheojax ...` consumer.
        csv_file = tmp_path / "sample.csv"
        csv_file.write_text(
            "time,stress\n"
            "0.1,4.756\n"
            "0.5,3.894\n"
            "1.0,3.033\n"
            "2.0,1.839\n"
            "3.0,1.115\n"
            "4.0,0.677\n"
            "5.0,0.410\n"
        )

        result = main(
            [
                str(tmp_path / "*.csv"),
                "--model",
                "maxwell",
                "--test-mode",
                "relaxation",
                "--max-iter",
                "50",
                "--json",
            ]
        )

        captured = capsys.readouterr()
        assert result == 0
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 1
        assert parsed[0]["status"] == "success"


class TestMainParallel:
    @pytest.mark.unit
    def test_parallel_load_preserves_file_order_and_succeeds(self, tmp_path, capsys):
        # --parallel used to be a documented no-op. It now loads files
        # concurrently (I/O only) while still fitting sequentially (JAX JIT
        # is not thread-safe) — this pins both that it actually runs and
        # that results stay in glob order despite out-of-order completion.
        relaxation_csv = (
            "time,stress\n"
            "0.1,4.756\n"
            "0.5,3.894\n"
            "1.0,3.033\n"
            "2.0,1.839\n"
            "3.0,1.115\n"
            "4.0,0.677\n"
            "5.0,0.410\n"
        )
        for name in ("a_sample.csv", "b_sample.csv", "c_sample.csv"):
            (tmp_path / name).write_text(relaxation_csv)

        result = main(
            [
                str(tmp_path / "*.csv"),
                "--model",
                "maxwell",
                "--test-mode",
                "relaxation",
                "--max-iter",
                "50",
                "--parallel",
                "--workers",
                "3",
                "--json",
            ]
        )

        captured = capsys.readouterr()
        assert result == 0
        parsed = json.loads(captured.out)
        assert [Path(r["file"]).name for r in parsed] == [
            "a_sample.csv",
            "b_sample.csv",
            "c_sample.csv",
        ]
        assert all(r["status"] == "success" for r in parsed)

    @pytest.mark.unit
    def test_parallel_actually_overlaps_file_loads(self, tmp_path, monkeypatch):
        # The order/success test above uses identical per-file latency, so a
        # silently-sequential fallback would pass it too -- it can't tell
        # "concurrent" from "coincidentally correct." This one proves actual
        # overlap: sleep duration is inversely tied to submission order (the
        # first file submitted sleeps longest), so true concurrency finishes
        # in ~max(sleeps) while a sequential fallback takes ~sum(sleeps).
        names = ["a_sample.csv", "b_sample.csv", "c_sample.csv"]
        sleeps = {names[0]: 0.3, names[1]: 0.2, names[2]: 0.1}
        for name in names:
            (tmp_path / name).write_text("time,stress\n0.1,1.0\n0.2,0.9\n")

        def fake_auto_load(path, **kwargs):
            time.sleep(sleeps[Path(path).name])
            return RheoData(
                x=np.array([0.1, 0.2]), y=np.array([1.0, 0.9]), domain="time"
            )

        class _InstantModel:
            """No-op .fit() -- isolates the load-concurrency proof above
            from real NLSQ fit time."""

            class _Param:
                def __init__(self, value):
                    self.value = value

            def __init__(self):
                self.parameters = {"G0": self._Param(1.0)}

            def fit(self, x, y, **kwargs):
                return self

        monkeypatch.setattr("rheojax.io.auto_load", fake_auto_load)
        monkeypatch.setattr(
            "rheojax.core.registry.ModelRegistry.create",
            lambda name: _InstantModel(),
        )

        start = time.perf_counter()
        result = main(
            [
                str(tmp_path / "*.csv"),
                "--model",
                "maxwell",
                "--test-mode",
                "relaxation",
                "--parallel",
                "--workers",
                "3",
                "--json",
            ]
        )
        elapsed = time.perf_counter() - start

        assert result == 0
        # Concurrent: ~max(sleeps) = 0.3s. Sequential fallback: ~sum = 0.6s.
        # 0.5s threshold gives headroom for scheduling jitter without being
        # reachable by the sequential path.
        assert elapsed < 0.5, (
            f"took {elapsed:.2f}s -- looks sequential, not concurrent "
            f"(--parallel may have silently fallen back)"
        )
