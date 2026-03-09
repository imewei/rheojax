"""Tests for rheojax.cli._globals — shared parser and global flag handling."""

from __future__ import annotations

import argparse

import pytest

from rheojax.cli._globals import apply_globals, create_global_parser


class TestCreateGlobalParser:
    @pytest.mark.smoke
    def test_returns_argument_parser(self):
        parser = create_global_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    @pytest.mark.unit
    def test_has_verbose_flag(self):
        parser = create_global_parser()
        ns = parser.parse_args(["-v"])
        assert ns.verbose == 1

    @pytest.mark.unit
    def test_verbose_count_accumulates(self):
        parser = create_global_parser()
        ns = parser.parse_args(["-vv"])
        assert ns.verbose == 2

    @pytest.mark.unit
    def test_has_quiet_flag(self):
        parser = create_global_parser()
        ns = parser.parse_args(["-q"])
        assert ns.quiet is True

    @pytest.mark.unit
    def test_no_json_flag_on_global_parser(self):
        """--json is handled per-command, not in the global parser."""
        parser = create_global_parser()
        # Global parser should NOT have --json — each command defines its own
        with pytest.raises(SystemExit):
            parser.parse_args(["--json"])

    @pytest.mark.unit
    def test_has_no_color_flag(self):
        parser = create_global_parser()
        ns = parser.parse_args(["--no-color"])
        assert ns.no_color is True

    @pytest.mark.unit
    def test_has_log_level_flag(self):
        parser = create_global_parser()
        ns = parser.parse_args(["--log-level", "DEBUG"])
        assert ns.log_level == "DEBUG"

    @pytest.mark.unit
    def test_default_values(self):
        parser = create_global_parser()
        ns = parser.parse_args([])
        assert ns.verbose == 0
        assert ns.quiet is False
        assert ns.no_color is False
        assert ns.log_level is None

    @pytest.mark.unit
    def test_verbose_and_quiet_are_mutually_exclusive(self):
        parser = create_global_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["-v", "-q"])

    @pytest.mark.unit
    def test_log_level_choices_accepted(self):
        parser = create_global_parser()
        for level in ("DEBUG", "INFO", "WARNING", "ERROR"):
            ns = parser.parse_args(["--log-level", level])
            assert ns.log_level == level


class TestApplyGlobals:
    def _make_ns(self, **kwargs) -> argparse.Namespace:
        defaults = dict(verbose=0, quiet=False, no_color=False, log_level=None)
        defaults.update(kwargs)
        return argparse.Namespace(**defaults)

    @staticmethod
    def _patch_globals(monkeypatch):
        """Patch configure_logging and silence the module logger for clean testing."""
        from unittest.mock import MagicMock

        calls: list[str] = []
        monkeypatch.setattr(
            "rheojax.cli._globals.configure_logging",
            lambda level="WARNING": calls.append(level),
        )
        # Silence the module-level logger to avoid a pre-existing bug where
        # logger.debug() is called with `level=` as a keyword argument, which
        # conflicts with LoggerAdapter.log()'s positional `level` parameter.
        import rheojax.cli._globals as _mod

        monkeypatch.setattr(_mod, "logger", MagicMock())
        return calls

    @pytest.mark.smoke
    def test_apply_globals_calls_configure_logging(self, monkeypatch):
        calls = self._patch_globals(monkeypatch)
        apply_globals(self._make_ns(verbose=1))
        assert calls == ["INFO"]

    @pytest.mark.unit
    def test_apply_globals_quiet_uses_error_level(self, monkeypatch):
        calls = self._patch_globals(monkeypatch)
        apply_globals(self._make_ns(quiet=True))
        assert calls == ["ERROR"]

    @pytest.mark.unit
    def test_apply_globals_log_level_overrides_verbose(self, monkeypatch):
        calls = self._patch_globals(monkeypatch)
        apply_globals(self._make_ns(verbose=1, log_level="ERROR"))
        assert calls == ["ERROR"]

    @pytest.mark.unit
    def test_apply_globals_vv_sets_debug(self, monkeypatch):
        calls = self._patch_globals(monkeypatch)
        apply_globals(self._make_ns(verbose=2))
        assert calls == ["DEBUG"]

    @pytest.mark.unit
    def test_apply_globals_default_uses_warning(self, monkeypatch):
        calls = self._patch_globals(monkeypatch)
        apply_globals(self._make_ns())
        assert calls == ["WARNING"]
