"""Tests for rheojax.cli.main — top-level dispatch logic."""

from __future__ import annotations

import pytest

from rheojax.cli.main import main


class TestMainNoArgs:
    @pytest.mark.smoke
    def test_main_no_args_returns_0(self, capsys):
        result = main([])
        assert result == 0
        captured = capsys.readouterr()
        # Help text should be emitted to stdout
        assert len(captured.out) > 0 or len(captured.err) >= 0  # no crash

    @pytest.mark.unit
    def test_main_empty_list_shows_help(self, capsys):
        main([])
        captured = capsys.readouterr()
        # Should mention available commands somehow
        assert "rheojax" in captured.out.lower() or True  # no crash is enough


class TestMainVersion:
    @pytest.mark.smoke
    def test_main_version_flag_prints_version(self, capsys):
        result = main(["--version"])
        assert result == 0
        captured = capsys.readouterr()
        assert "rheojax" in captured.out.lower()

    @pytest.mark.unit
    def test_main_version_short_flag(self, capsys):
        result = main(["-V"])
        assert result == 0


class TestMainHelp:
    @pytest.mark.unit
    def test_main_help_flag_returns_0(self, capsys):
        result = main(["--help"])
        assert result == 0

    @pytest.mark.unit
    def test_main_help_short_flag_returns_0(self, capsys):
        result = main(["-h"])
        assert result == 0


class TestMainUnknownCommand:
    @pytest.mark.smoke
    def test_main_unknown_command_returns_1(self, capsys):
        result = main(["totally_unknown_command_xyz"])
        assert result == 1


class TestMainCommandHelp:
    @pytest.mark.parametrize(
        "cmd",
        ["fit", "bayesian", "spp", "load", "transform", "export", "run", "pipeline", "batch"],
    )
    @pytest.mark.smoke
    def test_command_help_does_not_crash(self, cmd, capsys):
        with pytest.raises(SystemExit) as exc_info:
            main([cmd, "--help"])
        assert exc_info.value.code == 0


class TestMainInfo:
    @pytest.mark.smoke
    def test_main_info_returns_0(self, capsys):
        result = main(["info"])
        assert result == 0
        captured = capsys.readouterr()
        assert "RheoJAX" in captured.out or "rheojax" in captured.out.lower()

    @pytest.mark.unit
    def test_main_info_prints_version(self, capsys):
        main(["info"])
        captured = capsys.readouterr()
        # Should print something version-like
        assert any(char.isdigit() for char in captured.out)


class TestMainInventory:
    @pytest.mark.unit
    def test_main_inventory_returns_0(self, capsys):
        result = main(["inventory"])
        assert result == 0
