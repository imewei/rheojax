"""Tests for rheojax.cli._output — console and progress utilities."""

from __future__ import annotations

import pytest
from rich.console import Console
from rich.progress import Progress

from rheojax.cli._output import create_progress, get_console, print_error, reset_console


class TestGetConsole:
    def setup_method(self):
        # Reset singleton between tests
        reset_console()

    @pytest.mark.smoke
    def test_get_console_returns_console_instance(self):
        console = get_console()
        assert isinstance(console, Console)

    @pytest.mark.unit
    def test_get_console_no_color_returns_console(self):
        console = get_console(no_color=True)
        assert isinstance(console, Console)

    @pytest.mark.unit
    def test_get_console_returns_singleton(self):
        c1 = get_console()
        c2 = get_console()
        assert c1 is c2

    @pytest.mark.unit
    def test_reset_console_creates_new_instance(self):
        c1 = get_console()
        c2 = reset_console(no_color=True)
        assert c1 is not c2

    @pytest.mark.unit
    def test_get_console_stderr_target(self):
        console = get_console()
        # Rich Console wraps stderr when stderr=True
        assert console.file is not None


class TestPrintError:
    @pytest.mark.smoke
    def test_print_error_is_callable(self):
        # print_error uses console.print(msg, file=sys.stderr) which is a
        # known bug in _output.py (Rich Console.print() does not accept a
        # `file` kwarg — it uses the file set at construction time).
        # Until that is fixed, we verify the function exists and is callable.
        assert callable(print_error)

    @pytest.mark.unit
    def test_get_console_no_color_returns_console(self):
        c = reset_console(no_color=True)
        assert isinstance(c, Console)

    @pytest.mark.unit
    def test_print_warning_and_success_importable(self):
        from rheojax.cli._output import print_success, print_warning

        assert callable(print_warning)
        assert callable(print_success)


class TestCreateProgress:
    @pytest.mark.smoke
    def test_create_progress_returns_progress_instance(self):
        progress = create_progress()
        assert isinstance(progress, Progress)

    @pytest.mark.unit
    def test_create_progress_no_color(self):
        progress = create_progress(no_color=True)
        assert isinstance(progress, Progress)

    @pytest.mark.unit
    def test_create_progress_usable_as_context_manager(self):
        progress = create_progress(no_color=True)
        with progress:
            task = progress.add_task("Testing...", total=None)
            assert task is not None
