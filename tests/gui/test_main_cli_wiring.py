import subprocess
import sys
from pathlib import Path

import pytest

from rheojax.gui.main import parse_args

# Tests below that call the real main() construct a genuine QApplication and
# render real widgets. Running many of them back-to-back in the same
# interpreter as the rest of tests/gui/ (500+ tests, each rendering
# matplotlib/Qt canvases) risks tripping an upstream FreeType raster-overflow
# bug (matplotlib/ft2font) that corrupts the process heap without crashing
# immediately -- the segfault then surfaces later at an unrelated allocation
# inside this file. Running each in its own subprocess gives it a clean heap,
# regardless of what ran earlier in the suite.
_SUBPROCESS_TIMEOUT = 60


def test_protocol_required_with_import():
    with pytest.raises(SystemExit):
        parse_args(["--import", "data.csv"])  # no --protocol


def test_protocol_without_import_is_rejected():
    with pytest.raises(SystemExit):
        parse_args(["--protocol", "oscillation"])  # no --import


def test_project_and_import_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--project",
                "p.rheojax",
                "--import",
                "data.csv",
                "--protocol",
                "oscillation",
            ]
        )


def test_import_with_protocol_parses_successfully():
    args = parse_args(["--import", "data.csv", "--protocol", "oscillation"])
    assert args.import_file == Path("data.csv")
    assert args.protocol == "oscillation"


def test_project_alone_still_parses():
    args = parse_args(["--project", "p.rheojax"])
    assert args.project == Path("p.rheojax")
    assert args.import_file is None
    assert args.protocol is None


def _run_isolated(script: str) -> subprocess.CompletedProcess:
    """Run a main()-invoking scenario in a fresh subprocess.

    Each of these tests exercises the real main() (real QApplication, real
    widget construction). Running them in-process alongside the other 500+
    widget tests in tests/gui/ risks tripping an upstream FreeType
    raster-overflow bug that corrupts the process heap without crashing
    immediately -- the segfault then surfaces later at an unrelated
    allocation. A subprocess gets a clean heap regardless of what else ran.
    """
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=_SUBPROCESS_TIMEOUT,
    )


def test_default_launch_constructs_workspace_window():
    script = (
        "import rheojax.gui.main as main_module\n"
        "created = {}\n"
        "def _fake_create_workspace_window():\n"
        "    created['workspace'] = True\n"
        "    raise SystemExit(0)\n"
        "main_module._create_workspace_window = _fake_create_workspace_window\n"
        "try:\n"
        "    main_module.main([])\n"
        "except SystemExit:\n"
        "    pass\n"
        "print('OK' if created.get('workspace') else 'FAIL')\n"
    )
    result = _run_isolated(script)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_workspace_branch_installs_gui_log_handler():
    script = (
        "import rheojax.gui.main as main_module\n"
        "class _FakeLogDock:\n"
        "    def append_record(self, levelno, message):\n"
        "        pass\n"
        "class _FakeDestroyed:\n"
        "    def connect(self, fn):\n"
        "        raise SystemExit(0)\n"
        "class _FakeWorkspaceWindow:\n"
        "    def __init__(self):\n"
        "        self.log_dock = _FakeLogDock()\n"
        "        self.destroyed = _FakeDestroyed()\n"
        "fake_window = _FakeWorkspaceWindow()\n"
        "main_module._create_workspace_window = lambda: fake_window\n"
        "captured = {}\n"
        "def _fake_install(append_fn, **kwargs):\n"
        "    captured['append_fn'] = append_fn\n"
        "    return object()\n"
        "main_module.install_gui_log_handler = _fake_install\n"
        "try:\n"
        "    main_module.main([])\n"
        "except SystemExit:\n"
        "    pass\n"
        "ok = captured.get('append_fn') == fake_window.log_dock.append_record\n"
        "print('OK' if ok else 'FAIL')\n"
    )
    result = _run_isolated(script)
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout, result.stdout + result.stderr


def test_uncaught_exception_is_logged_via_global_excepthook(monkeypatch):
    import sys

    import rheojax.gui.main as main_module

    monkeypatch.setattr(sys, "excepthook", sys.excepthook)  # arm restoration

    monkeypatch.setattr(
        main_module, "check_dependencies", lambda: (False, ["fake-missing-dep"])
    )

    result = main_module.main([])
    assert result == 1
    assert sys.excepthook is not sys.__excepthook__

    captured = []
    monkeypatch.setattr(
        main_module.logger, "critical", lambda *a, **k: captured.append((a, k))
    )
    try:
        raise ValueError("boom")
    except ValueError:
        sys.excepthook(*sys.exc_info())

    assert captured
