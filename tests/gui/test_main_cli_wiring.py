from pathlib import Path

import pytest

from rheojax.gui.main import parse_args


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


def test_no_flags_parses_with_legacy_false():
    args = parse_args([])
    assert args.legacy is False


def test_legacy_flag_parses():
    args = parse_args(["--legacy"])
    assert args.legacy is True


def test_workspace_flag_still_parses_for_back_compat():
    args = parse_args(["--workspace"])
    assert args.workspace is True


def _reuse_qapplication_singleton(monkeypatch):
    """Qt allows only one QApplication per process; main() unconditionally
    constructs one, so back-to-back main() calls in the same test session
    must reuse the existing singleton instead of each trying to create one.
    Delegates attribute access (e.g. the static QApplication.setAttribute(...)
    call in main()) to the real class; only the constructor call is intercepted.
    """
    import rheojax.gui.compat as compat

    real_qapplication_cls = compat.QApplication

    class _QApplicationProxy:
        def __getattr__(self, name):
            return getattr(real_qapplication_cls, name)

        def __call__(self, *args, **kwargs):
            return real_qapplication_cls.instance() or real_qapplication_cls(
                *args, **kwargs
            )

    monkeypatch.setattr(compat, "QApplication", _QApplicationProxy())


def test_default_launch_constructs_workspace_window(monkeypatch, tmp_path):
    import rheojax.gui.main as main_module

    _reuse_qapplication_singleton(monkeypatch)
    created = {}

    def _fake_create_workspace_window():
        created["workspace"] = True
        raise SystemExit(0)  # short-circuit before app.exec() actually blocks

    monkeypatch.setattr(
        main_module, "_create_workspace_window", _fake_create_workspace_window
    )
    with pytest.raises(SystemExit):
        main_module.main([])
    assert created.get("workspace") is True


def test_legacy_flag_constructs_main_window(monkeypatch):
    import rheojax.gui.main as main_module

    _reuse_qapplication_singleton(monkeypatch)
    created = {}

    class _FakeMainWindow:
        def __init__(self, *a, **k):
            created["legacy"] = True
            raise SystemExit(0)

    monkeypatch.setattr(
        "rheojax.gui.app.main_window.RheoJAXMainWindow", _FakeMainWindow
    )
    with pytest.raises(SystemExit):
        main_module.main(["--legacy"])
    assert created.get("legacy") is True


class _FakeLogDock:
    def append_record(self, levelno, message):
        pass


class _FakeDestroyed:
    def connect(self, fn):
        # Short-circuits before app.exec() would otherwise block, same as the
        # existing tests above but one step later (after handler install).
        raise SystemExit(0)


class _FakeWorkspaceWindow:
    def __init__(self):
        self.log_dock = _FakeLogDock()
        self.destroyed = _FakeDestroyed()


def test_workspace_branch_installs_gui_log_handler(monkeypatch):
    import rheojax.gui.main as main_module

    _reuse_qapplication_singleton(monkeypatch)
    fake_window = _FakeWorkspaceWindow()
    monkeypatch.setattr(main_module, "_create_workspace_window", lambda: fake_window)

    captured = {}

    def _fake_install(append_fn, **kwargs):
        captured["append_fn"] = append_fn
        return object()

    monkeypatch.setattr(main_module, "install_gui_log_handler", _fake_install)

    with pytest.raises(SystemExit):
        main_module.main([])

    assert captured.get("append_fn") == fake_window.log_dock.append_record


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
