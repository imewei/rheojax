import pytest

pytest.importorskip("PySide6")

from rheojax.gui.foundation.notifier import DatasetLibraryNotifier


def test_notifier_changed_signal_fires(qtbot):
    notifier = DatasetLibraryNotifier()
    with qtbot.waitSignal(notifier.changed, timeout=1000):
        notifier.changed.emit()
