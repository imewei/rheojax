import pytest

pytest.importorskip("PySide6")

from rheojax.gui.workspace.pipeline.cancel_runnable import CancelWorkerRunnable


class _FakeWorker:
    def __init__(self):
        self.cancel_called = False

    def cancel(self):
        self.cancel_called = True


def test_cancel_worker_runnable_calls_cancel():
    worker = _FakeWorker()
    runnable = CancelWorkerRunnable(worker)
    runnable.run()
    assert worker.cancel_called is True


def test_cancel_worker_runnable_runs_via_threadpool(qtbot):
    from PySide6.QtCore import QThreadPool

    worker = _FakeWorker()
    pool = QThreadPool.globalInstance()
    pool.start(CancelWorkerRunnable(worker))
    pool.waitForDone(2000)
    assert worker.cancel_called is True
