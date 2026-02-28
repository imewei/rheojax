"""Tests for subprocess worker isolation."""

import multiprocessing as mp
import os
import time

import pytest


# ---------------------------------------------------------------------------
# Module-level helper functions for subprocess tests.
# These MUST be at module level so they are picklable with the "spawn"
# start method (default on macOS Python 3.12+).
# ---------------------------------------------------------------------------


def _child_wait_for_cancel(cancel_event, q):
    """Top-level function for cross-process test (must be picklable)."""
    from rheojax.gui.jobs.cancellation import ProcessCancellationToken

    child_token = ProcessCancellationToken(event=cancel_event)
    child_token.wait(timeout=5.0)
    q.put(child_token.is_cancelled())


def _work_fn_success(progress_queue, cancel_event):
    """Work function that returns a simple result."""
    return {"answer": 42}


def _work_fn_raises(progress_queue, cancel_event):
    """Work function that raises ValueError."""
    raise ValueError("oops")


def _work_fn_cancellation(progress_queue, cancel_event):
    """Work function that raises CancellationError."""
    from rheojax.gui.jobs.cancellation import CancellationError

    raise CancellationError("user cancelled")


def _work_fn_with_progress(progress_queue, cancel_event):
    """Work function that sends a progress message then returns."""
    progress_queue.put(
        {"type": "progress", "percent": 50, "total": 100, "message": "halfway"}
    )
    return "done"


def _work_fn_returns_value(progress_queue, cancel_event):
    """Work function that returns a dict with 'value' key."""
    return {"value": 99}


def _work_fn_raises_runtime(progress_queue, cancel_event):
    """Work function that raises RuntimeError."""
    raise RuntimeError("boom")


def _work_fn_sleeps_long(progress_queue, cancel_event):
    """Work function that sleeps for 60 seconds (for cancel tests)."""
    import time

    time.sleep(60)


class TestProcessCancellationToken:
    """ProcessCancellationToken uses mp.Event for cross-process signaling."""

    def test_initial_state_not_cancelled(self):
        from rheojax.gui.jobs.cancellation import ProcessCancellationToken

        token = ProcessCancellationToken()
        assert not token.is_cancelled()

    def test_cancel_sets_event(self):
        from rheojax.gui.jobs.cancellation import ProcessCancellationToken

        token = ProcessCancellationToken()
        token.cancel()
        assert token.is_cancelled()

    def test_check_raises_after_cancel(self):
        from rheojax.gui.jobs.cancellation import (
            CancellationError,
            ProcessCancellationToken,
        )

        token = ProcessCancellationToken()
        token.cancel()
        with pytest.raises(CancellationError):
            token.check()

    def test_cross_process_cancellation(self):
        """Cancel in parent, observe in child."""
        from rheojax.gui.jobs.cancellation import ProcessCancellationToken

        token = ProcessCancellationToken()
        result_queue = mp.Queue()

        p = mp.Process(target=_child_wait_for_cancel, args=(token.event, result_queue))
        p.start()
        time.sleep(0.1)
        token.cancel()
        p.join(timeout=10)
        assert not p.is_alive()
        assert result_queue.get(timeout=5) is True

    def test_wait_returns_on_cancel(self):
        from rheojax.gui.jobs.cancellation import ProcessCancellationToken

        token = ProcessCancellationToken()
        token.cancel()
        assert token.wait(timeout=1.0) is True

    def test_wait_returns_false_on_timeout(self):
        from rheojax.gui.jobs.cancellation import ProcessCancellationToken

        token = ProcessCancellationToken()
        assert token.wait(timeout=0.05) is False

    def test_reset(self):
        from rheojax.gui.jobs.cancellation import ProcessCancellationToken

        token = ProcessCancellationToken()
        token.cancel()
        token.reset()
        assert not token.is_cancelled()


# ===========================================================================
# Tests for _subprocess_entry
# ===========================================================================


class TestSubprocessEntry:
    """_subprocess_entry runs a target function in a child process."""

    def test_successful_function_puts_completed(self):
        from rheojax.gui.jobs.process_adapter import _subprocess_entry

        result_queue = mp.Queue()
        cancel_event = mp.Event()

        p = mp.Process(
            target=_subprocess_entry,
            args=(_work_fn_success, result_queue, cancel_event),
        )
        p.start()
        p.join(timeout=30)
        assert p.exitcode == 0

        msg = result_queue.get(timeout=5)
        assert msg["type"] == "completed"
        assert msg["result"]["answer"] == 42

    def test_exception_puts_failed(self):
        from rheojax.gui.jobs.process_adapter import _subprocess_entry

        result_queue = mp.Queue()
        cancel_event = mp.Event()

        p = mp.Process(
            target=_subprocess_entry,
            args=(_work_fn_raises, result_queue, cancel_event),
        )
        p.start()
        p.join(timeout=30)

        msg = result_queue.get(timeout=5)
        assert msg["type"] == "failed"
        assert "oops" in msg["error"]
        assert "traceback" in msg

    def test_cancellation_puts_cancelled(self):
        from rheojax.gui.jobs.process_adapter import _subprocess_entry

        result_queue = mp.Queue()
        cancel_event = mp.Event()

        p = mp.Process(
            target=_subprocess_entry,
            args=(_work_fn_cancellation, result_queue, cancel_event),
        )
        p.start()
        p.join(timeout=30)

        msg = result_queue.get(timeout=5)
        assert msg["type"] == "cancelled"

    def test_progress_messages_forwarded(self):
        from rheojax.gui.jobs.process_adapter import _subprocess_entry

        result_queue = mp.Queue()
        cancel_event = mp.Event()

        p = mp.Process(
            target=_subprocess_entry,
            args=(_work_fn_with_progress, result_queue, cancel_event),
        )
        p.start()
        p.join(timeout=30)

        messages = []
        while not result_queue.empty():
            messages.append(result_queue.get(timeout=1))

        types = [m["type"] for m in messages]
        assert "progress" in types
        assert "completed" in types

        progress_msg = next(m for m in messages if m["type"] == "progress")
        assert progress_msg["percent"] == 50
        assert progress_msg["total"] == 100
        assert progress_msg["message"] == "halfway"


# ===========================================================================
# Tests for ProcessWorkerAdapter (requires PySide6)
# ===========================================================================

# Check PySide6 availability
try:
    from rheojax.gui.compat import QObject  # noqa: F811

    _HAS_PYSIDE6 = True
except ImportError:
    _HAS_PYSIDE6 = False

_SKIP_QT = not _HAS_PYSIDE6 or (
    not os.environ.get("DISPLAY") and not os.environ.get("QT_QPA_PLATFORM")
)


@pytest.mark.skipif(_SKIP_QT, reason="PySide6 not available or no display")
class TestProcessWorkerAdapter:
    """ProcessWorkerAdapter wraps work_fn in mp.Process with Qt signals."""

    @pytest.fixture(autouse=True)
    def _setup_qt(self, qapp):
        pass

    def test_successful_run_emits_completed(self):
        from rheojax.gui.jobs.process_adapter import ProcessWorkerAdapter

        adapter = ProcessWorkerAdapter(_work_fn_returns_value)
        completed_results = []
        adapter.signals.completed.connect(lambda r: completed_results.append(r))
        adapter.run()

        assert len(completed_results) == 1
        assert completed_results[0]["value"] == 99

    def test_failed_run_emits_failed(self):
        from rheojax.gui.jobs.process_adapter import ProcessWorkerAdapter

        adapter = ProcessWorkerAdapter(_work_fn_raises_runtime)
        errors = []
        adapter.signals.failed.connect(lambda msg: errors.append(msg))
        adapter.run()

        assert len(errors) == 1
        assert "boom" in errors[0]

    def test_cancel_terminates_process(self):
        import threading

        from rheojax.gui.jobs.process_adapter import ProcessWorkerAdapter

        adapter = ProcessWorkerAdapter(
            _work_fn_sleeps_long,
            process_timeout=1.0,
            kill_timeout=1.0,
        )
        run_thread = threading.Thread(target=adapter.run, daemon=True)
        run_thread.start()
        time.sleep(0.5)
        adapter.cancel()
        run_thread.join(timeout=15)
        assert not run_thread.is_alive()

    def test_progress_messages_emitted(self):
        from rheojax.gui.jobs.process_adapter import ProcessWorkerAdapter

        adapter = ProcessWorkerAdapter(_work_fn_with_progress)
        progress_msgs = []
        adapter.signals.progress.connect(
            lambda p, t, m: progress_msgs.append((p, t, m))
        )
        completed_results = []
        adapter.signals.completed.connect(lambda r: completed_results.append(r))
        adapter.run()

        assert any(p == 50 for p, _, _ in progress_msgs)
        assert len(completed_results) == 1

    def test_cancellation_emits_cancelled_signal(self):
        from rheojax.gui.jobs.process_adapter import ProcessWorkerAdapter

        adapter = ProcessWorkerAdapter(_work_fn_cancellation)
        cancelled_count = []
        adapter.signals.cancelled.connect(lambda: cancelled_count.append(1))
        adapter.run()

        assert len(cancelled_count) == 1
