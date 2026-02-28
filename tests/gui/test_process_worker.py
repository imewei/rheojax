"""Tests for subprocess worker isolation."""

import multiprocessing as mp
import time

import pytest


def _child_wait_for_cancel(cancel_event, q):
    """Top-level function for cross-process test (must be picklable)."""
    from rheojax.gui.jobs.cancellation import ProcessCancellationToken

    child_token = ProcessCancellationToken(event=cancel_event)
    child_token.wait(timeout=5.0)
    q.put(child_token.is_cancelled())


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
