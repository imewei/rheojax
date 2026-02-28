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


# ===========================================================================
# Tests for run_fit_isolated
# ===========================================================================


class TestRunFitIsolated:
    """run_fit_isolated is a pure function for subprocess NLSQ fitting."""

    def test_basic_fit_returns_result(self):
        import multiprocessing as mp
        import numpy as np
        from rheojax.gui.jobs.subprocess_fit import run_fit_isolated

        # Maxwell relaxation: G(t) = G0 * exp(-t/tau), where tau = eta/G0
        t = np.linspace(0.01, 5, 100)
        G0, eta = 1000.0, 1000.0  # tau = eta/G0 = 1.0
        G_t = G0 * np.exp(-t * G0 / eta)

        result = run_fit_isolated(
            model_name="maxwell",
            x_data=t, y_data=G_t,
            test_mode="relaxation",
            initial_params={"G0": 500.0, "eta": 500.0},
            options={"max_iter": 500},
            progress_queue=mp.Queue(),
            cancel_event=mp.Event(),
        )

        assert result["success"]
        assert result["model_name"] == "maxwell"
        assert isinstance(result["parameters"], dict)
        assert "G0" in result["parameters"]
        assert "eta" in result["parameters"]
        assert result["parameters"]["G0"] > 0
        assert result["parameters"]["eta"] > 0
        assert result["fit_time"] > 0
        assert result["timestamp"]  # non-empty ISO string

    def test_all_arrays_are_numpy(self):
        import multiprocessing as mp
        import numpy as np
        from rheojax.gui.jobs.subprocess_fit import run_fit_isolated

        t = np.linspace(0.01, 5, 100)
        G_t = 1000.0 * np.exp(-t / 1.0)

        result = run_fit_isolated(
            model_name="maxwell",
            x_data=t, y_data=G_t,
            test_mode="relaxation",
            initial_params={},
            options={},
            progress_queue=mp.Queue(),
            cancel_event=mp.Event(),
        )

        if result.get("x_fit") is not None:
            assert isinstance(result["x_fit"], np.ndarray)
        if result.get("y_fit") is not None:
            assert isinstance(result["y_fit"], np.ndarray)
        if result.get("residuals") is not None:
            assert isinstance(result["residuals"], np.ndarray)


# ===========================================================================
# Tests for run_bayesian_isolated
# ===========================================================================


@pytest.mark.slow
class TestRunBayesianIsolated:
    """run_bayesian_isolated is a pure function for subprocess NUTS sampling."""

    def test_basic_bayesian_returns_result(self):
        import multiprocessing as mp

        import numpy as np

        from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated

        t = np.linspace(0.01, 5, 50)
        rng = np.random.default_rng(42)
        # Maxwell: G(t) = G0 * exp(-t * G0/eta) -- use G0=1000, eta=1000 (tau=1)
        G_t = 1000.0 * np.exp(-t / 1.0) + rng.normal(0, 10, len(t))

        result = run_bayesian_isolated(
            model_name="maxwell",
            x_data=t,
            y_data=G_t,
            test_mode="relaxation",
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            warm_start={"G0": 1000.0, "eta": 1000.0},
            priors={},
            seed=42,
            progress_queue=mp.Queue(),
            cancel_event=mp.Event(),
        )

        assert result["model_name"] == "maxwell"
        assert "posterior_samples" in result
        assert isinstance(result["posterior_samples"], dict)
        assert len(result["posterior_samples"]) > 0
        for name, samples in result["posterior_samples"].items():
            assert isinstance(samples, np.ndarray), f"{name} is not NumPy"
        assert result["mcmc_time"] > 0
        assert result["timestamp"]  # non-empty ISO string
        assert result["num_warmup"] == 50
        assert result["num_samples"] == 100
        assert result["num_chains"] == 1

    def test_inference_data_is_none(self):
        import multiprocessing as mp

        import numpy as np

        from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated

        t = np.linspace(0.01, 5, 50)
        G_t = 1000.0 * np.exp(-t / 1.0)

        result = run_bayesian_isolated(
            model_name="maxwell",
            x_data=t,
            y_data=G_t,
            test_mode="relaxation",
            num_warmup=25,
            num_samples=50,
            num_chains=1,
            warm_start={"G0": 1000.0, "eta": 1000.0},
            priors={},
            seed=0,
            progress_queue=mp.Queue(),
            cancel_event=mp.Event(),
        )

        assert result.get("inference_data") is None

    def test_result_has_diagnostics(self):
        import multiprocessing as mp

        import numpy as np

        from rheojax.gui.jobs.subprocess_bayesian import run_bayesian_isolated

        t = np.linspace(0.01, 5, 50)
        G_t = 1000.0 * np.exp(-t / 1.0)

        result = run_bayesian_isolated(
            model_name="maxwell",
            x_data=t,
            y_data=G_t,
            test_mode="relaxation",
            num_warmup=25,
            num_samples=50,
            num_chains=1,
            warm_start={"G0": 1000.0, "eta": 1000.0},
            priors={},
            seed=0,
            progress_queue=mp.Queue(),
            cancel_event=mp.Event(),
        )

        assert "r_hat" in result
        assert "ess" in result
        assert "divergences" in result
        assert isinstance(result["divergences"], int)
        assert "diagnostics_valid" in result
        assert isinstance(result["diagnostics_valid"], bool)
        assert "credible_intervals" in result
        assert isinstance(result["credible_intervals"], dict)


# ===========================================================================
# Tests for worker isolation config
# ===========================================================================


class TestWorkerIsolationConfig:
    """get_worker_isolation_mode reads RHEOJAX_WORKER_ISOLATION env var."""

    def test_default_is_subprocess(self):
        from rheojax.gui.jobs.process_adapter import get_worker_isolation_mode

        os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)
        assert get_worker_isolation_mode() == "subprocess"

    def test_thread_fallback(self):
        from rheojax.gui.jobs.process_adapter import get_worker_isolation_mode

        os.environ["RHEOJAX_WORKER_ISOLATION"] = "thread"
        try:
            assert get_worker_isolation_mode() == "thread"
        finally:
            os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)

    def test_subprocess_explicit(self):
        from rheojax.gui.jobs.process_adapter import get_worker_isolation_mode

        os.environ["RHEOJAX_WORKER_ISOLATION"] = "subprocess"
        try:
            assert get_worker_isolation_mode() == "subprocess"
        finally:
            os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)


# ===========================================================================
# Tests for result reconstruction
# ===========================================================================


class TestResultReconstruction:
    """fit_result_from_dict / bayesian_result_from_dict reconstruct dataclasses."""

    def test_fit_result_from_dict(self):
        from rheojax.gui.jobs.process_adapter import fit_result_from_dict
        from rheojax.gui.state.store import FitResult

        d = {
            "model_name": "maxwell",
            "parameters": {"G0": 1000.0, "eta": 1000.0},
            "chi_squared": 0.01,
            "success": True,
            "message": "Converged",
            "timestamp": "2026-02-28T12:00:00",
            "r_squared": 0.999,
            "mpe": 0.1,
            "fit_time": 2.5,
            "num_iterations": 42,
            "dataset_id": "test",
        }
        result = fit_result_from_dict(d)
        assert isinstance(result, FitResult)
        assert result.model_name == "maxwell"
        assert result.success
        assert result.parameters == {"G0": 1000.0, "eta": 1000.0}
        assert result.chi_squared == 0.01
        assert result.r_squared == 0.999
        assert result.dataset_id == "test"

    def test_fit_result_from_dict_missing_fields(self):
        from rheojax.gui.jobs.process_adapter import fit_result_from_dict
        from rheojax.gui.state.store import FitResult

        d = {
            "model_name": "maxwell",
            "parameters": {},
            "success": False,
            "message": "Failed",
        }
        result = fit_result_from_dict(d)
        assert isinstance(result, FitResult)
        assert not result.success
        assert result.chi_squared == 0.0

    def test_bayesian_result_from_dict(self):
        import numpy as np

        from rheojax.gui.jobs.process_adapter import bayesian_result_from_dict
        from rheojax.gui.state.store import BayesianResult

        d = {
            "model_name": "maxwell",
            "dataset_id": "test",
            "posterior_samples": {"G0": np.array([900, 1000, 1100])},
            "summary": {"G0": {"mean": 1000.0}},
            "r_hat": {"G0": 1.001},
            "ess": {"G0": 500.0},
            "divergences": 0,
            "credible_intervals": {"G0": (900.0, 1100.0)},
            "mcmc_time": 30.0,
            "timestamp": "2026-02-28T12:00:00",
            "num_warmup": 100,
            "num_samples": 200,
            "num_chains": 2,
            "inference_data": None,
            "diagnostics_valid": True,
        }
        result = bayesian_result_from_dict(d)
        assert isinstance(result, BayesianResult)
        assert result.model_name == "maxwell"
        assert result.dataset_id == "test"
        assert result.inference_data is None
        assert result.num_warmup == 100
        assert result.num_samples == 200
        assert result.num_chains == 2
        assert result.divergences == 0
        assert result.diagnostics_valid is True

    def test_bayesian_result_from_dict_missing_fields(self):
        from rheojax.gui.jobs.process_adapter import bayesian_result_from_dict
        from rheojax.gui.state.store import BayesianResult

        d = {
            "model_name": "maxwell",
            "dataset_id": "test",
            "posterior_samples": {},
        }
        result = bayesian_result_from_dict(d)
        assert isinstance(result, BayesianResult)
        assert result.mcmc_time == 0.0
        assert result.divergences == 0


# ===========================================================================
# Tests for make_fit_worker / make_bayesian_worker factories
# ===========================================================================


class TestMakeFitWorker:
    """make_fit_worker returns FitWorker in thread mode, ProcessWorkerAdapter otherwise."""

    def test_thread_mode_returns_fit_worker(self):
        import numpy as np

        os.environ["RHEOJAX_WORKER_ISOLATION"] = "thread"
        try:
            from rheojax.gui.jobs.fit_worker import FitWorker
            from rheojax.gui.jobs.process_adapter import make_fit_worker

            # Use a minimal mock for data
            class FakeData:
                x = np.linspace(0, 1, 10)
                y = np.ones(10)
                metadata = {"test_mode": "relaxation"}
                _explicit_test_mode = "relaxation"

            worker = make_fit_worker(
                model_name="maxwell",
                data=FakeData(),
                initial_params={"G0": 1000.0},
                options={},
                dataset_id="test",
            )
            assert isinstance(worker, FitWorker)
        finally:
            os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)

    @pytest.mark.skipif(_SKIP_QT, reason="PySide6 not available or no display")
    def test_subprocess_mode_returns_adapter(self, qapp):
        import numpy as np

        os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)
        from rheojax.gui.jobs.process_adapter import (
            ProcessWorkerAdapter,
            make_fit_worker,
        )

        class FakeData:
            x = np.linspace(0, 1, 10)
            y = np.ones(10)
            metadata = {"test_mode": "relaxation"}
            _explicit_test_mode = "relaxation"

        worker = make_fit_worker(
            model_name="maxwell",
            data=FakeData(),
            initial_params={"G0": 1000.0},
            options={},
            dataset_id="test",
        )
        assert isinstance(worker, ProcessWorkerAdapter)


class TestMakeBayesianWorker:
    """make_bayesian_worker returns BayesianWorker or ProcessWorkerAdapter."""

    def test_thread_mode_returns_bayesian_worker(self):
        import numpy as np

        os.environ["RHEOJAX_WORKER_ISOLATION"] = "thread"
        try:
            from rheojax.gui.jobs.bayesian_worker import BayesianWorker
            from rheojax.gui.jobs.process_adapter import make_bayesian_worker

            class FakeData:
                x_data = np.linspace(0, 1, 10)
                y_data = np.ones(10)
                y2_data = None
                test_mode = "relaxation"
                metadata = {}

            worker = make_bayesian_worker(
                model_name="maxwell",
                data=FakeData(),
                num_warmup=50,
                num_samples=100,
                num_chains=1,
                seed=42,
                dataset_id="test",
            )
            assert isinstance(worker, BayesianWorker)
        finally:
            os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)

    @pytest.mark.skipif(_SKIP_QT, reason="PySide6 not available or no display")
    def test_subprocess_mode_returns_adapter(self, qapp):
        import numpy as np

        os.environ.pop("RHEOJAX_WORKER_ISOLATION", None)
        from rheojax.gui.jobs.process_adapter import (
            ProcessWorkerAdapter,
            make_bayesian_worker,
        )

        class FakeData:
            x_data = np.linspace(0, 1, 10)
            y_data = np.ones(10)
            y2_data = None
            test_mode = "relaxation"
            metadata = {}

        worker = make_bayesian_worker(
            model_name="maxwell",
            data=FakeData(),
            num_warmup=50,
            num_samples=100,
            num_chains=1,
            seed=42,
            dataset_id="test",
        )
        assert isinstance(worker, ProcessWorkerAdapter)
