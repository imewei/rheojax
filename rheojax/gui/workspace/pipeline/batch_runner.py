"""QRunnable driving PipelineExecutionService.execute() once per selected dataset, sequentially.

This is the WT-side half of the job-commit split (design spec §3.3): _prepare_job_record() builds
a plain-dict record shape (still holding raw, in-memory PhaseResult.result values -- persistence
to job_results/*.hdf5 happens later, at Save time, in save_project_v2(); see the design note
above this task's Interfaces section) but never mutates AppState directly -- that happens only
in PipelineController._commit_job_result() (controller.py), the GUI-thread slot connected to
dataset_run_finished.
"""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QRunnable

from rheojax.gui.foundation.pipeline_bridge import pipeline_context_from_library
from rheojax.gui.workspace.pipeline.models import FitStepResult
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PipelineBatchRunner(QRunnable):
    def __init__(
        self,
        service,
        steps,
        selected_dataset_ids,
        library,
        stop_requested,
        app_state=None,
        batch_job_id: str | None = None,
    ) -> None:
        super().__init__()
        self._service = service
        self._steps = steps
        self._selected_dataset_ids = selected_dataset_ids
        self._library = library
        self._stop_requested = stop_requested
        self._app_state = app_state
        # Caller (WorkspaceWindow._on_pipeline_run_requested) pre-registers this key in
        # active_jobs synchronously on the GUI thread, before QThreadPool.start()
        # returns, so a double-click "Run All" sees a non-empty active_jobs even before
        # this thread reaches its first per-dataset registration below. We own clearing
        # it once the batch reaches a terminal state.
        self._batch_job_id = batch_job_id

    def run(self) -> None:
        from rheojax.gui.services.pipeline_execution_service import (
            WorkerIsolationRequiredError,
        )

        try:
            # ponytail: stop_requested is checked here (between datasets) and again
            # inside self._service.execute() (between steps, and between a fit
            # step's NLSQ/NUTS phases) -- but once a phase's worker.run() actually
            # starts (a blocking subprocess call inside _run_worker_phase), nothing
            # polls stop_requested again until that call returns. Cancelling
            # mid-NUTS-run genuinely can't be made finer-grained without the
            # worker/subprocess protocol itself supporting mid-run interruption;
            # upgrade path is a cancellation channel into ProcessWorkerAdapter, not
            # another check here.
            for idx, dataset_id in enumerate(self._selected_dataset_ids):
                if self._stop_requested.is_set():
                    break
                # Register synchronously (plain dict write, not via the Qt signal)
                # before execute() can start mutating the library. dataset_run_started
                # is Qt.AutoConnection, which resolves to a queued cross-thread
                # connection when run() executes on a QThreadPool worker thread --
                # relying on it alone leaves a window where Save can see an empty
                # active_jobs and serialize the library while this thread is
                # concurrently writing to it. `app_state` is optional so tests that
                # construct this runner directly (with no AppState) are unaffected.
                if self._app_state is not None:
                    with self._app_state.active_jobs.lock:
                        self._app_state.active_jobs.by_id[dataset_id] = {
                            "status": "running"
                        }
                self._service.dataset_run_started.emit(dataset_id)
                ctx = pipeline_context_from_library(self._library, [dataset_id])
                steps_for_dataset = self._substitute_dataset_id(self._steps, dataset_id)
                fatal = False
                error_msg = None
                try:
                    result = self._service.execute(
                        steps=steps_for_dataset,
                        initial_context=ctx,
                        library=self._library,
                        stop_requested=self._stop_requested,
                    )
                    record = self._prepare_job_record(dataset_id, result)
                except WorkerIsolationRequiredError as exc:
                    # Misconfigured environment -- would fail identically for every
                    # remaining dataset, so this is the batch's last iteration too.
                    error_msg = str(exc)
                    record = self._failed_record(dataset_id, error_msg)
                    fatal = True
                except Exception as exc:  # pragma: no cover - defensive backstop
                    # execute() already converts ordinary step failures into a
                    # status="failed" PipelineRunResult internally; this only catches an
                    # unexpected bug elsewhere in the call chain (e.g. context/record
                    # construction). Without this, such an exception would propagate out
                    # of run() with dataset_run_finished never emitted, leaving this
                    # dataset's active_jobs entry stuck forever (nothing else clears it).
                    logger.error(
                        "Unexpected error running pipeline for dataset",
                        dataset_id=dataset_id,
                        error=str(exc),
                        exc_info=True,
                    )
                    record = self._failed_record(dataset_id, str(exc))
                self._service.dataset_run_finished.emit(dataset_id, record)
                if fatal:
                    # Every dataset still queued behind this one would hit the exact
                    # same fatal precondition error -- record each as "skipped" (via
                    # the same dataset_run_finished path PipelineController already
                    # commits to job_history) instead of leaving it with no
                    # job_history/active_jobs entry at all, which was indistinguishable
                    # from "never got to it yet" when reviewing results afterwards.
                    remaining = self._selected_dataset_ids[idx + 1 :]
                    if remaining:
                        logger.warning(
                            "Skipping remaining datasets after fatal precondition error",
                            skipped_count=len(remaining),
                            reason=error_msg,
                        )
                    for skipped_id in remaining:
                        self._service.dataset_run_finished.emit(
                            skipped_id, self._skipped_record(skipped_id, error_msg)
                        )
                    break
        finally:
            # Runs on every exit path (normal completion, stop_requested break, fatal
            # break, empty selected_dataset_ids) so the pre-registered sentinel never
            # outlives the batch, regardless of how it ends.
            if self._app_state is not None and self._batch_job_id is not None:
                with self._app_state.active_jobs.lock:
                    self._app_state.active_jobs.by_id.pop(self._batch_job_id, None)

    @staticmethod
    def _failed_record(dataset_id: str, error: str) -> dict:
        return {
            "dataset_id": dataset_id,
            "status": "failed",
            "error": error,
            "step_results": {},
        }

    @staticmethod
    def _skipped_record(dataset_id: str, reason: str | None) -> dict:
        return {
            "dataset_id": dataset_id,
            "status": "skipped",
            "error": f"Batch aborted before this dataset could run: {reason}",
            "step_results": {},
        }

    @staticmethod
    def _substitute_dataset_id(steps, dataset_id: str) -> list:
        substituted = []
        for step in steps:
            new_config = dict(step.config)
            if "path" in new_config and "{id}" in new_config["path"]:
                new_config["path"] = new_config["path"].format(id=dataset_id)
            substituted.append(
                type(step)(id=step.id, step_type=step.step_type, config=new_config)
            )
        return substituted

    def _prepare_job_record(self, dataset_id: str, result) -> dict:
        step_results_out: dict[str, Any] = {}
        for step_id, step_result in result.step_results.items():
            if isinstance(step_result, FitStepResult):
                step_results_out[step_id] = {
                    "step_type": "fit",
                    "nlsq": self._phase_to_dict(step_result.nlsq),
                    "nuts": self._phase_to_dict(step_result.nuts)
                    if step_result.nuts
                    else None,
                }
            else:
                step_results_out[step_id] = {"step_type": "other", **step_result}

        return {
            "dataset_id": dataset_id,
            "status": result.status,
            "error": result.error,
            "step_results": step_results_out,
        }

    @staticmethod
    def _phase_to_dict(phase) -> dict:
        # Deliberately NOT calling write_result_arrays() here -- see this task's design note.
        return {"status": phase.status, "error": phase.error, "result": phase.result}
