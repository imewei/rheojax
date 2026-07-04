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


class PipelineBatchRunner(QRunnable):
    def __init__(self, service, steps, selected_dataset_ids, library, stop_requested) -> None:
        super().__init__()
        self._service = service
        self._steps = steps
        self._selected_dataset_ids = selected_dataset_ids
        self._library = library
        self._stop_requested = stop_requested

    def run(self) -> None:
        from rheojax.gui.services.pipeline_execution_service import (
            WorkerIsolationRequiredError,
        )

        for dataset_id in self._selected_dataset_ids:
            if self._stop_requested.is_set():
                break
            self._service.dataset_run_started.emit(dataset_id)
            ctx = pipeline_context_from_library(self._library, [dataset_id])
            steps_for_dataset = self._substitute_dataset_id(self._steps, dataset_id)
            fatal = False
            try:
                result = self._service.execute(
                    steps=steps_for_dataset, initial_context=ctx, library=self._library,
                    stop_requested=self._stop_requested,
                )
                record = self._prepare_job_record(dataset_id, result)
            except WorkerIsolationRequiredError as exc:
                # Misconfigured environment -- would fail identically for every
                # remaining dataset, so this is the batch's last iteration too.
                record = self._failed_record(dataset_id, str(exc))
                fatal = True
            except Exception as exc:  # pragma: no cover - defensive backstop
                # execute() already converts ordinary step failures into a
                # status="failed" PipelineRunResult internally; this only catches an
                # unexpected bug elsewhere in the call chain (e.g. context/record
                # construction). Without this, such an exception would propagate out
                # of run() with dataset_run_finished never emitted, leaving this
                # dataset's active_jobs entry stuck forever (nothing else clears it).
                record = self._failed_record(dataset_id, str(exc))
            self._service.dataset_run_finished.emit(dataset_id, record)
            if fatal:
                break

    @staticmethod
    def _failed_record(dataset_id: str, error: str) -> dict:
        return {"dataset_id": dataset_id, "status": "failed", "error": error, "step_results": {}}

    @staticmethod
    def _substitute_dataset_id(steps, dataset_id: str) -> list:
        substituted = []
        for step in steps:
            new_config = dict(step.config)
            if "path" in new_config and "{id}" in new_config["path"]:
                new_config["path"] = new_config["path"].format(id=dataset_id)
            substituted.append(type(step)(id=step.id, step_type=step.step_type, config=new_config))
        return substituted

    def _prepare_job_record(self, dataset_id: str, result) -> dict:
        step_results_out: dict[str, Any] = {}
        for step_id, step_result in result.step_results.items():
            if isinstance(step_result, FitStepResult):
                step_results_out[step_id] = {
                    "step_type": "fit",
                    "nlsq": self._phase_to_dict(step_result.nlsq),
                    "nuts": self._phase_to_dict(step_result.nuts) if step_result.nuts else None,
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
