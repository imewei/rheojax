"""PipelineController: Pipeline mode's WorkflowController plus the sole GUI-thread commit path
for a finished batch job (design spec §3.3)."""

from __future__ import annotations

import uuid
from collections.abc import Callable

from rheojax.gui.workspace.controller import WorkflowController

# ponytail: Qt.AutoConnection (the connect() default) resolves to Direct for
# same-thread emission and Queued for cross-thread emission, so it already
# guarantees GUI-thread delivery when PipelineBatchRunner emits from a worker
# thread. An explicit QueuedConnection would always defer to the next event
# loop iteration -- which breaks pytest-qt's waitSignal() (its internal
# listener fires synchronously and short-circuits the loop before a queued
# slot on this object gets a chance to run).


class PipelineController(WorkflowController):
    STEP_IDS = ["configure_run"]

    def __init__(
        self,
        app_state,
        service,
        on_dirty: Callable[[], None],
        epoch: int = 0,
        guard: Callable[[int, Callable], Callable] | None = None,
        notify: Callable[[], None] | None = None,
    ) -> None:
        self._state = app_state
        self._service = service
        self._on_dirty = on_dirty
        self._notify = notify
        super().__init__(steps=[])
        on_started = self._on_dataset_run_started
        on_finished = self._commit_job_result
        # guard() no-ops these handlers once WorkspaceWindow._epoch advances past
        # `epoch` (a rebuild happened), so a stale controller left connected to the
        # persistent PipelineExecutionService can't write into a discarded AppState.
        if guard is not None:
            on_started = guard(epoch, on_started)
            on_finished = guard(epoch, on_finished)
        self._service.dataset_run_started.connect(on_started)
        self._service.dataset_run_finished.connect(on_finished)
        # Keep the exact connected callables so WorkspaceWindow._dispose_workspace
        # can disconnect them from the persistent service on rebuild -- without
        # this, every New/Open leaves the old controller (and the AppState it
        # closed over) connected and reachable for the life of the window.
        self._started_slot = on_started
        self._finished_slot = on_finished

    def _on_dataset_run_started(self, dataset_id: str) -> None:
        # Registers ONE dataset's active_jobs entry right before it starts running -- not the
        # whole batch upfront. If the batch is cancelled mid-run, only the currently-running
        # dataset is ever "active"; PipelineBatchRunner.run()'s stop_requested check (batch_runner.py)
        # is what actually prevents it from advancing to the next queued dataset, so there's
        # never a queued-but-never-cleared active_jobs entry (design spec §3.3's job-lifecycle
        # rule; see this plan's window.py Task 10/11 for why pre-registering the whole batch was
        # a bug).
        self._state.active_jobs.by_id[dataset_id] = {"status": "running"}

    def _commit_job_result(self, dataset_id: str, record: dict) -> None:
        self._state.active_jobs.by_id.pop(dataset_id, None)
        job_id = uuid.uuid4().hex
        self._state.job_history.by_id[job_id] = record
        self._state.pipeline.job_id = job_id
        self._on_dirty()
        if self._notify is not None:
            # Pipeline transform steps add/store into the library directly on the
            # worker thread (PipelineExecutionService._execute_pipeline_transform),
            # bypassing WorkspaceWindow._commit_dataset()'s notifier emission. By the
            # time this GUI-thread slot runs, that thread's execute() call has already
            # returned (dataset_run_finished fires after it), so the library mutation
            # has already happened -- reading/notifying about it here is safe and is
            # what makes a pipeline-produced dataset show up in LibraryRail at all.
            self._notify()


def build_pipeline_controller(
    app_state, service, epoch: int = 0, guard=None, notify=None
):
    from rheojax.gui.workspace.controller import Step
    from rheojax.gui.workspace.pipeline.step1_configure_run import (
        PipelineConfigureRunStep,
    )

    body = PipelineConfigureRunStep(app_state.pipeline, app_state.library)
    ctl = PipelineController(
        app_state,
        service,
        on_dirty=lambda: setattr(app_state.project, "dirty", True),
        epoch=epoch,
        guard=guard,
        notify=notify,
    )
    ctl.steps = [
        Step(
            id="configure_run",
            title="Configure & Run",
            is_ready=body.is_ready,
            validate=lambda: True,
        )
    ]
    ctl.reached = {0}
    return ctl, [body]
