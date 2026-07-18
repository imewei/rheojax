<!-- Generated: 2026-07-18 | Files scanned: ~115 (rheojax/gui) | Token estimate: ~700 -->

# GUI (PySide6 desktop app, `rheojax-gui`)

Single-shell app. No web frontend — this replaces the template's "page tree"/"component
hierarchy" with the Qt widget tree + Redux-style state store.

## Shell

`WorkspaceWindow` (`gui/workspace/window.py`) — sole entry point. 3-step-wizard model: Fit /
Transform / Pipeline modes, switched via toolbar `QToolButton`s (`_fit_btn`/`_tx_btn`/
`_pipeline_btn`). Each mode owns a `StepperCanvas` (`gui/workspace/stepper_canvas.py`) — a numbered
step rail + `QStackedWidget` body, gated by `WorkflowController.reached` (forward-locked,
backward-revisitable). `Ctrl+1..9` jump steps; `Ctrl+K` opens a command palette.

## Wizard Steps

```
Fit:       1 Protocol&Model → 2 Data → 3 NLSQ → 4 NUTS → 5 Visualize → 6 Export
Transform: 1 Slots → 2 Config → 3 Run → 4 Visualize → 5 Export
Pipeline:  1 Configure&Run  (batch execution via PipelineExecutionService)
```
Built by `fit_controller.build_fit_controller(AppState)` /
`transform_controller.build_transform_controller(AppState)` — each returns
`(WorkflowController, [step widgets])`. All three built eagerly in
`WorkspaceWindow._build_workspace`, not lazily per mode-switch.

## State (Redux-style, `gui/state/` + `gui/foundation/state.py`)

`AppState` → `FitState` / `TransformState` / `PipelineState` / `DatasetLibrary` /
`ActiveJobsState` (`by_id: dict[str, dict]`, tracks in-flight background jobs for
Close/New/Open confirmation) / `UiState` (theme, mode). `StateStore`/`StateSignals`
(`gui/state/store.py`, `signals.py`) — granular per-domain Qt signals, not one broad
`state_changed` fan-out.

## Services (`gui/services/`) — GUI's only path to `core`/`pipeline`

`DataService`, `TransformService`, `ModelService` (`FAMILY_LABELS` for model-picker grouping),
`BayesianService`, `PlotService`, `ExportService`, `PipelineExecutionService`. GUI code never
imports JAX directly — always through these.

## Background Jobs (`gui/jobs/`)

`QRunnable`+`QThreadPool` workers, dispatched off the GUI thread and registered in
`AppState.active_jobs.by_id` so window-close waits for them: `FitWorker`, `BayesianWorker`,
`ImportWorker`, `ExportWorker`, `PipelineBatchRunner`. Established pattern (post-PR-#90): hold a
Python reference to the worker for its lifetime (parentless `signals` QObject risks GC mid-run
otherwise), register/deregister via a `job_id` in `active_jobs.by_id`.

## Plotting

`gui/widgets/pyqtgraph_canvas.py` (`RheoPlotCanvas`/`PyQtGraphCanvas`) — interactive PyQtGraph,
downsampling+clip-to-view enabled (PR #90). `gui/widgets/arviz_canvas.py`/`residuals_panel.py` —
matplotlib-backed diagnostic plots (separate from the interactive canvas).

## Key Files

- `gui/main.py` — entry point, JAX status probe (deferred via QTimer as of PR #90)
- `gui/workspace/window.py` — shell, menus, mode/job/theme management (~1400 lines)
- `gui/workspace/fit/fit_controller.py`, `transform/transform_controller.py` — wizard assembly
- `gui/foundation/state.py` — dataclass state tree
- `gui/foundation/project_codec.py` — project save/load (v2 schema)
