# RheoJAX Workspace Shell

The default GUI shell for RheoJAX (`rheojax-gui` with no flags). A mode-based
interface — Fit, Transform, Pipeline — that replaces the legacy page-based
window (`rheojax/gui/app/`, `pages/`, `state/`, still available via
`rheojax-gui --legacy`) as the default entry point.

## Relationship to `foundation/`

This package is the UI layer only. All state, persistence, and cross-cutting
services live in `rheojax/gui/foundation/`:

- `foundation/state.py` — `AppState` and its slices (`FitState`,
  `TransformState`, `PipelineState`, `ProjectState`, `UiState`,
  `JobHistoryState`), the single state object each mode's controller reads
  and mutates.
- `foundation/library.py` — the in-memory dataset library (`DatasetRef` +
  payload store) shared across all three modes.
- `foundation/project_codec.py` — `.rheojax` v2 project file encode/decode.
- `foundation/import_service.py` — data import shared by CLI `--import` and
  the in-app import flow.
- `foundation/notifier.py`, `foundation/invalidation.py`,
  `foundation/pipeline_bridge.py`, `foundation/metrics.py`,
  `foundation/priors.py`, `foundation/contract.py` — supporting services
  (change notification, step invalidation, pipeline execution glue, priors,
  and the step-controller contract each mode's controller implements).

`workspace/` never talks to the legacy `app/`/`pages/`/`state/` package, and
vice versa — they are two independent shells over the same `rheojax` core
API, selected at startup by `rheojax/gui/main.py` based on the `--legacy`
flag.

## The Three Modes

`WorkspaceWindow` (`window.py`) hosts a toolbar with three mode buttons
(`WorkspaceWindow.MODES = ("fit", "transform", "pipeline")`) and a central
splitter of library rail / step canvas / inspector panel. Switching modes
swaps the step canvas; each mode owns its own controller and step sequence.

- **Fit** (`fit/`) — single-dataset step-by-step model fitting: pick a
  protocol and model (`step1_protocol_model.py`), select data
  (`step2_data.py`), run NLSQ (`step3_nlsq.py`), optionally run NUTS
  (`step4_nuts.py`), visualize (`step5_visualize.py`), export
  (`step6_export.py`).
- **Transform** (`transform/`) — single-dataset transform application: pick
  a transform (`step1_pick.py`), configure its slots (`step2_slots.py`), run
  it (`step3_run.py`), visualize (`step4_visualize.py`), export
  (`step5_export.py`).
- **Pipeline** (`pipeline/`) — batch orchestration across many datasets. A
  single configure/run step (`step1_configure_run.py`) lets a user assemble
  a sequence of transform/fit/export steps, select which library datasets to
  run them against, and hit "Run All". There is deliberately no per-step
  "Run Step" button — running one step interactively is what Fit/Transform
  modes are for. `batch_runner.py` executes the queued jobs on a
  `QThreadPool` worker; `cancel_runnable.py` and `models.py` support
  cancellation and the pipeline's step/job data model.

Each mode's controller (`fit_controller.py`, `transform_controller.py`,
`pipeline/controller.py`) drives a `StepperCanvas` (`stepper_canvas.py`) —
a linear step sequence with forward-only unlocking: a step becomes reachable
once the previous step is both ready and valid.

## File Menu / Project Model

`WorkspaceWindow._build_file_menu()` adds New / Open / Save / Save As /
Close under **File**. Save and Open round-trip through
`foundation/project_codec.py`'s `.rheojax` v2 format: a zip archive with
JSON metadata/state files (`fit.json`, `transform.json`, `pipeline.json`,
`job_history.json`, `project.json`, `ui.json`) plus HDF5 payloads for
dataset and result arrays, an allowlist of archive members, and SHA-256
checksums verified on load.

`WorkspaceWindow` tracks a `dirty` flag on `AppState.project` (set whenever
the dataset library changes) and prompts to save unsaved changes before
New/Open/Close. If jobs are still running in the background (pipeline batch
or fit/transform steps), it also prompts to cancel them first.

## Package Structure

```
rheojax/gui/
├── foundation/                 # State, persistence, and shared services (see above)
└── workspace/
    ├── window.py                # WorkspaceWindow: toolbar, File menu, mode switching
    ├── controller.py            # Shared step-controller base
    ├── stepper_canvas.py        # Linear step UI shared by all three modes
    ├── library_rail.py          # Dataset library panel (left)
    ├── inspector.py             # Inspector panel (right)
    ├── fit/                     # Fit mode: steps 1-6
    ├── transform/                # Transform mode: steps 1-5, transform slot specs
    └── pipeline/                 # Pipeline mode: configure/run step, batch runner, models
```
