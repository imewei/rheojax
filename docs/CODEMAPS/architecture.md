<!-- Generated: 2026-07-18 | Files scanned: ~342 source (excl. tests) | Token estimate: ~650 -->

# RheoJAX Architecture

JAX-accelerated rheology analysis framework: 53 constitutive models (22 families), 11 data
transforms, NLSQ+Bayesian(NumPyro) dual-track fitting, scikit-learn-style API + PySide6 GUI.

## Module Graph

```
rheojax/
├── core/        (17 src)  BaseModel/BaseTransform, RheoData, Registry, FitOrchestrator, NumPyro builder
├── models/      (105 src) 53 models / 22 families, lazy __getattr__ registration
├── transforms/  (12 src)  FFT, mastercurve/TTS-SRFS, SPP decomposition, Prony — eager-imported
├── io/          (27 src)  TRIOS/Anton Paar/CSV/Excel/HDF5 readers+writers, format auto-detect
├── pipeline/    (6 src)   Pipeline/BayesianPipeline fluent API, PipelineBuilder, BatchPipeline
├── cli/         (17 src)  `rheojax`/`rj` entry point: fit/bayesian/spp/pipeline/batch/info
├── gui/         (115 src) PySide6 desktop app (`rheojax-gui`) — see frontend.md
├── utils/       (33 src)  Mittag-Leffler, Prony series, NLSQ helpers, uncertainty, init heuristics
├── parallel/    (4 src)   Process-pool fit/bayesian/batch parallelism
└── logging/     (6 src)   Structured logging shared by core/cli/gui
```

Dependency direction: everything → `core`. `models`/`transforms` register into `core.Registry` via
decorators. `pipeline` and `cli` orchestrate `core` + `models`/`transforms`. `gui` calls `core`/
`pipeline`/`io` through a service layer (never touches JAX directly — see frontend.md).

## Data Flow (fit path)

```
file (CSV/HDF5/TRIOS) --io.readers--> RheoData --core.FitOrchestrator--
    --> NLSQ (utils.optimization) --warm-start--> NumPyro NUTS (core.numpyro_model_builder)
    --> FitResult (params + ArviZ diagnostics: R-hat/ESS/BFMI)
```

Same RheoData container flows through `transforms` (FFT/mastercurve/SPP/...) independently of the
fit path; `pipeline.Pipeline`/`BayesianPipeline` chain transform → fit steps declaratively.

## Key Invariants (see root CLAUDE.md for full detail)

- **float64 forced** at package import (`rheojax/__init__.py`); internal code must use
  `core.jax_config.safe_import_jax()`, never bare `import jax`.
- **Lazy registration**: `models`, `PersistentProcessPool`, and the root package's `models`
  attribute all use `__getattr__` — `_ensure_all_registered()` forces eager load when needed
  (CLI, GUI model-selection step, tests).
- **NLSQ → NUTS handoff** is the standard Bayesian path (`core/fit_orchestrator.py`).
- **Single GUI shell**: `WorkspaceWindow` (`gui/workspace/window.py`) is the only entry point;
  legacy `RheoJAXMainWindow`/`--legacy` CLI flag removed (BREAKING, see CHANGELOG `[Unreleased]`).

## Entry Points

| Command | File |
|---|---|
| `rheojax` / `rj` (CLI) | `rheojax/cli/main.py` |
| `rheojax-gui` / `rj-gui` | `rheojax/gui/main.py` → `WorkspaceWindow` |
| Library import | `import rheojax` → `rheojax/__init__.py` (triggers x64 config) |

See `backend.md` (core/pipeline/cli), `frontend.md` (gui), `data.md` (RheoData/io), and
`dependencies.md` (external stack).
