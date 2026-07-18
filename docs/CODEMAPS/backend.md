<!-- Generated: 2026-07-18 | Files scanned: ~40 (core+pipeline+cli) | Token estimate: ~550 -->

# Core / Pipeline / CLI ("backend" layer — no HTTP server; library + CLI)

No web routes. Equivalent surface: CLI subcommands and the fluent Pipeline API, both sitting on
`core`'s Registry + FitOrchestrator.

## CLI Commands → Core Call Chain

```
rheojax fit <file>       → cli.fit         → core.Registry.get(model) → FitOrchestrator.fit_nlsq
rheojax bayesian <file>  → cli.bayesian    → FitOrchestrator.fit_nlsq → fit_bayesian (NUTS warm-start)
rheojax spp <file>       → cli.spp         → transforms.SPPDecomposer
rheojax load/transform   → cli.load/transform → io.readers.auto_load → transforms.Registry
rheojax export           → cli.export      → io.writers (HDF5/Excel/NPZ, SPP MATLAB-compat)
rheojax pipeline <yaml>  → cli.pipeline    → pipeline.PipelineBuilder.from_yaml
rheojax batch            → cli.batch       → pipeline.BatchPipeline (+ parallel.PersistentProcessPool)
rheojax info/inventory   → cli.info        → core.Registry introspection
```

## Core Registry

`core/registry.py` — `Registry` (singleton), `ModelRegistry`, `TransformRegistry`. Models/
transforms self-register via `@ModelRegistry.register`/`@TransformRegistry.register` decorators.
`PluginInfo.doc` = `inspect.getdoc(plugin_class)`, used for CLI `info` output and (as of PR #90)
GUI model-picker tooltips.

## FitOrchestrator (`core/fit_orchestrator.py`)

- `fit_nlsq(model, data)` → NLSQ-wrapped optimizer (`utils/optimization.py`)
- `fit_bayesian(model, data, nlsq_result=...)` → warm-starts NumPyro NUTS via
  `core/numpyro_model_builder.py`; diagnostics via `core/arviz_utils.py` (arviz 1.x kwarg shim)

## Pipeline API (`pipeline/`)

- `Pipeline`/`BayesianPipeline` — fluent, chainable transform→fit steps
- `PipelineBuilder` — YAML → Pipeline (used by `cli pipeline` and `cli batch`)
- `BatchPipeline` — multi-dataset batch execution, routes through `parallel/pool.py`
- Preset workflows: mastercurve, model comparison, creep transforms (`pipeline/workflows.py`)

## Key Files

- `core/base.py` — `BaseModel`/`BaseTransform` abstract contracts
- `core/data.py` — `RheoData` container (see data.md)
- `core/registry.py` — plugin registry (722+ lines)
- `core/fit_orchestrator.py` — NLSQ↔Bayesian orchestration
- `core/numpyro_model_builder.py` — NumPyro model construction
- `core/arviz_utils.py` — ArviZ 1.x compat shim
- `cli/main.py` — argparse entry, subcommand dispatch
- `pipeline/builder.py` — YAML pipeline loader

## Envelope I/O

CLI pipeline/batch runs read/write a JSON "envelope" format (input config + output results) —
see `cli/` for envelope schema, not a database.
