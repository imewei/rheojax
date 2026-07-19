# Modernization Assessment: rheojax

## Executive Summary

RheoJAX is a JAX-accelerated rheological analysis framework (Python 3.12+): 53 constitutive models across 22 families, 11 data transforms, dual NLSQ/Bayesian (NumPyro NUTS) fitting, a CLI, and a PySide6 desktop GUI. At ~179 KSLOC of source (340 files) plus ~117 KSLOC of tests (404 files), it is a large, actively maintained codebase built on a modern stack (`uv`, JAX/diffrax/NumPyro, PySide6, ruff+mypy, pytest with tiered markers) — this is **not** a stale legacy system in need of a stack migration. Risk is concentrated in a handful of god-object files (`Pipeline`, `WorkspaceWindow`, `nlsq_optimize()`), one duplicated model-family implementation (fluidity), and a pre-existing dead-test-class defect that silently drops ~600 lines of test coverage. Security posture is strong — one Medium-severity CSV/formula-injection finding in Excel export, everything else (YAML loading, zip extraction, subprocess IPC, dynamic model dispatch, dependencies) checked clean. Headline recommendation: **refactor-in-place**, not a rewrite — target the god objects and the fluidity duplication, fix the dead-test-class defect, and patch the Excel export sanitization.

## System Inventory

**Tool used:** `scc`/`cloc`/`lizard` unavailable in this environment; fell back to `find`+`wc -l` (LOC) and decision-keyword grep counts (`if|elif|for|while|except`, as a complexity proxy) per the modernize-assess fallback procedure. Figures are reproducible with the commands below.

| Scope | Files | LOC |
|---|---|---|
| `rheojax/` (source) | 340 | 178,843 |
| `tests/` | 404 | 117,288 |
| **Total** | **744** | **296,131** |

### Source LOC by subpackage

| Subpackage | Files | LOC |
|---|---|---|
| models | 105 | 77,009 |
| gui | 100 | 31,398 |
| io | 27 | 15,062 |
| utils | 38 | 18,787 |
| core | 17 | 10,618 |
| transforms | 12 | 7,061 |
| visualization | 7 | 6,371 |
| cli | 17 | 4,961 |
| pipeline | 6 | 5,421 |
| logging | 6 | 1,371 |
| parallel | 4 | 660 |

### Highest decision-keyword-density files (complexity proxy — top 10 of 25 scanned)

| File | Decision-keyword count |
|---|---|
| `rheojax/utils/optimization.py` | 207 |
| `rheojax/io/readers/trios/txt.py` | 154 |
| `rheojax/io/readers/anton_paar.py` | 136 |
| `rheojax/pipeline/base.py` | 135 |
| `rheojax/core/parameters.py` | 135 |
| `rheojax/gui/workspace/window.py` | 131 |
| `rheojax/io/readers/trios/csv.py` | 128 |
| `rheojax/gui/services/export_service.py` | 125 |
| `rheojax/gui/services/data_service.py` | 125 |
| `rheojax/models/multimode/generalized_maxwell.py` | 119 |

Note: `trios/txt.py` and `anton_paar.py` are large but already decomposed into ~25-30 small single-purpose functions each (per the tech-debt scan) — a readability/file-length concern, not a god-object pattern. `optimization.py`, `pipeline/base.py`, and `workspace/window.py` are, and are discussed below.

### Technology fingerprint

- **Language/runtime:** Python 3.12+/3.13, JAX ecosystem (`jax`, `jaxlib`, `diffrax`, `numpyro`, `arviz` 1.x split packages), `nlsq` for GPU-accelerated NLSQ.
- **Build/deps:** `uv` + `pyproject.toml` (setuptools backend); float64 enforced at import (`jax.config.update("jax_enable_x64", True)`); optional CUDA 12/13 GPU extras (mutually exclusive).
- **GUI:** PySide6 6.10+ / qtpy / PyQtGraph — single shell (`WorkspaceWindow`), legacy `RheoJAXMainWindow`/`--legacy`/`--workspace` confirmed fully removed (zero grep hits).
- **Data stores:** no database — file-based I/O (TRIOS/Anton Paar/CSV/Excel readers, HDF5/Excel/NPZ writers, JSON envelopes for CLI, `.rheojax` v2 ZIP+JSON+HDF5 GUI project archives).
- **Integration points:** CLI (`rheojax`/`rj` entry points), YAML pipeline configs (`cli/_yaml_runner.py`, schema-validated via `yaml.safe_load`), GUI subprocess-isolated background workers (`multiprocessing`, spawn method).
- **Test presence:** 404 test files, 117 KSLOC, tiered pytest markers (`smoke`/`unit`/`integration`/`validation`/`benchmark`), `--cov=rheojax` in default `addopts`. One known coverage gap (see Documentation Gaps #3).

## Architecture-at-a-Glance

10 functional domains identified (see `ARCHITECTURE.mmd` for the full Mermaid diagram):

| Domain | Key paths | Depends on | Responsibility |
|---|---|---|---|
| **core** | `core/{base,data,parameters,fit_result,fit_orchestrator,registry,bayesian,numpyro_model_builder,arviz_utils,...}.py` | `utils` (top-level); `models`/`transforms`/`io` (lazy, function-local only) | Base classes, shared DTOs (`RheoData`, `ParameterSet`, `FitResult`, `BayesianResult`), registries, NLSQ→NUTS orchestration. Architectural hub. |
| **models** | `models/` — 22 families, 53 models | `core`, `transforms`, `utils` | Constitutive models, self-registering via `@ModelRegistry.register`. |
| **transforms** | `transforms/` — 11 transforms | `core`, `io`, `utils` | Data transforms, eagerly registered (no lazy loading, unlike models). |
| **io** | `io/{readers,writers,analysis_exporter,...}` | `core`, `utils` | Format readers/writers, SPP MATLAB export, format auto-detection. |
| **pipeline + parallel** | `pipeline/{base,builder,workflows,bayesian,batch}.py`, `parallel/` (4 files) | `core`, `models`, `transforms`, `io`, `visualization` | Fluent orchestration API. `parallel` is a thin process-pool with exactly one internal consumer (`pipeline/workflows.py`) — folded in rather than given its own row. |
| **cli** | `cli/{main,fit,bayesian,spp,cmd_*,_yaml_runner,...}.py` | `core`, `io`, `models`, `transforms`, `pipeline` (narrow) | `rheojax`/`rj` entry point; YAML pipeline runner; JSON envelope I/O. |
| **gui** | `gui/{workspace,foundation,services,jobs,dialogs,widgets}` (100 files) | `core`, `io`, `models`, `transforms`, `utils`; **not** `pipeline`, **not** `parallel` | PySide6 desktop app, single `WorkspaceWindow` shell, Redux-style state (`foundation/state.py`), subprocess-isolated background jobs. |
| **utils** | `utils/` (38 files) | `core` (top-level, mutual dependency) | Optimization (NLSQ helpers), per-family kernels, initialization heuristics. |
| **visualization** | `visualization/{plotter,fit_plotter,...}.py` | `core`, `utils` | Matplotlib static plotting, consumed only by `pipeline/base.py`. Architecturally disjoint from the GUI's PyQtGraph canvas. |
| **logging** (cross-cutting) | `logging/` (6 files) | none internal | Structured logging, leaf utility imported by all other domains. |

**Key architectural findings from the domain-dependency sweep:**

1. **GUI never imports `rheojax.pipeline`.** `gui/services/pipeline_execution_service.py` independently reimplements pipeline-wizard step execution (driving `ModelService`/`TransformService`/`DataService` directly) rather than reusing `Pipeline`/`PipelineBuilder`. This duplicates orchestration logic between the CLI's YAML runner and the GUI wizard — not documented anywhere as an intentional split. Worth an SME check: deliberate (UI-specific checkpoint semantics) or unreconciled drift?
2. **`core ↔ utils` is a genuine mutual dependency** (top-level both directions) — any change to `core.parameters`/`core.jax_config` must be checked against all 38 `utils` files.
3. **Two independent plotting stacks** (matplotlib `visualization/` vs. GUI's native PyQtGraph canvas) with no cross-reference — a styling/behavior fix in one won't propagate to the other (cf. the previously-known FreeType blit+colorbar bug, which was stack-specific).
4. **`core → {models, transforms, io}` is entirely lazy/function-local**, by design, to avoid import cycles while still letting `core` own the registries these packages populate.

## Production Runtime Profile

No telemetry available — no observability/APM MCP server, batch job logs, or runtime exports were supplied. This is a desktop CLI/GUI scientific tool, not a service with production request telemetry; step skipped per the assessment procedure.

## Technical Debt

Ranked by remediation value (top 10, from a dedicated parallel scan):

1. **Fluidity model family: triplicated bounds-clamping closure + duplicated `_predict_saos_jit`** — `models/fluidity/_base.py:202`, `local.py:564,578`, `nonlocal_model.py:778,792`. Identical `_clipped()` closure copy-pasted into two subclasses instead of living once in the shared base; `_predict_saos_jit` is byte-for-byte duplicated between the two subclasses, self-flagged by the authors (`# TODO (FL-010)`). A bug fixed in one copy silently persists in the sibling. Effort: S/M.
2. **Silent, unlogged fallback degrades prediction-interval rigor** — `utils/optimization.py:1198-1213`. Leverage-weighted CI computation falls back to constant-MSE approximation on `except Exception` with no `logger.warning`, unlike a sibling except-block a few lines below that does log. Users get less-rigorous CIs with zero indication. Effort: S.
3. **Dead/deprecated ODE kernel shipped in production, JIT-unsafe if reintroduced** — `models/dmt/_kernels.py:367-401`. Explicitly documented as dead + JIT-unsafe, raises `DeprecationWarning`, but nothing calls it. Effort: S (delete).
4. **Manually-synced duplicate CLI template list** — `cli/cmd_pipeline.py:26-35`. Hardcoded fallback tuple must be kept in sync with `_templates.py`'s `TEMPLATES` dict by hand; bare `except Exception` also masks real import errors. Effort: S.
5. **`Pipeline` god object** — `pipeline/base.py:47-1482`, 1,490 lines / 25 public methods spanning load/transform/fit/predict/plot/save/export — the main user-facing fluent API with no composition into smaller collaborators. Effort: L.
6. **`WorkspaceWindow` god object** — `gui/workspace/window.py:46`, 1,519 lines / 73 methods, even after existing controller delegation for fit/transform steps. Partial decomposition already proves the extraction pattern works. Effort: L.
7. **`nlsq_optimize()` monolith** — `utils/optimization.py:1458-1873`, 415 lines, the single highest-fan-in function in the repo (184 edges per graphify), despite well-factored helpers already sitting in the same file. Effort: M.
8. **Copy-pasted atomic-write boilerplate (3x)** — `io/spp_export.py`, identical tmp-file/`os.replace`/cleanup-on-exception pattern repeated three times. Effort: S.
9. **23 hardcoded, inconsistent `max_steps` values at `diffeqsolve()` call sites** across 12 model families, ranging 1M-16M with no shared constant or documented rationale for the differences. Effort: S (or "document," pending SME confirmation of numerical-stiffness rationale).
10. **Misleading "(deprecated)" comment on the name actually used everywhere** — `core/test_modes.py:90-92`: `TestMode` is commented deprecated but is the name 38 files actually import; it was renamed only to dodge pytest's `Test*` collection heuristic. Risk: a future contributor "fixes" 38 call sites that don't need fixing. Effort: S (fix the comment).

*(Two pre-existing, already-tracked issues were excluded from re-reporting per scope: the `tests/core/test_base.py` duplicate `TestBaseModel` class, and the `cli/cmd_batch.py --parallel` no-op flag — both covered under Documentation Gaps below.)*

## Security Findings

| CWE | Title | file:line | Severity | Fix |
|---|---|---|---|---|
| CWE-1236 | CSV/Formula injection via unsanitized file-derived metadata in Excel export | `io/analysis_exporter.py:684-686`, `io/writers/excel_writer.py` (Units column, sourced from `io/readers/csv_reader.py:420-434`) | Medium | Sanitize any string value sourced from file-derived metadata/headers before writing to a spreadsheet cell (prefix leading `=`/`+`/`-`/`@` with `'`), via one shared helper used by every Excel-writing sink. |

**Checked and clean:** YAML pipeline config parsing (`yaml.safe_load`, no CWE-502 risk); path traversal in YAML steps and multi-file glob loaders (explicit `..`/absolute-path rejection); GUI `.rheojax` v2 project archive extraction (allowlisted member names, size caps, checksum verification, resolved-path containment — no zip-slip); HDF5/NPZ I/O (`allow_pickle=False`); subprocess-isolated GUI workers (spawn method, no pickle-from-file, no eval/exec); dynamic model/transform dispatch (`importlib.import_module` targets are a fixed developer-authored table, not user-controlled strings); `subprocess` usage (fixed executable names, no `shell=True`); no hardcoded credentials found anywhere; `pip-audit` reports no known dependency vulnerabilities.

No credentials were discovered, so no `SECRETS.local.md` was created and no `.gitignore` update was needed for this pass.

## Documentation Gaps

Top 5 undocumented behaviors a new engineer would need explained:

1. **`cli batch --parallel`/`--workers` flags are undocumented in the user guide** (`docs/source/user_guide/04_practical_guides/batch_processing.rst` has zero mentions of `--parallel`) despite being defined in `cmd_batch.py:108-116` — and being a self-documented no-op (`cmd_batch.py:268-270` logs "reserved for future use; running sequentially"). A user reading `--help` would expect a speedup and silently get none, with no doc explaining why.
2. **GUI's `pipeline_execution_service.py` reimplements pipeline-wizard execution** rather than reusing `rheojax.pipeline.Pipeline`/`PipelineBuilder` — no architecture doc explains whether this is an intentional UI-specific divergence (different checkpoint/retry semantics) or unreconciled duplication.
3. **`tests/core/test_base.py` has a duplicate `TestBaseModel` class** that shadows ~600 lines of tests, which silently never run. Zero documentation or warning anywhere in the repo; a new contributor could assume test coverage that doesn't exist. (Pre-existing, deliberately left unfixed per prior project decision — but still undocumented.)
4. **`rheojax.parallel` (process-pool module) exists and works but is only wired into `pipeline/workflows.py`** — not exposed as a documented parallel-batch option even though the whole module is implemented and functional; ties into gap #1.
5. **Two independent plotting stacks with no cross-reference** — `rheojax/visualization/` (matplotlib, used by `pipeline`) vs. the GUI's native PyQtGraph canvas (`gui/widgets/pyqtgraph_canvas.py`) are maintained completely separately with no doc noting that a fix in one won't propagate to the other.

## Relative Scale

**Method:** COCOMO-II basic figure, `2.94 × (KSLOC)^1.10` (nominal scale factors), computed from the `find`+`wc -l` fallback LOC counts above (no `scc`/`cloc` available).

| Scope | KSLOC | COCOMO-II index |
|---|---|---|
| `rheojax/` (source) | 178.8 | **883** |
| `tests/` | 117.3 | 555 |
| Total | 296.1 | 1,538 |

**This is a relative complexity/scale index only** — useful for ranking rheojax against other systems or tracking growth over time. It is **not** a timeline or cost estimate: the underlying COCOMO person-month formula assumes traditional human-team productivity curves, which agentic transformation does not follow. No schedule, cost, or date should be inferred from this number.

## Recommended Modernization Pattern

**Refactor (in-place)** — one-paragraph rationale: rheojax is not a legacy system needing a stack migration. It runs a current, well-chosen stack (Python 3.12+, JAX/diffrax/NumPyro, `uv`, PySide6, ruff+mypy, tiered pytest) with strong architectural discipline (float64 enforcement, lazy-loading to avoid import cycles, a documented single GUI shell, clean registry-based model/transform dispatch) and a clean security posture (one Medium finding, everything else already hardened). The real risk surface is concentrated and well-scoped: a handful of god objects (`Pipeline`, `WorkspaceWindow`, `nlsq_optimize()`), one duplicated model-family implementation (fluidity), a pre-existing dead-test-class defect, and undocumented GUI/CLI behavioral gaps. None of this requires a cross-stack rewrite or a same-stack version bump — it requires targeted, in-place refactoring guided by the Technical Debt list above, plus the one Excel-export sanitization fix.

**Routes to:** `/modernize-uplift` (Replatform / Refactor-in-place, same-stack).
