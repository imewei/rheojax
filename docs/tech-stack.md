# RheoJAX Tech Stack

> Complete dependency and tooling reference for the RheoJAX project.

## Project Metadata

| Field | Value |
|-------|-------|
| **Name** | rheojax |
| **Version** | 0.6.0 |
| **Python** | >=3.12 (supports 3.12, 3.13) |
| **Build System** | setuptools (>=68.0) + wheel + setuptools-scm |
| **License** | MIT |
| **Package Manager** | uv (lockfile: `uv.lock`) |
| **Entry Points** | `rheojax` (CLI), `rheojax-gui` (GUI) |
| **Repository** | https://github.com/imewei/rheojax |
| **Documentation** | https://rheojax.readthedocs.io |

---

## Core Dependencies

### JAX Ecosystem (Numerical Engine)

| Package | Version | Purpose |
|---------|---------|---------|
| **jax** | >=0.8.3 | Functional numerical computing (JIT, vmap, grad) |
| **jaxlib** | >=0.8.3 | XLA compiler backend |
| **interpax** | >=0.3.12 | JIT-safe interpolation (replaces scipy.interpolate) |
| **diffrax** | >=0.7.1 | JAX-native ODE/SDE solver (Tsit5, Kvaerno5, etc.) |
| **nlsq** | >=0.6.10 | GPU-accelerated non-linear least squares (imported before JAX) |
| **optimistix** | (via diffrax) | Root-finding and optimization |
| **optax** | (via jax) | Gradient-based optimization schedules |
| **equinox** | (via diffrax) | Neural network / pytree utilities |
| **lineax** | (via diffrax) | Linear algebra solvers |

### Probabilistic Programming & Bayesian Inference

| Package | Version | Purpose |
|---------|---------|---------|
| **numpyro** | >=0.20.0, <1.0.0 | Probabilistic programming (NUTS/HMC sampler) |
| **arviz** | >=0.23.4, <1.0.0 | Bayesian diagnostics (R-hat, ESS, BFMI) |
| **arviz-base** | (via arviz) | Core ArviZ data structures |
| **arviz-stats** | (via arviz) | Statistical computations |
| **arviz-plots** | (via arviz) | Posterior visualization |
| **xarray** | >=2026.1.0, <2027.0.0 | Labeled multi-dimensional arrays (InferenceData) |

### Scientific Computing

| Package | Version | Purpose |
|---------|---------|---------|
| **numpy** | >=2.3.5, <3.0.0 | Array operations, host-side computation |
| **scipy** | >=1.17.0, <2.0.0 | Fallback optimization, special functions |
| **pandas** | >=2.3.3, <3.0.0 | Data loading, tabular I/O |
| **scikit-learn** | >=1.8.0 | API conventions, metrics utilities |
| **matplotlib** | >=3.10.8, <4.0.0 | Publication-quality static plots |

### GUI & Interactive Visualization

| Package | Version | Purpose |
|---------|---------|---------|
| **PySide6** | >=6.10.2 | Qt6 desktop GUI framework |
| **qtpy** | >=2.4.3 | Qt abstraction layer |
| **pyqtgraph** | >=0.14.0 | High-performance interactive plots |
| **qasync** | >=0.28.0 | asyncio + Qt event loop integration |

### File I/O

| Package | Version | Purpose |
|---------|---------|---------|
| **h5py** | >=3.15.1 | HDF5 read/write |
| **openpyxl** | >=3.1.5 | Excel .xlsx read/write |
| **xlrd** | >=2.0.2 | Legacy .xls reading |

### Utilities

| Package | Version | Purpose |
|---------|---------|---------|
| **rich** | >=14.0.0 | Terminal formatting, progress bars |
| **pyyaml** | >=6.0.2 | YAML configuration parsing |

---

## Optional GPU Dependencies

```toml
[project.optional-dependencies]
gpu_cuda13 = [
    "jax-cuda13-plugin>=0.8.0",
    "jax-cuda13-pjrt>=0.8.0",
]
gpu_cuda12 = [
    "jax-cuda12-plugin>=0.8.0",
    "jax-cuda12-pjrt>=0.8.0",
]
```

Install with: `make install-jax-gpu` (auto-detects CUDA version) or `uv sync --extra gpu_cuda13`.

---

## Development Dependencies

### Testing

| Package | Version | Purpose |
|---------|---------|---------|
| **pytest** | >=9.0.2 | Test framework |
| **pytest-xdist** | >=3.8.0 | Parallel test execution (capped at 4 workers) |
| **pytest-timeout** | >=2.3.1 | Per-test timeout (default 120s) |
| **pytest-cov** | | Coverage reporting |
| **pytest-qt** | | GUI widget testing |
| **pytest-image-diff** | | Visual regression testing |
| **hypothesis** | | Property-based testing |

### Code Quality

| Tool | Version | Purpose |
|------|---------|---------|
| **black** | >=26.1.0 | Code formatting (line-length=88) |
| **ruff** | >=0.15.0 | Linting (E, W, F, I, C, B, UP, S rules) |
| **mypy** | >=1.19.1 | Static type checking (strict equality) |
| **pre-commit** | | Git hook management |

### Documentation

| Package | Version | Purpose |
|---------|---------|---------|
| **sphinx** | 9.1.0 | Documentation generator |
| **furo** | 2025.12.19 | Sphinx theme |
| **sphinx-autodoc-typehints** | | Auto type annotation docs |
| **sphinx-copybutton** | | Code block copy buttons |
| **myst-parser** | | Markdown in Sphinx |

### Interactive Development

| Package | Version | Purpose |
|---------|---------|---------|
| **ipython** | 9.10.0 | Enhanced REPL |
| **jupyter** | 1.1.1 | Notebook environment |

---

## Build & Quality Commands

### Makefile Targets

| Target | Command | Purpose |
|--------|---------|---------|
| `make install` | `uv sync` | Editable install |
| `make install-dev` | `uv sync --all-extras` | + development deps |
| `make install-jax-gpu` | Auto-detect CUDA | GPU JAX backend |
| `make test` | `pytest` | Full test suite (~4963 tests) |
| `make test-smoke` | `pytest -m smoke` | Critical tests (~1838, CI gate) |
| `make test-fast` | `pytest -m "not slow..."` | Exclude slow Bayesian (~4714) |
| `make test-parallel` | `pytest -n $XDIST_WORKERS` | Parallel (default 4 workers) |
| `make test-ci` | `pytest -m smoke` | CI gate (matches GitHub Actions) |
| `make test-ci-full` | `pytest -m "not slow..."` | Extended CI (~4714 tests) |
| `make test-coverage` | `pytest --cov-report=html` | Coverage with HTML report |
| `make format` | `black . && ruff --fix` | Auto-format + lint fix |
| `make quick` | `black + ruff + smoke` | Fast quality check |
| `make lint` | `ruff check .` | Lint only |
| `make type` | `mypy .` | Type checking |
| `make docs` | `sphinx-build docs docs/_build` | Build HTML docs |
| `make clean` | Remove artifacts | Preserves .venv/, .claude/ |
| `make gpu-check` | `python -c "import jax..."` | Verify GPU backend |
| `make gpu-diagnose` | Plugin conflict check | Debug GPU issues |

### Pytest Configuration

```toml
[tool.pytest.ini_options]
addopts = [
    "-v", "--strict-markers", "--tb=short", "--color=yes",
    "--cov=rheojax", "--cov-report=term-missing",
    "--dist=loadscope", "--maxfail=10", "--timeout=120"
]
```

**Test Markers:**
- **Tiers:** `smoke` (~1838 tests, <6 min), `unit`, `integration`, `validation`, `benchmark`
- **Execution:** `slow` (>30s), `gpu`, `macos_only`, `crash_test`
- **Content:** `notebook_smoke`, `notebook_comprehensive`, `io`, `visual`, `sgr`, `spp`, `gui`

### Ruff Configuration

```toml
[tool.ruff]
line-length = 88
target-version = "py312"
select = ["E", "W", "F", "I", "C", "B", "UP", "S"]
ignore = ["E501", "B008", "C901", "S101", "S110", "S112", "S607", "S311"]
```

Per-file ignores: `__init__.py` (F401 unused imports), model/transform files (E402 import order), test files (relaxed rules), notebooks (all rules).

### Mypy Configuration

```toml
[tool.mypy]
python_version = "3.12"
ignore_missing_imports = true
strict_equality = true
```

Overrides: `rheojax.models.*` (relaxed name-defined/any-return), `rheojax.gui.*` / `rheojax.cli.*` (ignore errors for PySide6 stubs).

---

## CI/CD

- **Platform:** GitHub Actions (`.github/workflows/`)
- **Workflows:**
  - `ci.yml` — lint (3.12 + 3.13), quality (radon), dependency-review, test (3 OS × 2 Py), docs, build, audit
  - `release.yml` — tag-triggered PyPI publish with build provenance attestation, SBOM, GitHub Release (reuses CI via `workflow_call`)
  - `security.yml` — CodeQL, Semgrep SAST, Gitleaks, Trivy (push/PR + weekly schedule)
  - `dependabot-auto-merge.yml` — auto-squash-merge minor/patch dependency updates
- **CI gate:** Smoke tests (~1838 tests) on 3 OS × 2 Python versions
- **Local CI:** `make quick` (format + lint + smoke)

---

## Test Infrastructure

### Test Organization

```
tests/
├── core/           # base, data, parameters, bayesian, float64, deformation_mode
├── models/         # 26 model test suites (including hvm/, hvnm/)
├── transforms/     # 12 transform test suites
├── pipeline/       # Pipeline + BayesianPipeline
├── utils/          # optimization, compatibility, initialization, modulus_conversion
├── io/             # readers + DMTA + writers
├── gui/            # PySide6 widget tests
├── integration/    # End-to-end (NLSQ → NUTS)
├── validation/     # pyrheo, hermes-rheo (local only)
└── conftest.py     # Shared fixtures, JAX reset, MCMC config
```

### Key Fixtures

- **Session-scoped:** `mcmc_config` (adaptive: fast CI = 200/200, full = 2000/1000)
- **Function-scoped:** Test data (oscillation, relaxation, creep, rotation)
- **Autouse:** JAX cache reset (`gc.collect()` + `jax.clear_caches()`)

---

## Documentation Structure

```
docs/
├── source/
│   ├── conf.py              # Sphinx config (furo theme)
│   ├── index.rst            # Main entry
│   ├── installation.rst     # Setup guide
│   ├── quickstart.rst       # Getting started
│   ├── user_guide/          # 6 guide sections
│   ├── models/              # 20 model family directories
│   ├── transforms/          # Transform documentation
│   ├── api/                 # API reference
│   ├── architecture/        # Architecture docs
│   ├── developer/           # Developer guide
│   └── examples/            # Example gallery
├── examples/                # 170 Jupyter notebooks
│   ├── basic/ (5), advanced/ (10), bayesian/ (9)
│   ├── tnt/ (30), fluidity/ (24), hvnm/ (15), vlb/ (16)
│   ├── hvm/ (13), fikh/ (12), ikh/ (12), itt_mct/ (12)
│   ├── dmta/ (8), transforms/ (8), verification/ (7)
│   ├── epm/ (6), giesekus/ (6), hl/ (6), sgr/ (6), stz/ (6), dmt/ (6)
│   ├── io/ (1)
│   └── data/                # Sample datasets by protocol
└── internal/                # Internal documentation
```

---

## Scripts

| Script | Purpose |
|--------|---------|
| `run_all_notebooks.py` | Batch notebook runner with retry/timeout |
| `run_single_notebook_96h.py` | Single notebook with 96-hour timeout |
| `run_profiling.py` | Performance profiling |
| `micro_benchmarks.py` | Microbenchmark suite |
| `validate_model_docs.py` | Documentation validation |
| `clean_notebook_outputs.py` | Strip outputs for git |
| `golden_data/` | Visual regression test baselines |

---

## Performance Characteristics

- **NLSQ fitting:** 2-10x faster than SciPy (JAX JIT + GPU)
- **XLA cache:** `~/.cache/rheojax/jax_cache/` (764-1552ms cold → <10ms warm)
- **Bayesian (NUTS):** NumPyro on JAX, GPU-accelerated when available
- **Parallel:** Up to 4 workers (CPU), auto-detected GPU count
- **OOM management:** xdist worker detection → sequential mode, per-test timeout
