# RheoJAX GUI Package

Qt-based graphical user interface for RheoJAX rheological analysis.

## Architecture Overview

RheoJAX GUI is a mode-based interface with Fit, Transform, and Pipeline
modes, launched by `rheojax-gui`. It is located in `foundation/` (core
models) and `workspace/` (UI shell), and includes project save/load and
batch pipeline execution. See `workspace/README.md` for the workspace
shell's own package structure and mode breakdown.

## Package Structure

```
rheojax/gui/
├── __init__.py                    # Package init with main() entry point
├── main.py                        # Application entry point
├── foundation/                    # Per-window state, library, invalidation cascade
│   ├── __init__.py
│   ├── state.py                   # AppState/FitState/TransformState/PipelineState dataclasses
│   ├── invalidation.py            # Cross-step cascade (apply_cascade/register_step)
│   └── library.py                 # DatasetLibrary
├── workspace/                     # WorkspaceWindow shell (fit/transform/pipeline steps)
├── services/                      # RheoJAX API integration layer
│   ├── __init__.py
│   ├── data_service.py            # Data loading/validation
│   ├── transform_service.py       # Transform operations
│   ├── model_service.py           # Model fitting
│   ├── bayesian_service.py        # Bayesian inference
│   ├── plot_service.py            # Visualization
│   └── export_service.py          # Result export
├── jobs/                          # Background worker pool
│   ├── __init__.py
│   ├── worker_pool.py             # Thread pool manager
│   ├── fit_worker.py              # Model fitting jobs
│   ├── bayesian_worker.py         # MCMC sampling jobs
│   └── cancellation.py            # Cancellation tokens
├── widgets/                       # Custom reusable widgets
│   ├── __init__.py
│   ├── parameter_table.py         # Parameter editor
│   ├── plot_canvas.py             # Matplotlib/PyQtGraph canvas
│   ├── priors_editor.py           # Prior distribution editor
│   ├── residuals_panel.py         # Residual analysis
│   └── arviz_canvas.py            # ArviZ diagnostics
├── dialogs/                       # Modal dialogs
│   ├── __init__.py
│   ├── column_mapper.py           # Column mapping dialog
│   ├── fitting_options.py         # NLSQ configuration
│   ├── bayesian_options.py        # NUTS configuration
│   ├── export_options.py          # Export configuration
│   ├── preferences.py             # Application settings
│   └── about.py                   # About dialog
├── resources/                     # Static assets
│   ├── __init__.py
│   ├── icons/                     # Icon files
│   │   └── .gitkeep
│   └── styles/                    # Themes and styles
│       ├── __init__.py
│       ├── light.qss              # Light theme
│       ├── dark.qss               # Dark theme
│       └── plot_styles/           # Matplotlib styles
│           └── __init__.py
└── utils/                         # Utility modules
    ├── __init__.py
    ├── config.py                  # Configuration management
    ├── jax_utils.py               # JAX utilities (USES safe_import_jax)
    ├── provenance.py              # Provenance tracking
    └── seeds.py                   # Random seed management (USES safe_import_jax)
```

## Architecture

### State Management

`foundation/state.py` holds one `AppState` per window -- plain mutable
dataclasses (`FitState`/`TransformState`/`PipelineState`/...), passed
directly to step-widget bodies and mutated in place. Cross-step
invalidation (editing step N clears downstream state) is centralized in
`foundation/invalidation.py`'s `apply_cascade()`/`register_step()`.

### Service Layer

Abstracts RheoJAX API calls with:
- Error handling and validation
- Progress callbacks
- Type-safe interfaces

### Background Workers

Thread pool for long-running operations:
- Model fitting with NLSQ
- Bayesian inference with NUTS
- Transform computations
- Cancellation support

## Requirements

Install all dependencies (GUI deps are included):
```bash
uv sync
```

**Core Dependencies:**
- PySide6 >= 6.7.0
- matplotlib >= 3.8.0
- Additional from main RheoJAX requirements

## Usage

### Launch GUI

```python
from rheojax.gui import main
main()
```

Or from command line:
```bash
python -m rheojax.gui
```

### Programmatic Access

```python
from rheojax.gui.services import DataService, ModelService

# Load data
data_service = DataService()
rheo_data = data_service.load_file('data.csv')

# Fit model
model_service = ModelService()
model = model_service.fit_model('maxwell', rheo_data, test_mode='relaxation')
```

## Development Status

**Current Version**: 0.6.0

All modules are currently **stub implementations** with:
- Complete docstrings
- Type annotations
- Interface definitions
- Example usage

**Next Steps:**
1. Implement core state management (Store, Actions, Signals)
2. Implement service layer with RheoJAX integration
3. Implement MainWindow and page navigation
4. Implement custom widgets (PlotCanvas, ParameterTable, etc.)
5. Add Qt stylesheets and themes
6. Comprehensive testing

## Important Notes

### JAX Import Pattern

Modules that use JAX **MUST** use the safe import pattern:

```python
from rheojax.core.jax_config import safe_import_jax

jax, jnp = safe_import_jax()
```

**Current modules with JAX:**
- `jobs/fit_worker.py`
- `jobs/bayesian_worker.py`
- `utils/jax_utils.py`
- `utils/seeds.py`

### Float64 Critical

GUI operations that call RheoJAX models inherit float64 configuration automatically through `safe_import_jax()`.

## Contributing

When implementing GUI components:

1. **Follow Architecture**: Use service layer, don't call RheoJAX directly from widgets
2. **State Management**: All state changes through Actions, read via Selectors
3. **Background Work**: Use WorkerPool for long operations (>100ms)
4. **Error Handling**: Comprehensive try-catch, user-friendly messages
5. **Progress Feedback**: Always show progress for operations >1 second
6. **Type Safety**: Complete type hints, no `Any` in public APIs
7. **Documentation**: Docstrings with examples for all public methods

## License

Same as main RheoJAX package.
