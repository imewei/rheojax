# RheoJAX GUI Package

Qt-based graphical user interface for RheoJAX rheological analysis.

## Package Structure

```
rheojax/gui/
├── __init__.py                    # Package init with main() entry point
├── main.py                        # Application entry point
├── app/                           # Application core
│   ├── __init__.py
│   ├── main_window.py             # Central window
│   ├── menu_bar.py                # Application menu
│   ├── toolbar.py                 # Quick-access toolbar
│   └── status_bar.py              # Status and progress bar
├── state/                         # Redux-inspired state management
│   ├── __init__.py
│   ├── store.py                   # Central state store
│   ├── signals.py                 # Qt signals for reactivity
│   ├── actions.py                 # State mutation actions
│   └── selectors.py               # State query utilities
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
├── pages/                         # Page-based navigation
│   ├── __init__.py
│   ├── home_page.py               # Landing page
│   ├── data_page.py               # Data management
│   ├── transform_page.py          # Transform application
│   ├── fit_page.py                # Model fitting
│   ├── bayesian_page.py           # Bayesian inference
│   ├── diagnostics_page.py        # MCMC diagnostics
│   └── export_page.py             # Result export
├── widgets/                       # Custom reusable widgets
│   ├── __init__.py
│   ├── dataset_tree.py            # Dataset hierarchy tree
│   ├── parameter_table.py         # Parameter editor
│   ├── plot_canvas.py             # Matplotlib canvas
│   ├── multi_view.py              # Multi-panel plots
│   ├── model_browser.py           # Model library browser
│   ├── quick_fit_strip.py         # Quick fit toolbar
│   ├── pipeline_chips.py          # Pipeline visualization
│   ├── jax_status.py              # GPU status indicator
│   ├── priors_editor.py           # Prior distribution editor
│   ├── residuals_panel.py         # Residual analysis
│   └── arviz_canvas.py            # ArviZ diagnostics
├── dialogs/                       # Modal dialogs
│   ├── __init__.py
│   ├── import_wizard.py           # Data import wizard
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

### State Management (Redux-inspired)

- **Store**: Central immutable state container
- **Actions**: State mutation operations
- **Signals**: Qt signals for reactive UI updates
- **Selectors**: Memoized state queries

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

### Page-Based Navigation

Workflow-oriented pages:
1. **Home**: Quick start and examples
2. **Data**: Load and validate data
3. **Transform**: Apply mastercurve, FFT, SRFS
4. **Fit**: Model fitting with NLSQ
5. **Bayesian**: MCMC sampling
6. **Diagnostics**: ArviZ plots (trace, pair, forest)
7. **Export**: Save results

## Requirements

Install GUI dependencies:
```bash
pip install rheojax[gui]
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
