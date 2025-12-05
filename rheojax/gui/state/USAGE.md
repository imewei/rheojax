# RheoJAX GUI State Management - Usage Guide

Complete state management system for the RheoJAX GUI with immutable updates, undo/redo, and Qt signal integration.

## Architecture

### Core Components

1. **StateStore** - Singleton state container with immutable updates
2. **StateSignals** - Qt signals for reactive UI updates
3. **actions** - State mutation functions
4. **selectors** - State query functions (computed properties)

### State Structure

```python
AppState
├── project_path: Path
├── project_name: str
├── is_modified: bool
├── datasets: dict[str, DatasetState]
├── active_dataset_id: str
├── active_model_name: str
├── model_params: dict[str, ParameterState]
├── fit_results: dict[str, FitResult]
├── bayesian_results: dict[str, BayesianResult]
├── pipeline_state: PipelineState
├── jax_device: str
├── jax_memory_used: int
├── jax_memory_total: int
├── transform_history: list[TransformRecord]
├── current_seed: int
├── auto_save_enabled: bool
├── theme: str
└── recent_projects: list[Path]
```

## Quick Start

### 1. Initialize Store and Signals

```python
from rheojax.gui.state import StateStore, StateSignals

# Create store (singleton)
store = StateStore()

# Create Qt signals
signals = StateSignals()

# Connect signals to store
store.set_signals(signals)
```

### 2. Connect Signal Handlers

```python
# Connect to specific signals
signals.dataset_added.connect(on_dataset_added)
signals.fit_completed.connect(on_fit_completed)
signals.state_changed.connect(on_state_changed)

def on_dataset_added(dataset_id: str):
    print(f"Dataset {dataset_id} was added")

def on_fit_completed(model_name: str, dataset_id: str):
    print(f"Fit completed: {model_name} on {dataset_id}")
```

### 3. Load Data

```python
from rheojax.gui.state import actions
import jax.numpy as jnp

# Load a dataset
dataset_id = actions.load_dataset(
    file_path=Path("data.csv"),
    name="My Dataset",
    test_mode="oscillation",
    x_data=jnp.array([1.0, 2.0, 3.0]),
    y_data=jnp.array([10.0, 20.0, 30.0]),
    y2_data=jnp.array([5.0, 10.0, 15.0]),  # Optional G''
    metadata={"temperature": 25.0}
)
```

### 4. Select Model and Parameters

```python
from rheojax.gui.state import ParameterState

# Define parameters
params = {
    "eta": ParameterState(
        name="eta",
        value=100.0,
        min_bound=1.0,
        max_bound=1000.0,
        unit="Pa·s",
        description="Viscosity"
    ),
    "tau": ParameterState(
        name="tau",
        value=1.0,
        min_bound=0.01,
        max_bound=100.0,
        unit="s",
        description="Relaxation time"
    ),
}

# Select model
actions.select_model("maxwell", params)
```

### 5. Store Fit Results

```python
from rheojax.gui.state import FitResult
from datetime import datetime

result = FitResult(
    model_name="maxwell",
    dataset_id=dataset_id,
    parameters={"eta": 95.5, "tau": 1.2},
    r_squared=0.98,
    mpe=1.5,
    chi_squared=0.005,
    fit_time=0.25,
    timestamp=datetime.now(),
    num_iterations=50,
    convergence_message="Success"
)

actions.store_fit_result(result)
```

### 6. Query State

```python
from rheojax.gui.state import selectors

# Get active dataset
dataset = selectors.get_active_dataset()
print(f"Dataset: {dataset.name}, Mode: {dataset.test_mode}")

# Get model parameters
params = selectors.get_model_param_dict()
print(f"Parameters: {params}")

# Check if fit available
if selectors.is_fit_available():
    result = selectors.get_active_fit_result()
    print(f"R²: {result.r_squared:.4f}")

# Get pipeline progress
progress = selectors.get_pipeline_progress()
print(f"Pipeline: {progress*100:.1f}% complete")
```

## Common Patterns

### Dataset Management

```python
# Load dataset
dataset_id = actions.load_dataset(path, name, test_mode, x, y)

# Set active dataset
actions.set_active_dataset(dataset_id)

# Update dataset
actions.update_dataset(dataset_id, is_modified=True)

# Remove dataset
actions.remove_dataset(dataset_id)

# Query datasets
active = selectors.get_active_dataset()
all_datasets = selectors.get_all_datasets()
count = selectors.get_dataset_count()
```

### Model & Parameters

```python
# Select model with parameters
actions.select_model(model_name, parameters)

# Update single parameter
actions.update_parameter("eta", 150.0)

# Reset to defaults
actions.reset_parameters(default_params)

# Query model state
model_name = selectors.get_active_model_name()
params = selectors.get_model_param_dict()
bounds = selectors.get_model_param_bounds()
```

### Fitting Workflow

```python
# Start fit (emits signal)
actions.start_fit(model_name, dataset_id)

# ... perform fit in background ...

# Store result
actions.store_fit_result(result)

# Or handle failure
actions.fail_fit(model_name, dataset_id, "Error message")

# Query results
result = selectors.get_active_fit_result()
all_results = selectors.get_all_fit_results()
is_available = selectors.is_fit_available()
```

### Bayesian Inference

```python
# Start Bayesian inference
actions.start_bayesian(model_name, dataset_id)

# Store result
bayesian_result = BayesianResult(
    model_name=model_name,
    dataset_id=dataset_id,
    posterior_samples=idata,
    r_hat={"eta": 1.01, "tau": 1.00},
    ess={"eta": 1500, "tau": 1600},
    divergences=0,
    credible_intervals={"eta": (90.0, 110.0), "tau": (0.8, 1.4)},
    mcmc_time=5.2,
    timestamp=datetime.now()
)
actions.store_bayesian_result(bayesian_result)

# Query Bayesian results
result = selectors.get_active_bayesian_result()
is_available = selectors.is_bayesian_available()
```

### Pipeline Management

```python
from rheojax.gui.state import PipelineStep, StepStatus

# Update pipeline steps
actions.set_pipeline_step(PipelineStep.LOAD, StepStatus.COMPLETE)
actions.set_pipeline_step(PipelineStep.FIT, StepStatus.ACTIVE)

# Query pipeline
status = selectors.get_pipeline_step_status(PipelineStep.LOAD)
current = selectors.get_current_pipeline_step()
progress = selectors.get_pipeline_progress()  # 0.0 to 1.0
```

### JAX Device & Memory

```python
# Set device
actions.set_jax_device("cuda")

# Update memory (periodically)
actions.update_jax_memory(used_bytes=1024*1024*500, total_bytes=1024*1024*8000)

# Query JAX state
device = selectors.get_jax_device()
used, total = selectors.get_jax_memory_usage()
percent = selectors.get_jax_memory_percent()
```

### Settings

```python
# Theme
actions.set_theme("dark")
theme = selectors.get_theme()

# Random seed
actions.set_seed(42)
seed = selectors.get_current_seed()

# Auto-save
actions.set_auto_save(True)
enabled = selectors.is_auto_save_enabled()
```

### Project Management

```python
# Save project
actions.save_project(Path("project.rheojax"))

# Load project
state = load_project_from_file(path)  # Your implementation
actions.load_project(path, state)

# Query project state
name = selectors.get_project_name()
is_modified = selectors.is_project_modified()
recent = selectors.get_recent_projects()
```

### Transform Provenance

```python
# Record transform
actions.add_transform_record(
    source_id=source_dataset_id,
    target_id=target_dataset_id,
    transform_name="mastercurve",
    parameters={"reference_temp": 60.0},
    seed=42
)

# Query transform history
history = selectors.get_transform_history_for_dataset(dataset_id)
lineage = selectors.get_dataset_lineage(dataset_id)
```

### Undo/Redo

```python
# Check availability
can_undo = selectors.can_undo()
can_redo = selectors.can_redo()

# Perform undo/redo
store.undo()
store.redo()

# Note: Settings actions (theme, seed, auto-save) and memory updates
# don't track undo by default (track_undo=False)
```

## Advanced Usage

### Custom State Updates

```python
# Direct state update with custom logic
def updater(state: AppState) -> AppState:
    # Create modified state
    new_datasets = state.datasets.copy()
    new_datasets["new_id"] = DatasetState(...)

    return AppState(**{
        **state.__dict__,
        "datasets": new_datasets,
        "is_modified": True
    })

store.update_state(updater, track_undo=True, emit_signal=True)
```

### Batch Updates

```python
# Multiple updates in single transaction
updaters = [
    lambda s: AppState(**{**s.__dict__, "current_seed": 42}),
    lambda s: AppState(**{**s.__dict__, "theme": "dark"}),
]

store.batch_update(updaters, track_undo=True)
```

### State Subscriptions

```python
# Subscribe to all state changes
def on_state_change(state: AppState):
    print(f"State changed: {len(state.datasets)} datasets")

store.subscribe(on_state_change)

# Unsubscribe
store.unsubscribe(on_state_change)
```

## Signal Reference

### Dataset Signals
- `dataset_added(str)` - Dataset ID
- `dataset_removed(str)` - Dataset ID
- `dataset_updated(str)` - Dataset ID
- `dataset_selected(str)` - Dataset ID

### Model Signals
- `model_selected(str)` - Model name
- `model_params_changed(str)` - Model name

### Fit Signals
- `fit_started(str, str)` - Model name, Dataset ID
- `fit_progress(str, int)` - Job ID, Progress (0-100)
- `fit_completed(str, str)` - Model name, Dataset ID
- `fit_failed(str, str, str)` - Model name, Dataset ID, Error

### Bayesian Signals
- `bayesian_started(str, str)` - Model name, Dataset ID
- `bayesian_progress(str, int)` - Job ID, Progress (0-100)
- `bayesian_completed(str, str)` - Model name, Dataset ID
- `bayesian_failed(str, str, str)` - Model name, Dataset ID, Error

### Pipeline Signals
- `pipeline_step_changed(str, str)` - Step name, Status

### Transform Signals
- `transform_applied(str, str)` - Transform name, Dataset ID

### JAX Signals
- `jax_device_changed(str)` - Device name
- `jax_memory_updated(int, int)` - Used bytes, Total bytes

### UI Signals
- `theme_changed(str)` - Theme name
- `state_changed()` - General state update

### Project Signals
- `project_loaded(str)` - Project path
- `project_saved(str)` - Project path

## Best Practices

1. **Always use actions** - Never modify state directly
2. **Use selectors for queries** - Don't access store._state directly
3. **Connect signals early** - Set up signal handlers during initialization
4. **Leverage immutability** - State is immutable; updates create new objects
5. **Track undo selectively** - Use `track_undo=False` for transient updates
6. **Batch related updates** - Use `batch_update()` for multiple changes
7. **Clean up subscriptions** - Unsubscribe when components are destroyed

## Testing

```python
# Reset store for testing
StateStore.reset()

# Create fresh store
store = StateStore()

# Mock signals if PySide6 not available
import sys
from unittest.mock import MagicMock
sys.modules["PySide6"] = MagicMock()
sys.modules["PySide6.QtCore"] = MagicMock()
```

## Type Safety

All state objects are properly typed with dataclasses:
- `DatasetState` - Dataset with metadata
- `ParameterState` - Model parameter with bounds
- `FitResult` - NLSQ fit result
- `BayesianResult` - Bayesian inference result
- `PipelineState` - Pipeline execution state
- `TransformRecord` - Transform provenance

Enums for type-safe constants:
- `PipelineStep` - LOAD, TRANSFORM, FIT, BAYESIAN, EXPORT
- `StepStatus` - PENDING, ACTIVE, COMPLETE, WARNING, ERROR

## Performance Considerations

1. **Cloning** - State objects implement efficient `.clone()` methods
2. **JAX Arrays** - Array references are kept (not deep-copied) for performance
3. **Undo Stack** - Limited to 50 entries by default (`_max_undo_size`)
4. **Signal Emission** - Can be disabled with `emit_signal=False`
5. **Memory Updates** - Don't track undo to avoid stack pollution
