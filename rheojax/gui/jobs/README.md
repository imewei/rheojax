# Background Job System

Complete background job system for running JAX operations off the main UI thread.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                         Main Thread                          │
│  ┌──────────────┐         ┌─────────────────────┐          │
│  │   GUI/UI     │◄───────►│    WorkerPool       │          │
│  │  Components  │ signals │  (QThreadPool)      │          │
│  └──────────────┘         └──────────┬──────────┘          │
└───────────────────────────────────────┼──────────────────────┘
                                        │ submit()
                    ┌───────────────────┼───────────────────┐
                    │                   │                   │
            ┌───────▼────────┐  ┌──────▼──────┐   ┌───────▼────────┐
            │  FitWorker     │  │ Bayesian-   │   │  Custom        │
            │  (QRunnable)   │  │ Worker      │   │  Workers       │
            │                │  │             │   │                │
            │ - NLSQ fit     │  │ - NUTS      │   │ - Transform    │
            │ - Progress     │  │ - Warmup    │   │ - Analysis     │
            │ - Cancellation │  │ - Sampling  │   │ - Export       │
            └────────┬───────┘  └──────┬──────┘   └────────┬───────┘
                     │                 │                    │
              ┌──────▼─────────────────▼────────────────────▼──────┐
              │         CancellationToken (thread-safe)            │
              │  - cancel(), is_cancelled(), check()               │
              │  - Error storage and retrieval                     │
              └────────────────────────────────────────────────────┘
```

## Components

### 1. CancellationToken

Thread-safe cancellation mechanism:

```python
from rheojax.gui.jobs import CancellationToken, CancellationError

token = CancellationToken()

# In UI thread
token.cancel()  # Request cancellation

# In worker thread
token.check()  # Raises CancellationError if cancelled
if token.is_cancelled():
    # Cleanup and exit
    return

# Error handling
try:
    risky_operation()
except Exception as e:
    token.set_error(e)

error = token.get_error()  # Retrieve stored error
```

### 2. WorkerPool

PySide6 QThreadPool-based job manager:

```python
from rheojax.gui.jobs import WorkerPool, FitWorker

# Create pool
pool = WorkerPool(max_threads=4)

# Connect signals
pool.job_started.connect(lambda job_id: print(f"Job {job_id} started"))
pool.job_progress.connect(lambda job_id, curr, total, msg:
    print(f"{job_id}: {curr}/{total} - {msg}"))
pool.job_completed.connect(lambda job_id, result:
    print(f"Job {job_id} completed: {result}"))
pool.job_failed.connect(lambda job_id, error:
    print(f"Job {job_id} failed: {error}"))
pool.job_cancelled.connect(lambda job_id:
    print(f"Job {job_id} cancelled"))

# Submit worker
worker = FitWorker(model_name='maxwell', data=rheo_data)
job_id = pool.submit(worker)

# Cancel if needed
pool.cancel(job_id)

# Check status
is_busy = pool.is_busy()
active_count = pool.get_active_count()

# Shutdown
pool.shutdown(wait=True, timeout_ms=30000)
```

**Signals:**

- `job_started(job_id: str)` - Job started execution
- `job_progress(job_id: str, current: int, total: int, message: str)` - Progress update
- `job_completed(job_id: str, result: object)` - Job completed successfully
- `job_failed(job_id: str, error_message: str)` - Job failed with error
- `job_cancelled(job_id: str)` - Job was cancelled

### 3. FitWorker

NLSQ model fitting worker:

```python
from rheojax.gui.jobs import FitWorker, FitResult, CancellationToken
from rheojax.core.data import RheoData

# Prepare data
rheo_data = RheoData(x=time, y=stress, test_mode='relaxation')

# Create worker
token = CancellationToken()
worker = FitWorker(
    model_name='maxwell',
    data=rheo_data,
    initial_params={'G0': 1e6, 'tau': 1.0},
    options={'max_iter': 5000, 'ftol': 1e-8},
    cancel_token=token
)

# Connect signals
worker.signals.progress.connect(
    lambda iteration, loss, msg: print(f"Iteration {iteration}: {loss:.6e}")
)
worker.signals.completed.connect(
    lambda result: print(f"Fit completed: R²={result.r_squared:.4f}")
)
worker.signals.failed.connect(
    lambda error: print(f"Fit failed: {error}")
)
worker.signals.cancelled.connect(
    lambda: print("Fit cancelled")
)

# Submit to pool
job_id = pool.submit(worker)
```

**FitResult attributes:**

- `model_name: str` - Name of fitted model
- `parameters: dict[str, float]` - Fitted parameter values
- `r_squared: float` - R² goodness of fit
- `mpe: float` - Mean percentage error
- `chi_squared: float` - Chi-squared statistic
- `fit_time: float` - Fitting duration in seconds
- `timestamp: datetime` - Completion timestamp
- `n_iterations: int` - Number of iterations
- `success: bool` - Optimization convergence status

### 4. BayesianWorker

MCMC sampling worker with NumPyro:

```python
from rheojax.gui.jobs import BayesianWorker, BayesianResult

# Create worker (with NLSQ warm-start)
worker = BayesianWorker(
    model_name='maxwell',
    data=rheo_data,
    num_warmup=1000,
    num_samples=2000,
    num_chains=4,
    warm_start={'G0': 1e6, 'tau': 1.0},  # From NLSQ fit
    priors={},  # Optional custom priors
    seed=42,
    cancel_token=token
)

# Connect signals
worker.signals.progress.connect(
    lambda chain, sample, total: print(f"Chain {chain}: {sample}/{total}")
)
worker.signals.stage_changed.connect(
    lambda stage: print(f"Stage: {stage}")
)
worker.signals.divergence_detected.connect(
    lambda count: print(f"Warning: {count} divergences detected")
)
worker.signals.completed.connect(
    lambda result: print(f"Sampling completed in {result.sampling_time:.2f}s")
)

# Submit to pool
job_id = pool.submit(worker)
```

**BayesianResult attributes:**

- `model_name: str` - Name of fitted model
- `posterior_samples: dict[str, array]` - Posterior samples for each parameter
- `summary: dict[str, dict]` - Summary statistics (mean, std, quantiles)
- `diagnostics: dict` - R-hat, ESS, divergences
- `num_samples: int` - Samples per chain
- `num_chains: int` - Number of chains
- `sampling_time: float` - Total sampling duration
- `timestamp: datetime` - Completion timestamp
- `credible_intervals: dict` - 95% HDI for each parameter

## Complete Example: NLSQ → Bayesian Workflow

```python
from PySide6.QtWidgets import QMainWindow, QPushButton, QProgressBar, QLabel
from rheojax.gui.jobs import WorkerPool, FitWorker, BayesianWorker
from rheojax.core.data import RheoData
import numpy as np

class RheoAnalysisWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Create worker pool
        self.pool = WorkerPool(max_threads=4)

        # Connect pool signals
        self.pool.job_started.connect(self.on_job_started)
        self.pool.job_progress.connect(self.on_job_progress)
        self.pool.job_completed.connect(self.on_job_completed)
        self.pool.job_failed.connect(self.on_job_failed)
        self.pool.job_cancelled.connect(self.on_job_cancelled)

        # UI components
        self.fit_button = QPushButton("Fit Model (NLSQ)")
        self.bayesian_button = QPushButton("Bayesian Inference")
        self.cancel_button = QPushButton("Cancel")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("Ready")

        # Connect buttons
        self.fit_button.clicked.connect(self.start_fit)
        self.bayesian_button.clicked.connect(self.start_bayesian)
        self.cancel_button.clicked.connect(self.cancel_jobs)

        # Track jobs
        self.current_job_id = None
        self.fit_result = None

    def start_fit(self):
        """Start NLSQ fitting in background."""
        # Load data
        rheo_data = RheoData(
            x=self.time_data,
            y=self.stress_data,
            test_mode='relaxation'
        )

        # Create worker
        worker = FitWorker(
            model_name='maxwell',
            data=rheo_data,
            options={'max_iter': 5000}
        )

        # Connect worker signals
        worker.signals.progress.connect(
            lambda iter, loss, msg: self.update_progress(msg)
        )
        worker.signals.completed.connect(self.on_fit_completed)

        # Submit to pool
        self.current_job_id = self.pool.submit(worker)
        self.fit_button.setEnabled(False)

    def on_fit_completed(self, result):
        """Handle NLSQ fit completion."""
        self.fit_result = result
        self.status_label.setText(
            f"Fit completed: R²={result.r_squared:.4f}, "
            f"Time={result.fit_time:.2f}s"
        )

        # Enable Bayesian inference with warm-start
        self.bayesian_button.setEnabled(True)
        self.fit_button.setEnabled(True)

    def start_bayesian(self):
        """Start Bayesian inference with NLSQ warm-start."""
        if self.fit_result is None:
            self.status_label.setText("Run NLSQ fit first!")
            return

        # Load data
        rheo_data = RheoData(
            x=self.time_data,
            y=self.stress_data,
            test_mode='relaxation'
        )

        # Create worker with warm-start
        worker = BayesianWorker(
            model_name='maxwell',
            data=rheo_data,
            num_warmup=1000,
            num_samples=2000,
            num_chains=4,
            warm_start=self.fit_result.parameters,  # Use NLSQ result
            seed=42
        )

        # Connect worker signals
        worker.signals.progress.connect(self.update_bayesian_progress)
        worker.signals.stage_changed.connect(
            lambda stage: self.status_label.setText(f"Stage: {stage}")
        )
        worker.signals.divergence_detected.connect(
            lambda count: self.status_label.setText(
                f"Warning: {count} divergences"
            )
        )
        worker.signals.completed.connect(self.on_bayesian_completed)

        # Submit to pool
        self.current_job_id = self.pool.submit(worker)
        self.bayesian_button.setEnabled(False)

    def on_bayesian_completed(self, result):
        """Handle Bayesian inference completion."""
        self.status_label.setText(
            f"Bayesian completed: {result.num_chains} chains, "
            f"{result.sampling_time:.2f}s"
        )

        # Display diagnostics
        for param_name, samples in result.posterior_samples.items():
            r_hat = result.diagnostics['r_hat'][param_name]
            ess = result.diagnostics['ess'][param_name]
            ci_lower, ci_upper = result.credible_intervals[param_name]
            print(f"{param_name}: R-hat={r_hat:.4f}, ESS={ess:.0f}, "
                  f"95% CI=[{ci_lower:.2e}, {ci_upper:.2e}]")

        self.bayesian_button.setEnabled(True)

    def cancel_jobs(self):
        """Cancel all active jobs."""
        if self.current_job_id:
            self.pool.cancel(self.current_job_id)

    def on_job_started(self, job_id):
        """Handle job start."""
        self.status_label.setText(f"Job {job_id[:8]} started")
        self.cancel_button.setEnabled(True)

    def on_job_progress(self, job_id, current, total, message):
        """Update progress bar."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(current)
        self.status_label.setText(message)

    def on_job_completed(self, job_id, result):
        """Handle job completion."""
        self.cancel_button.setEnabled(False)
        self.progress_bar.setValue(0)

    def on_job_failed(self, job_id, error_message):
        """Handle job failure."""
        self.status_label.setText(f"Error: {error_message}")
        self.cancel_button.setEnabled(False)
        self.fit_button.setEnabled(True)
        self.bayesian_button.setEnabled(True)

    def on_job_cancelled(self, job_id):
        """Handle job cancellation."""
        self.status_label.setText("Job cancelled")
        self.cancel_button.setEnabled(False)
        self.fit_button.setEnabled(True)
        self.bayesian_button.setEnabled(True)

    def update_progress(self, message):
        """Update status with progress message."""
        self.status_label.setText(message)

    def update_bayesian_progress(self, chain, sample, total):
        """Update Bayesian sampling progress."""
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(sample)

    def closeEvent(self, event):
        """Shutdown worker pool on window close."""
        self.pool.shutdown(wait=True, timeout_ms=5000)
        event.accept()
```

## Best Practices

### 1. Always Use safe_import_jax()

```python
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()  # Enforces float64
```

### 2. Import Models Inside Worker.run()

```python
def run(self):
    # Import inside run() to avoid JAX initialization issues
    from rheojax.models import ModelRegistry
    model_class = ModelRegistry.get(self._model_name)
    model = model_class()
    # ... rest of implementation
```

### 3. Check Cancellation Regularly

```python
def long_operation(self):
    for i in range(1000):
        self.cancel_token.check()  # Raises CancellationError if cancelled
        # ... do work
```

### 4. Use NLSQ Warm-Start for Bayesian

```python
# Step 1: NLSQ fit (fast)
fit_worker = FitWorker(model_name='maxwell', data=rheo_data)
pool.submit(fit_worker)

# Step 2: Bayesian with warm-start (2-5x faster convergence)
def on_fit_completed(result):
    bayesian_worker = BayesianWorker(
        model_name='maxwell',
        data=rheo_data,
        warm_start=result.parameters  # Use NLSQ result
    )
    pool.submit(bayesian_worker)
```

### 5. Handle Errors Gracefully

```python
worker.signals.failed.connect(lambda error:
    QMessageBox.critical(self, "Error", f"Operation failed: {error}")
)
```

### 6. Shutdown Pool on Application Exit

```python
def closeEvent(self, event):
    self.pool.shutdown(wait=True, timeout_ms=5000)
    event.accept()
```

## Performance Considerations

1. **Thread Count**: Default 4 threads is optimal for most systems
   - CPU-bound: `max_threads = cpu_count`
   - I/O-bound: `max_threads = 2 * cpu_count`

2. **NLSQ Performance**: 5-270x faster than scipy on CPU

3. **Bayesian Warm-Start**: 2-5x faster convergence with NLSQ initialization

4. **Progress Callbacks**: Keep lightweight to avoid blocking worker threads

5. **Memory**: JAX operations run in separate threads but share GPU memory

## Thread Safety

- ✅ **CancellationToken**: Fully thread-safe
- ✅ **WorkerPool**: Thread-safe job submission/cancellation
- ✅ **Signals**: PySide6 signals are thread-safe (queued connections)
- ⚠️ **JAX Arrays**: Immutable but compilation cache is shared
- ❌ **UI Components**: Never access directly from worker threads

## Troubleshooting

### "PySide6 is required"

```bash
pip install PySide6
```

### "Model not found in registry"

Check model name spelling:
```python
from rheojax.models import ModelRegistry
print(ModelRegistry.list())  # Show available models
```

### Worker hangs without progress

Ensure progress callbacks are lightweight and check cancellation:
```python
def progress_callback(iter, loss):
    self.cancel_token.check()  # Add cancellation checks
    self.signals.progress.emit(iter, loss, f"Iteration {iter}")
```

### High divergences in Bayesian inference

Use NLSQ warm-start and increase warmup:
```python
BayesianWorker(
    warm_start=fit_result.parameters,  # NLSQ result
    num_warmup=2000,  # Increase from 1000
    num_samples=2000
)
```
