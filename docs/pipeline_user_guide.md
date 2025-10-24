# Pipeline API User Guide

The Pipeline API provides a fluent, intuitive interface for rheological analysis workflows. This guide demonstrates common usage patterns and workflows.

## Table of Contents

1. [Basic Pipeline Usage](#basic-pipeline-usage)
2. [Model Comparison Workflow](#model-comparison-workflow)
3. [Mastercurve Construction](#mastercurve-construction)
4. [Data Conversion Pipelines](#data-conversion-pipelines)
5. [Batch Processing](#batch-processing)
6. [Pipeline Builder](#pipeline-builder)

---

## Basic Pipeline Usage

### Simple Analysis Workflow

The most basic workflow involves loading data, fitting a model, and visualizing results:

```python
from rheo.pipeline import Pipeline

# Create and execute pipeline with method chaining
pipeline = (Pipeline()
    .load('relaxation_data.csv', x_col='time', y_col='stress')
    .fit('maxwell')
    .plot(style='publication')
    .save('results.hdf5'))

# Get the fitted model
model = pipeline.get_last_model()
params = model.get_params()
print(f"Fitted parameters: {params}")

# Get execution history
history = pipeline.get_history()
for step in history:
    print(f"Step: {step}")
```

### Load-Transform-Fit Workflow

Apply data transforms before model fitting:

```python
from rheo.pipeline import Pipeline

pipeline = (Pipeline()
    .load('noisy_data.csv', x_col='time', y_col='modulus')
    .transform('smooth', window_size=5)
    .transform('derivative')  # Chain multiple transforms
    .fit('zener', method='L-BFGS-B')
    .plot(include_prediction=True, style='presentation'))

# Access the result
data = pipeline.get_result()
print(f"Data shape: {data.shape}")
print(f"Data domain: {data.domain}")
```

### Working with Predictions

Generate predictions from fitted models:

```python
import numpy as np
from rheo.pipeline import Pipeline

# Fit model
pipeline = (Pipeline()
    .load('data.csv', x_col='t', y_col='G')
    .fit('maxwell'))

# Predict on original data
predictions = pipeline.predict()

# Predict on new time points
new_times = np.logspace(-2, 3, 100)
new_predictions = pipeline.predict(X=new_times)

# Access prediction data
print(f"Predicted values: {new_predictions.y}")
print(f"Prediction metadata: {new_predictions.metadata}")
```

---

## Model Comparison Workflow

Compare multiple models on the same dataset to find the best fit:

```python
from rheo.pipeline import ModelComparisonPipeline
from rheo.core.data import RheoData
import numpy as np

# Load or create data
t = np.logspace(-2, 2, 50)
G_t = 1000 * np.exp(-t / 1.0) + 500 * np.exp(-t / 10.0)
data = RheoData(x=t, y=G_t, x_units='s', y_units='Pa', domain='time')

# Compare multiple models
models_to_compare = ['maxwell', 'zener', 'springpot', 'fractional_maxwell']
pipeline = ModelComparisonPipeline(models_to_compare)
pipeline.run(data)

# Get best model by different metrics
best_rmse = pipeline.get_best_model(metric='rmse', minimize=True)
best_r2 = pipeline.get_best_model(metric='r_squared', minimize=False)
best_aic = pipeline.get_best_model(metric='aic', minimize=True)

print(f"Best model (RMSE): {best_rmse}")
print(f"Best model (R²): {best_r2}")
print(f"Best model (AIC): {best_aic}")

# View comparison table
table = pipeline.get_comparison_table()
for model_name, metrics in table.items():
    print(f"\n{model_name}:")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²: {metrics['r_squared']:.4f}")
    print(f"  AIC: {metrics['aic']:.2f}")
    print(f"  Parameters: {metrics['n_params']}")

# Get detailed results for specific model
best_result = pipeline.get_model_result(best_aic)
print(f"\nBest model parameters: {best_result['parameters']}")

# Access fitted model for further analysis
best_model = best_result['model']
```

---

## Mastercurve Construction

Construct mastercurves from multi-temperature rheological data:

```python
from rheo.pipeline import MastercurvePipeline

# Define file paths and corresponding temperatures
file_paths = [
    'data_273K.csv',
    'data_298K.csv',
    'data_323K.csv',
    'data_348K.csv'
]
temperatures = [273.15, 298.15, 323.15, 348.15]  # in Kelvin

# Create mastercurve pipeline with reference temperature
pipeline = MastercurvePipeline(reference_temp=298.15)

# Run mastercurve construction
pipeline.run(
    file_paths=file_paths,
    temperatures=temperatures,
    format='csv'
)

# Get the mastercurve data
mastercurve = pipeline.get_result()

# Get shift factors
shift_factors = pipeline.get_shift_factors()
for temp, shift in shift_factors.items():
    print(f"T = {temp:.2f} K: a_T = {shift:.4e}")

# Visualize mastercurve
pipeline.plot(style='publication', show=True)

# Save mastercurve
pipeline.save('mastercurve.hdf5')
```

---

## Data Conversion Pipelines

### Creep to Relaxation Conversion

Convert creep compliance J(t) to relaxation modulus G(t):

```python
from rheo.pipeline import CreepToRelaxationPipeline
from rheo.core.data import RheoData
import numpy as np

# Load creep data
t = np.logspace(-2, 3, 100)
J_t = 1e-3 * (1 + t / 5.0)  # Creep compliance
creep_data = RheoData(
    x=t,
    y=J_t,
    x_units='s',
    y_units='1/Pa',
    metadata={'test_mode': 'creep'}
)

# Convert to relaxation modulus
pipeline = CreepToRelaxationPipeline()
pipeline.run(creep_data, method='approximate')

# Get relaxation modulus
G_t_data = pipeline.get_result()
print(f"Conversion method: {G_t_data.metadata['conversion_method']}")

# Use converted data for further analysis
comparison = ModelComparisonPipeline(['maxwell', 'zener'])
comparison.run(G_t_data)
```

### Frequency to Time Domain Conversion

Convert dynamic modulus G*(ω) to relaxation modulus G(t):

```python
from rheo.pipeline import FrequencyToTimePipeline
import numpy as np
from rheo.core.data import RheoData

# Create frequency domain data
omega = np.logspace(-2, 2, 50)
G_prime = 1000 * omega**2 / (1 + omega**2)
G_double_prime = 1000 * omega / (1 + omega**2)
G_star = G_prime + 1j * G_double_prime

freq_data = RheoData(
    x=omega,
    y=G_star,
    x_units='rad/s',
    y_units='Pa',
    domain='frequency'
)

# Convert to time domain
pipeline = FrequencyToTimePipeline()
pipeline.run(freq_data, time_range=(0.01, 100), n_points=100)

# Get time domain data
time_data = pipeline.get_result()
print(f"Time range: {time_data.x.min():.3e} to {time_data.x.max():.3e} s")
```

---

## Batch Processing

Process multiple datasets with the same pipeline configuration:

```python
from rheo.pipeline import Pipeline, BatchPipeline

# Create a template pipeline
template = (Pipeline()
    .fit('maxwell')
    .plot(show=False, style='publication'))

# Create batch processor
batch = BatchPipeline(template)

# Process all CSV files in a directory
batch.process_directory(
    'experimental_data/',
    pattern='*.csv',
    format='csv',
    x_col='time',
    y_col='modulus'
)

# Get processing statistics
stats = batch.get_statistics()
print(f"Total files: {stats['total_files']}")
print(f"Success rate: {stats['success_rate']:.1%}")
print(f"Mean R²: {stats.get('mean_r_squared', 0):.4f}")
print(f"Mean RMSE: {stats.get('mean_rmse', 0):.4e}")

# Export summary to Excel
batch.export_summary('batch_results.xlsx', format='excel')

# Filter results
batch.apply_filter(lambda path, data, metrics: metrics.get('r_squared', 0) > 0.95)
print(f"High-quality fits: {len(batch)}")

# Access individual results
for file_path, data, metrics in batch.get_results():
    print(f"{file_path}: R² = {metrics.get('r_squared', 0):.4f}")

# Process specific files
file_list = ['sample1.csv', 'sample2.csv', 'sample3.csv']
batch.clear()  # Clear previous results
batch.process_files(file_list, format='csv', x_col='t', y_col='G')
```

### Batch Processing with Custom Template

```python
from rheo.pipeline import Pipeline, BatchPipeline

# Create more complex template
template = (Pipeline()
    .transform('smooth', window_size=5)
    .fit('zener', method='L-BFGS-B')
    .plot(include_prediction=True, show=False))

# Process recursively through subdirectories
batch = BatchPipeline(template)
batch.process_directory(
    'data/',
    pattern='*.txt',
    recursive=True,  # Search subdirectories
    format='csv',
    delimiter='\t'  # Tab-separated
)

# Export to CSV for further analysis
batch.export_summary('summary.csv', format='csv')
```

---

## Pipeline Builder

For programmatic pipeline construction with validation:

```python
from rheo.pipeline import PipelineBuilder

# Build pipeline step by step
builder = PipelineBuilder()
builder.add_load_step('data.csv', format='csv', x_col='time', y_col='stress')
builder.add_transform_step('smooth', window_size=5)
builder.add_fit_step('maxwell', method='L-BFGS-B', use_jax=True)
builder.add_plot_step(style='publication', show=False)
builder.add_save_step('output.hdf5')

# Build and execute
pipeline = builder.build()

# Or use method chaining
pipeline = (PipelineBuilder()
    .add_load_step('data.csv', x_col='t', y_col='G')
    .add_transform_step('smooth', window_size=3)
    .add_fit_step('zener')
    .add_plot_step()
    .build())
```

### Pipeline Builder with Validation

```python
from rheo.pipeline import PipelineBuilder

# The builder validates pipeline structure
builder = PipelineBuilder()

# This will fail validation (no load step first)
builder.add_fit_step('maxwell')
try:
    pipeline = builder.build()  # Raises ValueError
except ValueError as e:
    print(f"Validation error: {e}")

# Correct order
builder.clear()
builder.add_load_step('data.csv', x_col='t', y_col='G')
builder.add_fit_step('maxwell')
pipeline = builder.build()  # Success!

# Skip validation if needed (advanced use)
builder.clear()
builder.add_fit_step('maxwell')  # Invalid order
pipeline = builder.build(validate=False)  # Succeeds, but may fail at runtime
```

### Inspect Builder Steps

```python
builder = (PipelineBuilder()
    .add_load_step('data.csv')
    .add_fit_step('maxwell')
    .add_plot_step())

# View steps before building
steps = builder.get_steps()
print(f"Number of steps: {len(builder)}")
for step_type, kwargs in steps:
    print(f"- {step_type}: {kwargs}")

# Clear and rebuild
builder.clear()
assert len(builder) == 0
```

---

## Advanced Patterns

### Pipeline Cloning

Clone pipelines for parallel analysis:

```python
from rheo.pipeline import Pipeline

# Original pipeline
pipeline1 = (Pipeline()
    .load('data.csv', x_col='t', y_col='G')
    .fit('maxwell'))

# Clone for alternative analysis
pipeline2 = pipeline1.clone()
pipeline2.fit('zener')  # Different model

# Compare results
r2_maxwell = pipeline1.get_last_model().score(
    pipeline1.data.x,
    pipeline1.data.y
)
r2_zener = pipeline2.get_last_model().score(
    pipeline2.data.x,
    pipeline2.data.y
)

print(f"Maxwell R²: {r2_maxwell:.4f}")
print(f"Zener R²: {r2_zener:.4f}")
```

### Pipeline Reset

Reset pipeline to initial state:

```python
pipeline = Pipeline()
pipeline.load('data.csv', x_col='t', y_col='G')
pipeline.fit('maxwell')

print(f"Before reset: {len(pipeline.history)} steps")

pipeline.reset()
print(f"After reset: {len(pipeline.history)} steps")  # 0

# Reuse pipeline
pipeline.load('different_data.csv', x_col='t', y_col='G')
pipeline.fit('zener')
```

---

## Error Handling

### Handling Missing Data

```python
from rheo.pipeline import Pipeline

pipeline = Pipeline()

try:
    pipeline.load('nonexistent.csv')
except FileNotFoundError as e:
    print(f"File not found: {e}")

# Operations without data
try:
    pipeline.fit('maxwell')  # No data loaded
except ValueError as e:
    print(f"Error: {e}")  # "No data loaded. Call load() first."
```

### Batch Error Handling

```python
from rheo.pipeline import BatchPipeline, Pipeline

batch = BatchPipeline(Pipeline())
batch.process_files(
    ['good.csv', 'bad.csv', 'missing.csv'],
    format='csv'
)

# Check errors
errors = batch.get_errors()
for file_path, error in errors:
    print(f"Error in {file_path}: {error}")

# Process only successful results
results = batch.get_results()
print(f"Successfully processed: {len(results)} files")
```

---

## Best Practices

### 1. Use Method Chaining for Readability

```python
# Good: Clear, readable workflow
pipeline = (Pipeline()
    .load('data.csv', x_col='time', y_col='stress')
    .transform('smooth', window_size=5)
    .fit('maxwell')
    .plot(style='publication'))

# Less ideal: Multiple statements
pipeline = Pipeline()
pipeline.load('data.csv', x_col='time', y_col='stress')
pipeline.transform('smooth', window_size=5)
pipeline.fit('maxwell')
pipeline.plot(style='publication')
```

### 2. Save Intermediate Results

```python
pipeline = (Pipeline()
    .load('data.csv', x_col='t', y_col='G')
    .transform('smooth', window_size=5)
    .save('smoothed_data.hdf5')  # Save intermediate result
    .fit('maxwell')
    .save('fitted_results.hdf5'))  # Save final result
```

### 3. Use Model Comparison for Unknown Systems

```python
# Don't assume a model
# pipeline.fit('maxwell')  # May not be best fit

# Compare multiple models first
comparison = ModelComparisonPipeline(['maxwell', 'zener', 'springpot'])
comparison.run(data)
best = comparison.get_best_model(metric='aic')

# Then use best model
pipeline = Pipeline(data=data).fit(best)
```

### 4. Validate Data Before Fitting

```python
from rheo.pipeline import Pipeline

pipeline = Pipeline().load('data.csv', x_col='t', y_col='G')
data = pipeline.get_result()

# Check data quality
print(f"Data points: {len(data.x)}")
print(f"Data range: {data.x.min():.2e} to {data.x.max():.2e}")
print(f"NaN values: {np.isnan(data.y).sum()}")

# Proceed if quality is good
if len(data.x) > 10 and not np.isnan(data.y).any():
    pipeline.fit('maxwell')
else:
    print("Data quality insufficient for fitting")
```

---

## Performance Tips

### 1. Use JAX Gradients for Faster Fitting

```python
# JAX gradients (default, faster)
pipeline.fit('maxwell', use_jax=True)

# Numerical gradients (slower)
pipeline.fit('maxwell', use_jax=False)
```

### 2. Batch Processing is More Efficient

```python
# Inefficient: Loop over files
for file in file_list:
    pipeline = Pipeline().load(file).fit('maxwell')

# Efficient: Use batch processing
batch = BatchPipeline(Pipeline().fit('maxwell'))
batch.process_files(file_list)
```

### 3. Choose Appropriate Optimization Methods

```python
# Fast, general purpose
pipeline.fit('maxwell', method='L-BFGS-B')

# More robust for complex models
pipeline.fit('fractional_maxwell', method='trust-constr')

# Auto-select based on bounds
pipeline.fit('maxwell', method='auto')
```

---

## Integration Examples

### Combine with Plotting

```python
import matplotlib.pyplot as plt
from rheo.pipeline import Pipeline

# Create figure with subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Fit and plot on custom axes
pipeline = Pipeline().load('data.csv', x_col='t', y_col='G')
pipeline.fit('maxwell')

# Manual plotting with custom styling
predictions = pipeline.predict()
ax1.loglog(pipeline.data.x, pipeline.data.y, 'o', label='Data')
ax1.loglog(predictions.x, predictions.y, '-', label='Fit')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Modulus (Pa)')
ax1.legend()

# Use built-in plotting
pipeline.plot(show=False)

plt.tight_layout()
plt.show()
```

### Export for Further Analysis

```python
from rheo.pipeline import Pipeline
import pandas as pd

pipeline = (Pipeline()
    .load('data.csv', x_col='t', y_col='G')
    .fit('maxwell'))

# Get results
model = pipeline.get_last_model()
data = pipeline.get_result()
predictions = pipeline.predict()

# Create analysis DataFrame
df = pd.DataFrame({
    'time': data.x,
    'measured': data.y,
    'predicted': predictions.y,
    'residual': data.y - predictions.y
})

# Export for other tools
df.to_csv('analysis_results.csv', index=False)

# Export parameters
params_df = pd.DataFrame([model.get_params()])
params_df.to_csv('model_parameters.csv', index=False)
```

---

This user guide covers the main usage patterns for the Pipeline API. For more detailed information on specific models, transforms, and advanced features, refer to the API documentation.
