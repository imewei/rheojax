# Fix for 06-frequentist-model-selection.ipynb - Oscillation Mode

## Problem

The notebook currently uses `|G*|` (complex modulus magnitude) as scalar data, but models in oscillation mode return `[G', G"]` as a 2D array with shape `(50, 2)`. This causes:

1. **Shape mismatch during optimization**: `residuals = y - y_pred` fails with `shapes=[(50,), (50, 2)]`
2. **Complex number errors**: Optimization internally creates complex predictions but data is real scalar

## Root Cause

The optimization utilities in `rheojax/utils/optimization.py` expect oscillation data as:
- **Complex format**: `G* = G' + iG"` (preferred, enables proper weighting of real/imaginary parts)
- **NOT scalar magnitude**: `|G*| = sqrt(G'^2 + G"^2)`

When data is scalar but models return 2D `[G', G"]`, the residual calculation fails.

## Solution: Use Complex Data

The notebook already generates `G_storage_noisy` (G') and `G_loss_noisy` (G"), so we should combine them into complex format.

### Cell 4 Fix - Change RheoData Creation

**BEFORE (lines ~35-43 in cell 4):**
```python
# For model fitting, we'll use |G*| (complex modulus magnitude)
G_star_true = np.sqrt(G_storage_true**2 + G_loss_true**2)
G_star_noisy = np.sqrt(G_storage_noisy**2 + G_loss_noisy**2)

# Create RheoData object
data = RheoData(
    x=frequency,
    y=G_star_noisy,
    x_units='rad/s',
    y_units='Pa',
    domain='frequency',
)
```

**AFTER:**
```python
# For model fitting, use complex modulus G* = G' + iG"
# This allows proper optimization with both storage and loss moduli
G_star_complex = G_storage_noisy + 1j * G_loss_noisy
G_star_true_complex = G_storage_true + 1j * G_loss_true

# Also keep magnitude for visualization
G_star_noisy = np.sqrt(G_storage_noisy**2 + G_loss_noisy**2)
G_star_true = np.sqrt(G_storage_true**2 + G_loss_true**2)

# Create RheoData object with COMPLEX data
data = RheoData(
    x=frequency,
    y=G_star_complex,  # Complex data: G' + iG"
    x_units='rad/s',
    y_units='Pa',
    domain='frequency',
)
```

### Why This Works

1. **Optimization**: `create_least_squares_objective` detects complex data and properly handles both real (G') and imaginary (G") components
2. **Residuals**: Computes separate residuals for G' and G" with proper normalization
3. **Models**: Return `[G', G"]` which gets automatically converted to complex format for comparison

### Cell 16 Fix - Update Visualization

Predictions from oscillation mode are 2D `[G', G"]`, so we need to convert to magnitude for plotting:

**BEFORE (line ~12 in cell 16):**
```python
if model_name in comparison_pipeline.results:
    predictions = comparison_pipeline.results[model_name]['predictions']
    r2 = comparison_pipeline.results[model_name]['r_squared']
```

**AFTER:**
```python
if model_name in comparison_pipeline.results:
    predictions_raw = comparison_pipeline.results[model_name]['predictions']

    # Convert predictions to magnitude for plotting
    # Oscillation mode returns [G', G"] with shape (n, 2)
    if predictions_raw.ndim == 2 and predictions_raw.shape[1] == 2:
        predictions = np.sqrt(predictions_raw[:, 0]**2 + predictions_raw[:, 1]**2)
    else:
        predictions = predictions_raw

    r2 = comparison_pipeline.results[model_name]['r_squared']
```

### Cell 18 Fix - Update Residual Analysis

Same fix for residual plots:

**BEFORE (line ~7 in cell 18):**
```python
if model_name in comparison_pipeline.results:
    predictions = comparison_pipeline.results[model_name]['predictions']
    residuals = (G_star_noisy - predictions) / G_star_noisy * 100
```

**AFTER:**
```python
if model_name in comparison_pipeline.results:
    predictions_raw = comparison_pipeline.results[model_name]['predictions']

    # Convert predictions to magnitude
    if predictions_raw.ndim == 2 and predictions_raw.shape[1] == 2:
        predictions = np.sqrt(predictions_raw[:, 0]**2 + predictions_raw[:, 1]**2)
    else:
        predictions = predictions_raw

    residuals = (G_star_noisy - predictions) / G_star_noisy * 100
```

## Alternative: ModelComparisonPipeline Enhancement (Already Implemented)

I've already added magnitude calculation to `ModelComparisonPipeline` at `rheojax/pipeline/workflows.py:247-253`:

```python
# Handle complex modulus (oscillation mode)
# Oscillation predictions return (n, 2) array: [G', G"]
# Convert to magnitude |G*| = sqrt(G'^2 + G"^2)
if y_pred.ndim == 2 and y_pred.shape[1] == 2:
    y_pred_magnitude = np.sqrt(y_pred[:, 0]**2 + y_pred[:, 1]**2)
else:
    y_pred_magnitude = y_pred
```

However, this doesn't fix the **optimization errors** which occur during `.fit()`, before the pipeline even gets predictions. The optimization itself fails because of the data format mismatch.

## Recommended Action

**Use complex data (Solution above)**. This is the proper way to fit oscillation data and matches the design of `rheojax.utils.optimization.create_least_squares_objective`.

Benefits:
- Properly weights G' and G" components
- Works with all 20 models in oscillation mode
- Matches standard rheology practice
- Enables accurate parameter estimation

## Testing the Fix

After applying the changes to cell 4, restart the kernel and run:

```python
# Verify data is complex
print(f"Data type: {data.y.dtype}")
print(f"Data shape: {data.y.shape}")
print(f"First value: {data.y[0]}")
# Expected: dtype=complex128, shape=(50,), first value shows G'+iG"
```

Then run the pipeline and verify all 5 models fit successfully.
