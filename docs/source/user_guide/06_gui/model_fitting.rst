.. _gui-model-fitting:

=============
Model Fitting
=============

The Fit page provides interactive NLSQ (Nonlinear Least Squares) model fitting
with real-time visualization.

Available Models
================

Models are organized by category:

Classical Models
----------------

- **Maxwell**: Single exponential relaxation
- **Kelvin-Voigt**: Elastic solid with viscous damping
- **Zener (Standard Linear Solid)**: Maxwell + elastic element

Fractional Models
-----------------

- **Fractional Maxwell**: Viscoelastic with fractional derivatives
- **Fractional Kelvin-Voigt**: Fractional creep response
- **Fractional Zener**: Fractional standard solid

Flow Models
-----------

- **Cross**: Shear-thinning flow
- **Carreau**: Smoothed shear-thinning
- **Power Law**: Simple shear-thinning
- **Herschel-Bulkley**: Yield stress + power law

Multi-Mode Models
-----------------

- **Generalized Maxwell (GMM)**: Multiple relaxation modes

SGR Models
----------

- **SGR Conventional**: Soft glassy rheology
- **SGR Generic**: GENERIC framework SGR

Model Selection
===============

Using the Model Browser
-----------------------

1. Navigate to the **Fit** page
2. In the **Model Browser** panel (left side):

   - Expand categories by clicking arrows
   - Click model name to select
   - Hover for model description tooltip

3. Selected model appears in the **Parameters** panel

Model Information
-----------------

After selecting a model:

- **Description**: Physical interpretation
- **Parameters**: List of model parameters
- **Compatible modes**: Supported test modes (oscillation, relaxation, etc.)

Parameter Configuration
=======================

The Parameter Table shows all model parameters:

Columns
-------

- **Parameter**: Name and description
- **Value**: Current/initial value
- **Min**: Lower bound
- **Max**: Upper bound
- **Fixed**: Lock parameter (don't optimize)
- **Unit**: Physical units

Setting Initial Values
----------------------

Good initial values improve convergence:

1. Click the **Value** cell
2. Enter your estimate
3. Press Enter to confirm

Or use **Auto Initialize** button for smart defaults based on data.

Setting Bounds
--------------

Constrain parameters to physically meaningful ranges:

1. Click **Min** or **Max** cell
2. Enter bound value
3. Bounds prevent non-physical results

Fixing Parameters
-----------------

Hold parameters constant during fitting:

1. Check the **Fixed** checkbox
2. Parameter won't be optimized
3. Useful for known values or constraints

Running the Fit
===============

Starting a Fit
--------------

1. Ensure data is loaded and model selected
2. Configure parameters (or use defaults)
3. Click **"Fit Model"** button

Progress
--------

During fitting:

- Progress bar shows optimization iterations
- Current parameter values update in real-time
- Fit quality metrics displayed

Stopping a Fit
--------------

Click **"Cancel"** to stop early (results may be partial).

Fit Results
===========

Quality Metrics
---------------

After fitting completes:

- **R²**: Coefficient of determination (closer to 1 = better)
- **χ²**: Chi-squared statistic
- **MPE**: Mean percentage error
- **RMSE**: Root mean square error

Fitted Parameters
-----------------

The Parameter Table updates with:

- Optimized parameter values
- Standard errors (if available)
- Correlation matrix

Plot Visualization
==================

The plot canvas shows:

Data and Fit
------------

- **Data points**: Experimental measurements
- **Fit curve**: Model prediction
- **Residuals**: Optional residual subplot

Plot Controls
-------------

- **Zoom**: Mouse wheel or toolbar
- **Pan**: Click and drag
- **Reset**: Double-click or toolbar button
- **Log scale**: Toggle buttons for X/Y axes

Multi-Dataset Fitting
---------------------

Compare fits across datasets:

1. Load multiple datasets
2. Fit each independently
3. Use **Multi-View** to compare side-by-side

Residual Analysis
=================

Open the **Residuals Panel** (View menu):

Available Plots
---------------

- **Residuals vs Fitted**: Check for systematic bias
- **Q-Q Plot**: Test normality of residuals
- **Histogram**: Residual distribution
- **Scale-Location**: Check heteroscedasticity
- **Autocorrelation**: Check independence

Good Fit Indicators
-------------------

- Residuals randomly scattered around zero
- Q-Q plot follows diagonal line
- No patterns in autocorrelation

Advanced Options
================

Optimization Settings
---------------------

Access via **Settings > Fitting Options**:

- **Algorithm**: NLSQ algorithm variant
- **Max Iterations**: Iteration limit
- **Tolerance**: Convergence criteria (ftol, xtol)
- **Multi-start**: Number of random initializations

Warm Start
----------

Use previous fit results as starting point:

1. Fit with model A
2. Switch to similar model B
3. Enable **Warm Start** checkbox
4. Fit model B (starts from model A values)

Batch Fitting
-------------

Fit multiple datasets with same model:

1. Load all datasets
2. Configure model once
3. Click **Fit All Datasets**

Tips for Good Fits
==================

1. **Start simple**: Try simpler models first
2. **Check data range**: Ensure data spans model features
3. **Initial values**: Use Auto Initialize or manual estimates
4. **Bounds**: Set physically meaningful constraints
5. **Check residuals**: Look for systematic patterns
6. **Compare models**: Use R² and AIC for model selection
