.. _gui-transforms:

==========
Transforms
==========

The Transform page provides data transformation tools for rheological analysis.

Available Transforms
====================

Mastercurve (Time-Temperature Superposition)
--------------------------------------------

Shift multi-temperature data to a reference temperature:

**Usage**:

1. Load datasets at different temperatures
2. Select **Mastercurve** transform
3. Set reference temperature
4. Choose shift method (auto or manual)
5. Apply transform

**Parameters**:

- **Reference Temperature**: T_ref in °C or K
- **Auto Shift**: Automatic shift factor calculation
- **Manual Shifts**: Enter known shift factors

**Output**:

- Mastercurve at T_ref
- Shift factors (a_T)
- Williams-Landel-Ferry (WLF) fit (if applicable)

FFT (Fourier Transform)
-----------------------

Convert between time and frequency domains:

**Usage**:

1. Select time-domain dataset
2. Choose **FFT** transform
3. Configure window and padding
4. Apply transform

**Parameters**:

- **Window**: Hanning, Hamming, Blackman, None
- **Zero Padding**: Extend data for resolution
- **Nyquist Warning**: Alert for aliasing

**Output**:

- Frequency-domain G*(ω)
- G'(ω) and G''(ω)

IFFT (Inverse Fourier Transform)
--------------------------------

Convert frequency to time domain:

**Usage**:

1. Select frequency-domain dataset
2. Choose **IFFT** transform
3. Apply transform

**Output**:

- Time-domain G(t)

Derivatives
-----------

Numerical differentiation:

**Usage**:

1. Select dataset
2. Choose **Derivative** transform
3. Select order (1st or 2nd)
4. Choose method

**Parameters**:

- **Order**: 1 (first derivative), 2 (second derivative)
- **Method**: Central difference, Savitzky-Golay
- **Smoothing**: Window size for noise reduction

**Output**:

- dG/dt or d²G/dt²

SRFS (Strain-Rate Frequency Superposition)
------------------------------------------

Collapse flow curves at different shear rates:

**Usage**:

1. Load datasets at different shear rates
2. Select **SRFS** transform
3. Set reference shear rate
4. Apply transform

**Parameters**:

- **Reference Rate**: γ̇_ref
- **Power Law Exponent**: For scaling

**Output**:

- Master flow curve
- Shift factors

Using Transforms
================

Basic Workflow
--------------

1. Navigate to **Transform** page
2. Select source dataset from dropdown
3. Choose transform from list
4. Configure parameters
5. Click **Apply Transform**
6. Review result in preview
7. **Accept** to add as new dataset

Transform Preview
-----------------

Before accepting:

- View transformed data plot
- Check data ranges
- Verify expected behavior
- Adjust parameters if needed

Transform History
-----------------

All transforms are tracked:

- Source dataset
- Transform type
- Parameters used
- Timestamp
- Random seed (if applicable)

This enables reproducibility and audit trails.

Chaining Transforms
-------------------

Apply multiple transforms sequentially:

1. Apply first transform
2. Accept result as new dataset
3. Select new dataset
4. Apply next transform
5. Repeat as needed

Transform Parameters
====================

Mastercurve Settings
--------------------

**Shift Method**:

- **Auto WLF**: Automatic WLF fit
- **Auto Arrhenius**: Automatic Arrhenius fit
- **Manual**: User-specified shift factors

**Overlap Region**:

- Minimum overlap decades for shifting
- Default: 0.5 decades

**Optimization**:

- JAX-accelerated shift optimization
- Multi-start for robustness

FFT Settings
------------

**Windowing**:

- Reduces spectral leakage
- Hanning recommended for most cases

**Padding**:

- Zero-padding increases frequency resolution
- Powers of 2 for efficiency

**Detrending**:

- Remove DC offset
- Linear detrend option

Derivative Settings
-------------------

**Method Comparison**:

- **Central Difference**: Fast, sensitive to noise
- **Savitzky-Golay**: Smoother, preserves features

**Window Size**:

- Larger = smoother, less detail
- Smaller = noisier, more detail
- Odd numbers only

Output Handling
===============

New Dataset
-----------

Transforms create new datasets:

- Original preserved
- Transform result added to dataset list
- Naming: `original_name_transform`

Metadata
--------

Transformed datasets include:

- Source dataset reference
- Transform parameters
- Provenance chain

Export
------

Export transformed data:

1. Select transformed dataset
2. Go to **Export** page
3. Choose format

Tips and Best Practices
=======================

Mastercurve
-----------

1. Ensure sufficient temperature range
2. Check for thermorheological simplicity
3. Use auto-shift first, then refine
4. Validate WLF parameters against literature

FFT
---

1. Use windowing to reduce artifacts
2. Ensure time data is evenly spaced
3. Check Nyquist frequency
4. Zero-pad for smooth spectra

Derivatives
-----------

1. Always smooth noisy data first
2. Start with larger windows
3. Compare methods on same data
4. Validate against known solutions

SRFS
----

1. Verify power-law scaling behavior
2. Check for shear banding (discontinuities)
3. Use consistent strain amplitudes
