.. _tensorial_epm:

Tensorial Elasto-Plastic Model (EPM)
=====================================

The Tensorial EPM extends the scalar :doc:`lattice_epm` to track the **full stress tensor**, enabling predictions of **normal stress differences** (N₁, N₂), anisotropic yielding, and flow-induced microstructure. This is critical for capturing non-Newtonian behaviors like rod climbing (Weissenberg effect), die swell, and flow instabilities.

.. contents:: Table of Contents
    :local:
    :depth: 2

Physical Interpretation
-----------------------

Tensorial vs Scalar EPM
~~~~~~~~~~~~~~~~~~~~~~~~

The scalar EPM models only the shear component σ_xy. While this captures flow curves and avalanches, it misses:

- **Normal stress differences**: N₁ = σ_xx - σ_yy (rod climbing) and N₂ = σ_yy - σ_zz (secondary flows)
- **Anisotropic yielding**: Materials with directional microstructure (fiber suspensions, liquid crystals)
- **Flow instabilities**: Edge fracture, shear banding driven by normal stress gradients

The Tensorial EPM tracks [σ_xx, σ_yy, σ_xy] at each lattice site, with σ_zz = ν(σ_xx + σ_yy) from the plane strain constraint.

Tensorial Eshelby Propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a site yields plastically, the stress redistribution couples all components:

.. math::

    \frac{\partial \sigma_{ij}}{\partial t} = \mu \dot{\gamma} \delta_{ij}
    - \frac{\sigma_{ij}}{\tau_{ij}^{pl}} f(\sigma_{eff}, \sigma_c)
    + \sum_{kl} \mathcal{G}_{ij,kl}(\mathbf{q}) \dot{\gamma}^{pl}_{kl}

The tensorial propagator :math:`\mathcal{G}_{ij,kl}` derives from the Eshelby inclusion solution in plane strain:

.. math::

    \mathcal{G}_{ij,kl}(\mathbf{q}) = C_{ijmn} \frac{q_m q_n}{|\mathbf{q}|^2}

where C is the elastic stiffness tensor with shear modulus μ and Poisson ratio ν.

Yield Criteria
~~~~~~~~~~~~~~

**Von Mises (Isotropic)**:

.. math::

    \sigma_{eff} = \frac{1}{\sqrt{2}} \sqrt{(\sigma_{xx} - \sigma_{yy})^2 + (\sigma_{yy} - \sigma_{zz})^2 + (\sigma_{zz} - \sigma_{xx})^2 + 6\sigma_{xy}^2}

**Hill (Anisotropic)**:

.. math::

    \sigma_{eff} = \sqrt{H (\sigma_{xx} - \sigma_{yy})^2 + 2N \sigma_{xy}^2}

where H and N are anisotropy parameters (H=1, N=3 recovers von Mises in plane stress).

Prandtl-Reuss Flow Rule
~~~~~~~~~~~~~~~~~~~~~~~~

Plastic flow is component-wise with independent timescales:

.. math::

    \dot{\gamma}^{pl}_{ij} = \frac{\sigma'_{ij}}{\tau^{pl}_{ij}} \Theta(\sigma_{eff} - \sigma_c)

where :math:`\sigma'_{ij}` is the deviatoric stress. Separate τ_pl_shear and τ_pl_normal allow modeling materials with different relaxation times for shear and dilation.

Mathematical Formulation
------------------------

Plane Strain Constraint
~~~~~~~~~~~~~~~~~~~~~~~

For 2D flow with out-of-plane confinement:

.. math::

    \sigma_{zz} = \nu (\sigma_{xx} + \sigma_{yy})

This couples the in-plane components and leads to non-zero N₂.

Normal Stress Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    N_1 &= \sigma_{xx} - \sigma_{yy} \\
    N_2 &= \sigma_{yy} - \sigma_{zz} = (1 - \nu) \sigma_{yy} - \nu \sigma_{xx}

Typical experimental observations:
- Polymer melts: N₁ > 0 (rod climbing), |N₂| ≪ N₁
- Shear banding: Large gradients in N₁ correlate with band boundaries

Fitting to Normal Stress Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loss function for combined fitting is:

.. math::

    \mathcal{L} = \sum_i (\sigma_{xy,i}^{pred} - \sigma_{xy,i}^{data})^2 + w_{N1} \sum_i (N_{1,i}^{pred} - N_{1,i}^{data})^2

Set ``w_N1 > 1`` to prioritize normal stress accuracy.

API Reference
-------------

.. autoclass:: rheojax.models.epm.tensor.TensorialEPM
    :members:
    :undoc-members:
    :show-inheritance:

Parameters
----------

.. list-table:: TensorialEPM Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``mu``
     - Pa
     - [0.1, 100.0]
     - Shear modulus (elastic stiffness)
   * - ``nu``
     - dimensionless
     - [0.3, 0.5]
     - Poisson's ratio (plane strain), avoid 0.5 (incompressible singularity)
   * - ``tau_pl_shear``
     - s
     - [0.01, 100.0]
     - Plastic relaxation time for shear components (σ_xy)
   * - ``tau_pl_normal``
     - s
     - [0.01, 100.0]
     - Plastic relaxation time for normal stresses (σ_xx, σ_yy)
   * - ``sigma_c_mean``
     - Pa
     - [0.1, 10.0]
     - Mean local yield threshold
   * - ``sigma_c_std``
     - Pa
     - [0.0, 1.0]
     - Disorder strength (std dev of thresholds)
   * - ``w_N1``
     - dimensionless
     - [0.1, 10.0]
     - Weight for N₁ in combined fitting loss
   * - ``hill_H``
     - dimensionless
     - [0.1, 5.0]
     - Hill anisotropy parameter H (normal stress coupling)
   * - ``hill_N``
     - dimensionless
     - [0.1, 5.0]
     - Hill anisotropy parameter N (shear amplification)

**Configuration (not fitted)**:

.. list-table:: Configuration Parameters
   :header-rows: 1
   :widths: 18 18 64

   * - Parameter
     - Default
     - Description
   * - ``L``
     - 64
     - Lattice size (L×L grid)
   * - ``dt``
     - 0.01
     - Time step for integration
   * - ``yield_criterion``
     - "von_mises"
     - Yield criterion: "von_mises" (isotropic) or "hill" (anisotropic)
   * - ``seed``
     - 0
     - Random seed for threshold initialization (reproducibility)

Usage Examples
--------------

Basic Flow Curve with N₁ Prediction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.models.epm.tensor import TensorialEPM
    from rheojax.core.data import RheoData
    import numpy as np

    # Initialize model
    model = TensorialEPM(L=64, dt=0.01, mu=1.0, nu=0.48)

    # Define shear rates
    shear_rates = np.logspace(-2, 1, 10)
    data = RheoData(x=shear_rates, y=None, initial_test_mode="flow_curve")

    # Predict flow curve (returns σ_xy with N₁ in metadata)
    result = model.predict(data, smooth=True)

    print(f"Shear stress: {result.y}")
    print(f"Normal stress N₁: {result.metadata['N1']}")

Fitting to Shear-Only Data (Backward Compatible)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Experimental shear stress data
    gamma_dot = np.array([0.01, 0.1, 1.0, 10.0])
    sigma_xy_exp = np.array([0.5, 1.2, 2.8, 5.1])

    rheo_data = RheoData(x=gamma_dot, y=sigma_xy_exp, initial_test_mode="flow_curve")

    # Fit model to shear stress only (standard workflow)
    model = TensorialEPM(L=32)  # Smaller L for faster fitting
    model.fit(rheo_data, max_iter=100)

    print(f"Fitted mu: {model.params.get_value('mu'):.3f}")
    print(f"Fitted sigma_c_mean: {model.params.get_value('sigma_c_mean'):.3f}")

Fitting to Combined [σ_xy, N₁] Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Experimental data with normal stresses
    gamma_dot = np.array([0.1, 1.0, 10.0])
    sigma_xy_exp = np.array([1.2, 2.8, 5.1])
    N1_exp = np.array([0.3, 1.5, 4.2])  # N₁ typically ~ σ_xy in magnitude

    # Combine data (requires custom fitting workflow)
    # Option 1: Use metadata to pass N₁ targets (future feature)
    # Option 2: Manually construct multi-objective loss

    # Current best practice: Fit shear first, then validate N₁
    rheo_data = RheoData(x=gamma_dot, y=sigma_xy_exp, initial_test_mode="flow_curve")
    model = TensorialEPM(L=32, w_N1=2.0)  # Higher weight for N₁
    model.fit(rheo_data, max_iter=200)

    # Check N₁ predictions
    pred_result = model.predict(rheo_data)
    N1_pred = pred_result.metadata["N1"]
    print(f"N₁ RMSE: {np.sqrt(np.mean((N1_pred - N1_exp)**2)):.3f}")

Comparison of Yield Criteria
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Von Mises (isotropic)
    model_vm = TensorialEPM(L=32, yield_criterion="von_mises")
    model_vm.fit(rheo_data)

    # Hill (anisotropic)
    model_hill = TensorialEPM(L=32, yield_criterion="hill")
    model_hill.params.set_value("hill_H", 1.5)  # Stronger normal coupling
    model_hill.params.set_value("hill_N", 2.0)  # Weaker shear resistance
    model_hill.fit(rheo_data)

    # Compare predictions
    pred_vm = model_vm.predict(rheo_data)
    pred_hill = model_hill.predict(rheo_data)

Visualization
~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.visualization.epm_plots import (
        plot_tensorial_fields,
        plot_von_mises_field,
        plot_normal_stress_ratio,
    )

    # Run a startup simulation to get stress field history
    t = np.linspace(0, 10, 100)
    startup_data = RheoData(x=t, y=None, initial_test_mode="startup")
    startup_data.metadata = {"gamma_dot": 1.0}

    result = model.predict(startup_data, smooth=False)

    # Plot tensorial stress components (from history if saved)
    # Note: Access to intermediate stress fields requires history tracking
    # See examples/tensorial_epm_visualization_demo.py

When to Use TensorialEPM vs LatticeEPM
---------------------------------------

.. list-table:: Model Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Use Case
     - LatticeEPM (Scalar)
     - TensorialEPM
   * - Flow curves (σ vs γ̇)
     - ✓ Faster (3-5x)
     - ✓ More accurate if N₁ ≠ 0
   * - Yield stress determination
     - ✓ Sufficient
     - ✓ Accounts for anisotropy
   * - Normal stress differences
     - ✗
     - ✓ Required
   * - Shear banding analysis
     - ~ Qualitative
     - ✓ Quantitative (N₁ gradients)
   * - Rod climbing / die swell
     - ✗
     - ✓ Required
   * - Anisotropic materials
     - ✗
     - ✓ Hill criterion
   * - Computational cost
     - 1x (baseline)
     - 3-5x (9 propagator components vs 1)
   * - Memory usage
     - 1x
     - 3x (stress tensor storage)

**Recommendation**: Start with LatticeEPM for flow curve fitting. Use TensorialEPM when:

1. Normal stress data is available
2. Material shows strong anisotropy (e.g., fiber suspensions)
3. Analyzing flow instabilities (shear banding, edge fracture)
4. Modeling 3D extrusion flows

Troubleshooting
---------------

N₁ Predictions Are Too Small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cause**: Insufficient disorder (σ_c_std too low) or ν too high (approaching incompressible limit)
- **Fix**: Increase ``sigma_c_std`` to 0.2-0.5 or reduce ``nu`` to 0.40-0.45

Fitting Fails to Converge
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cause**: Combined [σ_xy, N₁] loss has competing gradients
- **Fix**: Fit shear first with ``w_N1=0``, then refine with ``w_N1 > 0``

Von Mises vs Hill Give Similar Results
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cause**: Hill parameters (H, N) default to isotropic values
- **Fix**: Set ``hill_H`` ≠ 1 or ``hill_N`` ≠ 3 to activate anisotropy

GPU Memory Overflow
~~~~~~~~~~~~~~~~~~~

- **Cause**: Large L or long simulation times
- **Fix**: Reduce ``L`` to 32 or 48 for fitting, use ``L=64+`` only for production simulations

References
----------

1. **Hébraud-Lequeux Theory**: Hébraud, P., & Lequeux, F. (1998). *Phys. Rev. Lett.*, 81, 2934.
2. **EPM Framework**: Bocquet, L., Colin, A., & Ajdari, A. (2009). *Phys. Rev. Lett.*, 103, 036001.
3. **Eshelby Tensor**: Eshelby, J. D. (1957). *Proc. R. Soc. Lond. A*, 241, 376.
4. **Hill Anisotropy**: Hill, R. (1948). *Proc. R. Soc. Lond. A*, 193, 281.
5. **Normal Stress Differences**: Bird, R. B., Armstrong, R. C., & Hassager, O. (1987). *Dynamics of Polymeric Liquids*, Vol. 1.
