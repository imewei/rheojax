.. _tensorial_epm:

Tensorial Elasto-Plastic Model (EPM)
=====================================

Quick Reference
---------------

- **Use when:** Full stress tensor modeling, normal stress differences (:math:`N_1, N_2`), anisotropic yielding, flow instabilities

- **Parameters:** 9 (:math:`\mu, \nu, \tau_{\text{pl,shear}}, \tau_{\text{pl,normal}}, \sigma_{c,\text{mean}}, \sigma_{c,\text{std}}`, w_N1, hill_H, hill_N)

- **Key equation:** :math:`\partial_t \sigma_{ij} = \mu \dot{\gamma} \delta_{ij} - \frac{\sigma_{ij}}{\tau_{ij}^{pl}} f(\sigma_{eff}, \sigma_c) + \sum_{kl} \mathcal{G}_{ij,kl}(\mathbf{q}) \dot{\gamma}^{pl}_{kl}`

- **Test modes:** flow_curve, startup, relaxation, creep, oscillation

- **Material examples:** Rod climbing polymer melts, fiber suspensions, anisotropic gels, flow-induced microstructure

Overview
--------

The Tensorial EPM extends the scalar :doc:`lattice_epm` to track the **full stress tensor**, enabling predictions of **normal stress differences** (:math:`N_1, N_2`), anisotropic yielding, and flow-induced microstructure. This is critical for capturing non-Newtonian behaviors like rod climbing (Weissenberg effect), die swell, and flow instabilities.

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - :math:`\sigma_{ij}`
     - Pa
     - Stress tensor components [:math:`\sigma_{xx}`, :math:`\sigma_{yy}`, :math:`\sigma_{xy}`]
   * - :math:`\sigma_{zz}`
     - Pa
     - Out-of-plane stress (from plane strain constraint)
   * - :math:`N_1`
     - Pa
     - First normal stress difference (:math:`\sigma_{xx} - \sigma_{yy}`)
   * - :math:`N_2`
     - Pa
     - Second normal stress difference (:math:`\sigma_{yy} - \sigma_{zz}`)
   * - :math:`\sigma_{eff}`
     - Pa
     - Effective stress (von Mises or Hill criterion)
   * - :math:`\dot{\gamma}^p_{ij}`
     - 1/s
     - Plastic strain rate tensor (deviatoric)
   * - :math:`\mathcal{G}_{ij,kl}`
     - —
     - Tensorial Eshelby propagator (4th-order)
   * - :math:`\nu`
     - —
     - Poisson's ratio (plane strain constraint)
   * - :math:`H, N`
     - —
     - Hill anisotropy parameters

Physical Interpretation
-----------------------

Tensorial vs Scalar EPM
~~~~~~~~~~~~~~~~~~~~~~~~

The scalar EPM models only the shear component :math:`\sigma_xy`. While this captures flow curves and avalanches, it misses:

- **Normal stress differences**: :math:`N_1 = \sigma_xx - \sigma_yy` (rod climbing) and :math:`N_2 = \sigma_yy - \sigma_zz` (secondary flows)
- **Anisotropic yielding**: Materials with directional microstructure (fiber suspensions, liquid crystals)
- **Flow instabilities**: Edge fracture, shear banding driven by normal stress gradients

The Tensorial EPM tracks [:math:`\sigma_xx, \sigma_yy, \sigma_xy`] at each lattice site, with :math:`\sigma_zz = \nu(\sigma_xx + \sigma_yy)` from the plane strain constraint.

Tensorial Eshelby Propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a site yields plastically, the stress redistribution couples all components:

.. math::

    \boxed{
    \frac{\partial \sigma_{ij}}{\partial t} = \mu \dot{\gamma} \delta_{ij}
    - \frac{\sigma_{ij}}{\tau_{ij}^{pl}} f(\sigma_{eff}, \sigma_c)
    + \sum_{kl} \mathcal{G}_{ij,kl}(\mathbf{q}) \dot{\gamma}^{pl}_{kl}
    }

This is the tensorial extension of the scalar EPM evolution equation (see
:ref:`Discrete State Variables <epm_model>` in :doc:`lattice_epm`).

The tensorial propagator :math:`\mathcal{G}_{ij,kl}` derives from the Eshelby inclusion solution in plane strain:

.. math::

    \mathcal{G}_{ij,kl}(\mathbf{q}) = C_{ijmn} \frac{q_m q_n}{|\mathbf{q}|^2}

where C is the elastic stiffness tensor with shear modulus :math:`\mu` and Poisson ratio :math:`\nu`.

Yield Criteria
~~~~~~~~~~~~~~

**Von Mises (Isotropic)**:

.. math::

    \boxed{
    \sigma_{eff} = \frac{1}{\sqrt{2}} \sqrt{(\sigma_{xx} - \sigma_{yy})^2 + (\sigma_{yy} - \sigma_{zz})^2 + (\sigma_{zz} - \sigma_{xx})^2 + 6\sigma_{xy}^2}
    }

**Hill (Anisotropic)**:

.. math::

    \boxed{
    \sigma_{eff} = \sqrt{H (\sigma_{xx} - \sigma_{yy})^2 + 2N \sigma_{xy}^2}
    }

where H and N are anisotropy parameters (H=1, N=3 recovers von Mises in plane stress).

.. note::

    For protocol-specific governing equations (flow curve, startup, relaxation, creep,
    oscillation), see the :ref:`epm-flow-curve` section in :doc:`lattice_epm`.
    The tensorial extension uses the same protocol structure with the full stress tensor.

Prandtl-Reuss Flow Rule
~~~~~~~~~~~~~~~~~~~~~~~~

Plastic flow is component-wise with independent timescales:

.. math::

    \dot{\gamma}^{pl}_{ij} = \frac{\sigma'_{ij}}{\tau^{pl}_{ij}} \Theta(\sigma_{eff} - \sigma_c)

where :math:`\sigma'_{ij}` is the deviatoric stress. Separate :math:`\tau_{\text{pl,shear}}` and :math:`\tau_{\text{pl,normal}}` allow modeling materials with different relaxation times for shear and dilation.

Physical Foundations
--------------------

Why Track the Full Stress Tensor?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Many complex fluids exhibit **normal stress differences** during shear flow:

- **Polymer melts**: Rod climbing (Weissenberg effect), die swell, extrudate swell
- **Fiber suspensions**: Normal stresses from fiber orientation and rotation
- **Anisotropic gels**: Directional microstructure leads to non-isotropic yielding

The scalar EPM (:math:`\sigma_xy` only) misses these phenomena because:

1. Normal components :math:`\sigma_xx, \sigma_yy` evolve independently under shear
2. Yielding in one direction affects stress redistribution in all directions
3. Anisotropic yield criteria (Hill) require full tensor

Tensorial Eshelby Solution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a site yields plastically with strain rate :math:`\dot{\gamma}^{pl}_{kl}`, the stress redistribution to site :math:`\mathbf{r}` is given by the 4th-order Eshelby tensor:

.. math::

    \Delta \sigma_{ij}(\mathbf{r}) = \mathcal{G}_{ij,kl}(\mathbf{r}) \dot{\gamma}^{pl}_{kl}

In Fourier space (plane strain, 2D):

.. math::

    \tilde{\mathcal{G}}_{ij,kl}(\mathbf{q}) = C_{ijmn} \frac{q_m q_n}{|\mathbf{q}|^2}

where C is the elastic stiffness tensor. For isotropic elasticity with shear modulus :math:`\mu` and Poisson ratio :math:`\nu`:

.. math::

    C_{ijkl} = \mu \left[ \delta_{ik}\delta_{jl} + \delta_{il}\delta_{jk} + \frac{2\nu}{1-2\nu} \delta_{ij}\delta_{kl} \right]

**Key property**: The propagator couples all stress components, so a plastic event in :math:`\sigma_xy affects \sigma_xx and \sigma_yy`, and vice versa.

Plane Strain and Normal Stress Generation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In 2D flow with out-of-plane confinement (e.g., narrow gap rheometry), the strain :math:`\varepsilon_zz` = 0 but stress :math:`\sigma_zz` ≠ 0. The constraint is:

.. math::

    \sigma_{zz} = \nu (\sigma_{xx} + \sigma_{yy})

This coupling generates **non-zero** :math:`N_2`:

.. math::

    N_2 = \sigma_{yy} - \sigma_{zz} = (1 - \nu) \sigma_{yy} - \nu \sigma_{xx}

even when :math:`N_1 = \sigma_xx - \sigma_yy` might be small. This is a purely geometric effect of confinement.

Governing Equations
------------------------

Plane Strain Constraint
~~~~~~~~~~~~~~~~~~~~~~~

For 2D flow with out-of-plane confinement:

.. math::

    \boxed{
    \sigma_{zz} = \nu (\sigma_{xx} + \sigma_{yy})
    }

This couples the in-plane components and leads to non-zero :math:`N_2`.

Normal Stress Differences
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. math::

    \boxed{
    N_1 = \sigma_{xx} - \sigma_{yy}
    }

.. math::

    \boxed{
    N_2 = \sigma_{yy} - \sigma_{zz} = (1 - \nu) \sigma_{yy} - \nu \sigma_{xx}
    }

Typical experimental observations:
- Polymer melts: :math:`N_1` > 0 (rod climbing), :math:`|N_2|` ≪ :math:`N_1`
- Shear banding: Large gradients in :math:`N_1` correlate with band boundaries

Fitting to Normal Stress Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The loss function for combined fitting is:

.. math::

    \mathcal{L} = \sum_i (\sigma_{xy,i}^{pred} - \sigma_{xy,i}^{data})^2 + w_{N1} \sum_i (N_{1,i}^{pred} - N_{1,i}^{data})^2

Set ``w_N1 > 1`` to prioritize normal stress accuracy.

Validity and Assumptions
------------------------

**Valid for:**

- Materials where **normal stresses are measurable** and significant (:math:`N_1/\sigma_xy` > 0.1)
- **Anisotropic materials** with directional microstructure (fibers, liquid crystals)
- **Flow instabilities** driven by normal stress gradients (shear banding, edge fracture)
- **Confined geometries** where plane strain applies (narrow gap, slit flow)

**Assumptions:**

- **Plane strain constraint**: :math:`\varepsilon_zz` = 0 (appropriate for 2D confined flow)
- **Isotropic elasticity** (unless Hill criterion used for anisotropy)
- **Quenched disorder** in yield thresholds (same as scalar EPM)
- **No inertia** (overdamped dynamics)

**Not appropriate for:**

- Pure shear measurements where :math:`N_1` is not measured (use LatticeEPM instead)
- 3D bulk flows without confinement (requires full 3D tensor implementation)
- Very compressible materials (model assumes :math:`\nu` ≈ 0.4-0.5)

What You Can Learn
------------------

From fitting TensorialEPM to experimental data, you can extract insights about normal stress generation, anisotropic yielding, and flow instabilities in soft matter.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

:math:`\sigma_c` **(Yield Stress Threshold)**:
   Local critical stress for plastic yielding in von Mises or Hill criterion.
   *For graduate students*: For von Mises, :math:`\sigma_{\text{eff}} = \sqrt{\frac{1}{2}\boldsymbol{\tau}:\boldsymbol{\tau}}` must exceed :math:`\sigma_c` for plastic flow. Connects to microscopic cage breaking or bond rupture energetics.
   *For practitioners*: Design parameter for processing stresses. :math:`\sigma_c` sets minimum stress for continuous flow.

:math:`N_1, N_2` **(Normal Stress Differences)**:
   First (:math:`N_1 = \sigma_xx - \sigma_yy`) and second (:math:`N_2 = \sigma_yy - \sigma_zz`) normal stress differences from tensorial stress state.
   *For graduate students*: :math:`N_1` arises from upper-convected Maxwell backbone (chain stretching, particle alignment). :math:`N_2` from plane strain constraint: :math:`N_2 = (1-\nu)\sigma_yy - \nu\sigma_xx. Ratio N_1/\sigma_xy` ~ Wi (Weissenberg number) quantifies elasticity.
   *For practitioners*: Measure :math:`N_1` to predict rod climbing (Weissenberg effect), die swell, and edge fracture. :math:`N_1/\sigma_xy` > 0.5 indicates strong elastic effects.

**Hill H, N (Anisotropy Parameters)**:
   Hill criterion parameters quantifying directional yield resistance (H for normal, N for shear).
   *For graduate students*: Effective stress: :math:`\sigma_eff,Hill =` √[H(:math:`\sigma_xx-\sigma_yy`)\ :math:`^2 + 2N\sigma_xy^2`]. H=1, N=3 recovers von Mises. Microstructurally, H and N relate to fiber orientation tensor or crystallographic texture.
   *For practitioners*: Fit H and N from biaxial or combined loading tests. Use to predict forming limits and failure modes in anisotropic materials.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from TensorialEPM Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - :math:`N_1/\sigma_xy` < 0.1
     - Weakly elastic
     - Pastes, concentrated suspensions
     - Minimal normal stress effects
   * - :math:`N_1/\sigma_xy` = 0.1-1
     - Moderate elasticity
     - Emulsions, soft colloids
     - Rod climbing, moderate die swell
   * - :math:`N_1/\sigma_xy` > 1
     - Strongly elastic
     - Polymer melts, fiber suspensions
     - Strong Weissenberg effect, edge fracture
   * - H=1, N=3 (isotropic)
     - von Mises yielding
     - Isotropic gels, foams
     - Symmetric flow patterns
   * - H≠1 or N≠3 (anisotropic)
     - Directional yielding
     - Fiber composites, liquid crystals
     - Asymmetric instabilities, orientation-dependent strength

Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Fit shear stress only (w_N1 = 0)**

Start with shear stress data to get baseline parameters:

- ``mu``, ``sigma_c_mean``, ``tau_pl_shear``

This is equivalent to scalar EPM fitting.

**Step 2: Add normal stress constraint (w_N1 = 1)**

Refine parameters to match both :math:`\sigma_xy and N_1`:

- Adjust ``nu`` (Poisson ratio) to control :math:`N_1` magnitude
- Adjust ``tau_pl_normal`` if :math:`N_1` relaxation differs from shear

**Step 3: Test anisotropy (Hill criterion)**

If isotropic fit fails (:math:`R^2` < 0.9 for :math:`N_1`):

- Switch to ``yield_criterion='hill'``
- Fit ``hill_H`` and ``hill_N`` while holding other parameters fixed

Parameter Bounds and Physical Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Physical Constraint
   * - ``nu``
     - 0.40-0.49
     - Avoid 0.5 (incompressible singularity); :math:`N_1` sensitive to :math:`\nu`
   * - ``tau_pl_shear``
     - 0.01-10 s
     - Match shear stress relaxation timescale
   * - ``tau_pl_normal``
     - 0.1-10× tau_pl_shear
     - Often similar, but can differ for anisotropic materials
   * - ``w_N1``
     - 0.1-10
     - Higher weight = prioritize :math:`N_1 fit over \sigma_xy`
   * - ``hill_H``
     - 0.5-2.0
     - H = 1, N = 3 recovers von Mises (isotropic)
   * - ``hill_N``
     - 1.5-5.0
     - N controls shear-normal coupling

Common Fitting Issues
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Issue
     - Solution
   * - :math:`N_1` predictions too small
     - Increase ``sigma_c_std`` (disorder) or reduce ``nu`` to 0.42-0.45
   * - :math:`N_1` predictions too large
     - Increase ``nu`` toward 0.48 or reduce disorder
   * - Shear fit good, :math:`N_1` fit poor
     - Increase ``w_N1`` to 2-5; consider anisotropy (Hill)
   * - Convergence fails with w_N1 > 0
     - Fit shear first (w_N1=0), then refine with w_N1=1
   * - GPU memory overflow
     - Reduce ``L`` to 32 or 48; batch process long time series

API Reference
-------------

.. autoclass:: rheojax.models.epm.tensor.TensorialEPM
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:

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
     - Plastic relaxation time for shear components (:math:`\sigma_xy`)
   * - ``tau_pl_normal``
     - s
     - [0.01, 100.0]
     - Plastic relaxation time for normal stresses (:math:`\sigma_xx, \sigma_yy`)
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
     - Weight for :math:`N_1` in combined fitting loss
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

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from rheojax.models.epm import TensorialEPM
    import numpy as np

    # Create model instance
    model = TensorialEPM(L=32, dt=0.01)

    # Fit to flow curve data
    gamma_dot = np.logspace(-2, 1, 10)
    stress_exp = np.array([0.5, 1.2, 2.8, 5.1, 8.7, 13.5, 19.8, 27.3, 36.2, 46.5])

    model.fit(gamma_dot, stress_exp, test_mode='flow_curve')

    # Predict stress (including normal stress differences)
    gamma_dot_new = np.logspace(-2, 1, 30)
    sigma_pred = model.predict(gamma_dot_new, test_mode='flow_curve')

Advanced Usage Examples
------------------------

Basic Flow Curve with :math:`N_1` Prediction
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

Fitting to Combined [:math:`\sigma_xy, N_1`] Data
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
   * - Flow curves (:math:`\sigma vs \dot{\gamma}`)
     - ✓ Faster (3-5x)
     - ✓ More accurate if :math:`N_1` ≠ 0
   * - Yield stress determination
     - ✓ Sufficient
     - ✓ Accounts for anisotropy
   * - Normal stress differences
     - ✗
     - ✓ Required
   * - Shear banding analysis
     - ~ Qualitative
     - ✓ Quantitative (:math:`N_1` gradients)
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

:math:`N_1` Predictions Are Too Small
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cause**: Insufficient disorder (:math:`\sigma_{c,\text{std}}` too low) or :math:`\nu` too high (approaching incompressible limit)
- **Fix**: Increase ``sigma_c_std`` to 0.2-0.5 or reduce ``nu`` to 0.40-0.45

Fitting Fails to Converge
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Cause**: Combined [:math:`\sigma_xy, N_1`] loss has competing gradients
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

.. [1] Hébraud, P. and Lequeux, F. "Mode-coupling theory for the pasty rheology of soft
   glassy materials." *Physical Review Letters*, 81, 2934 (1998).
   https://doi.org/10.1103/PhysRevLett.81.2934

.. [2] Bocquet, L., Colin, A., and Ajdari, A. "Kinetic theory of plastic flow in soft
   glassy materials." *Physical Review Letters*, 103, 036001 (2009).
   https://doi.org/10.1103/PhysRevLett.103.036001

.. [3] Eshelby, J. D. "The determination of the elastic field of an ellipsoidal inclusion,
   and related problems." *Proceedings of the Royal Society A*, 241, 376-396 (1957).
   https://doi.org/10.1098/rspa.1957.0133

.. [4] Hill, R. "A theory of the yielding and plastic flow of anisotropic metals."
   *Proceedings of the Royal Society A*, 193, 281-297 (1948).
   https://doi.org/10.1098/rspa.1948.0045

.. [5] Bird, R. B., Armstrong, R. C., and Hassager, O. *Dynamics of Polymeric Liquids*,
   Vol. 1: Fluid Mechanics, 2nd Edition. Wiley (1987). ISBN: 978-0471802457

.. [6] Picard, G., Ajdari, A., Lequeux, F., and Bocquet, L. "Elastic consequences of a
   single plastic event: A step towards the microscopic modeling of the flow of yield
   stress fluids." *European Physical Journal E*, 15, 371-381 (2004).
   https://doi.org/10.1140/epje/i2004-10054-8

.. [7] Nicolas, A., Ferrero, E. E., Martens, K., and Barrat, J.-L. "Deformation and flow
   of amorphous solids: Insights from elastoplastic models." *Reviews of Modern Physics*,
   90, 045006 (2018). https://doi.org/10.1103/RevModPhys.90.045006

.. [8] Larson, R. G. "Constitutive equations for polymer melts and solutions."
   *Butterworths Series in Chemical Engineering*, Boston (1988). ISBN: 978-0409901191

.. [9] Coussot, P. "Yield stress fluid flows: A review of experimental data."
   *Journal of Non-Newtonian Fluid Mechanics*, 211, 31-49 (2014).
   https://doi.org/10.1016/j.jnnfm.2014.05.006

.. [10] Saramito, P. "A new elastoviscoplastic model based on the Herschel-Bulkley
    viscoplastic model." *Journal of Non-Newtonian Fluid Mechanics*, 158, 154-161 (2009).
    https://doi.org/10.1016/j.jnnfm.2008.12.001

See Also
--------

- :doc:`index` — EPM family overview and comparison
- :doc:`lattice_epm` — Scalar EPM for faster fitting when :math:`N_1` data unavailable
- :doc:`/models/sgr/sgr_conventional` — Mean-field SGR (complementary theory)
- :doc:`/models/hl/hebraud_lequeux` — Mean-field HL (EPM limiting case)
- :py:func:`rheojax.visualization.epm_plots.plot_tensorial_fields` — Visualization functions for tensor fields


Bayesian Inference for Tensorial EPM
====================================

Tensorial EPM supports the full NLSQ → NUTS Bayesian pipeline for uncertainty
quantification of the extended parameter set.

NLSQ → NUTS Workflow
--------------------

.. code-block:: python

   from rheojax.models import TensorialEPM
   from rheojax.core.data import RheoData
   import numpy as np

   # Load experimental data with normal stresses
   gamma_dot = np.logspace(-2, 1, 20)
   sigma_xy_exp = np.array([...])  # Shear stress
   N1_exp = np.array([...])        # First normal stress difference

   # Create model with tensor capability
   model = TensorialEPM(
       L=32,           # Moderate lattice for fitting
       dt=0.01,
       yield_criterion="von_mises",
       w_N1=2.0        # Weight for N₁ in combined loss
   )

   # Step 1: NLSQ fitting for point estimates
   rheo_data = RheoData(x=gamma_dot, y=sigma_xy_exp, initial_test_mode="flow_curve")
   model.fit(rheo_data, max_iter=300)

   # Step 2: Bayesian inference with 4 chains
   result = model.fit_bayesian(
       rheo_data,
       test_mode="flow_curve",
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Step 3: Analyze posteriors and diagnostics
   print(result.summary)
   print(f"R-hat (max): {max(result.diagnostics['r_hat'].values()):.4f}")
   print(f"ESS (min): {min(result.diagnostics['ess'].values()):.0f}")
   print(f"Divergences: {result.diagnostics['divergences']:.1%}")

Prior Specification for Tensor Parameters
-----------------------------------------

TensorialEPM uses weakly informative priors tailored to the tensor structure:

.. list-table:: Default Priors for TensorialEPM Parameters
   :widths: 20 35 45
   :header-rows: 1

   * - Parameter
     - Prior Distribution
     - Rationale
   * - :math:`\mu` (shear modulus)
     - LogNormal(log(10), 1.5)
     - Positive, spans 0.1-1000 Pa
   * - :math:`\nu` (Poisson ratio)
     - Beta(8, 2) scaled to [0.3, 0.5]
     - Constrained near incompressible
   * - :math:`\sigma_{c,\text{mean}}`
     - HalfNormal(10)
     - Positive, order of yield stress
   * - :math:`\sigma_{c,\text{std}}`
     - HalfNormal(2)
     - Positive, smaller than mean
   * - :math:`\tau_{\text{pl,shear}}`
     - LogNormal(log(1), 1)
     - Positive, log-uniform over decades
   * - :math:`\tau_{\text{pl,normal}}`
     - LogNormal(log(1), 1)
     - Often similar to shear, but free
   * - w_N1
     - Fixed (not sampled)
     - User-specified weight
   * - hill_H
     - HalfNormal(1)
     - Centered on isotropic (H=1)
   * - hill_N
     - HalfNormal(3)
     - Centered on von Mises (N=3)

**Customizing priors:**

.. code-block:: python

   from numpyro import distributions as dist

   custom_priors = {
       "nu": dist.Uniform(0.40, 0.48),  # Tighter constraint on ν
       "tau_pl_normal": dist.LogNormal(
           loc=jnp.log(model.params.get_value("tau_pl_shear")),
           scale=0.5  # Prior centered on shear value
       ),
   }

   result = model.fit_bayesian(
       rheo_data,
       priors=custom_priors,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

Multi-Chain Diagnostics for Tensor Models
-----------------------------------------

Tensorial EPM posteriors can exhibit complex geometry due to parameter correlations.
Use 4 chains (default) and check diagnostics carefully:

.. list-table:: Diagnostic Guidelines for TensorialEPM
   :widths: 25 25 50
   :header-rows: 1

   * - Diagnostic
     - Target
     - Action if Failed
   * - R-hat (all params)
     - < 1.05
     - Increase num_samples, check for multimodality
   * - ESS (all params)
     - > 400
     - Increase num_samples or reduce correlation
   * - ESS bulk/tail ratio
     - > 0.5
     - Indicates chain mixing issues
   * - Divergences
     - < 1%
     - Reduce step size, reparameterize
   * - BFMI (energy)
     - > 0.3
     - Indicates sampling difficulties

**Common parameter correlations:**

- :math:`\nu` **↔** :math:`\sigma_{c,\text{std}}`: Higher Poisson ratio compensates for lower disorder
- :math:`\tau_{\text{pl,shear}}` **↔** :math:`\tau_{\text{pl,normal}}`: Often correlated; consider fixing ratio
- **hill_H ↔ hill_N**: Strong correlation in anisotropic fits

.. code-block:: python

   import arviz as az

   # Convert to ArviZ format
   idata = result.to_arviz()

   # Pair plot to visualize correlations
   az.plot_pair(
       idata,
       var_names=["nu", "sigma_c_std", "tau_pl_shear", "tau_pl_normal"],
       divergences=True,
       figsize=(10, 10)
   )

   # Energy plot (BFMI diagnostic)
   az.plot_energy(idata)

   # Rank plot (chain mixing)
   az.plot_rank(idata, var_names=["nu", "sigma_c_mean"])

Combined [:math:`\sigma_xy, N_1`] Fitting Strategy
------------------------------------

Fitting to both shear and normal stress data requires careful balancing:

**Strategy 1: Sequential fitting**

1. Fit :math:`\sigma_{xy}` only (w_N1 = 0) to get :math:`\mu, \sigma_c, \tau_{\text{pl}}`
2. Fix shear parameters, fit :math:`\nu, \tau_{\text{pl,normal}}` from :math:`N_1`
3. Joint refinement with w_N1 = 1

.. code-block:: python

   # Step 1: Shear-only fit
   model_shear = TensorialEPM(L=32, w_N1=0)
   model_shear.fit(rheo_data, max_iter=200)

   # Step 2: Fix shear params, fit normal
   model_normal = TensorialEPM(L=32, w_N1=10)
   model_normal.params = model_shear.params.copy()
   model_normal.params.fix("mu")
   model_normal.params.fix("sigma_c_mean")
   model_normal.fit(rheo_data_with_N1, max_iter=100)

   # Step 3: Joint Bayesian
   model_joint = TensorialEPM(L=32, w_N1=2)
   model_joint.params = model_normal.params.copy()
   model_joint.params.unfix_all()

   result = model_joint.fit_bayesian(
       rheo_data_with_N1,
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )

**Strategy 2: Weighted joint fitting**

.. code-block:: python

   # Adjust w_N1 based on data quality
   # Higher weight if N₁ data is high-quality
   # Lower weight if N₁ is noisy

   model = TensorialEPM(L=32, w_N1=2.0)

   # Bayesian with both observables
   result = model.fit_bayesian_multi_observable(
       gamma_dot=gamma_dot,
       sigma_xy=sigma_xy_exp,
       N1=N1_exp,
       weights={"sigma_xy": 1.0, "N1": 2.0},
       num_warmup=1000,
       num_samples=2000,
       num_chains=4
   )


Advanced Normal Stress Physics
==============================

This section provides deeper insight into the physical origins and interpretation
of normal stress differences in TensorialEPM.

Microscopic Origin of Normal Stresses
-------------------------------------

Normal stress differences arise from **microstructural anisotropy** induced by shear:

**In colloidal systems:**

- Particles align along the compression axis (45° to flow)
- Cage distortion creates asymmetric stress distribution
- :math:`N_1` > 0: Extension along flow direction dominates

**In polymer melts:**

- Chain stretching and orientation along flow
- Entropic elasticity: stretched chains exert restoring force
- Classic result: :math:`N_1 \approx 2G`':math:`\gamma` for linear regime

**In fiber suspensions:**

- Fiber orientation tensor evolves under shear
- Jeffery orbits create periodic normal stress oscillations
- Steady-state :math:`N_1` depends on aspect ratio and concentration

Rate-Dependence of Normal Stresses
----------------------------------

The ratio :math:`N_1/\sigma_xy` typically shows characteristic rate-dependence:

**Low shear rates (linear regime):**

.. math::

   N_1 \sim \gamma \cdot \sigma_{xy} \sim \dot{\gamma}^2

In this regime, :math:`N_1/\sigma_xy` ~ Wi (Weissenberg number).

**Intermediate rates (non-linear):**

.. math::

   N_1 \sim \dot{\gamma}^\alpha \quad \text{with } 0.5 < \alpha < 2

Power-law scaling with exponent depending on microstructure.

**High shear rates (shear thinning):**

.. math::

   N_1 / \sigma_{xy} \rightarrow \text{const}

Plateau ratio indicates fully aligned microstructure.

**Diagnostic:** :math:`N_1/\sigma_xy` **ratio**

.. code-block:: python

   def diagnose_normal_stress_ratio(gamma_dot, sigma_xy, N1):
       """Diagnose material behavior from N₁/σ_xy ratio."""
       ratio = N1 / sigma_xy

       # Weissenberg number interpretation
       Wi = ratio  # In linear regime

       # Regime detection
       if np.mean(ratio) < 0.1:
           regime = "weakly elastic (colloidal suspension)"
       elif np.mean(ratio) < 1.0:
           regime = "moderate elasticity (emulsion, soft colloid)"
       else:
           regime = "strongly elastic (polymer melt, fiber suspension)"

       # Rate dependence exponent
       from scipy import stats
       log_gd = np.log10(gamma_dot)
       log_ratio = np.log10(ratio)
       slope, _, r_value, _, _ = stats.linregress(log_gd, log_ratio)

       return {
           "mean_ratio": np.mean(ratio),
           "regime": regime,
           "rate_exponent": slope,
           "linearity_r2": r_value**2
       }


Anisotropic Yielding: Hill Criterion Details
=============================================

The Hill criterion extends von Mises to **anisotropic** yield behavior, essential
for materials with directional microstructure.

Hill Criterion Derivation
-------------------------

For materials with principal axes of anisotropy, Hill (1948) proposed:

.. math::

   \sigma_{eff}^2 = F(\sigma_{yy} - \sigma_{zz})^2 + G(\sigma_{zz} - \sigma_{xx})^2
                   + H(\sigma_{xx} - \sigma_{yy})^2 + 2L\sigma_{yz}^2 + 2M\sigma_{zx}^2 + 2N\sigma_{xy}^2

In plane stress (:math:`\sigma_zz = \sigma_xz = \sigma_yz` = 0):

.. math::

   \sigma_{eff} = \sqrt{H (\sigma_{xx} - \sigma_{yy})^2 + 2N \sigma_{xy}^2}

**Parameter interpretation:**

- **H**: Resistance to normal stress difference (:math:`\sigma_xx - \sigma_yy`)
- **N**: Resistance to shear stress :math:`\sigma_xy`
- **H = 1, N = 3**: Recovers von Mises (isotropic)
- **H > 1**: Stronger in normal direction (resists :math:`N_1` generation)
- **N > 3**: Stronger in shear (higher yield stress for same :math:`\sigma_xy`)

Connection to Fiber Orientation Tensor
--------------------------------------

For fiber suspensions, Hill parameters relate to the orientation tensor **a**:

.. math::

   H \sim 1 + c_H \langle p_x^2 - p_y^2 \rangle

   N \sim 3 + c_N \langle p_x^2 p_y^2 \rangle

where **p** is the fiber orientation unit vector and c_H, c_N are material constants.

**Fully aligned fibers (p_x = 1):**
H > 1 (strong normal resistance), yield is dominated by shear

**Random orientation:**
H ≈ 1, N ≈ 3 (recovers von Mises)

Visualization: Yield Surfaces
-----------------------------

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt

   def plot_yield_surface(H=1.0, N=3.0, sigma_c=1.0):
       """Plot Hill yield surface in (σ_xx - σ_yy, σ_xy) space."""
       theta = np.linspace(0, 2*np.pi, 100)

       # Parametric form of yield surface
       # H(σ_xx - σ_yy)² + 2N σ_xy² = σ_c²
       sigma_normal = sigma_c * np.cos(theta) / np.sqrt(H)
       sigma_shear = sigma_c * np.sin(theta) / np.sqrt(2*N)

       plt.figure(figsize=(6, 6))
       plt.plot(sigma_normal, sigma_shear, 'b-', linewidth=2,
                label=f'Hill (H={H}, N={N})')

       # Reference: von Mises (H=1, N=3)
       sigma_normal_vm = sigma_c * np.cos(theta)
       sigma_shear_vm = sigma_c * np.sin(theta) / np.sqrt(6)
       plt.plot(sigma_normal_vm, sigma_shear_vm, 'k--', alpha=0.5,
                label='von Mises')

       plt.xlabel(r'$\sigma_{xx} - \sigma_{yy}$ (Pa)')
       plt.ylabel(r'$\sigma_{xy}$ (Pa)')
       plt.axis('equal')
       plt.legend()
       plt.title('Hill Yield Surface')
       plt.grid(True, alpha=0.3)
       return plt.gcf()


Enhanced Experimental Protocols
===============================

This section details experimental considerations for obtaining high-quality data
suitable for TensorialEPM fitting.

Normal Stress Measurement Techniques
------------------------------------

**Cone-plate geometry:**

- Preferred for :math:`N_1` measurement (uniform shear rate)
- Edge fracture limits high rates (:math:`N_1` ~ :math:`\sigma_xy` at critical Wi)
- Requires thrust bearing calibration

**Parallel plates:**

- :math:`N_1` requires correction: :math:`N_1 = 2F_N/(\piR^2)` (1 + d ln F_N / d ln :math:`\dot{\gamma}`)
- Gap variation affects accuracy
- Useful for temperature sweeps

**Capillary rheometry:**

- :math:`N_1` from exit pressure (Bagley plot)
- High shear rates accessible
- End corrections essential

.. list-table:: Geometry Comparison for Normal Stress
   :widths: 25 25 25 25
   :header-rows: 1

   * - Geometry
     - :math:`N_1` Accuracy
     - Rate Range
     - Limitations
   * - Cone-plate
     - Excellent
     - 10\ :math:`^{-3 - 10^2}` 1/s
     - Edge fracture at high Wi
   * - Parallel plate
     - Good (with correction)
     - 10\ :math:`^{-2 - 10^3}` 1/s
     - Gap-dependent correction
   * - Capillary
     - Moderate
     - :math:`10^1 - 10^5` 1/s
     - End effects, indirect

Shear Banding Detection
-----------------------

TensorialEPM can capture shear banding driven by normal stress gradients. Detection methods:

**Rheo-PIV (Particle Image Velocimetry):**

- Direct velocity profile measurement
- Banding: linear profile with sharp gradient
- Resolution: ~10 :math:`\mum` typical

**Rheo-SANS (Small-Angle Neutron Scattering):**

- Microstructure orientation during flow
- Bands have different alignment signatures
- Requires contrast-matched samples

**Stress fluctuations:**

- Banding creates stress fluctuations at band boundaries
- Monitor :math:`N_1` noise: high variance indicates instability

Data Quality Checklist
----------------------

Before fitting TensorialEPM, verify data quality:

.. list-table:: Data Quality Checklist
   :widths: 35 65
   :header-rows: 1

   * - Check
     - Requirement
   * - Steady state reached
     - Wait 5-10× :math:`\tau_relax` before recording
   * - :math:`N_1` noise level
     - < 10% of signal at each rate
   * - Edge fracture
     - Check visual/torque for high-rate anomalies
   * - Inertia correction
     - Apply for high rates if Re > 0.1
   * - Gap uniformity
     - Verify < 1% variation for parallel plates
   * - Temperature stability
     - ±0.1°C during measurement
   * - Sample loading
     - Consistent protocol (rest time, preshear)


Full Bayesian Workflow with Tensor Diagnostics
----------------------------------------------

Complete workflow for publication-quality TensorialEPM analysis:

.. code-block:: python

   from rheojax.models import TensorialEPM
   from rheojax.pipeline.bayesian import BayesianPipeline
   import arviz as az
   import numpy as np

   # Load combined stress data
   data = RheoData.from_csv(
       "tensorial_data.csv",
       x_col="gamma_dot",
       y_col="sigma_xy",
       metadata_cols=["N1", "N2"]
   )

   # Initialize pipeline
   pipeline = BayesianPipeline()

   (pipeline
       .load(data)
       .model(TensorialEPM, L=32, dt=0.01, yield_criterion="von_mises", w_N1=2.0)
       .fit_nlsq(max_iter=300)
       .fit_bayesian(
           num_warmup=1000,
           num_samples=2000,
           num_chains=4,
           seed=42
       )
   )

   # Tensor-specific diagnostics
   idata = pipeline.result.to_arviz()

   # 1. Check ν posterior (should not pile against bounds)
   az.plot_posterior(idata, var_names=["nu"])
   assert idata.posterior["nu"].mean() < 0.495, "ν hitting bound"

   # 2. Compare τ_pl_shear vs τ_pl_normal
   ratio = (idata.posterior["tau_pl_normal"] /
            idata.posterior["tau_pl_shear"])
   print(f"τ_pl_normal/τ_pl_shear = {ratio.mean():.2f} "
         f"[{np.percentile(ratio, 2.5):.2f}, {np.percentile(ratio, 97.5):.2f}]")

   # 3. N₁ prediction uncertainty
   gamma_dot_pred = np.logspace(-2, 1, 50)
   N1_samples = pipeline.predict_posterior_samples(
       gamma_dot_pred,
       n_samples=100,
       observable="N1"
   )
   N1_mean = N1_samples.mean(axis=0)
   N1_hdi = az.hdi(N1_samples, hdi_prob=0.95)

   # 4. Save with full metadata
   pipeline.save_results(
       "tensorial_fit.hdf5",
       include_diagnostics=True,
       include_predictions=True
   )
