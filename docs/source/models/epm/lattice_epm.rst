.. _epm_model:

Elasto-Plastic Models (EPM)
===========================

Quick Reference
---------------

**Use when:** Spatially-resolved modeling of amorphous solids, plastic avalanches, shear banding

**Parameters:** 6 (μ, σ_c_mean, σ_c_std, τ_pl, L, dt)

**Key equation:** :math:`\partial_t \sigma_{ij} = \mu \dot{\gamma}(t) - \mu \dot{\gamma}^{pl}_{ij} + \sum_{kl} \mathcal{G}_{ij,kl} \dot{\gamma}^{pl}_{kl}`

**Test modes:** flow_curve, startup, relaxation, creep, oscillation

**Material examples:** Metallic glasses, colloidal gels, pastes, dense granular suspensions, foams

Overview
--------

The Elasto-Plastic Model (EPM) is a mesoscopic lattice-based framework for modeling the rheology of amorphous solids (glasses, gels, pastes, dense suspensions). Unlike mean-field models (like Hebraud-Lequeux or Soft Glassy Rheology), EPMs explicitly resolve **spatial heterogeneity**, **plastic avalanches**, and **non-local stress redistribution**.

This implementation leverages **JAX** for high-performance FFT-based computations on GPU/TPU.

Notation Guide
--------------

.. list-table::
   :widths: 15 15 70
   :header-rows: 1

   * - Symbol
     - Units
     - Description
   * - σ
     - Pa
     - Local shear stress at lattice site
   * - γ̇
     - 1/s
     - Macroscopic applied shear rate
   * - γ̇ᵖ
     - 1/s
     - Local plastic strain rate at yielded sites
   * - σ_c
     - Pa
     - Local yield stress threshold (varies spatially)
   * - μ
     - Pa
     - Shear modulus of elastic matrix
   * - τ_pl
     - s
     - Plastic relaxation time for yielded blocks
   * - :math:`\mathcal{G}(\mathbf{r})`
     - —
     - Eshelby propagator (stress redistribution kernel)
   * - L
     - —
     - Lattice size (L × L grid)

Physical Interpretation & Assumptions
-------------------------------------

The Mesoscopic View
~~~~~~~~~~~~~~~~~~~
We discretize the material into a lattice of mesoscopic blocks of size $\xi$ (the correlation length of plastic events).
Each block is coarse-grained enough to be treated as a continuum element with a local stress $\sigma_{ij}$ and strain $\gamma_{ij}$, but small enough that plastic yielding is a discrete, local event.

The Physics of Plasticity
~~~~~~~~~~~~~~~~~~~~~~~~~
The dynamics are governed by the interplay of three mechanisms:

1.  **Elastic Loading**: The entire lattice is driven by a macroscopic shear rate $\dot{\gamma}$. Each block accumulates elastic stress.
2.  **Local Yielding**: If the local stress $|\sigma_i|$ exceeds a local yield threshold $\sigma_{c,i}$, the block yields. This is a "plastic event" or "Shear Transformation Zone (STZ) flip".
3.  **Stress Redistribution**: A plastic event at site $j$ releases local stress but must satisfy force balance ($\nabla \cdot \sigma = 0$). This stress is redistributed to neighbors via the **Eshelby Propagator** $\mathcal{G}_{ij}$.

Assumptions
~~~~~~~~~~~
*   **Scalar Approximation**: We model only the shear component $\sigma_{xy}$.
*   **Athermal Limit**: Yielding is purely stress-driven (zero temperature), though "smooth" yielding can approximate thermal activation.
*   **Periodic Boundary Conditions**: The system is an infinite repeating lattice.
*   **Overdamped Dynamics**: Inertia is neglected.

Mathematical Formulation
------------------------

Evolution Equation
~~~~~~~~~~~~~~~~~~
The time evolution of the local stress $\sigma(\mathbf{r}, t)$ is given by:

.. math::

    \frac{\partial \sigma_{ij}}{\partial t} = \underbrace{\mu \dot{\gamma}(t)}_{\text{Elastic Loading}}
    - \underbrace{\mu \dot{\gamma}^{pl}_{ij}}_{\text{Plastic Relaxation}}
    + \underbrace{\sum_{kl} \mathcal{G}_{ij,kl} \dot{\gamma}^{pl}_{kl}}_{\text{Redistribution}}

where:
*   $\mu$ is the shear modulus.
*   $\dot{\gamma}(t)$ is the macroscopic applied shear rate.
*   $\dot{\gamma}^{pl}$ is the local plastic strain rate.
*   $\mathcal{G}$ is the elastic propagator.

Physical Foundations
--------------------

Mesoscopic Coarse-Graining
~~~~~~~~~~~~~~~~~~~~~~~~~~

The EPM operates at a length scale ξ (correlation length of plastic events, typically 10-100 particle diameters in colloidal systems). At this scale:

- The material is **homogeneous enough** for continuum elasticity to apply
- Plastic yielding is **localized** to discrete regions (blocks)
- Spatial **correlations** between yielding events become important

This mesoscopic view differs from:

- **Microscopic models** (molecular dynamics): Track individual particles
- **Macroscopic models** (continuum plasticity): Smear plasticity into a continuous field

Stress Redistribution via Eshelby Propagator
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When a block yields plastically, it releases local stress. However, **mechanical equilibrium** (∇·σ = 0) requires this stress to be redistributed to neighboring blocks. The Eshelby propagator describes this redistribution:

.. math::

    \mathcal{G}(\mathbf{r}) = \text{stress at } \mathbf{r} \text{ due to plastic event at origin}

In 2D Fourier space, the propagator has characteristic **quadrupolar symmetry** ("four-leaf clover"):

.. math::

    \tilde{\mathcal{G}}(\mathbf{q}) = -4 \mu \frac{q_x^2 q_y^2}{(q_x^2 + q_y^2)^2} \quad \text{for } \mathbf{q} \neq 0

**Key properties:**

- :math:`\tilde{\mathcal{G}}(0) = 0`: Plastic events conserve total stress (controlled by boundary loading)
- **Long-range coupling**: :math:`\mathcal{G}(\mathbf{r}) \sim 1/r^2` in real space (power-law decay)
- **Quadrupolar structure**: Stress redistribution has four lobes (compression/extension pattern)

This long-range interaction is what leads to **avalanche** dynamics: one yielding event can trigger neighbors to yield, creating cascades of plasticity.

Governing Equations
-------------------

Yield Criteria
~~~~~~~~~~~~~~
We implement two modes of yielding:

1.  **Hard Mode** (Simulation):

    .. math::
        \dot{\gamma}^{pl} = \frac{\sigma}{\tau_{pl}} \Theta(|\sigma| - \sigma_c)

    Standard threshold dynamics. Used for physical validation.

2.  **Smooth Mode** (Inference):

    .. math::
        \dot{\gamma}^{pl} = \frac{\sigma}{\tau_{pl}} \frac{1}{2} \left[ 1 + \tanh\left(\frac{|\sigma| - \sigma_c}{w}\right) \right]

    A differentiable approximation that allows gradients to backpropagate through the yield surface for NLSQ/HMC fitting.

Numerical Implementation
------------------------

Spectral Method (FFT)
~~~~~~~~~~~~~~~~~~~~~
Direct summation of the stress redistribution is $O(L^4)$ or $O(L^2)$ with a cutoff.
We use **Fast Fourier Transforms (FFT)** to perform the convolution in $O(L^2 \log L)$ time.

1.  Compute $\dot{\gamma}^{pl}(\mathbf{r})$.
2.  FFT to Fourier space: $\tilde{\dot{\gamma}}^{pl}(\mathbf{q})$.
3.  Multiply by propagator: $\tilde{\sigma}^{redist}(\mathbf{q}) = \tilde{\mathcal{G}}(\mathbf{q}) \tilde{\dot{\gamma}}^{pl}(\mathbf{q})$.
4.  Inverse FFT to real space.

This allows us to simulate large systems ($L=64, 128, 256$) efficiently on GPUs.

Time Integration
~~~~~~~~~~~~~~~~
We use a semi-implicit or explicit Euler scheme with a small time step $dt$.
The yield thresholds $\sigma_{c,i}$ are **renewed** (drawn from a Gaussian distribution) whenever a site yields, introducing the necessary quenched disorder that leads to avalanches.

Validity and Assumptions
------------------------

**Valid for:**

- **Athermal plasticity**: Yielding driven by stress, not thermal activation (T ≈ 0)
- **Overdamped dynamics**: Inertia negligible (quasi-static or low Stokes number)
- **2D simple shear**: Single shear component σ_xy (for scalar EPM)
- **Periodic systems**: Infinite lattice (no boundary effects)

**Assumptions:**

- **Quenched disorder**: Yield thresholds σ_c,i drawn from Gaussian, renewed upon yielding
- **Elastic homogeneity**: Uniform shear modulus μ throughout
- **Mean-field-like yield**: Local yield criterion (no cooperative yielding beyond Eshelby coupling)

**Not appropriate for:**

- Thermal systems where k_B T ~ barrier heights
- High-frequency dynamics (inertial effects)
- Systems where plasticity is diffusive rather than avalanche-like

What You Can Learn
------------------

From fitting EPM to experimental data, you can extract insights about mesoscopic plasticity, avalanche dynamics, and spatial heterogeneity in amorphous solids.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**σ_c (Yield Stress Threshold)**:
   The local stress at which mesoscopic blocks yield plastically. Typically lower than macroscopic yield stress due to spatial averaging.
   *For graduate students*: σ_c represents the energy barrier for local plastic rearrangements in the free energy landscape. For colloidal gels, σ_c ~ bond strength; for glasses, σ_c ~ activation barrier height.
   *For practitioners*: Use σ_c to predict onset of yielding in processing. Lower σ_c = easier to flow but potentially less stable structures.

**σ_c_std (Disorder Strength)**:
   Standard deviation of local yield thresholds across the material, quantifying microstructural heterogeneity.
   *For graduate students*: Disorder drives avalanche criticality. Larger σ_c_std → broader avalanche size distributions, power-law exponents closer to τ ≈ 2.0 (with disorder) vs. 1.5 (mean-field). Controls correlation length ξ_corr of yielding events.
   *For practitioners*: High disorder correlates with pronounced shear banding. Monitor σ_c_std/σ_c ratio to predict flow instabilities.

**α (Disorder Parameter)**:
   Related parameter quantifying yield threshold variability, α = σ_c_std/σ_c.
   *For graduate students*: Critical parameter in mean-field elastoplastic theory. α → 0 recovers deterministic plasticity; α >> 1 leads to extreme heterogeneity and arrested dynamics.
   *For practitioners*: α > 0.3 indicates strong spatial heterogeneity requiring spatially-resolved models.

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Material Classification from EPM Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Behavior
     - Typical Materials
     - Processing Implications
   * - σ_c = 10-100 Pa, α < 0.2
     - Homogeneous yielding
     - Monodisperse colloids, simple gels
     - Uniform flow, minimal banding
   * - σ_c = 50-500 Pa, α = 0.2-0.5
     - Moderate heterogeneity
     - Emulsions, pastes, polydisperse suspensions
     - Possible shear banding, flow instabilities
   * - σ_c = 100-1000 Pa, α > 0.5
     - Strong heterogeneity, avalanches
     - Metallic glasses, dense granular media
     - Shear localization, stick-slip
   * - τ_pl < 0.1 s
     - Fast plastic relaxation
     - Soft colloids, concentrated emulsions
     - Rapid stress relaxation, smooth flow
   * - τ_pl > 1 s
     - Slow plastic relaxation
     - Glassy polymers, hard colloids
     - Stress overshoots, memory effects

Experimental Protocol Integration
---------------------------------

The model supports standard rheological protocols via `_predict(test_mode=...)`.

Flow Curve
~~~~~~~~~~
*   **Protocol**: Constant $\dot{\gamma}$.
*   **Observable**: Steady-state stress $\Sigma_{ss} = \langle \sigma \rangle$.
*   **Prediction**: Herschel-Bulkley behavior $\Sigma = \Sigma_y + A \dot{\gamma}^n$.

Creep (Stress Control)
~~~~~~~~~~~~~~~~~~~~~~
*   **Protocol**: Constant Stress $\Sigma_{target}$.
*   **Implementation**: Since the EPM is strain-rate driven, we use an **Adaptive P-Controller** (PID loop) to adjust $\dot{\gamma}(t)$ dynamically:

    .. math::
        \dot{\gamma}_{t+1} = \dot{\gamma}_t + K_p (\Sigma_{target} - \langle \sigma \rangle_t)

*   **Observable**: Strain $\gamma(t)$ vs time.

Oscillation (SAOS/LAOS)
~~~~~~~~~~~~~~~~~~~~~~~
*   **Protocol**: $\dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)$.
*   **Observable**: Lissajous figures (Stress vs Strain).
*   **Analysis**: Can capture non-linear harmonic generation and yielding transitions within a cycle.

Bayesian Inference (NLSQ → NUTS)
--------------------------------

EPM models now support the full NLSQ → NUTS Bayesian inference pipeline, enabling:

*   **Point estimates** via GPU-accelerated NLSQ optimization
*   **Posterior distributions** via NumPyro's NUTS sampler
*   **Uncertainty quantification** with credible intervals
*   **Convergence diagnostics** (R-hat, ESS, divergences)

The key requirement is the ``model_function()`` method, which provides a differentiable
forward model for both NLSQ and NumPyro.

Smooth Yielding Approximation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Bayesian inference requires gradients through the yield surface. EPM uses a smooth
``tanh`` approximation (``smooth=True``) during fitting:

.. math::

    \dot{\gamma}^{pl} = \frac{\sigma}{\tau_{pl}} \frac{1}{2} \left[ 1 + \tanh\left(\frac{|\sigma| - \sigma_c}{w}\right) \right]

This enables backpropagation while closely approximating the hard threshold behavior.

Example: NLSQ → NUTS Pipeline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.models.epm import LatticeEPM
    import numpy as np

    # Create model and synthetic data
    model = LatticeEPM(L=32, dt=0.01)
    gamma_dot = np.logspace(-2, 1, 30)
    # Example experimental data
    stress = 10.0 * gamma_dot**0.5 + 5.0  # Herschel-Bulkley-like

    # Step 1: NLSQ fitting (fast point estimation)
    model.fit(gamma_dot, stress, test_mode="flow_curve", max_iter=500)

    # Step 2: Bayesian inference (warm-started from NLSQ)
    result = model.fit_bayesian(
        gamma_dot,
        stress,
        test_mode="flow_curve",
        num_warmup=500,
        num_samples=1000,
        num_chains=4,  # Multiple chains for R-hat diagnostics
        seed=42,
    )

    # Step 3: Analyze posteriors
    print(result.summary)  # Parameter means, std, credible intervals

    # Convergence diagnostics
    print(f"R-hat: {result.diagnostics['r_hat']}")
    print(f"ESS: {result.diagnostics['ess']}")

    # Credible intervals
    intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
    for name, (lower, upper) in intervals.items():
        print(f"{name}: [{lower:.3f}, {upper:.3f}]")

Parameters
----------

.. list-table::
   :header-rows: 1
   :widths: 15 10 15 60

   * - Parameter
     - Symbol
     - Units
     - Description
   * - ``mu``
     - :math:`\mu`
     - Pa
     - Shear modulus of the elastic matrix
   * - ``sigma_c_mean``
     - :math:`\bar{\sigma}_c`
     - Pa
     - Mean local yield stress threshold
   * - ``sigma_c_std``
     - :math:`\delta\sigma_c`
     - Pa
     - Standard deviation of local yield stress (disorder)
   * - ``tau_pl``
     - :math:`\tau_{pl}`
     - s
     - Plastic relaxation time for yielded blocks
   * - ``L``
     - :math:`L`
     - —
     - Lattice size (L × L grid)
   * - ``dt``
     - :math:`\Delta t`
     - s
     - Time step for numerical integration

Fitting Guidance
----------------

Initialization Strategy
~~~~~~~~~~~~~~~~~~~~~~~

**Step 1: Flow curve fitting (fastest)**

Start with steady-state flow curve data to constrain:

- ``sigma_c_mean``: Should approximate macroscopic yield stress (or slightly below)
- ``mu``: Elastic modulus (can initialize from SAOS data if available)

**Step 2: Startup shear refinement**

Use transient startup data to refine:

- ``tau_pl``: Controls stress overshoot decay rate
- ``sigma_c_std``: Controls overshoot magnitude and fluctuations

**Step 3: Use small lattice for fitting**

- **L = 8-16** for parameter estimation (fast, 0.5-2 min per fit)
- **L = 32-64** for validation and spatial analysis (10-30 min)
- **L = 128+** only for production simulations (hours)

Parameter Bounds and Physical Constraints
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Parameter
     - Typical Range
     - Physical Constraint
   * - ``mu``
     - 10-10000 Pa
     - Match SAOS elastic modulus G' if available
   * - ``sigma_c_mean``
     - 0.5-2× macroscopic σ_y
     - Lower bound: σ_y/2; upper bound: 2σ_y
   * - ``sigma_c_std``
     - 0.1-0.5× sigma_c_mean
     - Larger disorder = stronger shear banding
   * - ``tau_pl``
     - 0.01-10 s
     - Should be << experimental timescale
   * - ``L``
     - 8-128
     - Fitting: 8-16; Production: 32-128
   * - ``dt``
     - 0.001-0.05
     - Must resolve τ_pl (dt < τ_pl/10)

Common Fitting Issues
~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 35 65
   :header-rows: 1

   * - Issue
     - Solution
   * - Fit converges but predictions unrealistic
     - Reduce L to 8-12 for faster iteration; check dt stability
   * - Large NLSQ residuals
     - Switch to ``use_log_residuals=True`` for flow curves
   * - Bayesian divergences > 5%
     - Increase ``num_warmup`` to 1000-2000; reduce L to 8
   * - R-hat > 1.1
     - Run longer chains (num_samples=2000+); check for multimodality
   * - Predictions too smooth (no avalanches)
     - Increase ``sigma_c_std`` (disorder) or use ``smooth=False``

Fitting Parameters
~~~~~~~~~~~~~~~~~~

EPM fitting supports these keyword arguments:

.. list-table:: Fitting Options
   :header-rows: 1
   :widths: 25 20 55

   * - Parameter
     - Default
     - Description
   * - ``test_mode``
     - (required)
     - Protocol: 'flow_curve', 'startup', 'relaxation', 'creep', 'oscillation'
   * - ``seed``
     - 42
     - Random seed for reproducibility
   * - ``gamma_dot``
     - 0.1
     - Shear rate for startup protocol
   * - ``gamma``
     - 0.1
     - Step strain for relaxation protocol
   * - ``stress``
     - 1.0
     - Target stress for creep protocol
   * - ``gamma0``
     - 0.01
     - Strain amplitude for oscillation
   * - ``omega``
     - 1.0
     - Angular frequency for oscillation
   * - ``max_iter``
     - 500
     - Maximum NLSQ iterations
   * - ``use_log_residuals``
     - True
     - Use log-space residuals (recommended)

Convergence Tips
~~~~~~~~~~~~~~~~

EPM models are stochastic due to the random yield thresholds. For robust inference:

1. **Use small lattices for fitting** (L=8-16): Faster and sufficient for parameter estimation
2. **Increase warmup samples**: EPM posteriors may have multimodal structure
3. **Check divergences**: >5% divergences suggests model-data mismatch
4. **Run multiple chains**: Essential for R-hat diagnostics

Expected diagnostics for well-converged EPM fits:

*   R-hat < 1.1 for all parameters
*   ESS > 400 per parameter
*   Divergences < 1%

Usage
-----

Basic Usage
~~~~~~~~~~~

.. code-block:: python

    from rheojax.models.epm import LatticeEPM
    import numpy as np

    # Create model instance
    model = LatticeEPM(L=16, dt=0.01)

    # Fit to flow curve data
    gamma_dot = np.logspace(-2, 1, 20)
    stress_exp = np.array([0.5, 0.8, 1.2, 1.8, 2.5, 3.4, 4.5, 5.8, 7.3, 9.1,
                           11.2, 13.6, 16.3, 19.4, 22.8, 26.5, 30.6, 35.0, 39.8, 44.9])

    model.fit(gamma_dot, stress_exp, test_mode='flow_curve')

    # Predict stress
    gamma_dot_new = np.logspace(-2, 1, 50)
    sigma_pred = model.predict(gamma_dot_new, test_mode='flow_curve')

Advanced Usage Examples
------------------------

Basic Flow Curve Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from rheojax.models.epm import LatticeEPM
    import numpy as np

    # Create model with small lattice for fitting
    model = LatticeEPM(L=16, dt=0.01)

    # Experimental flow curve data
    gamma_dot = np.logspace(-2, 1, 20)
    stress_exp = np.array([0.5, 0.8, 1.2, 1.8, 2.5, 3.4, 4.5, 5.8, 7.3, 9.1,
                           11.2, 13.6, 16.3, 19.4, 22.8, 26.5, 30.6, 35.0, 39.8, 44.9])

    # NLSQ fitting (fast)
    model.fit(gamma_dot, stress_exp, test_mode="flow_curve", max_iter=500)

    print(f"Fitted σ_c: {model.params.get_value('sigma_c_mean'):.2f} Pa")
    print(f"Fitted disorder: {model.params.get_value('sigma_c_std'):.3f} Pa")

Startup Shear with Stress Overshoot
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Simulate startup at constant shear rate
    t = np.linspace(0, 50, 500)
    gamma_dot_startup = 0.1  # 1/s

    # Predict using fitted parameters
    stress_startup = model.predict(
        t,
        test_mode="startup",
        gamma_dot=gamma_dot_startup
    )

    # Plot stress vs time (shows overshoot)
    import matplotlib.pyplot as plt
    plt.plot(t, stress_startup)
    plt.xlabel("Time (s)")
    plt.ylabel("Stress (Pa)")

Bayesian Inference with Uncertainty
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Bayesian inference (warm-started from NLSQ)
    result = model.fit_bayesian(
        gamma_dot,
        stress_exp,
        test_mode="flow_curve",
        num_warmup=500,
        num_samples=1000,
        num_chains=4,
        seed=42,
    )

    # Extract credible intervals
    intervals = model.get_credible_intervals(
        result.posterior_samples,
        credibility=0.95
    )

    for name, (lower, upper) in intervals.items():
        mean_val = result.posterior_samples[name].mean()
        print(f"{name}: {mean_val:.3f} [{lower:.3f}, {upper:.3f}]")

    # Check convergence
    print(f"R-hat (max): {max(result.diagnostics['r_hat'].values()):.4f}")
    print(f"ESS (min): {min(result.diagnostics['ess'].values()):.0f}")

Visualizing Spatial Fields
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import numpy as np
    from rheojax.visualization.epm_plots import plot_lattice_fields

    # Run simulation at higher resolution for visualization
    model_viz = LatticeEPM(L=64, dt=0.01)
    model_viz.params = model.params.copy()  # Use fitted parameters

    # Time array for startup simulation
    t = np.linspace(0, 50, 500)

    # Simulate and extract stress field
    stress_field = model_viz.predict(
        t,
        test_mode="startup",
        gamma_dot=1.0,
        return_fields=True  # Returns spatial arrays
    )

    # Plot stress heterogeneity
    plot_lattice_fields(
        stress_field,
        title="Stress Distribution at t=10s",
        cmap="viridis"
    )

API Reference
-------------

.. autoclass:: rheojax.models.epm.lattice.LatticeEPM
    :members:
    :undoc-members:
    :show-inheritance:
    :no-index:

.. autofunction:: rheojax.visualization.epm_plots.plot_lattice_fields
    :no-index:

Comparison: LatticeEPM vs TensorialEPM
---------------------------------------

RheoJAX provides two EPM implementations with different capabilities:

.. list-table:: Feature Comparison
   :header-rows: 1
   :widths: 30 35 35

   * - Feature
     - LatticeEPM (Scalar)
     - TensorialEPM
   * - Stress Components
     - σ_xy only
     - [σ_xx, σ_yy, σ_xy] + σ_zz
   * - Flow Curves
     - ✓ Fast
     - ✓ More accurate if N₁ ≠ 0
   * - Normal Stress Differences
     - ✗
     - ✓ N₁, N₂ predictions
   * - Yield Criteria
     - Scalar threshold
     - von Mises or Hill
   * - Anisotropic Materials
     - ✗
     - ✓ Hill criterion
   * - Computational Cost
     - 1x (baseline)
     - 3-5x slower
   * - Memory Usage
     - 1x
     - 3x (tensor storage)
   * - Fitting Speed
     - Fast
     - Moderate
   * - GPU Acceleration
     - ✓
     - ✓

**When to Use LatticeEPM**:
- Pure shear rheology (flow curves, yield stress)
- Fast parameter estimation
- Exploratory analysis
- No normal stress data available

**When to Use TensorialEPM** (:doc:`tensorial_epm`):
- Normal stress measurements available
- Anisotropic materials (fibers, liquid crystals)
- Flow instabilities (shear banding, edge fracture)
- Rod climbing or die swell phenomena

References
----------

.. [1] Picard, G., Ajdari, A., Lequeux, F., and Bocquet, L. "Elastic consequences of a
   single plastic event: A step towards the microscopic modeling of the flow of yield
   stress fluids." *European Physical Journal E*, 15, 371-381 (2004).
   https://doi.org/10.1140/epje/i2004-10054-8

.. [2] Nicolas, A., Ferrero, E. E., Martens, K., and Barrat, J.-L. "Deformation and flow
   of amorphous solids: Insights from elastoplastic models." *Reviews of Modern Physics*,
   90, 045006 (2018). https://doi.org/10.1103/RevModPhys.90.045006

.. [3] Martens, K., Bocquet, L., and Barrat, J.-L. "Connecting diffusion and dynamical
   heterogeneities in actively deformed amorphous systems." *Physical Review Letters*,
   106, 156001 (2011). https://doi.org/10.1103/PhysRevLett.106.156001

.. [4] Eshelby, J. D. "The determination of the elastic field of an ellipsoidal inclusion,
   and related problems." *Proceedings of the Royal Society A*, 241, 376-396 (1957).
   https://doi.org/10.1098/rspa.1957.0133

.. [5] Lin, J., Lerner, E., Rosso, A., and Wyart, M. "Scaling description of the yielding
   transition in soft amorphous solids at zero temperature." *Proceedings of the National
   Academy of Sciences*, 111, 14382-14387 (2014). https://doi.org/10.1073/pnas.1406391111

.. [6] Barrat, J.-L. and Lemaître, A. "Heterogeneities in amorphous systems under shear."
   *Dynamical Heterogeneities in Glasses, Colloids, and Granular Media*, Oxford University
   Press (2011). https://doi.org/10.1093/acprof:oso/9780199691470.003.0008

.. [7] Nicolas, A., Martens, K., Bocquet, L., and Barrat, J.-L. "Universal and non-universal
   features in coarse-grained models of flow in disordered solids." *Soft Matter*, 10,
   4648-4661 (2014). https://doi.org/10.1039/C4SM00395K

.. [8] Lemaitre, A. and Caroli, C. "Rate-dependent avalanche size in athermally sheared
   amorphous solids." *Physical Review Letters*, 103, 065501 (2009).
   https://doi.org/10.1103/PhysRevLett.103.065501

.. [9] Talamali, M., Petäjä, V., Vandembroucq, D., and Roux, S. "Strain localization and
   anisotropic correlations in a mesoscopic model of amorphous plasticity." *Comptes Rendus
   Mécanique*, 340, 275-288 (2012). https://doi.org/10.1016/j.crme.2012.02.010

.. [10] Budrikis, Z., Castellanos, D. F., Sandfeld, S., Zaiser, M., and Zapperi, S.
    "Universal features of amorphous plasticity." *Nature Communications*, 8, 15928 (2017).
    https://doi.org/10.1038/ncomms15928

See Also
--------

- :doc:`tensorial_epm` — Full stress tensor implementation
- :doc:`/user_guide/03_advanced_topics/index` — Advanced EPM workflows
- :py:func:`rheojax.visualization.epm_plots.plot_lattice_fields` — Visualization functions for spatial fields
