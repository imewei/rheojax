.. _epm_model:

Elasto-Plastic Models (EPM)
===========================

The Elasto-Plastic Model (EPM) is a mesoscopic lattice-based framework for modeling the rheology of amorphous solids (glasses, gels, pastes, dense suspensions). Unlike mean-field models (like Hebraud-Lequeux or Soft Glassy Rheology), EPMs explicitly resolve **spatial heterogeneity**, **plastic avalanches**, and **non-local stress redistribution**.

This implementation leverages **JAX** for high-performance FFT-based computations on GPU/TPU.

.. contents:: Table of Contents
    :local:
    :depth: 2

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

The Eshelby Propagator
~~~~~~~~~~~~~~~~~~~~~~
The propagator $\mathcal{G}(\mathbf{r})$ describes the elastic field of a plastic inclusion. In 2D Fourier space $\mathbf{q} = (q_x, q_y)$, it has a characteristic quadrupolar symmetry ("four-leaf clover"):

.. math::

    \tilde{\mathcal{G}}(\mathbf{q}) = -4 \mu \frac{q_x^2 q_y^2}{(q_x^2 + q_y^2)^2} \quad \text{for } \mathbf{q} \neq 0

Note that $\tilde{\mathcal{G}}(0) = 0$ to ensure plastic events do not change the mean stress (which is controlled by the walls/loading).

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
    stress = ...  # Your experimental data

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

API Reference
-------------

.. autoclass:: rheojax.models.epm.lattice.LatticeEPM
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: rheojax.visualization.epm_plots.plot_lattice_fields

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

See Also
--------

- :doc:`tensorial_epm` — Full stress tensor implementation
- :doc:`/user_guide/03_advanced_topics/index` — Advanced EPM workflows
- :ref:`epm_visualization` — Visualization functions
