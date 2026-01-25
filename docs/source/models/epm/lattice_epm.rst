.. _epm_model:

Elasto-Plastic Models (EPM)
===========================

Quick Reference
---------------

- **Use when:** Spatially-resolved modeling of amorphous solids, plastic avalanches, shear banding

- **Parameters:** 6 (μ, σ_c_mean, σ_c_std, τ_pl, L, dt)

- **Key equation:** :math:`\partial_t \sigma_{ij} = \mu \dot{\gamma}(t) - \mu \dot{\gamma}^{pl}_{ij} + \sum_{kl} \mathcal{G}_{ij,kl} \dot{\gamma}^{pl}_{kl}`

- **Test modes:** flow_curve, startup, relaxation, creep, oscillation

- **Material examples:** Metallic glasses, colloidal gels, pastes, dense granular suspensions, foams

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

Discrete State Variables
------------------------

The scalar EPM discretizes the material on a :math:`d`-dimensional periodic lattice
(typically :math:`d=2`) with :math:`N = L_x \times L_y` sites indexed by :math:`i`.

At each site :math:`i`:

- **Local shear stress**: :math:`\sigma_i(t)`
- **Local plastic strain**: :math:`\varepsilon_i^{pl}(t)`
- **Local yield threshold**: :math:`\sigma_{y,i}` (constant or disordered)
- **Local elastic modulus**: :math:`\mu_i` (often uniform :math:`\mu`)
- **(Optional)** Structural variable: age :math:`x_i`, effective temperature :math:`T_i`, etc.

Elastic Loading and Stress Redistribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The local stress evolves due to elastic loading from the macroscopic shear and
redistribution from plastic events at other sites:

.. math::

    \boxed{
    \sigma_i(t) = \mu \left[ \gamma(t) - \varepsilon_i^{pl}(t) \right] + \sum_j G_{ij} \, \varepsilon_j^{pl}(t)
    }

In rate form:

.. math::

    \boxed{
    \dot{\sigma}_i = \mu \, \dot{\gamma}(t) - \mu \, \dot{\varepsilon}_i^{pl} + \sum_j G_{ij} \, \dot{\varepsilon}_j^{pl}
    }

where:

- :math:`\mu \dot{\gamma}`: Affine elastic loading from imposed macroscopic shear
- :math:`G_{ij}`: Eshelby propagator (elastic kernel) discretized on the lattice
- The last two terms describe stress relaxation/redistribution from plastic strain rates

Effective Kernel
~~~~~~~~~~~~~~~~

In practice, it is common to combine terms into an **effective kernel** :math:`K_{ij}`
acting on plastic strain increments:

.. math::

    \boxed{
    \dot{\sigma}_i = \mu \dot{\gamma}(t) + \sum_j K_{ij} \, \dot{\varepsilon}_j^{pl},
    \qquad
    K_{ij} = G_{ij} - \mu \, \delta_{ij}
    }

This form is computationally convenient since the FFT convolution directly gives the
stress update from plastic strain rates.

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

.. note::

    For a 2D **scalar shear** model, a common alternative form is:

    .. math::

        \boxed{
        \hat{G}(\mathbf{q}) \propto -\frac{(q_x^2 - q_y^2)^2}{(q_x^2 + q_y^2)^2}
        }

    with :math:`\hat{G}(\mathbf{0}) = 0`. Variants exist depending on whether you model
    pure shear, simple shear, or include compressibility.

Numerically, the **effective kernel** is defined as :math:`\hat{K}(\mathbf{q}) = \hat{G}(\mathbf{q}) - \mu`,
and the convolution is computed via FFT:

.. math::

    \boxed{
    (K \ast \dot{\varepsilon}^{pl})_i = \mathcal{F}^{-1}\left( \hat{K}(\mathbf{q}) \cdot \mathcal{F}[\dot{\varepsilon}^{pl}] \right)_i
    }

where :math:`\mathcal{F}` denotes the discrete Fourier transform.

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

Local Yield Condition
~~~~~~~~~~~~~~~~~~~~~

A site becomes unstable when the local stress exceeds its yield threshold:

.. math::

    \boxed{| \sigma_i | \ge \sigma_{y,i}}

Once unstable, the plastic strain rate is determined by one of two dynamics:

Plastic Event Dynamics
~~~~~~~~~~~~~~~~~~~~~~

**Option A: Instantaneous Flip (Quasi-Static / Event-Driven)**

When unstable, site :math:`i` yields and its plastic strain jumps by a fixed increment:

.. math::

    \boxed{
    \varepsilon_i^{pl} \leftarrow \varepsilon_i^{pl} + \Delta\varepsilon_0 \, \mathrm{sign}(\sigma_i)
    }

where :math:`\Delta\varepsilon_0` may be drawn from a distribution.
Stress is then updated by redistribution to all sites:

.. math::

    \boxed{
    \sigma_k \leftarrow \sigma_k + K_{ki} \, \Delta\varepsilon_0 \, \mathrm{sign}(\sigma_i) \quad \forall k
    }

After redistribution, stability is re-checked (potentially triggering avalanches).

**Option B: Finite-Rate Maxwell / Viscoplastic Rule (Continuous Time Stepping)**

Introduce a plastic strain rate when the yield condition is met:

.. math::

    \boxed{
    \dot{\varepsilon}_i^{pl} =
    \begin{cases}
    0, & |\sigma_i| < \sigma_{y,i} \\[2pt]
    \displaystyle\frac{1}{\eta}\left(\sigma_i - \sigma_{y,i} \, \mathrm{sign}(\sigma_i)\right), & |\sigma_i| \ge \sigma_{y,i}
    \end{cases}
    }

Or a simpler **activated rule** with fixed rate:

.. math::

    \boxed{
    \dot{\varepsilon}_i^{pl} = \frac{1}{\tau} \, \mathrm{sign}(\sigma_i) \, \mathbf{1}_{|\sigma_i| \ge \sigma_{y,i}}
    }

The finite-rate rule is convenient for **LAOS** and explicit time integration since it
produces smooth time series.

Macroscopic Observables
~~~~~~~~~~~~~~~~~~~~~~~

The primary outputs from EPM simulations are spatially-averaged quantities:

**Macroscopic Shear Stress:**

.. math::

    \boxed{
    \Sigma(t) = \frac{1}{N} \sum_i \sigma_i(t)
    }

**Plastic Activity** (fraction of active sites or total plastic strain rate):

.. math::

    \Gamma(t) = \frac{1}{N} \sum_i \mathbf{1}_{|\sigma_i| \ge \sigma_{y,i}}
    \quad \text{or} \quad
    \Gamma(t) = \frac{1}{N} \sum_i |\dot{\varepsilon}_i^{pl}|

**Energy Dissipation Rate:**

.. math::

    \dot{W}_{\mathrm{diss}}(t) = \frac{1}{N} \sum_i \sigma_i \, \dot{\varepsilon}_i^{pl}

**Dissipated Energy Per Cycle** (for oscillatory protocols):

.. math::

    W_{\mathrm{diss,cycle}} = \oint \Sigma \, d\gamma

This integral over a complete strain cycle gives the area enclosed by the Lissajous figure,
representing energy lost to plastic dissipation.

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

The model supports standard rheological protocols via ``_predict(test_mode=...)``.
Below we provide the complete mathematical formulation for each protocol, using the
**finite-rate** plastic flow rule (Option B) unless noted otherwise.

.. _epm-flow-curve:

Flow Curve (Rotation / Steady Shear)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:**

.. math::

    \dot{\gamma}(t) = \dot{\gamma} = \text{const}

**Governing Equations:**

.. math::

    \boxed{
    \dot{\sigma} = \mu \dot{\gamma} + K \ast \dot{\varepsilon}^{pl},
    \qquad
    \dot{\varepsilon}^{pl} = \mathcal{R}(\sigma; \sigma_y, \eta, \tau)
    }

where :math:`\mathcal{R}` is the chosen plastic flow rule (finite-rate or activated).

**Flow Curve Extraction:**

Run to steady state and compute the time-averaged macroscopic stress:

.. math::

    \boxed{
    \Sigma(\dot{\gamma}) = \langle \Sigma(t) \rangle_{\mathrm{steady}}
    }

**Typical Scaling:**

Many EPMs produce **Herschel-Bulkley-like** flow curves:

.. math::

    \Sigma(\dot{\gamma}) \approx \Sigma_y + A \dot{\gamma}^n

The exponent :math:`n` depends on the flow rule and noise/activation:

- :math:`n \approx 0.5` near yield (diffusive scaling from avalanches)
- :math:`n \to 1` at high rates (linear viscous regime)

.. _epm-startup:

Start-up of Steady Shear
~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:**

.. math::

    \dot{\gamma}(t) =
    \begin{cases}
    0, & t < 0 \\
    \dot{\gamma}_0, & t \ge 0
    \end{cases}

starting from an initial condition (often :math:`\varepsilon^{pl} = 0`, :math:`\sigma`
drawn from a narrow distribution, or an "aged" state).

**Outputs:**

- Stress growth :math:`\Sigma(t)`
- Stress overshoot :math:`\Sigma_{\max}` and peak strain :math:`\gamma_{\max}`
- Avalanche statistics (for event-driven implementation)
- Transient shear banding (when spatial gradient direction is retained)

**Key Physics:**

1. **Elastic rise**: :math:`\Sigma \propto \mu \dot{\gamma}_0 t` at early times
2. **Overshoot**: Stress peak when first system-spanning avalanches occur
3. **Steady state**: Relaxation to the flow curve plateau

.. _epm-relaxation:

Stress Relaxation (Step Strain)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol (step strain):**

.. math::

    \gamma(t) = \gamma_0 \, H(t)
    \quad \Rightarrow \quad
    \dot{\gamma}(t) = \gamma_0 \, \delta(t)

**Implementation:**

1. Apply instantaneous affine loading at :math:`t = 0^+`:

   .. math::

       \sigma_i(0^+) = \sigma_i(0^-) + \mu \gamma_0

2. Then evolve at :math:`\dot{\gamma}(t > 0) = 0`.

**Governing Equations for** :math:`t > 0`:

.. math::

    \boxed{
    \dot{\sigma} = K \ast \dot{\varepsilon}^{pl},
    \qquad
    \dot{\varepsilon}^{pl} = \mathcal{R}(\sigma; \sigma_y, \eta, \tau)
    }

**Relaxation Modulus:**

.. math::

    \boxed{
    G(t) = \frac{\Sigma(t)}{\gamma_0}
    }

**Physics:**

Stress relaxes via "cascades"—an active site yields, triggering a neighbor, keeping
the system active long after the drive stops. This leads to slow, non-exponential
relaxation (power-law :math:`\sim t^{-\alpha}` or logarithmic :math:`\sim \ln t`).

.. _epm-creep:

Creep (Step Stress)
~~~~~~~~~~~~~~~~~~~

**Protocol (step stress):**

.. math::

    \Sigma(t) = \Sigma_0 \, H(t)

**Stress-Controlled Closure:**

Since EPM is naturally strain-rate driven, we determine :math:`\dot{\gamma}(t)`
dynamically so that:

.. math::

    \boxed{
    \frac{1}{N} \sum_i \sigma_i(t) = \Sigma_0
    }

**Simple Controller:**

.. math::

    \boxed{
    \dot{\gamma}_{n+1} = \dot{\gamma}_n + \lambda \left( \Sigma_0 - \Sigma_n \right)
    }

where :math:`\lambda` is the feedback gain (typically 0.1-1.0).

**Creep Strain:**

.. math::

    \gamma(t) = \int_0^t \dot{\gamma}(s) \, ds

**Behavior:**

- :math:`\Sigma_0 < \Sigma_y`: Strain rate decays to zero (**arrest**)
- :math:`\Sigma_0 > \Sigma_y`: Strain rate stabilizes to finite value (**flow**)
- Fluidization time :math:`t_f \sim (\Sigma_y - \Sigma_0)^{-\nu}` with :math:`\nu \approx 4-6`

.. _epm-oscillation:

Oscillation (SAOS and LAOS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Protocol:**

.. math::

    \gamma(t) = \gamma_0 \sin(\omega t),
    \qquad
    \dot{\gamma}(t) = \gamma_0 \omega \cos(\omega t)

**Governing Equations:**

.. math::

    \boxed{
    \dot{\sigma} = \mu \dot{\gamma}(t) + K \ast \dot{\varepsilon}^{pl},
    \qquad
    \dot{\varepsilon}^{pl} = \mathcal{R}(\sigma; \sigma_y, \eta, \tau)
    }

**SAOS Moduli** (Small Amplitude Oscillatory Shear, :math:`\gamma_0 \ll 1`):

Extract first harmonic from the stress response:

.. math::

    \Sigma(t) \approx \Sigma'_1 \sin(\omega t) + \Sigma''_1 \cos(\omega t)

Storage and loss moduli:

.. math::

    G' = \frac{\Sigma'_1}{\gamma_0}, \qquad G'' = \frac{\Sigma''_1}{\gamma_0}

**LAOS Harmonics** (Large Amplitude Oscillatory Shear):

At large :math:`\gamma_0`, the stress response contains higher harmonics:

.. math::

    \Sigma(t) = \sum_{n \ge 1} \left[ \Sigma'_n \sin(n\omega t) + \Sigma''_n \cos(n\omega t) \right]

Higher-order moduli:

.. math::

    G'_n = \frac{\Sigma'_n}{\gamma_0}, \qquad G''_n = \frac{\Sigma''_n}{\gamma_0}

**LAOS Physics:**

- Large :math:`\gamma_0` triggers yielding cycle-by-cycle
- The Eshelby propagator synchronizes these events
- Complex **Lissajous figures** with shear-banding signatures
- Intracycle yielding and strain stiffening/softening

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

- :doc:`index` — EPM family overview and comparison
- :doc:`tensorial_epm` — Full stress tensor implementation
- :doc:`/models/sgr/sgr_conventional` — Mean-field SGR (complementary theory)
- :doc:`/models/hl/hebraud_lequeux` — Mean-field HL (EPM limiting case)
- :doc:`/models/fluidity/fluidity_local` — Fluidity approach to plasticity
- :py:func:`rheojax.visualization.epm_plots.plot_lattice_fields` — Visualization functions for spatial fields


JAX Implementation Utilities
============================

This section provides JAX-first implementations of the core EPM components. These
utilities form the computational backbone of the ``LatticeEPM`` model.

.. note::

   For actual usage, prefer the high-level ``LatticeEPM`` class. These utilities
   are documented for advanced users and developers.

EPM Parameter Container
-----------------------

.. code-block:: python

    from __future__ import annotations
    from dataclasses import dataclass
    import jax.numpy as jnp

    Array = jnp.ndarray

    @dataclass(frozen=True)
    class EPMParams:
        """Immutable container for EPM simulation parameters."""
        Lx: int
        Ly: int
        mu: float = 1.0          # Shear modulus
        sig_y0: float = 1.0      # Mean yield stress
        eta: float = 1.0         # Viscoplastic viscosity
        tau: float = 1.0         # Activated timescale
        dt: float = 1e-3         # Time step
        sigy_logstd: float = 0.0 # Yield stress disorder (log-normal)

Fourier Grid and Eshelby Kernel
-------------------------------

.. code-block:: python

    def make_qgrid(Lx: int, Ly: int):
        """Create 2D wavevector grid for FFT operations."""
        qx = 2 * jnp.pi * jnp.fft.fftfreq(Lx)[:, None]
        qy = 2 * jnp.pi * jnp.fft.fftfreq(Ly)[None, :]
        return qx, qy

    def make_eshelby_kernel_hat(Lx: int, Ly: int, mu: float):
        """Construct effective kernel K̂(q) = Ĝ(q) - μ in Fourier space.

        Returns
        -------
        Khat : Array, shape (Lx, Ly)
            Effective kernel for FFT convolution
        """
        qx, qy = make_qgrid(Lx, Ly)
        q2 = qx * qx + qy * qy
        q2_safe = jnp.where(q2 == 0.0, 1.0, q2)

        # Quadrupolar Eshelby kernel: G(q) ∝ -(qx² - qy²)² / q⁴
        Ghat = -((qx * qx - qy * qy) ** 2) / (q2_safe * q2_safe)
        Ghat = jnp.where(q2 == 0.0, 0.0, Ghat)

        # Effective kernel
        Khat = Ghat - mu
        return Khat

Plastic Flow Rules
------------------

.. code-block:: python

    def plastic_rate_bingham(sigma: Array, sig_y: Array, eta: float) -> Array:
        """Bingham-like viscoplastic flow rule.

        ε̇ᵖˡ = (1/η) × sign(σ) × max(|σ| - σ_y, 0)
        """
        sgn = jnp.sign(sigma)
        overstress = jnp.maximum(jnp.abs(sigma) - sig_y, 0.0)
        return (overstress / eta) * sgn

Time-Stepping Kernel
--------------------

.. code-block:: python

    import jax

    @jax.jit
    def epm_step(
        sigma: Array,
        eps_pl: Array,
        sig_y: Array,
        gdot: float,
        Khat: Array,
        p: EPMParams,
    ) -> tuple[Array, Array]:
        """Single EPM time step with FFT-accelerated stress redistribution.

        Parameters
        ----------
        sigma : Array, shape (Lx, Ly)
            Current local stress field
        eps_pl : Array, shape (Lx, Ly)
            Current plastic strain field
        sig_y : Array, shape (Lx, Ly)
            Local yield thresholds
        gdot : float
            Applied macroscopic shear rate
        Khat : Array, shape (Lx, Ly)
            Precomputed effective kernel in Fourier space
        p : EPMParams
            Simulation parameters

        Returns
        -------
        sigma_new, eps_pl_new : Arrays
            Updated stress and plastic strain fields
        """
        # Plastic strain rate
        deps_pl = plastic_rate_bingham(sigma, sig_y, p.eta)

        # FFT convolution for stress redistribution
        deps_pl_hat = jnp.fft.fftn(deps_pl)
        conv = jnp.fft.ifftn(Khat * deps_pl_hat).real

        # Stress update: elastic loading + redistribution
        dsigma = p.mu * gdot + conv
        sigma_new = sigma + p.dt * dsigma
        eps_pl_new = eps_pl + p.dt * deps_pl

        return sigma_new, eps_pl_new

Strain-Rate Controlled Simulation
---------------------------------

.. code-block:: python

    @jax.jit
    def simulate_strain_rate_control(
        sigma0: Array,
        eps_pl0: Array,
        sig_y: Array,
        gdot_t: Array,
        Khat: Array,
        p: EPMParams,
    ):
        """Run EPM simulation with prescribed strain rate history.

        Uses jax.lax.scan for efficient compilation and execution.

        Parameters
        ----------
        sigma0, eps_pl0 : Arrays
            Initial stress and plastic strain fields
        sig_y : Array
            Yield threshold field
        gdot_t : Array, shape (Nt,)
            Strain rate at each time step
        Khat : Array
            Precomputed Eshelby kernel
        p : EPMParams
            Simulation parameters

        Returns
        -------
        Sigma_t : Array, shape (Nt,)
            Macroscopic stress time series
        sigmaT, epsT : Arrays
            Final stress and plastic strain fields
        """
        def body(carry, gdot):
            sigma, eps_pl = carry
            sigma, eps_pl = epm_step(sigma, eps_pl, sig_y, gdot, Khat, p)
            Sigma = jnp.mean(sigma)
            return (sigma, eps_pl), Sigma

        (sigmaT, epsT), Sigma_t = jax.lax.scan(body, (sigma0, eps_pl0), gdot_t)
        return Sigma_t, sigmaT, epsT

Stress-Controlled Simulation (Creep)
------------------------------------

.. code-block:: python

    @jax.jit
    def simulate_creep_stress_control(
        sigma0: Array,
        eps_pl0: Array,
        sig_y: Array,
        Sigma_target: float,
        Nt: int,
        Khat: Array,
        p: EPMParams,
        gdot0: float = 0.0,
        gain: float = 0.1,
    ):
        """Run EPM creep simulation with stress feedback control.

        Parameters
        ----------
        Sigma_target : float
            Target macroscopic stress
        Nt : int
            Number of time steps
        gain : float
            Feedback controller gain (λ in documentation)

        Returns
        -------
        Sigma_t : Array
            Macroscopic stress time series
        gdot_t : Array
            Strain rate time series
        sigmaT, epsT : Arrays
            Final fields
        """
        def body(carry, _):
            sigma, eps_pl, gdot = carry
            Sigma = jnp.mean(sigma)
            # P-controller: adjust strain rate
            gdot = gdot + gain * (Sigma_target - Sigma)
            sigma, eps_pl = epm_step(sigma, eps_pl, sig_y, gdot, Khat, p)
            return (sigma, eps_pl, gdot), (Sigma, gdot)

        (sigmaT, epsT, gdotT), (Sigma_t, gdot_t) = jax.lax.scan(
            body, (sigma0, eps_pl0, gdot0), xs=None, length=Nt
        )
        return Sigma_t, gdot_t, sigmaT, epsT

Quasi-Static Avalanche Relaxation
---------------------------------

For event-driven (quasi-static) simulations, use iterative relaxation:

.. code-block:: python

    def avalanche_relax(
        sigma: Array,
        sig_y: Array,
        Khat: Array,
        Delta_eps0: float,
        max_iters: int = 256,
    ):
        """Relax unstable sites via iterative yielding until stable.

        All unstable sites yield simultaneously per sub-iteration (JAX-friendly).

        Parameters
        ----------
        Delta_eps0 : float
            Fixed plastic strain increment per yield event

        Returns
        -------
        sigma : Array
            Relaxed (stable) stress field
        """
        def one_iter(sigma, _):
            unstable = jnp.abs(sigma) >= sig_y
            deps = Delta_eps0 * jnp.sign(sigma) * unstable
            deps_hat = jnp.fft.fftn(deps)
            conv = jnp.fft.ifftn(Khat * deps_hat).real
            sigma = sigma + conv
            return sigma, None

        sigma, _ = jax.lax.scan(one_iter, sigma, xs=jnp.arange(max_iters))
        return sigma

Field Initialization
--------------------

.. code-block:: python

    import jax.random

    def init_sigy(key, p: EPMParams):
        """Initialize yield threshold field with optional log-normal disorder."""
        if p.sigy_logstd <= 0:
            return p.sig_y0 * jnp.ones((p.Lx, p.Ly))
        z = jax.random.normal(key, (p.Lx, p.Ly))
        return p.sig_y0 * jnp.exp(p.sigy_logstd * z)

    def init_fields(p: EPMParams):
        """Initialize stress and plastic strain fields to zero."""
        sigma0 = jnp.zeros((p.Lx, p.Ly))
        eps0 = jnp.zeros((p.Lx, p.Ly))
        return sigma0, eps0


Advanced Physics
================

This section covers advanced theoretical aspects of EPM plasticity, connecting
the lattice model to broader statistical physics concepts.


Avalanche Dynamics
------------------

One of the defining features of EPM is its ability to capture **plastic avalanches**—
cascades of yielding events triggered by stress redistribution.

Avalanche Definition
~~~~~~~~~~~~~~~~~~~~

An avalanche is a sequence of plastic events (block yieldings) triggered by a single
driving event. The avalanche terminates when no more blocks exceed their yield thresholds.

**Detection algorithm:**

1. Identify primary yield event (block exceeds σ_c)
2. Propagate stress via Eshelby (FFT convolution)
3. Count secondary yields (blocks pushed over threshold by redistribution)
4. Repeat until no new yields occur
5. Total yield count = avalanche size S

**Code example:**

.. code-block:: python

   def detect_avalanches(model, stress_time_series, sigma_c):
       """Detect avalanches from stress time series.

       Parameters
       ----------
       stress_time_series : ndarray, shape (T, L, L)
           Spatiotemporal stress field
       sigma_c : ndarray, shape (L, L)
           Local yield thresholds

       Returns
       -------
       avalanche_sizes : list
           Sizes of detected avalanches
       """
       yielded = jnp.abs(stress_time_series) > sigma_c
       # Count connected yield events per timestep
       avalanche_sizes = []
       for t in range(len(yielded) - 1):
           new_yields = yielded[t+1] & ~yielded[t]
           if new_yields.sum() > 0:
               avalanche_sizes.append(int(new_yields.sum()))
       return avalanche_sizes

Avalanche Size Distribution
~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the critical regime near yielding, avalanche sizes follow a power-law distribution:

.. math::

   P(S) \sim S^{-\tau} \quad \text{with cutoff } S_c \sim L^{d_f}

where:

- **τ ≈ 1.5-2.0**: Universal exponent (mean-field τ = 3/2, with disorder corrections)
- **S_c**: Finite-size cutoff scaling with lattice size L
- **d_f ≈ 2**: Fractal dimension of avalanches in 2D

**Key observations:**

- Well below yield (σ < σ_y): Exponentially distributed (no criticality)
- At yield (σ ≈ σ_y): Power-law distribution (critical behavior)
- Above yield (σ > σ_y): Flowing state, continuous plastic activity

**Duration-size scaling:**

Avalanche duration T scales with size S:

.. math::

   T(S) \sim S^{\alpha} \quad \text{with } \alpha \approx 0.5

This scaling connects to the dynamical exponent z = 2 for overdamped dynamics.

**References:** Lin et al. PNAS 2014 [5], Budrikis et al. Nat. Commun. 2017 [10]


Yielding Transition Physics
---------------------------

The yielding transition in EPM exhibits **critical point behavior**, analogous to
absorbing phase transitions in statistical physics.

Critical Behavior
~~~~~~~~~~~~~~~~~

Near the macroscopic yield stress σ_y, several quantities diverge or vanish:

**Correlation length:**

.. math::

   \xi \sim |\sigma - \sigma_y|^{-\nu}

where ν is the correlation length exponent (ν ≈ 1 in mean-field).

**Relaxation time:**

.. math::

   \tau_{relax} \sim \xi^z \sim |\sigma - \sigma_y|^{-\nu z}

with dynamical exponent z ≈ 2.

**Mean-field exponents:**

.. list-table::
   :widths: 25 25 50
   :header-rows: 1

   * - Exponent
     - Value
     - Physical Meaning
   * - τ (avalanche)
     - 3/2
     - Avalanche size distribution P(S) ~ S^(-τ)
   * - ν (correlation)
     - 1
     - Correlation length ξ ~ (σ-σ_y)^(-ν)
   * - β (order param)
     - 1
     - Plastic rate γ̇_pl ~ (σ-σ_y)^β
   * - z (dynamical)
     - 2
     - Relaxation time τ ~ ξ^z

**Disorder corrections:**

Strong disorder (σ_c_std/σ_c_mean > 0.3) modifies these exponents:

- τ increases toward 2.0 (broader avalanche distribution)
- ν decreases (shorter correlations)

Dynamical Heterogeneity
~~~~~~~~~~~~~~~~~~~~~~~

Near yielding, the material exhibits **dynamical heterogeneity**: coexisting regions
with different local plastic activity. This is quantified by:

**Four-point correlator:**

.. math::

   \chi_4(t) = L^2 \left[ \langle \Delta(t)^2 \rangle - \langle \Delta(t) \rangle^2 \right]

where Δ(t) is the fraction of sites that have yielded by time t.

χ₄ peaks at the characteristic relaxation time, measuring the size of correlated
rearranging regions.

**Connection to absorbing phase transitions:**

The yielding transition maps onto the **directed percolation universality class**
in the athermal limit, or conserved directed percolation for stress-controlled protocols.

**References:** Nicolas et al. Rev. Mod. Phys. 2018 §IV [2], Martens et al. PRL 2011 [3]


Shear Banding Mechanisms
------------------------

EPM naturally captures **shear banding**—the localization of plastic flow into
narrow bands separated by solid-like regions.

Localization Criteria
~~~~~~~~~~~~~~~~~~~~~

Shear banding in EPM arises from two mechanisms:

1. **Disorder-driven localization**: High disorder (α = σ_c_std/σ_c_mean > 0.3)
   leads to heterogeneous yield thresholds. Regions with low σ_c yield first,
   creating stress concentrations that nucleate bands.

2. **Stress redistribution feedback**: The Eshelby propagator's quadrupolar
   symmetry creates positive feedback—plastic events along the flow direction
   destabilize neighboring regions.

**Banding criterion:**

.. math::

   \frac{\sigma_{c,std}}{\sigma_{c,mean}} > 0.3 \quad \text{(disorder threshold)}

**Band width:**

.. math::

   w_{band} \sim \xi_{corr} \sim L / \sqrt{\text{disorder}}

where ξ_corr is the correlation length of plastic events.

Detection Methods
~~~~~~~~~~~~~~~~~

**From velocity profile:**

.. code-block:: python

   def detect_shear_band(velocity_profile, y_positions, threshold=0.1):
       """Detect shear banding from velocity profile.

       Parameters
       ----------
       velocity_profile : ndarray
           Velocity v_x as function of y
       y_positions : ndarray
           y coordinates
       threshold : float
           Relative gradient threshold

       Returns
       -------
       band_width : float
           Width of the shear band (or NaN if no banding)
       """
       grad_v = jnp.gradient(velocity_profile, y_positions)
       max_grad = jnp.max(jnp.abs(grad_v))

       # Banding if localized high shear region
       high_shear = jnp.abs(grad_v) > threshold * max_grad
       band_width = jnp.sum(high_shear) * (y_positions[1] - y_positions[0])
       return band_width

**From stress gradients:**

Shear bands correlate with large spatial gradients in the stress field:

.. math::

   \nabla \sigma_{xy} > \sigma_{c,std} / \xi_{corr}

**Transient vs steady-state banding:**

- **Transient bands**: Appear during startup, may homogenize at long times
- **Steady-state bands**: Persist indefinitely; require strong disorder or thixotropy

**References:** Talamali et al. CR Mécanique 2012 [9], Nicolas et al. Soft Matter 2014 [7]


Connections to Other Models
---------------------------

EPM connects to several other rheological frameworks, providing physical insight
and enabling model selection.

Soft Glassy Rheology (SGR)
~~~~~~~~~~~~~~~~~~~~~~~~~~

SGR and EPM describe similar physics from different perspectives:

.. list-table::
   :widths: 30 35 35
   :header-rows: 1

   * - Aspect
     - SGR
     - EPM
   * - Approach
     - Mean-field trap model
     - Spatially resolved lattice
   * - Noise temperature
     - x (thermal + mechanical)
     - σ_c_std (quenched disorder)
   * - Yielding
     - Thermally activated escape
     - Stress-driven (athermal)
   * - Avalanches
     - Implicit (x < 1)
     - Explicit cascades
   * - Correlations
     - None (mean-field)
     - Quadrupolar (Eshelby)

**Mapping:**

The effective noise temperature x in SGR corresponds to disorder strength:

.. math::

   x \leftrightarrow \frac{\sigma_{c,std}}{\sigma_{c,mean}}

SGR's glass transition (x = 1) maps to the EPM critical disorder threshold.

Fluidity Models
~~~~~~~~~~~~~~~

Fluidity models describe plastic flow via a kinetic fluidity field φ(r, t).

**Connection:**

.. math::

   \phi = \frac{\dot{\gamma}^{pl}}{\sigma} = \frac{1}{\eta_{pl}}

The local plastic strain rate in EPM maps to fluidity:

- High φ → flowing (high plastic activity)
- Low φ → arrested (solid-like)

Fluidity diffusion in nonlocal models corresponds to Eshelby stress redistribution
in the mean-field limit.

Shear Transformation Zone (STZ) Theory
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

STZ theory views plastic events as activated transitions of local structural zones.

**Connection:**

Each EPM lattice block represents a **mesoscopic ensemble of STZs**. The yield
threshold σ_c corresponds to the activation barrier, and the plastic relaxation
time τ_pl to the STZ flip rate.

.. math::

   \dot{\gamma}^{pl}_{EPM} \sim n_{STZ} \cdot \nu_{flip} \cdot \gamma_0

where n_STZ is STZ density, ν_flip is flip rate, and γ₀ is elementary strain.

Hébraud-Lequeux Mean-Field Limit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Hébraud-Lequeux (HL) model is the **mean-field limit** of EPM:

- Replace Eshelby propagator with uniform redistribution
- All sites feel average stress, not local heterogeneity
- Exact solution possible, faster computation

EPM → HL when:

- L → ∞ and disorder → 0 (thermodynamic limit with weak disorder)
- Short-range interactions dominate (screened propagator)

The HL model provides analytic benchmarks for EPM predictions.

Depinning Universality Class
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The yielding transition in EPM belongs to the **depinning universality class**
(interfaces driven through random media):

- **Below threshold**: Pinned (solid-like)
- **At threshold**: Critical dynamics (avalanches)
- **Above threshold**: Steady flow (depinned)

**Shared features:**

- Power-law avalanche distributions
- Finite-size scaling: ξ ~ L at criticality
- Hysteresis in rate-controlled protocols

This connection enables use of field-theoretic methods from depinning transitions
to understand yielding rheology.

**References:** Nicolas et al. Rev. Mod. Phys. 2018 [2], Lemaitre & Caroli PRL 2009 [8]


Model Extensions
~~~~~~~~~~~~~~~~

Several extensions to the basic EPM framework are commonly used:

**1. Activated Yielding (Thermal/Athermal Noise)**

   Add thermal activation to the yield criterion:

   .. math::

      r_i = r_0 \exp\left( -\frac{\Delta E(\sigma_i)}{k_B T} \right)

   where :math:`\Delta E(\sigma_i) = E_0 (1 - |\sigma_i|/\sigma_{y,i})^p` (often :math:`p = 3/2`).

   This bridges EPM toward **SGR-like dynamics** and allows modeling of creep
   at sub-yield stresses.

**2. Aging (Time-Dependent Yield Thresholds)**

   Allow :math:`\sigma_{y,i}` to evolve:

   .. math::

      \dot{\sigma}_{y,i} = \frac{\sigma_{y,\infty} - \sigma_{y,i}}{t_{\mathrm{age}}} - \text{(rejuvenation from yielding)}

   This captures physical aging in glassy materials.

**3. Tensorial EPM for General Flow**

   Store the full deviatoric stress tensor :math:`[\sigma_{xx}, \sigma_{yy}, \sigma_{xy}]`
   at each site. Enables:

   - Normal stress predictions (N₁, N₂)
   - Anisotropic yield criteria (Hill, von Mises)
   - More accurate flow instability predictions

   See :doc:`tensorial_epm` for details.

**4. Shear Banding / Nonlocal Models**

   Retain explicit gradient direction (y-dependence in planar Couette):

   - Track velocity field :math:`v_x(y, t)`
   - Add stress diffusion :math:`D_\sigma \nabla^2 \sigma` to regularize bands
   - Couple to fluidity field for nonlocal relaxation

   Essential for predicting band width and transient banding dynamics.


Advanced Usage: Multi-Protocol Joint Fitting
---------------------------------------------

For robust parameter estimation, fit EPM to multiple protocols simultaneously:

.. code-block:: python

   from rheojax.models import LatticeEPM
   from rheojax.pipeline import Pipeline
   import numpy as np

   # Load multi-protocol data
   flow_data = RheoData.from_csv("flow_curve.csv", test_mode="flow_curve")
   startup_data = RheoData.from_csv("startup.csv", test_mode="startup")
   creep_data = RheoData.from_csv("creep.csv", test_mode="creep")

   # Create model with small lattice for fitting
   model = LatticeEPM(L=12, dt=0.01)

   # Joint fitting with protocol weights
   model.fit_multi_protocol(
       [flow_data, startup_data, creep_data],
       weights=[1.0, 2.0, 1.0],  # Emphasize startup
       max_iter=500
   )

   # Bayesian with multi-protocol likelihood
   result = model.fit_bayesian_multi_protocol(
       [flow_data, startup_data, creep_data],
       weights=[1.0, 2.0, 1.0],
       num_warmup=1000,
       num_samples=2000,
       num_chains=4,
       seed=42
   )

   # Validate on held-out protocol (LAOS)
   laos_data = RheoData.from_csv("laos.csv", test_mode="oscillation")
   laos_pred = model.predict(laos_data)


Parameter Sensitivity from Bayesian Posteriors
----------------------------------------------

Use Bayesian posteriors to understand parameter identifiability and correlations:

.. code-block:: python

   import arviz as az

   # After fit_bayesian()
   idata = result.to_arviz()

   # Pair plot reveals correlations
   az.plot_pair(idata, var_names=["sigma_c_mean", "sigma_c_std", "tau_pl"])

   # Sensitivity: which parameters are well-constrained?
   summary = az.summary(idata, hdi_prob=0.95)
   for param in summary.index:
       cv = summary.loc[param, "sd"] / summary.loc[param, "mean"]
       print(f"{param}: CV = {cv:.2f} ({'well-constrained' if cv < 0.3 else 'uncertain'})")

   # Prior-posterior comparison (diagnostic for informativeness)
   az.plot_dist_comparison(idata, var_names=["sigma_c_mean"])


Disorder Estimation from Stress Overshoot
-----------------------------------------

The stress overshoot in startup shear contains information about disorder strength:

**Physical basis:**

- Higher disorder → larger overshoot amplitude
- Overshoot position (peak strain) → τ_pl × γ̇
- Overshoot variability across runs → σ_c_std

.. code-block:: python

   def estimate_disorder_from_overshoot(gamma_dot_range, n_repeats=10):
       """Estimate disorder from overshoot variability.

       Higher variability across runs indicates stronger disorder.
       """
       model = LatticeEPM(L=16, dt=0.01)
       overshoot_cv = []  # Coefficient of variation

       for gamma_dot in gamma_dot_range:
           peaks = []
           for seed in range(n_repeats):
               t = np.linspace(0, 50/gamma_dot, 500)
               stress = model.predict(t, test_mode="startup",
                                       gamma_dot=gamma_dot, seed=seed)
               peaks.append(np.max(stress))

           cv = np.std(peaks) / np.mean(peaks)
           overshoot_cv.append(cv)

       # High CV indicates strong disorder
       return np.array(overshoot_cv)


Full BayesianPipeline Workflow
------------------------------

Complete pipeline from data loading to uncertainty quantification:

.. code-block:: python

   from rheojax.pipeline.bayesian import BayesianPipeline
   from rheojax.models import LatticeEPM

   # Initialize pipeline
   pipeline = BayesianPipeline()

   # Load and fit
   (pipeline
       .load("experimental_data.csv",
             x_col="gamma_dot",
             y_col="stress",
             test_mode="flow_curve")
       .model(LatticeEPM, L=12, dt=0.01)  # Small lattice for fitting
       .fit_nlsq(max_iter=500)            # Fast point estimate
       .fit_bayesian(                     # Full Bayesian
           num_warmup=1000,
           num_samples=2000,
           num_chains=4,
           seed=42
       )
   )

   # Diagnostics (ArviZ integration)
   (pipeline
       .plot_trace()                      # MCMC convergence
       .plot_pair(divergences=True)       # Parameter correlations
       .plot_forest(hdi_prob=0.95)        # Credible intervals
       .plot_energy()                     # HMC diagnostics
   )

   # Check convergence
   diagnostics = pipeline.get_diagnostics()
   assert max(diagnostics["r_hat"].values()) < 1.05, "R-hat too high"
   assert min(diagnostics["ess"].values()) > 400, "ESS too low"
   assert diagnostics["divergences"] < 0.01, "Too many divergences"

   # Save results
   (pipeline
       .save_results("epm_fit.hdf5")      # Full posteriors
       .save_summary("epm_summary.csv")   # Parameter estimates
   )

   # Production prediction at higher resolution
   model_prod = LatticeEPM(L=64, dt=0.01)
   model_prod.params = pipeline.model.params.copy()
   gamma_dot_fine = np.logspace(-2, 1, 100)
   stress_pred = model_prod.predict(gamma_dot_fine, test_mode="flow_curve")
