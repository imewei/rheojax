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

API Reference
-------------

.. autoclass:: rheojax.models.epm.lattice.LatticeEPM
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: rheojax.visualization.epm_plots.plot_lattice_fields
