Generalized Maxwell Model — Canonical Equations
================================================

This document collects the canonical equations for the Generalized Maxwell Model
(GMM / Prony series) as used in linear viscoelasticity. All equations below are
standard results derivable from the constitutive relation of a parallel
arrangement of N Maxwell elements plus an equilibrium spring.

References
----------
- Ferry, J. D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed. Wiley.
- Tschoegl, N. W. (1989). *The Phenomenological Theory of Linear Viscoelastic Behavior*. Springer.
- Baumgaertel, M. & Winter, H. H. (1989). Determination of discrete relaxation
  and retardation time spectra from dynamic mechanical data. *Rheol. Acta*, 28, 511–519.
- Honerkamp, J. & Weese, J. (1989). Determination of the relaxation spectrum by
  a regularization method. *Macromolecules*, 22(11), 4372–4377.
- Park, S. W. & Schapery, R. A. (1999). Methods of interconversion between
  linear viscoelastic material functions. Part I. *Int. J. Solids Struct.*, 36, 1653–1675.


1. Relaxation Modulus — Prony Series
------------------------------------

The N-mode discrete relaxation spectrum:

.. math::

   G(t) = G_e + \sum_{i=1}^{N} G_i \, \exp\!\left(-\frac{t}{\tau_i}\right)

where:

- :math:`G_e \geq 0` is the equilibrium (rubbery) modulus. For liquids,
  :math:`G_e = 0`; for crosslinked solids :math:`G_e > 0`.
- :math:`G_i > 0` are the relaxation strengths (mode amplitudes).
- :math:`\tau_i > 0` are the relaxation times.
- :math:`G_0 = G_e + \sum_i G_i` is the instantaneous (glassy) modulus at
  :math:`t = 0^+`.


2. Storage Modulus
------------------

Obtained by one-sided Fourier cosine transform of :math:`G(t)`:

.. math::

   G'(\omega) = G_e + \sum_{i=1}^{N} G_i \, \frac{\omega^2 \tau_i^2}
                {1 + \omega^2 \tau_i^2}

Limiting behavior:

- :math:`\omega \to 0`: :math:`G'(\omega) \to G_e`
- :math:`\omega \to \infty`: :math:`G'(\omega) \to G_0 = G_e + \sum_i G_i`


3. Loss Modulus
---------------

Obtained by one-sided Fourier sine transform of :math:`G(t)`:

.. math::

   G''(\omega) = \sum_{i=1}^{N} G_i \, \frac{\omega \tau_i}
                 {1 + \omega^2 \tau_i^2}

Each mode contributes a Debye peak at :math:`\omega = 1/\tau_i` with maximum
value :math:`G_i / 2`.


4. Complex Modulus and Complex Viscosity
----------------------------------------

The complex modulus is:

.. math::

   G^*(\omega) = G'(\omega) + i\, G''(\omega)
               = G_e + \sum_{i=1}^{N} G_i \, \frac{i\omega\tau_i}
                 {1 + i\omega\tau_i}

The complex viscosity :math:`\eta^*(\omega) = G^*(\omega) / (i\omega)`:

.. math::

   \eta^*(\omega) = \sum_{i=1}^{N} \frac{G_i \tau_i}{1 + i\omega\tau_i}
                  + \frac{G_e}{i\omega}

For liquids (:math:`G_e = 0`):

.. math::

   \eta^*(\omega) = \sum_{i=1}^{N} \frac{G_i \tau_i}{1 + i\omega\tau_i}

Components:

.. math::

   \eta'(\omega) = \sum_{i=1}^{N} \frac{G_i \tau_i}{1 + \omega^2 \tau_i^2}

.. math::

   \eta''(\omega) = \frac{G_e}{\omega} + \sum_{i=1}^{N}
                    \frac{G_i \omega \tau_i^2}{1 + \omega^2 \tau_i^2}


5. Creep Compliance
-------------------

The creep compliance :math:`J(t)` is the response to a step stress
:math:`\sigma_0`. For the GMM, :math:`J(t)` is the inverse Laplace transform
relation to :math:`G(t)`:

.. math::

   s\,\hat{G}(s)\,s\,\hat{J}(s) = 1

where :math:`\hat{G}(s)` and :math:`\hat{J}(s)` are the Laplace transforms.

**Exact retardation spectrum (Tschoegl 1989, Ch. 4):**

For an N-mode GMM with :math:`G_e > 0`, the creep compliance takes the
retardation form:

.. math::

   J(t) = J_g + \sum_{j=1}^{N} J_j \left[1 - \exp\!\left(-\frac{t}
          {\lambda_j}\right)\right]

where :math:`J_g = 1/G_0` is the glassy compliance, :math:`\lambda_j` are
the retardation times (roots of the polynomial :math:`\sum_i G_i \prod_{k
\neq i} (1 + s\tau_k) = 0`), and :math:`J_j` are the retardation strengths.

For liquids (:math:`G_e = 0`), an additional viscous flow term appears:

.. math::

   J(t) = J_g + \sum_{j=1}^{N-1} J_j \left[1 - \exp\!\left(-\frac{t}
          {\lambda_j}\right)\right] + \frac{t}{\eta_0}

**Numerical approach (backward-Euler, Park & Schapery 1999):**

The GMM constitutive ODE for each mode under creep loading
:math:`\sigma(t) = \sigma_0 H(t)` is:

.. math::

   \frac{d\sigma_i}{dt} = -\frac{\sigma_i}{\tau_i} + G_i \dot\varepsilon

with total stress balance :math:`\sigma_0 = G_e\,\varepsilon + \sum_i \sigma_i`.
The backward-Euler update uses exponential integration for unconditional stability:

.. math::

   \alpha_i = \exp(-\Delta t / \tau_i), \quad
   \beta_i = G_i \tau_i (1 - \alpha_i) / \Delta t

.. math::

   \Delta\varepsilon = \frac{\sigma_0 - \sum_i \alpha_i \sigma_i^n}
                       {G_e + \sum_i \beta_i}


6. Zero-Shear Viscosity
------------------------

.. math::

   \eta_0 = \int_0^\infty G(t)\,dt = \sum_{i=1}^{N} G_i \tau_i

This equals :math:`\lim_{\omega\to 0} \eta'(\omega)` and is well-defined only
for liquids (:math:`G_e = 0`).


7. Stress Growth Coefficient (Startup Flow)
--------------------------------------------

Under constant shear rate :math:`\dot\gamma`, the stress growth coefficient is:

.. math::

   \eta^+(t) = \frac{\sigma(t)}{\dot\gamma}
             = \sum_{i=1}^{N} G_i \tau_i \left[1 - \exp\!\left(-\frac{t}
               {\tau_i}\right)\right]

with :math:`\lim_{t\to\infty} \eta^+(t) = \eta_0`.


8. Fitting Methods for Spectrum Identification
-----------------------------------------------

**Non-Negative Least Squares (NNLS):**

Given fixed :math:`\tau_i` on a log-spaced grid, solve for :math:`G_i \geq 0`:

.. math::

   \min_{G_i \geq 0} \left\| \mathbf{y} - \mathbf{A}\,\mathbf{g} \right\|^2

where :math:`A_{kj} = \exp(-t_k/\tau_j)` for relaxation or the appropriate
kernel for :math:`G'(\omega)` / :math:`G''(\omega)`.

**Tikhonov Regularization (Honerkamp & Weese 1989):**

.. math::

   \min_{\mathbf{g}} \left\| \mathbf{y} - \mathbf{A}\,\mathbf{g} \right\|^2
   + \alpha \left\| \mathbf{L}\,\mathbf{g} \right\|^2

where :math:`\mathbf{L}` is typically a first- or second-order difference
operator and :math:`\alpha > 0` is the regularization parameter (chosen via
L-curve, GCV, or Bayesian evidence).

**Maximum Entropy (MEM):**

.. math::

   \min_{\mathbf{g}} \left\| \mathbf{y} - \mathbf{A}\,\mathbf{g} \right\|^2
   - \alpha \sum_j \left[ g_j \ln\!\left(\frac{g_j}{m_j}\right)
   - g_j + m_j \right]

where :math:`m_j` is a default model (flat prior). MEM naturally enforces
:math:`g_j > 0` and produces the smoothest spectrum consistent with the data.

**Two-Step NLSQ with Softmax Penalty (RheoJAX approach):**

Step 1: Fit with differentiable penalty encouraging :math:`G_i > 0`:

.. math::

   P(\mathbf{G}) = s \sum_i \ln\!\left(1 + \exp(-G_i/s)\right)

Step 2: If any :math:`G_i < 0`, refit with hard bounds :math:`G_i \geq 0`.

**Element Minimization (Baumgaertel & Winter 1989):**

Start from large N, iteratively reduce to find smallest N where
:math:`R^2 \geq R^2_\text{threshold}`. Uses warm-start from N+1 solution.


9. Discrete vs. Continuous Relaxation Spectrum
----------------------------------------------

The continuous spectrum :math:`H(\tau)` generalizes the discrete Prony series:

.. math::

   G(t) = G_e + \int_0^\infty H(\tau)\,\exp\!\left(-\frac{t}{\tau}\right)
          \,\frac{d\tau}{\tau}

.. math::

   G'(\omega) = G_e + \int_0^\infty H(\tau)\,\frac{\omega^2\tau^2}
                {1 + \omega^2\tau^2}\,\frac{d\tau}{\tau}

.. math::

   G''(\omega) = \int_0^\infty H(\tau)\,\frac{\omega\tau}
                 {1 + \omega^2\tau^2}\,\frac{d\tau}{\tau}

The discrete spectrum is formally:

.. math::

   H(\tau) = \sum_{i=1}^{N} G_i\,\tau_i\,\delta(\tau - \tau_i)

Recovering :math:`H(\tau)` from data is an ill-posed inverse problem,
motivating the regularization approaches in Section 8.

The retardation spectrum :math:`L(\lambda)` plays the analogous role for
creep compliance:

.. math::

   J(t) = J_g + \int_0^\infty L(\lambda)\left[1 - \exp\!\left(-\frac{t}
          {\lambda}\right)\right]\frac{d\lambda}{\lambda} + \frac{t}{\eta_0}


10. Interconversion Identities
-------------------------------

Key relationships between material functions (Ferry 1980, Ch. 3-4):

.. math::

   G'(\omega) = \omega \int_0^\infty G(t)\,\sin(\omega t)\,dt

.. math::

   G''(\omega) = \omega \int_0^\infty G(t)\,\cos(\omega t)\,dt

.. math::

   |G^*(\omega)| = \sqrt{G'^2 + G''^2}

.. math::

   \tan\delta = \frac{G''}{G'}

.. math::

   |\eta^*(\omega)| = \frac{|G^*(\omega)|}{\omega}

The Cox-Merz rule (empirical, for linear polymers):

.. math::

   |\eta^*(\omega)| \approx \eta(\dot\gamma)\big|_{\dot\gamma = \omega}
