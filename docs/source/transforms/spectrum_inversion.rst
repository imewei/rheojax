.. _transform-spectrum-inversion:

SpectrumInversion
=================

Overview
--------

The :class:`rheojax.transforms.SpectrumInversion` transform recovers the **continuous
relaxation spectrum** :math:`H(\tau)` from measured dynamic moduli :math:`G'(\omega)`,
:math:`G''(\omega)` or relaxation modulus :math:`G(t)`. This is a classical **ill-posed
inverse problem** in rheology, requiring regularization for stable solutions.

**Key Capabilities:**

- **Tikhonov regularization:** Automatic parameter selection via GCV (Generalized
  Cross-Validation) or manual control
- **Maximum entropy method:** Information-theoretic approach preserving positivity
- **Dual-source input:** Works from oscillation data :math:`G^*(\omega)` or relaxation
  data :math:`G(t)`
- **Non-negative spectrum:** Physical constraint :math:`H(\tau) \ge 0` enforced

The continuous relaxation spectrum provides a material fingerprint that reveals the
distribution of relaxation mechanisms—from single-mode Maxwellian fluids (delta function)
to broad distributions in polymer melts and filled systems.


Mathematical Theory
-------------------

Continuous Spectrum Definition
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The continuous relaxation spectrum :math:`H(\tau)` is defined through the integral
representation:

.. math::

   G(t) = G_e + \int_{-\infty}^{\infty} H(\tau) \exp(-t/\tau) \, d(\ln \tau)

The dynamic moduli are related to :math:`H(\tau)` by the **kernel integrals**:

.. math::

   G'(\omega) = G_e + \int_{-\infty}^{\infty} H(\tau) \frac{\omega^2 \tau^2}{1 + \omega^2 \tau^2} \, d(\ln \tau)

.. math::

   G''(\omega) = \int_{-\infty}^{\infty} H(\tau) \frac{\omega \tau}{1 + \omega^2 \tau^2} \, d(\ln \tau)

**Physical interpretation:** :math:`H(\tau) d(\ln \tau)` is the modulus contribution from
relaxation mechanisms with times between :math:`\tau` and :math:`\tau + d\tau`.

The Inverse Problem
~~~~~~~~~~~~~~~~~~~~

Recovering :math:`H(\tau)` from data is an **ill-posed Fredholm integral equation of the
first kind**. Discretizing on a log-spaced :math:`\tau` grid gives:

.. math::

   \mathbf{b} = A \mathbf{H} + \boldsymbol{\varepsilon}

where :math:`A` is the kernel matrix, :math:`\mathbf{H} = [H(\tau_1), \ldots, H(\tau_M)]`,
and :math:`\boldsymbol{\varepsilon}` is noise.

**Why ill-posed?** The singular values of :math:`A` decay rapidly—small noise in
:math:`\mathbf{b}` maps to large oscillations in :math:`\mathbf{H}` unless regularized.

Tikhonov Regularization
~~~~~~~~~~~~~~~~~~~~~~~~~

Tikhonov regularization stabilizes the inversion by penalizing solution norm:

.. math::

   \min_{\mathbf{H}} \left\{ \|A \mathbf{H} - \mathbf{b}\|^2 + \lambda^2 \|L \mathbf{H}\|^2 \right\}

- :math:`\lambda` controls the **trade-off** between data fidelity and smoothness
- :math:`L = I` (zeroth order): penalizes large :math:`H` values
- :math:`L = D_1` (first order): penalizes gradient (promotes smoothness)

**Solution:** :math:`\mathbf{H} = (A^T A + \lambda^2 L^T L)^{-1} A^T \mathbf{b}`

GCV Selection of λ (Implementation Detail)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The **Generalized Cross-Validation** criterion selects :math:`\lambda` by minimizing the
predicted leave-one-out error without actually performing :math:`n` separate inversions:

.. math::

   \text{GCV}(\lambda) = \frac{\|A \mathbf{H}_\lambda - \mathbf{b}\|^2}
   {\left[\text{tr}(I - M(\lambda))\right]^2}

where :math:`M(\lambda) = A(A^T A + \lambda^2 L^T L)^{-1} A^T` is the **influence matrix**
(also called the hat matrix). The numerator is the residual sum of squares; the denominator
penalizes overfitting by measuring the effective number of free parameters.

**Physical intuition:** GCV balances two extremes:

- :math:`\lambda \to 0`: Zero regularization. :math:`\mathbf{H}` fits the noise. Residual
  is small, but the denominator collapses (all data points are "fit"), so GCV → ∞.
- :math:`\lambda \to \infty`: Infinite regularization. :math:`\mathbf{H} \to 0`. Residual
  is large. GCV → ∞ again.
- **Optimal λ:** The minimum of the GCV curve sits between these extremes—enough
  regularization to suppress noise amplification, but not so much that physical features
  are lost.

**Fast SVD path (L = I):** When the regularization matrix is the identity (zeroth-order
Tikhonov, which is the RheoJAX default), the implementation uses the SVD of :math:`A`:

.. math::

   A = U \Sigma V^T

to express all quantities analytically via the **filter factors**
:math:`f_j = \sigma_j^2 / (\sigma_j^2 + \lambda^2)`:

.. math::

   \|\text{residual}\|^2 = \sum_j (1 - f_j)^2 (U^T \mathbf{b})_j^2 + \|\mathbf{b}_\perp\|^2

.. math::

   \text{tr}(I - M) = n - \sum_j f_j

where :math:`\mathbf{b}_\perp = (I - U U^T)\mathbf{b}` is the component of the data
orthogonal to the range of :math:`A` (constant across all :math:`\lambda`). The SVD is
computed once at :math:`O(\min(n, m)^2 \max(n, m))` cost, after which each of the 50
candidate :math:`\lambda` values (log-spaced from :math:`10^{-6}` to :math:`10^4`) is
evaluated in :math:`O(\min(n, m))` time—a significant speedup over the general case.

**General case (L ≠ I):** When a non-identity regularization matrix is used (e.g.,
first-derivative operator for smoothness), the fast SVD path is not available. Instead,
the implementation solves the normal equations for each candidate :math:`\lambda` via:

.. math::

   \mathbf{H}_\lambda = (A^T A + \lambda^2 L^T L)^{-1} A^T \mathbf{b}

and computes the influence trace as :math:`\text{tr}((A^T A + \lambda^2 L^T L)^{-1} A^T A)`.
This costs :math:`O(m^3)` per candidate but avoids forming the full :math:`n \times n`
influence matrix.

**Diagnostic:** The returned ``regularization_param`` in :class:`SpectrumResult` allows
you to verify the selected :math:`\lambda`. If the spectrum looks over-smoothed, you can
override with a smaller manual value; if it oscillates, use a larger one.

Maximum Entropy Method
~~~~~~~~~~~~~~~~~~~~~~~

The maximum entropy approach maximizes the **Shannon entropy** of :math:`H(\tau)` subject
to a data fidelity constraint:

.. math::

   \max_{\mathbf{H}} \left\{ S = -\sum_i H_i \ln(H_i / m_i) \right\}
   \quad \text{subject to} \quad \chi^2 \le \chi^2_{\text{target}}

where :math:`m_i` is a prior model (default: uniform). The solution is found via iterative
**multiplicative updates** (Bryan's algorithm):

.. math::

   H_i^{(k+1)} = H_i^{(k)} \exp\!\left( -\lambda \frac{\partial \chi^2}{\partial H_i} \Big/ \left(1 + \lambda \left|\frac{\partial \chi^2}{\partial H_i}\right|\right) \right)

**Advantages over Tikhonov:** Automatically enforces :math:`H(\tau) > 0`, produces
maximally non-committal spectra, and avoids oscillatory artifacts.


Parameters
----------

.. list-table:: SpectrumInversion Parameters
   :header-rows: 1
   :widths: 20 18 14 48

   * - Parameter
     - Type
     - Default
     - Description
   * - ``method``
     - str
     - ``"tikhonov"``
     - Inversion method: ``"tikhonov"`` or ``"max_entropy"``.
   * - ``n_tau``
     - int
     - ``100``
     - Number of :math:`\tau` points in the output spectrum.
   * - ``tau_range``
     - tuple | None
     - ``None``
     - Explicit :math:`(\tau_{\min}, \tau_{\max})`. Auto-detected if ``None``.
   * - ``regularization``
     - float | None
     - ``None``
     - Manual :math:`\lambda`. If ``None``, auto-selected via GCV.
   * - ``source``
     - str
     - ``"oscillation"``
     - Input data type: ``"oscillation"`` or ``"relaxation"``.
   * - ``G_e``
     - float
     - ``0.0``
     - Equilibrium modulus (Pa). Set to 0 for liquids.

Parameter Selection Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Method choice:**

- **Tikhonov:** Default, fast, well-understood. Best for clean data with moderate noise.
- **Maximum entropy:** Better for noisy data and when positivity is critical. Slightly
  slower due to iterative solution.

**n_tau:**

- **50–100:** Standard (adequate for most polymer systems)
- **200+:** High resolution (for resolving closely-spaced relaxation modes)
- **<50:** Coarse (fast screening)

**tau_range auto-detection logic:**

When ``tau_range=None`` (default), the :math:`\tau` grid boundaries are inferred from the
input data with a **1-decade safety margin** on each side:

*Oscillation source* (``source="oscillation"``):

The relationship :math:`\omega \sim 1/\tau` maps the frequency window to the relaxation
time window with an inversion:

.. math::

   \tau_{\min} = \frac{1}{10 \, \omega_{\max}}, \qquad
   \tau_{\max} = \frac{10}{\omega_{\min}}

For example, a frequency sweep from :math:`\omega = 0.01` to :math:`100` rad/s produces
:math:`\tau \in [10^{-3}, 10^{3}]` s—six decades, extending one decade beyond the data
on each side.

*Relaxation source* (``source="relaxation"``):

The time axis maps directly to relaxation times:

.. math::

   \tau_{\min} = \frac{t_{\min}}{10}, \qquad
   \tau_{\max} = 10 \, t_{\max}

For example, relaxation data from :math:`t = 0.001` to :math:`100` s produces
:math:`\tau \in [10^{-4}, 10^{3}]` s.

**Why the 10× safety margin?** The kernel functions
(:math:`\omega^2\tau^2/(1+\omega^2\tau^2)` for :math:`G'`,
:math:`\omega\tau/(1+\omega^2\tau^2)` for :math:`G''`) have significant sensitivity
beyond the strict :math:`1/\omega` boundary. A mode at :math:`\tau = 0.1/\omega_{\max}`
still contributes ~1% to :math:`G'(\omega_{\max})`, which is resolvable with good SNR.
Without the margin, edge modes are truncated and the spectrum appears artificially narrow.

**When to override:** Use explicit ``tau_range`` when:

- Data is noisy at the frequency extremes (tighten range to avoid fitting noise)
- You know the material has no modes outside a specific range (e.g., single-peak gel)
- You want to zoom into a specific part of the spectrum for higher resolution


Input / Output Specifications
-----------------------------

- **Input (oscillation)**: :class:`RheoData` with ``x`` = :math:`\omega` (rad/s),
  ``y`` = complex :math:`G^*` or ``(N, 2)`` array :math:`[G', G'']`
- **Input (relaxation)**: :class:`RheoData` with ``x`` = time (s), ``y`` = :math:`G(t)` (Pa)
- **Output**: :class:`RheoData` with ``x`` = :math:`\tau` (s), ``y`` = :math:`H(\tau)` (Pa)

Metadata includes :class:`SpectrumResult` with ``regularization_param``, ``residual_norm``,
and ``method``.


Usage
-----

From Oscillatory Data
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SpectrumInversion
   from rheojax.core.data import RheoData
   import numpy as np

   # Frequency sweep data
   omega = np.logspace(-2, 2, 50)
   G_prime = 1e4 * omega**2 / (1 + omega**2)  # Maxwell model
   G_double_prime = 1e4 * omega / (1 + omega**2)
   G_star = G_prime + 1j * G_double_prime

   data = RheoData(x=omega, y=G_star, metadata={'test_mode': 'oscillation'})

   # Recover relaxation spectrum
   inv = SpectrumInversion(method="tikhonov", n_tau=100)
   spectrum_data, info = inv.transform(data)

   tau = spectrum_data.x
   H_tau = spectrum_data.y

   # For a single Maxwell element, expect a peak at τ = 1 s
   peak_tau = tau[np.argmax(H_tau)]
   print(f"Peak relaxation time: {peak_tau:.3f} s")
   print(f"Regularization λ: {info['spectrum_result'].regularization_param:.4g}")

From Relaxation Data
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SpectrumInversion

   # Stress relaxation data
   t = np.logspace(-3, 2, 100)
   G_t = 5e3 * np.exp(-t / 0.1) + 2e3 * np.exp(-t / 10.0)

   relax_data = RheoData(x=t, y=G_t, metadata={'test_mode': 'relaxation'})

   inv = SpectrumInversion(source="relaxation", method="max_entropy", n_tau=80)
   spectrum_data, info = inv.transform(relax_data)

   # Expect two peaks at τ ≈ 0.1 s and τ ≈ 10 s
   import matplotlib.pyplot as plt
   plt.semilogx(spectrum_data.x, spectrum_data.y)
   plt.xlabel(r'$\tau$ (s)')
   plt.ylabel(r'$H(\tau)$ (Pa)')
   plt.title('Relaxation Spectrum')

Comparing Methods
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SpectrumInversion

   # Compare Tikhonov and MaxEnt on same data
   tik = SpectrumInversion(method="tikhonov", n_tau=100)
   mem = SpectrumInversion(method="max_entropy", n_tau=100)

   spec_tik, info_tik = tik.transform(osc_data)
   spec_mem, info_mem = mem.transform(osc_data)

   print(f"Tikhonov residual: {info_tik['spectrum_result'].residual_norm:.4g}")
   print(f"MaxEnt residual:   {info_mem['spectrum_result'].residual_norm:.4g}")


Validation and Quality Control
-------------------------------

Diagnostic Checks
~~~~~~~~~~~~~~~~~~

**1. Residual norm:** Measures data fidelity

- Low residual + smooth spectrum → good inversion
- Low residual + oscillatory spectrum → under-regularized (reduce :math:`\lambda`)
- High residual + smooth spectrum → over-regularized (increase :math:`n_\tau` or reduce :math:`\lambda`)

**2. Back-calculation test:**

.. code-block:: python

   # Verify: can H(τ) reproduce the original data?
   from rheojax.transforms import PronyConversion
   # Discretize H(τ) as Prony modes
   G_i = H_tau * np.diff(np.log(tau), append=np.log(tau[-1]) - np.log(tau[-2]))
   prony = PronyConversion(direction="time_to_freq")
   # ... compare reconstructed G'(ω), G''(ω) with original

Common Failure Modes
~~~~~~~~~~~~~~~~~~~~~

**1. Spurious oscillations:**

- **Cause:** Insufficient regularization (:math:`\lambda` too small)
- **Fix:** Increase ``regularization`` or switch to ``max_entropy``

**2. Over-smoothed spectrum:**

- **Cause:** Excessive regularization (:math:`\lambda` too large)
- **Fix:** Reduce ``regularization``, increase ``n_tau``

**3. Truncated spectrum:**

- **Cause:** ``tau_range`` does not span the full relaxation window
- **Fix:** Widen ``tau_range`` or let auto-detection handle it

**4. Negative H(τ) values:**

- **Cause:** Tikhonov can produce negative values before clamping
- **Fix:** Use ``max_entropy`` (inherently non-negative)


See Also
--------

- :doc:`prony_conversion` — Discrete Prony series (complementary parametric approach)
- :doc:`lve_envelope` — Uses Prony/spectrum parameters for startup prediction
- :doc:`fft` — Non-parametric time↔frequency interconversion
- :doc:`../models/gmm/generalized_maxwell` — Multi-mode Maxwell model (discrete spectrum)
- :doc:`../models/fractional/fractional_maxwell_model` — Power-law spectrum (fractional model)


API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.SpectrumInversion`


References
----------

1. Honerkamp, J. & Weese, J. (1993). "A nonlinear regularization method for the
   calculation of relaxation spectra." *Rheol. Acta*, 32, 65–73.
   DOI: `10.1007/BF00396678 <https://doi.org/10.1007/BF00396678>`_

2. Baumgaertel, M. & Winter, H.H. (1989). "Determination of discrete relaxation
   and retardation time spectra from dynamic mechanical data." *Rheol. Acta*,
   28, 511–519. DOI: `10.1007/BF01332922 <https://doi.org/10.1007/BF01332922>`_

3. Davies, A.R. & Anderssen, R.S. (1997). "Sampling localization in determining
   the relaxation spectrum." *J. Non-Newtonian Fluid Mech.*, 73, 163–179.
   DOI: `10.1016/S0377-0257(97)00056-6 <https://doi.org/10.1016/S0377-0257(97)00056-6>`_

4. Hansen, P.C. (1992). "Analysis of discrete ill-posed problems by means of
   the L-curve." *SIAM Review*, 34, 561–580.
   DOI: `10.1137/1034115 <https://doi.org/10.1137/1034115>`_

5. Bryan, R.K. (1990). "Maximum entropy analysis of oversampled data problems."
   *Eur. Biophys. J.*, 18, 165–174.
   DOI: `10.1007/BF02427376 <https://doi.org/10.1007/BF02427376>`_

6. Tschoegl, N.W. (1989). *The Phenomenological Theory of Linear Viscoelastic
   Behavior: An Introduction*. Springer-Verlag.
   DOI: `10.1007/978-3-642-73602-5 <https://doi.org/10.1007/978-3-642-73602-5>`_
