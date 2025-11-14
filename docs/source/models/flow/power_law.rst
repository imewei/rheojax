.. _model-power-law:

Power-Law (Ostwald–de Waele)
============================

Quick Reference
---------------

**Use when:** Straight lines on log-log plots, no plateaus visible, mid-rate behavior
**Parameters:** 2 (K, n)
**Key equation:** :math:`\eta(\dot{\gamma}) = K \dot{\gamma}^{n-1}`, n < 1 thinning, n > 1 thickening
**Test modes:** Flow (steady shear)
**Material examples:** Polymer melts (n≈0.3-0.5), paints, ketchup, inks, biological fluids

Overview
--------

The :class:`rheojax.models.PowerLaw` equation is the simplest description of
non-Newtonian flow. It assumes shear stress scales with a power of shear rate and lacks
finite plateaus. Use it as a quick diagnostic for mid-rate behavior or whenever the data
follow straight lines on log–log plots.

**Physical Basis**: The Power-Law model captures the structural alignment of molecules,
particles, or polymer chains in the flow direction, which reduces resistance to flow
(shear-thinning, n < 1) or induces hydrodynamic clustering and jamming (shear-thickening,
n > 1). It represents purely empirical behavior without a microstructural foundation.

**Key Limitation**: Valid **only in the intermediate shear rate range** where plateaus are
absent. Cannot predict zero-shear viscosity (η₀) or infinite-shear viscosity (η∞).

Equations
---------

.. math::

   \tau = K \dot{\gamma}^{n}, \qquad
   \eta(\dot{\gamma}) = K \dot{\gamma}^{n-1}

**Physical Interpretation**:

- **Shear stress**: Nonlinear relationship between stress and shear rate
- **Viscosity**: Monotonically decreases (n < 1) or increases (n > 1) without bounds
- **Log-log linearity**: :math:`\log \tau = \log K + n \log \dot{\gamma}` (straight line with slope n)

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 30 18 52

   * - Name
     - Units
     - Description / Constraints
   * - ``K``
     - Pa·s\ :sup:`n`
     - Consistency index; > 0. Sets magnitude of stress at :math:`\dot{\gamma}=1` s⁻¹.
       Related to material viscosity and molecular weight.
   * - ``n``
     - –
     - Flow index; < 1 shear-thinning, > 1 thickening, = 1 reduces to Newtonian of
       viscosity ``K``. Controls rate of viscosity change with shear rate.

Non-Newtonian Behavior
----------------------

**Shear-Thinning (Pseudoplastic, n < 1)**

Viscosity **decreases** with increasing shear rate. Typical values:

- **Polymer melts**: n ≈ 0.3–0.5 (PE, PP, PS in intermediate shear range)
- **Paints and coatings**: n ≈ 0.4–0.6 (easy spreading at high γ̇)
- **Food products**: n ≈ 0.5–0.7 (sauces, yogurt, fruit purees)
- **Inks**: n ≈ 0.3–0.5 (shear-thinning for printing)
- **Biological fluids**: n ≈ 0.4–0.8 (blood at intermediate shear rates)

**Mechanism**: Molecular/particle alignment along flow direction reduces resistance.
Polymer chains disentangle, particle aggregates break up, or anisotropic structures orient
with flow.

**Shear-Thickening (Dilatant, n > 1)**

Viscosity **increases** with increasing shear rate. Typical values:

- **Concentrated suspensions**: n ≈ 1.1–1.5 (corn starch >40 wt%, silica >50 vol%)
- **Particle dispersions**: n ≈ 1.1–1.3 near jamming transition

**Mechanism**: Particle jamming, hydrodynamic clustering, or formation of force chains
under shear. Discontinuous shear-thickening (DST) shows abrupt viscosity jumps.

Material Examples with Parameters
----------------------------------

**Polyethylene Melt (LDPE, 190°C)**

- n = 0.35 (strong shear-thinning)
- K = 8,000 Pa·s\ :sup:`n`
- Valid range: 1–1000 s⁻¹
- Zero-shear plateau below 0.1 s⁻¹ requires Carreau/Cross

**Tomato Ketchup**

- n = 0.23 (very shear-thinning)
- K = 15 Pa·s\ :sup:`n`
- Valid range: 10–500 s⁻¹
- Yield stress (≈10 Pa) requires Herschel-Bulkley below 1 s⁻¹

**Corn Starch Suspension (45 wt%)**

- n = 1.3 (shear-thickening)
- K = 0.05 Pa·s\ :sup:`n`
- Valid range: 10–300 s⁻¹
- Discontinuous jump at γ̇ ≈ 100 s⁻¹ not captured

**Water-Based Latex Paint**

- n = 0.52
- K = 2.5 Pa·s\ :sup:`n`
- Valid range: 1–100 s⁻¹
- Both plateaus exist but Power-Law fits mid-range well

Model Selection: When to Use Power-Law
---------------------------------------

**✓ Recommended When:**

- Log-log plot of η vs. γ̇ shows **straight line** over 1.5–2 decades
- **No clear plateaus** in measured shear rate range
- **Quick characterization** needed (2 parameters)
- Data spans **intermediate shear rates** only (1–1000 s⁻¹ typical)
- Comparative studies where consistency index K reflects material differences

**✗ Avoid When:**

- **Zero-shear plateau** visible at low γ̇ → Use :doc:`carreau` or :doc:`cross`
- **Yield stress** present → Use :doc:`bingham` or :doc:`herschel_bulkley`
- Data spans **>3 decades** with visible curvature at both ends
- **Extrapolation** needed beyond measured range (unphysical predictions)

**Model Hierarchy**:

1. Power-Law (2 params) — simplest, limited range
2. Carreau/Cross (4 params) — adds plateaus, full flow curve
3. Herschel-Bulkley (3 params) — adds yield stress to Power-Law

Usage
-----

.. code-block:: python

   import jax.numpy as jnp
   from rheojax.models import PowerLaw

   # Log-spaced shear rates (typical rotational rheometer range)
   gamma_dot = jnp.logspace(-2, 3, 90)
   tau_data = stress_curve(gamma_dot)  # Measured shear stress

   # Initialize and fit
   model = PowerLaw(K=2.0, n=0.6)
   model.fit(gamma_dot, tau_data, loss="logcosh")  # Robust to outliers

   # Predict and analyze
   tau_pred = model.predict(gamma_dot)
   eta_pred = tau_pred / gamma_dot

   # Extract fitted parameters
   K_fit = model.parameters.get_value('K')
   n_fit = model.parameters.get_value('n')

   print(f"Consistency index: {K_fit:.3e} Pa·s^{n_fit:.3f}")
   if n_fit < 1:
       print("Shear-thinning behavior")
   elif n_fit > 1:
       print("Shear-thickening behavior")

Experimental Protocol Recommendations
--------------------------------------

**Controlled Shear Rate (CSR) Test**

1. **Logarithmic shear rate sweep**: 0.01–1000 s⁻¹ (or narrower if plateaus visible)
2. **Points per decade**: 8–12 for accurate slope determination
3. **Equilibration time**: 10–60 s per point (longer for structured fluids)
4. **Pre-shear**: Apply γ̇ = 100 s⁻¹ for 60 s to erase shear history

**Controlled Shear Stress (CSS) Test** (alternative for yield stress materials)

- Better for low-γ̇ region where CSR may slip
- Stress range: 0.1–1000 Pa (material dependent)

**Log-Log Analysis for Parameter Estimation**

From linear regression on :math:`\log \tau = \log K + n \log \dot{\gamma}`:

.. code-block:: python

   import numpy as np

   # Linear fit in log-log space
   log_gamma = np.log(gamma_dot)
   log_tau = np.log(tau_data)
   coeffs = np.polyfit(log_gamma, log_tau, 1)

   n_initial = coeffs[0]  # Slope = n
   K_initial = np.exp(coeffs[1])  # Intercept = log(K)

   # Use as initial guess for nonlinear fit
   model.parameters.set_value('n', n_initial)
   model.parameters.set_value('K', K_initial)

Fitting Strategies
------------------

**1. Log-Space Fitting** (preferred for Power-Law)

.. code-block:: python

   # Fit in log-space to weight all decades equally
   log_tau = np.log(tau_data)
   log_tau_pred = np.log(model.predict(gamma_dot))
   residuals = log_tau - log_tau_pred

**2. Weighted Least Squares**

.. code-block:: python

   # Weight by inverse variance if noise is heteroscedastic
   weights = 1.0 / tau_data  # Proportional noise
   model.fit(gamma_dot, tau_data, sample_weight=weights)

**3. Robust Loss Functions**

.. code-block:: python

   # Huber loss: robust to outliers from wall slip or instrument limits
   model.fit(gamma_dot, tau_data, loss="huber")

Tips & Pitfalls
---------------

**Numerical Stability**

- **Clamp ``gamma_dot``** away from zero to prevent numerical issues when computing
  ``eta(γ̇) = K γ̇^(n-1)``. Use minimum γ̇ ≥ 10⁻⁴ s⁻¹.
- For n < 0.3, viscosity diverges rapidly at low γ̇; truncate data below instrument limits.
- For n > 1.5, viscosity growth at high γ̇ may exceed instrument torque; check transducer
  limits.

**Physical Constraints**

- Apply only within the rate window where the slope remains constant; there are no
  intrinsic plateaus.
- **Extrapolation is dangerous**: η → ∞ as γ̇ → 0 for n < 1 (unphysical).
- Use log residuals to avoid overweighting high-shear stresses.

**Model Upgrade Criteria**

- Upgrade to :doc:`carreau` or :doc:`cross` if experimental evidence shows Newtonian
  plateaus at low or high γ̇.
- Add yield stress via :doc:`herschel_bulkley` if τ vs γ̇ has positive intercept.
- Combine with :doc:`../../transforms/owchirp` for fast estimation of ``n`` via LAOS
  broadband sweeps.

**Common Mistakes**

- Fitting full flow curve including plateaus → poor fit at extremes
- Using Power-Law for process design outside measured γ̇ range
- Ignoring time-dependent effects (thixotropy) → n varies with shear history

Industrial Applications
-----------------------

**Polymer Processing**

- **Extrusion**: n controls die swell and pressure drop (lower n → higher swell)
- **Injection molding**: K affects filling time and cavity pressure
- **Fiber spinning**: Shear-thinning (n < 1) enables high draw ratios

**Coatings and Paints**

- **Brushing/rolling**: High-γ̇ shear-thinning (n ≈ 0.5) ensures easy application
- **Leveling**: Viscosity recovery after application (time-dependent, not captured)
- **Sagging**: Low-γ̇ behavior critical (requires Carreau with η₀)

**Food Industry**

- **Pumping and mixing**: K determines energy consumption
- **Texture perception**: n correlates with mouthfeel (n ≈ 0.3 = thick, n ≈ 0.7 = thin)

**Oil and Gas**

- **Drilling mud**: Shear-thinning (n ≈ 0.4–0.6) during circulation, structure recovery at
  rest (thixotropic)

References
----------

- W. Ostwald, "Über die Geschwindigkeitsfunktion der Viskosität disperser Systeme,"
  *Kolloid-Zeitschrift* 36, 99–117 (1925).
- H. de Waele, "Viscosity-temperature relation and the flow law of the oil," *Oil & Gas
  J.* 28, 146–147 (1929).
- R.G. Larson, *Constitutive Equations for Polymer Melts and Solutions*, Butterworths
  (1988).
- H.A. Barnes et al., *An Introduction to Rheology*, Elsevier (1989).
- J. Mewis and N.J. Wagner, *Colloidal Suspension Rheology*, Cambridge (2012).
- C.W. Macosko, *Rheology: Principles, Measurements, and Applications*, Wiley (1994).

See also
--------

- :doc:`carreau` and :doc:`cross` — introduce finite plateaus beyond the power-law regime.
- :doc:`bingham` and :doc:`herschel_bulkley` — add yield stresses on top of the scaling law.
- :doc:`../fractional/fractional_poynting_thomson` — captures similar power-law slopes in
  oscillatory measurements.
- :doc:`../../transforms/mutation_number` — detect when structural evolution invalidates
  steady-state power-law fits.
- :doc:`../../examples/transforms/05-flow-power-law` — notebook showing slope extraction
  from steady shear curves.
