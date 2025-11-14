.. _model-carreau-yasuda:

Carreau–Yasuda Model
====================

Quick Reference
---------------

**Use when:** Abrupt viscosity transitions, sharp changes between plateaus
**Parameters:** 5 (η₀, η∞, λ, n, a)
**Key equation:** :math:`\eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^{a}]^{(n-1)/a}`
**Test modes:** Flow (steady shear)
**Material examples:** Wormlike micelles, highly filled polymers, materials with sharp transitions

Overview
--------

The :class:`rheojax.models.CarreauYasuda` model extends the Carreau form by adding a
Yasuda exponent :math:`a` that controls transition sharpness. It is particularly useful
for fluids that exhibit abrupt changes in viscosity, such as wormlike micelles or highly
filled polymers.

Equations
---------

.. math::

   \eta(\dot{\gamma}) = \eta_{\infty} + (\eta_0 - \eta_{\infty})
       \left[1 + (\lambda \dot{\gamma})^{a}\right]^{\frac{n-1}{a}}

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 20 20 60

   * - Name
     - Units
     - Description / Constraints
   * - ``eta_0``
     - Pa·s
     - Zero-shear viscosity; > 0, typically ≥ ``eta_inf``.
   * - ``eta_inf``
     - Pa·s
     - Infinite-shear viscosity; ≥ 0.
   * - ``lambda``
     - s
     - Characteristic time constant; > 0.
   * - ``n``
     - –
     - Power index; < 1 for thinning, > 1 for thickening.
   * - ``a``
     - –
     - Transition sharpness; :math:`a = 2` reduces to Carreau.

Usage
-----

.. code-block:: python

   import jax.numpy as jnp
   from rheojax.models import CarreauYasuda

   gamma_dot = jnp.logspace(-3, 4, 200)
   eta_data = viscosity_curve(gamma_dot)

   model = CarreauYasuda(eta_0=1500.0, eta_inf=5.0, lambda_=0.4, n=0.25, a=1.3)
   params = model.fit(gamma_dot, eta_data, bounds={"a": (0.2, 3.0)})
   eta_pred = model.predict(gamma_dot, params=params)

Tips & Pitfalls
---------------

- ``a`` strongly correlates with ``lambda``; provide tight bounds or initial guesses based
  on the rate where viscosity is halfway between the plateaus.
- Keep ``eta_inf`` ≥ 0; negative values cause non-physical stress predictions.
- Rescale shear rates so the transition occurs near :math:`\dot{\gamma} = 1` s⁻¹ to reduce
  parameter covariance.
- When data do not show sharp transitions, fall back to :doc:`carreau` or
  :doc:`cross` to avoid overfitting.
- Use :doc:`../../transforms/smooth_derivative` on log–log viscosity curves to verify that
  the slope tends to :math:`n-1` at high shear.

References
----------

- K. Yasuda, R.C. Armstrong, and R.E. Cohen, "Shear flow properties of concentrated
  solutions of linear and star-branched polystyrenes," *Rheol. Acta* 20, 163–178 (1981).
- C.W. Macosko, *Rheology: Principles, Measurements, and Applications*, Wiley (1994).
- R.G. Larson, *Constitutive Equations for Polymer Melts and Solutions*, Butterworths
  (1988).
- H.A. Barnes, "Thixotropy—A Review," *J. Non-Newtonian Fluid Mech.* 70, 1–33 (1997).
- J. Mewis and N.J. Wagner, *Colloidal Suspension Rheology*, Cambridge (2012).

See also
--------

- :doc:`carreau` — special case with :math:`a = 2`.
- :doc:`cross` — alternate sigmoidal model with denominator exponent.
- :doc:`../fractional/fractional_maxwell_liquid` — viscoelastic analogue capturing similar
  slopes in oscillatory data.
- :doc:`../../transforms/owchirp` — broadband strain sweeps to estimate high-rate slopes.
- :doc:`../../examples/transforms/06-carreau-yasuda` — notebook comparing Carreau,
  Carreau–Yasuda, and Cross fits.
