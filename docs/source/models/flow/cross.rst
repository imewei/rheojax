.. _model-cross:

Cross Model
===========

Quick Reference
---------------

**Use when:** Well-characterized high-rate plateaus, tunable transition sharpness
**Parameters:** 4 (η₀, η∞, λ, m)
**Key equation:** :math:`\eta = \eta_{\infty} + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda\dot{\gamma})^m}`
**Test modes:** Flow (steady shear)
**Material examples:** Polymer melts, suspensions, fluids with well-defined plateaus

Overview
--------

The :class:`rheojax.models.Cross` form is another four-parameter generalized Newtonian
model. It transitions monotonically between two Newtonian plateaus with a tunable
exponent that controls the sharpness of the plateau-to-power-law switch. Cross is often
favored when high-rate plateaus are well characterized or when Carreau’s fixed square-law
exponent is too restrictive.

Equations
---------

.. math::

   \eta(\dot{\gamma}) = \eta_{\infty}
       + \frac{\eta_0 - \eta_{\infty}}{1 + (\lambda \dot{\gamma})^m}

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 24 20 56

   * - Name
     - Units
     - Description / Constraints
   * - ``eta_0``
     - Pa·s
     - Zero-shear viscosity; > 0 and typically ≥ ``eta_inf``.
   * - ``eta_inf``
     - Pa·s
     - Infinite-shear viscosity; ≥ 0.
   * - ``lambda``
     - s
     - Characteristic time; > 0.
   * - ``m``
     - –
     - Transition sharpness exponent; 0.2–2 captures most fluids.

Usage
-----

.. code-block:: python

   import jax.numpy as jnp
   from rheojax.models import Cross

   gamma_dot = jnp.logspace(-3, 5, 180)
   eta_data = viscosity_curve(gamma_dot)

   model = Cross(eta_0=800.0, eta_inf=4.0, lambda_=1.5, m=1.1)
   params = model.fit(gamma_dot, eta_data, penalty={"m": ("l2", 0.1)})
   eta_pred = model.predict(gamma_dot, params=params)

Tips & Pitfalls
---------------

- ``m`` approaching zero produces a step-like transition that is difficult to resolve;
  keep ``m`` ≥ 0.2 for numerical stability.
- ``eta_0``/``eta_inf`` become highly correlated if the corresponding plateaus are outside
  the measured range—apply bounds informed by temperature or molecular weight data.
- When ``m = 2`` the Cross model collapses to Carreau; use one or the other to avoid
  redundant parameters.
- Fit stresses instead of viscosities when measurement noise scales with load; the Cross
  form is still valid because :math:`\tau = \eta(\dot{\gamma}) \dot{\gamma}`.
- Differentiate stress-vs-rate curves with :doc:`../../transforms/smooth_derivative` to
  estimate the local slope and initialize ``m``.

References
----------

- M.M. Cross, "Rheology of non-Newtonian fluids: A new flow equation for pseudoplastic
  systems," *J. Colloid Sci.* 20, 417–437 (1965).
- R.G. Larson, *The Structure and Rheology of Complex Fluids*, Oxford (1999).
- H.A. Barnes et al., *An Introduction to Rheology*, Elsevier (1989).
- C.W. Macosko, *Rheology: Principles, Measurements, and Applications*, Wiley (1994).
- J. Mewis and N.J. Wagner, *Colloidal Suspension Rheology*, Cambridge (2012).

See also
--------

- :doc:`carreau` — uses a square-law exponent; choose based on data smoothness.
- :doc:`carreau_yasuda` — adds Yasuda’s extra exponent for even sharper transitions.
- :doc:`power_law` — approximates the Cross mid-rate slope when plateaus are unknown.
- :doc:`../fractional/fractional_burgers` — viscoelastic analogue for stress relaxation.
- :doc:`../../examples/transforms/04-non_newtonian-flow` — notebook comparing Carreau,
  Cross, and Herschel–Bulkley fits.
