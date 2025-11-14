.. _model-carreau:

Carreau Model
=============

Quick Reference
---------------

**Use when:** Smooth transition between Newtonian plateaus, well-defined mid-rate power-law
**Parameters:** 4 (η₀, η∞, λ, n)
**Key equation:** :math:`\eta = \eta_{\infty} + (\eta_0 - \eta_{\infty})[1 + (\lambda\dot{\gamma})^2]^{(n-1)/2}`
**Test modes:** Flow (steady shear)
**Material examples:** Polymer melts, food products, blood analogues, structured liquids

Overview
--------

The :class:`rheojax.models.Carreau` equation models shear-thinning or shear-thickening
fluids that transition smoothly between Newtonian plateaus at low and high shear rates.
It is widely used for polymer melts, food products, blood analogues, and other
structured liquids with a well-defined mid-rate power-law slope.

Equations
---------

.. math::

   \eta(\dot{\gamma}) = \eta_{\infty} + \left(\eta_0 - \eta_{\infty}\right)
   \left[1 + (\lambda \dot{\gamma})^2\right]^{\frac{n-1}{2}}

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
     - Infinite-shear viscosity; ≥ 0. Can be fixed to 0 for strong thinning.
   * - ``lambda``
     - s
     - Characteristic time controlling the transition location; > 0.
   * - ``n``
     - –
     - Power index; < 1 for thinning, > 1 for thickening, = 1 reduces to Newtonian.

Usage
-----

.. code-block:: python

   import jax.numpy as jnp
   from rheojax.models import Carreau

   gamma_dot = jnp.logspace(-2, 4, 150)
   eta_data = viscosity_curve(gamma_dot)

   model = Carreau(eta_0=1200.0, eta_inf=10.0, lambda_=0.8, n=0.35)
   params = model.fit(gamma_dot, eta_data, bounds={"n": (0.1, 1.2)})
   eta_pred = model.predict(gamma_dot, params=params)

Tips & Pitfalls
---------------

- Use log-spaced shear rates to capture both plateaus; otherwise ``eta_0`` and
  ``eta_inf`` become highly correlated.
- Keep ``eta_inf`` strictly non-negative. When high-rate data are unavailable, fixing
  ``eta_inf`` to a small constant improves stability.
- Provide an initial ``lambda`` from the rate where viscosity drops to
  :math:`\tfrac{1}{2}(\eta_0 + \eta_{\infty})`.
- When ``n`` is very close to 1, the Carreau model reduces to Cross; ensure the extra
  parameters are justified before fitting.
- Fit shear stress instead of viscosity if measurement noise is proportional to load.

References
----------

- P.J. Carreau, "Rheological equations from molecular network theories," *Trans. Soc.
  Rheol.* 16, 99–127 (1972).
- R.G. Larson, *Constitutive Equations for Polymer Melts and Solutions*, Butterworths
  (1988).
- C.W. Macosko, *Rheology: Principles, Measurements, and Applications*, Wiley (1994).
- H.A. Barnes, J.F. Hutton, and K. Walters, *An Introduction to Rheology*, Elsevier (1989).
- J. Mewis and N.J. Wagner, *Colloidal Suspension Rheology*, Cambridge (2012).

See also
--------

- :doc:`carreau_yasuda` — adds a Yasuda exponent for sharper transitions.
- :doc:`cross` — alternative sigmoidal form with denominator exponent.
- :doc:`power_law` — local approximation of the Carreau mid-rate region.
- :doc:`../../transforms/smooth_derivative` — compute :math:`d\log\eta/d\log\dot{\gamma}`
  to seed ``n``.
- :doc:`../../examples/transforms/03-caru-carreau-fit` — notebook demonstrating
  Carreau/Cross parameter estimation.
