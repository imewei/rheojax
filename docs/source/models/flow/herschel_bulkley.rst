.. _model-herschel-bulkley:

Herschel–Bulkley Model
======================

Quick Reference
---------------

**Use when:** Yield stress with power-law post-yield behavior, straight lines after yielding
**Parameters:** 3 (τ_y, K, n)
**Key equation:** :math:`\tau = \tau_y + K \dot{\gamma}^n` for :math:`\dot{\gamma} > 0`
**Test modes:** Flow (steady shear)
**Material examples:** Drilling muds, mayonnaise, toothpaste, colloidal gels

Overview
--------

The :class:`rheojax.models.HerschelBulkley` equation generalizes Bingham plastics by
combining a yield stress with a power-law post-yield slope. It is applicable whenever
log–log plots of stress vs. shear rate become straight lines after yielding, such as
drilling muds, mayonnaise, toothpaste, and colloidal gels.

Equations
---------

.. math::

   \tau = \tau_y + K \dot{\gamma}^n, \qquad \dot{\gamma} > 0

Apparent viscosity:

.. math::

   \eta(\dot{\gamma}) = \frac{\tau}{\dot{\gamma}} = \frac{\tau_y}{\dot{\gamma}} + K
   \dot{\gamma}^{n-1}.

Parameters
----------

.. list-table:: Parameter summary
   :header-rows: 1
   :widths: 22 20 58

   * - Name
     - Units
     - Description / Constraints
   * - ``tau_y``
     - Pa
     - Yield stress; ≥ 0.
   * - ``K``
     - Pa·s\ :sup:`n`
     - Consistency index; > 0.
   * - ``n``
     - –
     - Flow index; < 1 for thinning, > 1 for thickening, = 1 reduces to Bingham.

Usage
-----

.. code-block:: python

   import jax.numpy as jnp
   from rheojax.models import HerschelBulkley

   gamma_dot = jnp.logspace(-3, 3, 140)
   tau_data = stress_curve(gamma_dot)

   model = HerschelBulkley(tau_y=12.0, K=3.0, n=0.45)
   params = model.fit(gamma_dot, tau_data, bounds={"n": (0.05, 2.0)})
   tau_pred = model.predict(gamma_dot, params=params)

Tips & Pitfalls
---------------

- Filter or down-weight sub-yield data—the ideal model assumes rigid response until
  :math:`\tau > \tau_y`.
- Add a small floor to ``gamma_dot`` when reporting apparent viscosity to avoid division
  by zero.
- Use log–log fits for the post-yield segment to seed ``K`` and ``n`` before full
  regression.
- For materials with gradual yielding, consider blending with :doc:`carreau` or
  :doc:`cross` to capture smooth transitions.
- Bidirectional sweeps help detect hysteresis; the static Herschel–Bulkley equation does
  not model structural rebuild.

References
----------

- W.H. Herschel and R. Bulkley, "Konsistenzmessungen von Gummi-Benzollösungen," *Kolloid-
  Zeitschrift* 39, 291–300 (1926).
- N.D. Scott Blair, *Elementary Rheology*, Academic Press (1969).
- H.A. Barnes, "Thixotropy—a review," *J. Non-Newtonian Fluid Mech.* 70, 1–33 (1997).
- R.G. Larson, *The Structure and Rheology of Complex Fluids*, Oxford (1999).
- M. Piau, "Carbopol gels: a model for yield stress fluid," *J. Non-Newtonian Fluid
  Mech.* 144, 1–29 (2007).

See also
--------

- :doc:`bingham` — special case with :math:`n = 1` and linear post-yield slope.
- :doc:`power_law` — applies when :math:`\tau_y = 0`.
- :doc:`carreau` — shear-thinning without yield; combine for smoother transitions.
- :doc:`../../transforms/mutation_number` — monitor microstructural change during
  yielding experiments.
- :doc:`../../examples/advanced/02-yield-stress-fitting` — notebook comparing Bingham
  and Herschel–Bulkley fits on LAOS data.
