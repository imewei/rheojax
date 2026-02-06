.. _vlb_variant:

======================================
VLBVariant — Bell, FENE-P & Temperature
======================================

Quick Reference
===============

.. list-table:: Model Summary
   :widths: 30 70
   :header-rows: 0

   * - **Model Class**
     - ``VLBVariant``
   * - **Physics**
     - VLB transient network with force-dependent breakage, finite extensibility, and/or temperature dependence
   * - **Key Parameters**
     - :math:`G_0, k_d^0, \nu` (Bell), :math:`L_{\max}` (FENE), :math:`E_a, T_{\text{ref}}` (Temperature)
   * - **Protocols**
     - FLOW_CURVE, STARTUP, RELAXATION, CREEP, OSCILLATION, LAOS
   * - **Key Features**
     - Shear thinning, stress overshoot, bounded extensional viscosity, Arrhenius scaling
   * - **Reference**
     - Vernerey, Long & Brighenti (2017). *JMPS* 107, 1-20; Bell (1978). *Science* 200, 618

**Import:**

.. code-block:: python

   from rheojax.models import VLBVariant


Overview
========

``VLBVariant`` extends the VLB transient network with three composable
physics options:

1. **Bell breakage** — Force-dependent dissociation rate that produces
   shear thinning, stress overshoot, and nonlinear LAOS response
2. **FENE-P stress** — Finite extensibility that bounds the extensional
   viscosity and adds strain hardening
3. **Temperature dependence** — Arrhenius scaling of :math:`k_d` and
   thermal modulus :math:`G_0(T)`

These can be combined freely through constructor flags.  With all
flags off (``breakage="constant"``, ``stress_type="linear"``,
``temperature=False``), VLBVariant exactly reproduces VLBLocal.


Constructor Flags
=================

.. code-block:: python

   model = VLBVariant(
       breakage="constant",     # or "bell"
       stress_type="linear",    # or "fene"
       temperature=False,       # or True
   )


Parameters
==========

.. list-table::
   :widths: 15 15 15 55
   :header-rows: 1

   * - Parameter
     - When Present
     - Units
     - Description
   * - :math:`G_0`
     - Always
     - Pa
     - Network modulus
   * - :math:`k_d^0`
     - Always
     - 1/s
     - Unstressed dissociation rate
   * - :math:`\nu`
     - ``breakage="bell"``
     - —
     - Force sensitivity (0 = Newtonian, >1 = shear thinning)
   * - :math:`L_{\max}`
     - ``stress_type="fene"``
     - —
     - Maximum chain extensibility
   * - :math:`E_a`
     - ``temperature=True``
     - J/mol
     - Activation energy for bond dissociation
   * - :math:`T_{\text{ref}}`
     - ``temperature=True``
     - K
     - Reference temperature


Bell Breakage Model
===================

Theory
------

The Bell model (1978) makes the dissociation rate force-dependent:

.. math::

   k_d(\boldsymbol{\mu}) = k_d^0 \exp\!\left(\nu \cdot (\lambda_c - 1)\right)

where :math:`\lambda_c = \sqrt{\text{tr}(\boldsymbol{\mu})/3}` is the
normalized average chain stretch.

**Physical effects:**

- **Shear thinning:** :math:`\eta(\dot\gamma)` decreases with shear rate
  because stretched chains break faster, reducing effective cross-link density
- **Stress overshoot:** At startup, transient stretch exceeds steady state,
  causing :math:`\sigma_{\max} > \sigma_{ss}` at high Weissenberg numbers
- **Nonlinear LAOS:** Higher harmonics :math:`I_3/I_1 > 0` (unlike constant
  :math:`k_d` which is purely linear)
- **Faster relaxation:** Pre-stretched chains relax faster than predicted
  by the equilibrium :math:`k_d^0`

Usage
-----

.. code-block:: python

   from rheojax.models import VLBVariant

   model = VLBVariant(breakage="bell")
   model.parameters.set_value("G0", 1000.0)
   model.parameters.set_value("k_d_0", 1.0)
   model.parameters.set_value("nu", 5.0)

   # Shear-thinning flow curve
   gamma_dot = np.logspace(-2, 2, 50)
   sigma = model.predict_flow_curve(gamma_dot)

   # Stress overshoot in startup
   t = np.linspace(0.01, 20, 200)
   sigma_t = model.simulate_startup(t, gamma_dot=10.0)


FENE-P Finite Extensibility
============================

Theory
------

The FENE-P (finitely extensible nonlinear elastic with Peterlin closure)
model replaces the linear Hookean spring with a nonlinear spring:

.. math::

   \boldsymbol{\sigma} = G_0 \cdot f(\text{tr}(\boldsymbol{\mu})) \cdot
   (\boldsymbol{\mu} - \mathbf{I})

.. math::

   f = \frac{L_{\max}^2}{L_{\max}^2 - \text{tr}(\boldsymbol{\mu}) + 3}

At equilibrium (:math:`\boldsymbol{\mu} = \mathbf{I}`), :math:`f = 1` and
the stress reduces to the linear Hookean form.  As chains stretch toward
their maximum extensibility, :math:`f \to \infty`, creating a strongly
nonlinear stress response.

**Physical effects:**

- **Bounded extensional viscosity:** The Trouton ratio remains finite even
  at high extension rates (resolves the Hookean singularity at
  :math:`\dot\varepsilon = k_d/2`)
- **Strain hardening:** Stress increases super-linearly with strain
- **Large** :math:`L_{\max}` **recovers linear:** At :math:`L_{\max} \to \infty`,
  FENE-P reduces to the linear Hookean model

Usage
-----

.. code-block:: python

   model = VLBVariant(stress_type="fene")
   model.parameters.set_value("G0", 1000.0)
   model.parameters.set_value("k_d_0", 1.0)
   model.parameters.set_value("L_max", 10.0)


Temperature Dependence
======================

Theory
------

Arrhenius temperature scaling of the dissociation rate:

.. math::

   k_d(T) = k_d^0 \exp\!\left(-\frac{E_a}{R}\left(\frac{1}{T}
   - \frac{1}{T_{\text{ref}}}\right)\right)

Thermal modulus scaling (rubber elasticity):

.. math::

   G_0(T) = G_0^{\text{ref}} \cdot \frac{T}{T_{\text{ref}}}

At :math:`T = T_{\text{ref}}`, both reduce to their reference values.

Usage
-----

.. code-block:: python

   model = VLBVariant(temperature=True)
   model.parameters.set_value("G0", 1000.0)
   model.parameters.set_value("k_d_0", 1.0)
   model.parameters.set_value("E_a", 40000.0)  # J/mol
   model.parameters.set_value("T_ref", 300.0)

   # Predict at different temperatures
   sigma_300 = model.predict_flow_curve(gamma_dot, T=300.0)
   sigma_350 = model.predict_flow_curve(gamma_dot, T=350.0)


Combined Usage
==============

All flags compose naturally:

.. code-block:: python

   # Bell + FENE: shear thinning + bounded extension
   model = VLBVariant(breakage="bell", stress_type="fene")
   model.parameters.set_value("nu", 5.0)
   model.parameters.set_value("L_max", 10.0)

   # Full model with all extensions
   model = VLBVariant(
       breakage="bell",
       stress_type="fene",
       temperature=True,
   )


SAOS Behavior
=============

In the linear regime (SAOS), the Bell breakage reduces to constant
:math:`k_d` because at equilibrium the chain stretch is 1. Therefore
SAOS predictions are **always analytical Maxwell**, regardless of
breakage type.

This is exact — not an approximation — because SAOS probes
infinitesimally small deformations around equilibrium.


Bayesian Inference
==================

VLBVariant supports the standard NLSQ + NUTS pipeline:

.. code-block:: python

   # NLSQ warm start
   model.fit(omega, G_star, test_mode="oscillation")

   # Bayesian posterior
   result = model.fit_bayesian(
       omega, G_star, test_mode="oscillation",
       num_warmup=1000, num_samples=2000,
   )

The ``model_function`` is fully JAX-traceable, enabling gradient-based
NUTS sampling for all parameter combinations.


API Reference
=============

.. autoclass:: rheojax.models.vlb.VLBVariant
   :members:
   :undoc-members:
   :show-inheritance:
