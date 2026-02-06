HVNMLocal — Full Model Reference
=================================

Quick Reference
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - Class
     - ``HVNMLocal``
   * - Registry names
     - ``"hvnm_local"``, ``"hvnm"``
   * - Parameters
     - 13 (base) to 25 (all flags on)
   * - Parent class
     - ``HVNMBase(HVMBase(VLBBase(BaseModel)))``
   * - Feature flags
     - ``include_dissociative``, ``include_damage``, ``include_interfacial_damage``, ``include_diffusion``
   * - ODE state
     - 17 or 18 components (simple shear)


Notation Guide
--------------

.. list-table::
   :widths: 15 20 10 55
   :header-rows: 1

   * - Symbol
     - Parameter
     - Units
     - Description
   * - :math:`G_P`
     - ``G_P``
     - Pa
     - Permanent (covalent) network modulus
   * - :math:`G_E`
     - ``G_E``
     - Pa
     - Exchangeable (vitrimer) network modulus
   * - :math:`G_D`
     - ``G_D``
     - Pa
     - Dissociative (physical) network modulus
   * - :math:`\nu_0`
     - ``nu_0``
     - 1/s
     - Matrix TST attempt frequency
   * - :math:`E_a`
     - ``E_a``
     - J/mol
     - Matrix activation energy
   * - :math:`V_{act}`
     - ``V_act``
     - m³/mol
     - Matrix activation volume
   * - :math:`T`
     - ``T``
     - K
     - Temperature
   * - :math:`k_{d,D}`
     - ``k_d_D``
     - 1/s
     - Dissociative rate constant
   * - :math:`\beta_I`
     - ``beta_I``
     - —
     - Interphase reinforcement ratio :math:`G_I / G_E`
   * - :math:`\nu_0^{int}`
     - ``nu_0_int``
     - 1/s
     - Interfacial TST attempt frequency
   * - :math:`E_a^{int}`
     - ``E_a_int``
     - J/mol
     - Interfacial activation energy
   * - :math:`V_{act}^{int}`
     - ``V_act_int``
     - m³/mol
     - Interfacial activation volume
   * - :math:`\phi`
     - ``phi``
     - —
     - NP volume fraction
   * - :math:`R_{NP}`
     - ``R_NP``
     - m
     - NP radius
   * - :math:`\delta_m`
     - ``delta_m``
     - m
     - Mobile interphase thickness


Physical Foundations
--------------------

**4-Subnetwork Architecture**

The HVNM Cauchy stress in simple shear is:

.. math::

   \sigma_{tot} = \underbrace{(1-D) G_P X(\phi) \gamma}_{\text{permanent}}
   + \underbrace{G_E (\mu^E_{xy} - \mu^{E,nat}_{xy})}_{\text{exchangeable}}
   + \underbrace{G_D (\mu^D_{xy} - \delta_{xy})}_{\text{dissociative}}
   + \underbrace{(1-D_{int}) G_{I,eff} X_I (\mu^I_{xy} - \mu^{I,nat}_{xy})}_{\text{interphase}}

where:

- :math:`X(\phi) = 1 + 2.5\phi + 14.1\phi^2` is the Guth-Gold strain amplification
- :math:`G_{I,eff} = \beta_I G_E \phi_I` is the effective interphase modulus
- :math:`\phi_I` is the interphase volume fraction from NP geometry
- :math:`X_I = X(\phi_I)` is the interphase amplification factor

**Dual TST Kinetics**

Matrix and interfacial BER rates are independent:

.. math::

   k_{BER}^{mat} &= \nu_0 \exp\!\left(-\frac{E_a}{RT}\right) \cosh\!\left(\frac{V_{act} \sigma_{VM}^E}{RT}\right) \\
   k_{BER}^{int} &= \nu_0^{int} \exp\!\left(-\frac{E_a^{int}}{RT}\right) \cosh\!\left(\frac{V_{act}^{int} \sigma_{VM}^I}{RT}\right)

**I-Network Evolution**

The interphase distribution tensor evolves with amplified affine deformation:

.. math::

   \dot{\mu}^I_{xy} = X_I \dot{\gamma} (\mu^I_{xx} + 1)/2 - k_{BER}^{int}(\mu^I_{xy} - \mu^{I,nat}_{xy})


SAOS (Analytical)
-----------------

Three Maxwell modes plus amplified permanent plateau:

.. math::

   G'(\omega) = (1-D) G_P X + \frac{G_E \omega^2 \tau_m^2}{1 + \omega^2 \tau_m^2}
   + \frac{G_D \omega^2 \tau_D^2}{1 + \omega^2 \tau_D^2}
   + \frac{G_{I,eff} X_I \omega^2 \tau_I^2}{1 + \omega^2 \tau_I^2}

where :math:`\tau_m = 1/(2 k_{BER,0}^{mat})`, :math:`\tau_D = 1/k_{d,D}`,
:math:`\tau_I = 1/(2 k_{BER,0}^{int})`.


Limiting Cases
--------------

.. list-table::
   :widths: 30 40 30
   :header-rows: 1

   * - Limiting Case
     - Conditions
     - Factory Method
   * - HVM (unfilled vitrimer)
     - :math:`\phi = 0`
     - ``unfilled_vitrimer()``
   * - Filled elastomer
     - :math:`G_E = G_D = 0`
     - ``filled_elastomer()``
   * - Partial vitrimer NC
     - :math:`G_D = 0`
     - ``partial_vitrimer_nc()``
   * - Conventional filled rubber
     - :math:`G_E = 0`, frozen I
     - ``conventional_filled_rubber()``
   * - Matrix-only exchange
     - Frozen interphase
     - ``matrix_only_exchange()``


Usage Examples
--------------

**Flow curve with component decomposition:**

.. code-block:: python

   gamma_dot = np.logspace(-2, 2, 50)
   result = model.predict_flow_curve(gamma_dot, return_components=True)
   # result["stress"], result["sigma_D"], result["sigma_E"], result["sigma_I"]

**Relaxation with full state output:**

.. code-block:: python

   t = np.logspace(-3, 4, 200)
   result = model.simulate_relaxation(t, gamma_step=0.01, return_full=True)
   # result["G_t"], result["mu_E_xy"], result["mu_I_xy"], etc.

**LAOS with harmonic extraction:**

.. code-block:: python

   omega = 1.0
   t = np.linspace(0, 20 * 2*np.pi/omega, 2000)
   result = model.simulate_laos(t, gamma_0=0.5, omega=omega)
   harmonics = model.extract_laos_harmonics(result, n_harmonics=5)

**phi sweep (Payne effect):**

.. code-block:: python

   for phi_val in [0.0, 0.05, 0.10, 0.15, 0.20]:
       model.parameters.set_value("phi", phi_val)
       G_p, G_dp = model.predict_saos(omega)
       # G_P * X(phi) increases with phi


Numerical Notes
---------------

- **ODE solver**: diffrax Tsit5 with PIDController (rtol=1e-8, atol=1e-10)
- **State vector**: 17 components (18 with interfacial damage)
- **Creep**: Solves for :math:`\dot{\gamma}` from stress balance at each timestep
- **Stiff systems**: If TST feedback is strong, increase ``max_steps``
- **phi → 0**: Exact recovery of HVM predictions (verified to machine precision)
