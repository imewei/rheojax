HVM Model Reference
===================

This page provides the complete mathematical formulation and implementation
details for the Hybrid Vitrimer Model (HVM).


Quick Reference
---------------

.. list-table::
   :widths: 25 75
   :header-rows: 0

   * - **Full name**
     - Hybrid Vitrimer Model
   * - **Class**
     - :class:`~rheojax.models.hvm.HVMLocal`
   * - **Base class**
     - :class:`~rheojax.models.hvm._base.HVMBase` (extends VLBBase)
   * - **Registry names**
     - ``"hvm_local"``, ``"hvm"``
   * - **Parameters**
     - 6 (base) + 2 (dissociative) + 2 (damage) = 6-10
   * - **TST coupling**
     - ``kinetics="stress"`` (default) or ``"stretch"``
   * - **ODE solver**
     - diffrax Tsit5 with adaptive stepping
   * - **Bayesian**
     - Full NLSQ -> NUTS pipeline


Notation Guide
--------------

.. list-table::
   :widths: 15 50 15 20
   :header-rows: 1

   * - Symbol
     - Meaning
     - Units
     - Typical Range
   * - :math:`G_P`
     - Permanent network modulus
     - Pa
     - :math:`10^2` -- :math:`10^6`
   * - :math:`G_E`
     - Exchangeable network modulus
     - Pa
     - :math:`10^2` -- :math:`10^6`
   * - :math:`G_D`
     - Dissociative network modulus
     - Pa
     - :math:`10^1` -- :math:`10^5`
   * - :math:`\nu_0`
     - TST attempt frequency
     - 1/s
     - :math:`10^8` -- :math:`10^{12}`
   * - :math:`E_a`
     - Activation energy for BER
     - J/mol
     - 40k -- 150k
   * - :math:`V_{act}`
     - Activation volume
     - m\ :sup:`3`/mol
     - :math:`10^{-7}` -- :math:`10^{-4}`
   * - :math:`T`
     - Temperature
     - K
     - 250 -- 450
   * - :math:`k_d^D`
     - Dissociative bond rate
     - 1/s
     - :math:`10^{-3}` -- :math:`10^3`
   * - :math:`k_{BER}`
     - Bond exchange rate
     - 1/s
     - (computed from TST)
   * - :math:`\boldsymbol{\mu}^E`
     - E-network distribution tensor
     - --
     - (state variable)
   * - :math:`\boldsymbol{\mu}^E_{nat}`
     - E-network natural-state tensor
     - --
     - (state variable)
   * - :math:`\boldsymbol{\mu}^D`
     - D-network distribution tensor
     - --
     - (state variable)
   * - :math:`D`
     - Damage variable
     - --
     - [0, 1]
   * - :math:`R`
     - Universal gas constant
     - J/(mol K)
     - 8.314


Three-Subnetwork Architecture
-----------------------------

The HVM decomposes the total stress as:

.. math::

   \boldsymbol{\sigma} = \boldsymbol{\sigma}_P + \boldsymbol{\sigma}_E + \boldsymbol{\sigma}_D

**Permanent (P) network:**

Neo-Hookean elastic response from covalent crosslinks:

.. math::

   \boldsymbol{\sigma}_P = (1 - D) \, G_P (\mathbf{B} - \mathbf{I})

where :math:`\mathbf{B}` is the left Cauchy-Green tensor and :math:`D` is the
damage variable. In simple shear at strain :math:`\gamma`:

.. math::

   \sigma_{P,xy} = (1 - D) \, G_P \, \gamma

**Exchangeable (E) network:**

Stress arises from the deviation of :math:`\boldsymbol{\mu}^E` from its
evolving natural state :math:`\boldsymbol{\mu}^E_{nat}`:

.. math::

   \boldsymbol{\sigma}_E = G_E (\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})

This is the vitrimer hallmark: the natural state is **not** the identity tensor
but evolves to track the current deformation via BER.

**Dissociative (D) network:**

Standard upper-convected Maxwell:

.. math::

   \boldsymbol{\sigma}_D = G_D (\boldsymbol{\mu}^D - \mathbf{I})


Evolution Equations (Simple Shear)
----------------------------------

**E-network distribution tensor** (:math:`\boldsymbol{\mu}^E`):

.. math::

   \dot{\mu}^E_{xx} &= 2 \dot{\gamma} \, \mu^E_{xy} + k_{BER}(\mu^E_{nat,xx} - \mu^E_{xx}) \\
   \dot{\mu}^E_{yy} &= k_{BER}(\mu^E_{nat,yy} - \mu^E_{yy}) \\
   \dot{\mu}^E_{xy} &= \dot{\gamma} \, \mu^E_{yy} + k_{BER}(\mu^E_{nat,xy} - \mu^E_{xy})

**E-network natural-state tensor** (:math:`\boldsymbol{\mu}^E_{nat}`):

.. math::

   \dot{\mu}^E_{nat,ij} = k_{BER}(\mu^E_{ij} - \mu^E_{nat,ij})

This coupled evolution is the vitrimer hallmark: the natural state continuously
drifts toward the current state at rate :math:`k_{BER}`.

**D-network distribution tensor** (:math:`\boldsymbol{\mu}^D`):

.. math::

   \dot{\mu}^D_{xx} &= 2 \dot{\gamma} \, \mu^D_{xy} - k_d^D(\mu^D_{xx} - 1) \\
   \dot{\mu}^D_{yy} &= -k_d^D(\mu^D_{yy} - 1) \\
   \dot{\mu}^D_{xy} &= \dot{\gamma} \, \mu^D_{yy} - k_d^D \, \mu^D_{xy}

This is identical to the VLB single-network evolution.


TST Kinetics
------------

**Thermal BER rate** (zero stress):

.. math::

   k_{BER,0} = \nu_0 \exp\!\left(-\frac{E_a}{R T}\right)

**Stress-coupled BER rate** (``kinetics="stress"``):

.. math::

   k_{BER} = k_{BER,0} \cosh\!\left(\frac{V_{act} \, \sigma_{VM}^E}{R T}\right)

where :math:`\sigma_{VM}^E` is the von Mises stress of the E-network:

.. math::

   \sigma_{VM}^E = G_E \sqrt{(\sigma^E_{xx})^2 + (\sigma^E_{yy})^2
   - \sigma^E_{xx} \sigma^E_{yy} + 3(\sigma^E_{xy})^2}

with :math:`\sigma^E_{ij} = \mu^E_{ij} - \mu^E_{nat,ij}`.

**Stretch-coupled BER rate** (``kinetics="stretch"``):

.. math::

   \lambda_E = \sqrt{\frac{\text{tr}(\boldsymbol{\mu}^E)}{3}}
   \qquad
   k_{BER} = k_{BER,0} \cosh\!\left(\frac{V_{act} \, G_E(\lambda_E - 1)}{R T}\right)


Cooperative Damage Shielding
-----------------------------

When ``include_damage=True``, the damage variable :math:`D \in [0, 1]` evolves
as:

.. math::

   \dot{D} = \Gamma_0 (1 - D) \max(0, \lambda_{chain} - \lambda_{crit})

where:

- :math:`\Gamma_0` is the damage rate coefficient
- :math:`\lambda_{crit}` is the critical stretch for damage onset
- :math:`\lambda_{chain} = \sqrt{\text{tr}(\mathbf{B})/3}` is the effective chain stretch

The :math:`(1 - D)` factor provides cooperative shielding: damage slows as the
network degrades, preventing unphysical complete fracture.


Factor-of-2 in Relaxation
--------------------------

For the E-network under constant :math:`k_{BER}`, the stress difference
:math:`\Delta\mu_{ij} = \mu^E_{ij} - \mu^E_{nat,ij}` satisfies:

.. math::

   \dot{\Delta\mu}_{ij} = -2 k_{BER} \, \Delta\mu_{ij}

Therefore the E-network contribution to stress relaxes with effective time
constant:

.. math::

   \tau_{E,eff} = \frac{1}{2 k_{BER,0}}

A naive Maxwell fit yields :math:`\tau_{fit} = \tau_{E,eff} = \tau_E / 2`.
This is a fundamental vitrimer signature.


Analytical Solutions (Constant :math:`k_{BER}`)
------------------------------------------------

When TST feedback is negligible (linear regime, small deformations), the HVM
admits closed-form solutions.

**SAOS moduli:**

.. math::

   G'(\omega) &= G_P + \frac{G_E \omega^2 \tau_{E,eff}^2}{1 + \omega^2 \tau_{E,eff}^2}
   + \frac{G_D \omega^2 \tau_D^2}{1 + \omega^2 \tau_D^2} \\[6pt]
   G''(\omega) &= \frac{G_E \omega \tau_{E,eff}}{1 + \omega^2 \tau_{E,eff}^2}
   + \frac{G_D \omega \tau_D}{1 + \omega^2 \tau_D^2}

where :math:`\tau_{E,eff} = 1/(2 k_{BER,0})` and :math:`\tau_D = 1/k_d^D`.

**Relaxation modulus:**

.. math::

   G(t) = (1-D) \, G_P + G_E \, e^{-2 k_{BER,0} t} + G_D \, e^{-k_d^D t}

**Steady-state flow curve:**

At steady state, :math:`\sigma_E = 0` (the natural state fully tracks the
deformation), so:

.. math::

   \sigma_{ss} = G_P \gamma + \frac{G_D}{k_d^D} \dot{\gamma}


Limiting Cases
--------------

.. list-table::
   :widths: 25 20 30 25
   :header-rows: 1

   * - Case
     - Parameters
     - Equivalent Model
     - Factory Method
   * - Neo-Hookean
     - :math:`G_E=0, G_D=0`
     - Pure elastic
     - ``HVMLocal.neo_hookean(G_P)``
   * - Maxwell
     - :math:`G_P=0, G_E=0`
     - Single Maxwell element
     - ``HVMLocal.maxwell(G_D, k_d_D)``
   * - Zener (SLS)
     - :math:`G_E=0`
     - Standard Linear Solid
     - ``HVMLocal.zener(G_P, G_D, k_d_D)``
   * - Pure vitrimer
     - :math:`G_P=0, G_D=0`
     - E-network only
     - ``HVMLocal.pure_vitrimer(G_E, ...)``
   * - Partial vitrimer
     - :math:`G_D=0`
     - Meng et al. (2019)
     - ``HVMLocal.partial_vitrimer(G_P, G_E, ...)``
   * - Full HVM
     - All :math:`> 0`
     - Full 3-network
     - ``HVMLocal()``


Parameter Table
---------------

.. list-table::
   :widths: 12 12 15 10 51
   :header-rows: 1

   * - Parameter
     - Default
     - Bounds
     - Units
     - Description
   * - ``G_P``
     - 1e4
     - (0, 1e9)
     - Pa
     - Permanent network modulus (covalent crosslinks)
   * - ``G_E``
     - 1e4
     - (0, 1e9)
     - Pa
     - Exchangeable network modulus (vitrimer bonds)
   * - ``nu_0``
     - 1e10
     - (1e6, 1e14)
     - 1/s
     - TST attempt frequency
   * - ``E_a``
     - 80e3
     - (20e3, 200e3)
     - J/mol
     - Activation energy for bond exchange
   * - ``V_act``
     - 1e-5
     - (1e-8, 1e-2)
     - m\ :sup:`3`/mol
     - Activation volume (mechanochemical coupling)
   * - ``T``
     - 300
     - (200, 500)
     - K
     - Temperature (typically fixed)
   * - ``G_D``
     - 1e3
     - (0, 1e8)
     - Pa
     - Dissociative network modulus (when ``include_dissociative=True``)
   * - ``k_d_D``
     - 1.0
     - (1e-6, 1e6)
     - 1/s
     - Dissociative bond rate (when ``include_dissociative=True``)
   * - ``Gamma_0``
     - 1e-4
     - (0, 0.1)
     - 1/s
     - Damage rate coefficient (when ``include_damage=True``)
   * - ``lambda_crit``
     - 2.0
     - (1.001, 10)
     - --
     - Critical stretch for damage onset (when ``include_damage=True``)


Usage Examples
--------------

**SAOS with component decomposition:**

.. code-block:: python

   from rheojax.models import HVMLocal
   import numpy as np

   model = HVMLocal()
   model.parameters.set_value("G_P", 5000.0)
   model.parameters.set_value("G_E", 3000.0)
   model.parameters.set_value("G_D", 1000.0)

   omega = np.logspace(-3, 3, 100)
   G_prime, G_double_prime = model.predict_saos(omega)

**Stress relaxation:**

.. code-block:: python

   t = np.logspace(-2, 3, 200)
   G_t = model.simulate_relaxation(t, gamma_step=0.01)

**Startup with full trajectory output:**

.. code-block:: python

   t = np.linspace(0.01, 50, 300)
   result = model.simulate_startup(t, gamma_dot=1.0, return_full=True)
   # result keys: 'stress', 'strain', 'mu_E', 'mu_E_nat', 'mu_D', 'damage'

**Temperature sweep (Arrhenius):**

.. code-block:: python

   inv_T, log_k = model.arrhenius_plot_data(T_range=np.linspace(300, 450, 50))

**LAOS with harmonic extraction:**

.. code-block:: python

   t = np.linspace(0, 20 * 2 * np.pi, 2000)
   result = model.simulate_laos(t, gamma_0=0.5, omega=1.0)
   harmonics = model.extract_laos_harmonics(result, n_harmonics=5)


Numerical Implementation
-------------------------

**ODE state vector** (11 components in simple shear):

.. code-block:: text

   [mu_E_xx, mu_E_yy, mu_E_xy,           # E-network distribution (3)
    mu_E_nat_xx, mu_E_nat_yy, mu_E_nat_xy, # E-network natural state (3)
    mu_D_xx, mu_D_yy, mu_D_xy,           # D-network distribution (3)
    gamma,                                 # accumulated strain (1)
    D]                                     # damage variable (1)

**Solver**: diffrax ``Tsit5`` (explicit Runge-Kutta) with ``PIDController``
adaptive stepping (``rtol=1e-8``, ``atol=1e-10``).

**Initial conditions**: All tensors at identity
(:math:`\mu_{xx} = \mu_{yy} = 1`, :math:`\mu_{xy} = 0`),
:math:`\gamma = 0`, :math:`D = 0`.


Troubleshooting
---------------

**SAOS fit gives wrong relaxation time:**
Check for the factor-of-2: the fitted time constant from a Maxwell fit is
:math:`\tau_{E,eff} = 1/(2k_{BER,0})`, not the bond exchange time
:math:`\tau_E = 1/k_{BER,0}`.

**Steady-state stress grows without bound:**
The permanent network stress :math:`\sigma_P = G_P \gamma` grows linearly with
strain. This is physical for bounded strain protocols (relaxation, LAOS) but
produces unbounded stress in flow curve mode. Use subnetwork decomposition
to isolate contributions.

**ODE diverges at high shear rates:**
TST kinetics can create very stiff ODEs at high stress. Reduce ``gamma_dot``
or switch to ``kinetics="stretch"`` (less stiff coupling).

**Damage produces unphysical behavior:**
Ensure :math:`\lambda_{crit} > 1` (damage only activates under stretch beyond
equilibrium). Set ``Gamma_0`` small initially and increase gradually.
