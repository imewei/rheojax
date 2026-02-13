Knowledge Extraction from DMTA Data
====================================

DMTA measurements encode rich material information beyond simple modulus values.
This page describes the physical quantities that can be extracted from fitted
DMTA models in RheoJAX.

Glass Transition Temperature :math:`T_g`
-----------------------------------------

The glass transition is the most commonly extracted quantity from DMTA data.
Three conventions exist:

.. list-table::
   :header-rows: 1
   :widths: 30 35 35

   * - Method
     - Definition
     - Notes
   * - :math:`\tan(\delta)` peak
     - Temperature at max :math:`E''/E'`
     - `ISO 6721-11`_; highest :math:`T_g` value
   * - :math:`E''` peak
     - Temperature at max :math:`E''`
     - `ASTM D4065`_ preferred; closest to DSC :math:`T_g`
   * - :math:`E'` onset
     - Inflection point of :math:`E'(T)` drop
     - Lowest :math:`T_g`; onset of softening

.. important::

   Typical ordering: :math:`T_g^{E'\text{onset}} < T_g^{E''\text{peak}} < T_g^{\tan\delta\text{peak}}`
   with 5--15 °C difference between methods.

Relaxation Spectrum :math:`H(\tau)`
------------------------------------

The relaxation spectrum is a material property independent of deformation mode.
Several RheoJAX models give direct access:

- **Generalized Maxwell**: Discrete spectrum :math:`\{(G_i, \tau_i)\}` (Prony series)
- **Fractional models**: Continuous power-law spectrum :math:`H(\tau) \sim \tau^{-\alpha}`
  where :math:`\alpha` is the fractional order
- **VLB / HVM**: Molecular-level relaxation from bond exchange kinetics

.. code-block:: python

   # After fitting a GMM (see Workflow 2 in dmta_workflows),
   # extract discrete spectrum:

   # With modulus_type='tensile': parameters are E_i (Pa)
   for i in range(1, gmm._n_modes + 1):
       E_i = gmm.parameters.get_value(f"E_{i}")
       tau_i = gmm.parameters.get_value(f"tau_{i}")

   # With modulus_type='shear' + deformation_mode='tension':
   # parameters remain in G-space (G_i, Pa)
   for i in range(1, gmm_g._n_modes + 1):
       G_i = gmm_g.parameters.get_value(f"G_{i}")

.. seealso::

   :doc:`dmta_workflows` Workflow 2 for complete GMM fitting.

.. note::

   When using ``modulus_type='tensile'``, parameters are named ``E_inf``,
   ``E_1``, ..., ``E_N`` and represent tensile moduli directly.  When using
   the default ``modulus_type='shear'`` with ``deformation_mode='tension'``,
   the input E* is converted to G* internally, and parameters remain in
   G-space (``G_inf``, ``G_1``, ...).

Loss Tangent :math:`\tan\delta` Analysis
-----------------------------------------

The loss tangent :math:`\tan\delta = E''/E'` is the most information-rich
single quantity from DMTA:

- **Peak height**: Related to the damping capacity.  A tall, narrow peak
  signals a single dominant relaxation mechanism; a broad, low peak indicates
  a wide distribution (blends, copolymers, filled systems).

- **Peak width at half-height**: Quantifies the breadth of the glass
  transition.  For a single relaxation mechanism (Debye peak in
  :math:`E''(\omega)`), FWHM :math:`\approx` 1.14 decades; real polymers
  typically show 2--4 decades.

- **Peak asymmetry**: A high-frequency shoulder on :math:`\tan\delta`
  indicates sub-:math:`T_g` (beta) relaxations from local chain motions.

.. code-block:: python

   import numpy as np

   # From fitted model predictions
   E_pred = model.predict(omega, test_mode='oscillation')
   E_prime = E_pred.real
   E_double_prime = E_pred.imag
   tan_delta = E_double_prime / E_prime

   # Find Tg (tan delta peak)
   idx_peak = np.argmax(tan_delta)
   omega_peak = omega[idx_peak]
   tau_peak = 1.0 / omega_peak  # characteristic relaxation time

Storage Modulus Crossover and Modulus Drop
-------------------------------------------

Two key features of the :math:`E'(T)` or :math:`E'(\omega)` curve:

- **Modulus drop ratio** :math:`E'_{\text{glassy}} / E'_{\text{rubbery}}`
  indicates the degree of chain mobility change through :math:`T_g`.
  Typical values: 100--1000 for amorphous polymers, 10--100 for
  semi-crystalline (crystallites constrain rubbery plateau).

- **:math:`E' = E''` crossover frequency** :math:`\omega_c` gives a
  characteristic relaxation time :math:`\tau_c = 1/\omega_c`.  For a Maxwell
  element, this is exactly :math:`\tau`.

Molecular Weight from Plateau Modulus
--------------------------------------

For entangled polymers above :math:`T_g`, the entanglement molecular weight
:math:`M_e` is directly accessible:

.. math::

   M_e = \frac{\rho R T}{G_N^0} = \frac{\rho R T}{E_\infty / [2(1+\nu)]}

where :math:`\rho` is the polymer density, :math:`R` the gas constant, and
:math:`T` the absolute temperature.  This requires a clear rubbery plateau in
the DMTA data (i.e., the material must be entangled and above :math:`T_g`).

Plateau Modulus :math:`E_\infty` / :math:`G_N^0`
--------------------------------------------------

The equilibrium (rubbery plateau) modulus encodes crosslink density:

.. math::

   G_N^0 = \nu_e k_B T \quad \Rightarrow \quad
   \nu_e = \frac{G_N^0}{k_B T} = \frac{E_\infty}{2(1+\nu)\,k_B T}

where :math:`\nu_e` is the crosslink density.

- **Zener / Fractional Zener**: :math:`G_e` parameter = :math:`G_N^0`
- **VLB models**: :math:`G_0` parameter (sum of all network moduli)

Fractional Order :math:`\alpha` and Cooperativity
---------------------------------------------------

The fractional order from springpot-based models characterises
the breadth of the relaxation distribution:

- :math:`\alpha \to 0`: Purely elastic (Hookean spring)
- :math:`\alpha \approx 0.5`: Critical gel / moderate cooperativity
- :math:`\alpha \to 1`: Purely viscous (Newtonian dashpot)

The :math:`\alpha` value also correlates with the cooperativity of segmental
motion at :math:`T_g`:

- :math:`\alpha \approx 0.3`: Highly cooperative, rigid-chain polymers
  (e.g., polycarbonate, polysulfone) --- broad glass transition
- :math:`\alpha \approx 0.5`: Moderate cooperativity, flexible-chain
  polymers (e.g., polyisoprene, PDMS)
- :math:`\alpha \approx 0.7`: Low cooperativity, highly plasticised
  or low-:math:`M_w` systems --- narrow, single-mechanism relaxation

This provides a physics-based alternative to the empirical
Kohlrausch--Williams--Watts (KWW) stretched exponential parameter
:math:`\beta_{\text{KWW}} \approx \alpha`
(the Mittag-Leffler function :math:`E_\alpha(-(t/\tau)^\alpha)` approximates
:math:`\exp(-(t/\tau)^\beta)` with :math:`\beta = \alpha`; see Metzler &
Klafter, *J. Non-Cryst. Solids* **305**, 81--87, 2002).

.. seealso::

   :doc:`dmta_workflows` Workflow 3 for FZSS fitting with Bayesian
   uncertainty on :math:`\alpha`.

WLF / Arrhenius Activation Energy
----------------------------------

Temperature-dependent DMTA data (master curves) yield activation energy:

- **WLF**: :math:`\log a_T = -C_1(T - T_r) / (C_2 + T - T_r)`
- **Arrhenius**: :math:`E_a = 2.303\,R\,C_1 C_2` (from WLF at :math:`T > T_g + 100`)
- **HVM / HVNM**: Direct :math:`E_a` parameter from TST kinetics

.. seealso::

   :doc:`dmta_workflows` Workflow 2 for WLF extraction from multi-T DMTA, and
   :doc:`dmta_protocols` for recommended TTS pipeline steps.

Network Topology (Vitrimer Models)
------------------------------------

HVM and HVNM models extract topology-specific parameters from DMTA data:

- **Permanent crosslinks** :math:`G_P`: Covalent network modulus
- **Exchangeable bonds** :math:`G_E`: Vitrimer BER-active modulus
- **Bond exchange rate** :math:`k_{\text{BER}}`: Arrhenius with :math:`E_a, \nu_0`
- **Topology freezing** :math:`T_v`: Where :math:`\tau_{\text{BER}} \sim` experimental timescale

.. seealso::

   - :doc:`dmta_workflows` Workflow 5 for HVM fitting with tensile DMTA data
   - :doc:`/models/hvm/index` and :doc:`/models/hvnm/index` for vitrimer model details
   - :doc:`dmta_theory` --- mathematical background for E* |leftrightarrow| G* conversion
   - :doc:`dmta_models` --- model selection guide by material type
   - :doc:`dmta_numerical` --- bounds and convergence for DMTA fitting

.. |leftrightarrow| unicode:: U+2194

.. _ISO 6721-11: https://www.iso.org/standard/74988.html
.. _ASTM D4065: https://www.astm.org/d4065-20.html
