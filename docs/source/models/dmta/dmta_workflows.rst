DMTA Workflows
==============

This page presents end-to-end workflows for common DMTA analysis tasks.

Workflow 1: Direct Fit with Modulus Conversion
-----------------------------------------------

The simplest workflow: load E* data, fit with any model.

.. code-block:: python

   import numpy as np
   from rheojax.models import Maxwell

   # Synthetic E* data (rubber, nu=0.5 -> E = 3G)
   omega = np.logspace(-2, 3, 100)
   G0, eta = 1e6, 1e4
   tau = eta / G0
   G_star = G0 * (1j * omega * tau) / (1 + 1j * omega * tau)
   E_star = 3.0 * G_star  # E = 2(1+0.5)*G = 3G

   # Fit directly in tension mode
   model = Maxwell()
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.5,
   )

   # Parameters are in G-space
   print(f"G0 = {model.parameters.get_value('G0'):.0f} Pa")
   print(f"eta = {model.parameters.get_value('eta'):.0f} Pa.s")

   # predict() returns E*
   E_pred = model.predict(omega, test_mode='oscillation')

Workflow 2: Master Curve from Multi-T DMTA
---------------------------------------------

Time--temperature superposition collapses multi-temperature frequency sweeps
into a single master curve.

.. code-block:: python

   from rheojax.io import load_csv
   from rheojax.models import GeneralizedMaxwell
   from rheojax.transforms import Mastercurve

   # Load datasets at multiple temperatures
   datasets = []
   for T in [20, 40, 60, 80, 100, 120]:
       data = load_csv(
           f"dmta_T{T}C.csv",
           x_col="frequency (Hz)",
           y_cols=["E' (Pa)", "E'' (Pa)"],
           temperature=T + 273.15,
           deformation_mode='tension',
       )
       datasets.append(data)

   # Create master curve
   mc = Mastercurve(reference_temp=60 + 273.15, method='wlf')
   master, shifts = mc.transform(datasets)

   # Fit Prony series (native tensile mode)
   gmm = GeneralizedMaxwell(n_modes=10, modulus_type='tensile')
   gmm.fit(master.x, master.y, test_mode='oscillation')

   # Extract WLF parameters
   C1, C2 = shifts['wlf_C1'], shifts['wlf_C2']
   print(f"WLF: C1={C1:.1f}, C2={C2:.1f} K")

Workflow 3: NLSQ -> NUTS for Uncertainty
-------------------------------------------

Bayesian inference quantifies parameter uncertainty from DMTA data.

.. code-block:: python

   from rheojax.models import FractionalZenerSolidSolid

   model = FractionalZenerSolidSolid()

   # Step 1: NLSQ point estimate (with auto E->G conversion)
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.5,
   )

   # Step 2: NUTS with warm-start
   result = model.fit_bayesian(
       omega, E_star,
       test_mode='oscillation',
       num_warmup=1000,
       num_samples=2000,
   )

   # Diagnostics
   print(f"R-hat max: {result.diagnostics['r_hat_max']:.3f}")
   intervals = model.get_credible_intervals(
       result.posterior_samples, credibility=0.95
   )

Workflow 4: Loading DMTA CSV Files
-------------------------------------

The CSV reader auto-detects E'/E'' columns and sets ``deformation_mode``
in the RheoData metadata.

.. code-block:: python

   from rheojax.io import load_csv

   # Auto-detected from column names
   data = load_csv(
       "dmta_sweep.csv",
       x_col="omega (rad/s)",
       y_cols=["E' (Pa)", "E'' (Pa)"],
   )
   print(data.deformation_mode)  # "tension"

   # pyvisco format (f, E_stor, E_loss, T, Set)
   data = load_csv(
       "pyvisco_data.csv",
       x_col="f",
       y_cols=["E_stor", "E_loss"],
       deformation_mode='tension',  # explicit
   )

Workflow 5: Vitrimer DMTA with HVM
--------------------------------------

The HVM model has built-in Arrhenius kinetics, ideal for vitrimers showing
a topology-freezing transition :math:`T_v` in DMTA.

.. code-block:: python

   from rheojax.models import HVMLocal

   model = HVMLocal(kinetics="stress")
   model.parameters.set_value("E_a", 80e3)  # Activation energy (J/mol)

   # Fit isothermal SAOS at each temperature
   model.fit(
       omega, G_star,  # Already converted to G-space
       test_mode='oscillation',
   )

   # Or fit E* directly
   model.fit(
       omega, E_star,
       test_mode='oscillation',
       deformation_mode='tension',
       poisson_ratio=0.5,
   )

Workflow 6: Model Selection Query
-------------------------------------

Use the registry to find all DMTA-compatible models:

.. code-block:: python

   from rheojax.core.registry import ModelRegistry
   from rheojax.core.test_modes import DeformationMode
   from rheojax.core.inventory import Protocol

   # All models supporting oscillation + tension
   tension_models = ModelRegistry.find(
       protocol=Protocol.OSCILLATION,
       deformation_mode=DeformationMode.TENSION,
   )
   print(f"{len(tension_models)} models support tension DMTA")

   # Check if a specific model supports tension
   info = ModelRegistry.get_info("fractional_zener_ss")
   print(DeformationMode.TENSION in info.deformation_modes)  # True

Workflow 7: Bounds Handling for Real DMTA Data
-----------------------------------------------

Real polymer DMTA data can reach ~10 GPa in the glassy plateau.
``GeneralizedMaxwell`` with ``modulus_type='tensile'`` uses wider
default bounds (:math:`E_i \leq 10^{12}` Pa) that accommodate this range.

When fitting real data, disable element minimisation to avoid internal
sub-models reverting to default bounds:

.. code-block:: python

   from rheojax.models import GeneralizedMaxwell

   gmm = GeneralizedMaxwell(n_modes=10, modulus_type='tensile')
   gmm.fit(
       omega, E_star,
       test_mode='oscillation',
       optimization_factor=None,  # CRITICAL for real tensile data
   )

For other models, widen bounds manually if needed — see
:ref:`bounds widening <dmta-bounds-widening>` in :doc:`dmta_numerical` for
the code pattern and a complete discussion of parameter bounds.

Workflow 8: Cross-Domain Validation
--------------------------------------

Validate fitted models by predicting across domains (frequency |leftrightarrow| time):

.. |leftrightarrow| unicode:: U+2194

.. code-block:: python

   from rheojax.models import GeneralizedMaxwell
   import numpy as np

   # Fit to frequency-domain E*(omega)
   gmm_freq = GeneralizedMaxwell(n_modes=15, modulus_type='tensile')
   gmm_freq.fit(omega, E_star_freq, test_mode='oscillation',
                optimization_factor=None)

   # Predict E(t) relaxation from the frequency-domain Prony
   E_pred_relax = gmm_freq.predict(t, test_mode='relaxation')

   # Compare to measured E(t)
   R2 = 1 - np.sum((E_relax - E_pred_relax)**2) / np.sum((E_relax - np.mean(E_relax))**2)
   print(f"Cross-domain R² = {R2:.4f}")

.. warning::

   Cross-domain prediction requires sufficient Prony modes to represent the
   full relaxation spectrum.  For a master curve spanning 20+ decades, use
   ``n_modes >= 15``.  With ``n_modes=5`` (FAST_MODE), cross-domain R² can
   be pathologically low or even negative.

.. seealso::

   - :doc:`dmta_models` --- model selection guide for choosing the right model
   - :doc:`dmta_numerical` --- convergence criteria, bounds, and FAST_MODE settings
   - :doc:`dmta_protocols` --- ISO/ASTM protocol mapping and heating rate recommendations
   - :doc:`dmta_knowledge` --- physical quantities extractable from fitted parameters
