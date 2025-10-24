Models API
==========

This page documents all 20 rheological models implemented in rheo.

Overview
--------

rheo provides three families of models:

- **Classical Models** (3): Spring-dashpot combinations
- **Fractional Models** (11): Power-law viscoelastic behavior
- **Non-Newtonian Flow Models** (6): Shear-rate dependent viscosity

All models inherit from :class:`rheo.core.base.BaseModel` and follow the scikit-learn API pattern with ``fit()``, ``predict()``, and ``score()`` methods.

Model Registry
--------------

Access models through the registry:

.. code-block:: python

   from rheo.core.registry import ModelRegistry

   # List all available models
   models = ModelRegistry.list_models()

   # Create model by name
   model = ModelRegistry.create('maxwell')

   # Get model information
   info = ModelRegistry.get_info('fractional_maxwell_gel')

Classical Models
----------------

Maxwell Model
~~~~~~~~~~~~~

.. autoclass:: rheo.models.Maxwell
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Two-parameter viscoelastic model with a spring and dashpot in series.

   **Parameters**:
      - G_s (Pa): Shear modulus of spring element
      - eta_s (Pa·s): Viscosity of dashpot element

   **Test Modes**: Relaxation, Oscillation

   **Example**:

   .. code-block:: python

      from rheo.models import Maxwell
      import numpy as np

      # Create model
      maxwell = Maxwell()

      # Fit to oscillation data
      omega = np.logspace(-1, 2, 50)  # rad/s
      G_star = 1e5 / (1 + 1j * omega * 0.1)  # Synthetic data
      maxwell.fit(omega, np.abs(G_star))

      # Get parameters
      G_s = maxwell.parameters.get_value('G_s')
      eta_s = maxwell.parameters.get_value('eta_s')
      tau = eta_s / G_s

      print(f"Relaxation time: {tau:.3f} s")

Zener Model
~~~~~~~~~~~

.. autoclass:: rheo.models.Zener
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Three-parameter standard linear solid (SLS) model.

   **Parameters**:
      - G_s (Pa): Equilibrium modulus
      - G_p (Pa): Parallel spring modulus
      - eta_p (Pa·s): Parallel dashpot viscosity

   **Test Modes**: Relaxation, Creep, Oscillation

   **Example**:

   .. code-block:: python

      from rheo.models import Zener

      zener = Zener()
      zener.fit(omega, G_star)

      # Zener shows solid-like plateau at low frequency
      # and relaxation at high frequency

SpringPot Model
~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.SpringPot
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Two-parameter fractional power-law element.

   **Parameters**:
      - V (Pa·s^α): Fractional stiffness
      - alpha (-): Fractional order (0 < alpha < 1)

   **Test Modes**: Oscillation, Relaxation

   **Example**:

   .. code-block:: python

      from rheo.models import SpringPot

      springpot = SpringPot()
      springpot.fit(omega, G_star)

      alpha = springpot.parameters.get_value('alpha')
      # alpha ≈ 0: solid-like
      # alpha ≈ 1: liquid-like
      # 0 < alpha < 1: power-law viscoelastic

Fractional Maxwell Family
--------------------------

FractionalMaxwellGel
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalMaxwellGel
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Spring in series with SpringPot (gel-like behavior).

   **Parameters**:
      - G_s (Pa): Spring modulus
      - V (Pa·s^α): SpringPot stiffness
      - alpha (-): Fractional order

   **Example**:

   .. code-block:: python

      from rheo.models import FractionalMaxwellGel

      fmg = FractionalMaxwellGel()
      fmg.fit(omega, G_star)

      # Good for materials with elastic plateau and power-law relaxation

FractionalMaxwellLiquid
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalMaxwellLiquid
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: SpringPot in series with dashpot (liquid-like with memory).

   **Parameters**:
      - V (Pa·s^α): SpringPot stiffness
      - alpha (-): Fractional order
      - eta_s (Pa·s): Dashpot viscosity

FractionalMaxwellModel
~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalMaxwellModel
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: General fractional Maxwell with two SpringPots in series.

   **Parameters**: 4 parameters for maximum flexibility

FractionalKelvinVoigt
~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalKelvinVoigt
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Spring and SpringPot in parallel (solid-like with slow relaxation).

   **Parameters**:
      - G_p (Pa): Parallel spring modulus
      - V (Pa·s^α): SpringPot stiffness
      - alpha (-): Fractional order
      - eta_p (Pa·s, optional): Parallel dashpot viscosity

Fractional Zener Family
------------------------

FractionalZenerSolidLiquid (FZSL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalZenerSolidLiquid
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FZSL
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Fractional Maxwell + spring in parallel (solid + fractional liquid).

   **Parameters**:
      - G_s (Pa): Series spring modulus
      - eta_s (Pa·s): Series dashpot viscosity
      - V (Pa·s^α): SpringPot stiffness
      - alpha (-): Fractional order

   **Example**:

   .. code-block:: python

      from rheo.models import FZSL  # Short alias

      fzsl = FZSL()
      fzsl.fit(omega, G_star)

      # Good for polymer melts with plateau and power-law flow

FractionalZenerSolidSolid (FZSS)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalZenerSolidSolid
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FZSS
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Two springs + SpringPot (double elastic plateau).

   **Parameters**:
      - G_s (Pa): Series spring modulus
      - G_p (Pa): Parallel spring modulus
      - V (Pa·s^α): SpringPot stiffness
      - alpha (-): Fractional order

FractionalZenerLiquidLiquid (FZLL)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalZenerLiquidLiquid
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FZLL
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Most general fractional Zener (two dashpots + SpringPot).

   **Parameters**: 4 parameters for complex behavior

FractionalKelvinVoigtZener (FKVZ)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalKelvinVoigtZener
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FKVZ
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Fractional Kelvin-Voigt + spring in series.

   **Parameters**: 4 parameters

Advanced Fractional Models
---------------------------

FractionalBurgersModel (FBM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalBurgersModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FBM
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Maxwell + Fractional Kelvin-Voigt in series (creep + relaxation).

   **Parameters**: 5 parameters for complex time-dependent behavior

   **Example**:

   .. code-block:: python

      from rheo.models import FBM

      fbm = FBM()
      fbm.fit(time, creep_compliance)

      # Excellent for materials showing both creep and relaxation

FractionalPoyntingThomson (FPT)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalPoyntingThomson
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FPT
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Fractional Kelvin-Voigt + spring in series (alternative formulation).

   **Parameters**: 5 parameters

FractionalJeffreysModel (FJM)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.FractionalJeffreysModel
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

.. autoclass:: rheo.models.FJM
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Two dashpots + SpringPot (liquid-like with fractional element).

   **Parameters**: 4 parameters

Non-Newtonian Flow Models
--------------------------

These models describe shear-rate dependent viscosity for steady shear flows.

**Test Mode**: Rotation (steady shear) only

PowerLaw
~~~~~~~~

.. autoclass:: rheo.models.PowerLaw
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Simple power-law: τ = K·γ̇^n

   **Parameters**:
      - K (Pa·s^n): Consistency index
      - n (-): Flow index (n < 1: shear thinning, n > 1: shear thickening)

   **Example**:

   .. code-block:: python

      from rheo.models import PowerLaw
      import numpy as np

      # Shear thinning data
      shear_rate = np.logspace(-2, 2, 50)  # 1/s
      viscosity = 10 * shear_rate**(-0.3)  # Synthetic
      stress = viscosity * shear_rate

      power_law = PowerLaw()
      power_law.fit(shear_rate, stress)

      K = power_law.parameters.get_value('K')
      n = power_law.parameters.get_value('n')

      print(f"Flow index n = {n:.3f}")
      # n < 1: shear thinning

Carreau
~~~~~~~

.. autoclass:: rheo.models.Carreau
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Smooth transition from Newtonian to power-law behavior.

   **Equation**: η = η_∞ + (η_0 - η_∞) · [1 + (λ·γ̇)²]^((n-1)/2)

   **Parameters**:
      - eta_0 (Pa·s): Zero-shear viscosity
      - eta_inf (Pa·s): Infinite-shear viscosity
      - lambda (s): Time constant
      - n (-): Power-law index

   **Example**:

   .. code-block:: python

      from rheo.models import Carreau

      carreau = Carreau()
      carreau.fit(shear_rate, viscosity)

      # Good for polymer solutions

CarreauYasuda
~~~~~~~~~~~~~

.. autoclass:: rheo.models.CarreauYasuda
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Extended Carreau with transition sharpness control.

   **Equation**: η = η_∞ + (η_0 - η_∞) · [1 + (λ·γ̇)^a]^((n-1)/a)

   **Parameters**: 5 parameters (adds parameter 'a' for transition sharpness)

Cross
~~~~~

.. autoclass:: rheo.models.Cross
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Alternative to Carreau for polymer solutions.

   **Equation**: η = η_∞ + (η_0 - η_∞) / [1 + (K·γ̇)^m]

   **Parameters**:
      - eta_0 (Pa·s): Zero-shear viscosity
      - eta_inf (Pa·s): Infinite-shear viscosity
      - K (s^m): Consistency parameter
      - m (-): Rate constant

HerschelBulkley
~~~~~~~~~~~~~~~

.. autoclass:: rheo.models.HerschelBulkley
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Power-law with yield stress (τ_0).

   **Equation**: τ = τ_0 + K·γ̇^n

   **Parameters**:
      - tau_0 (Pa): Yield stress
      - K (Pa·s^n): Consistency index
      - n (-): Flow index

   **Example**:

   .. code-block:: python

      from rheo.models import HerschelBulkley

      hb = HerschelBulkley()
      hb.fit(shear_rate, stress)

      tau_0 = hb.parameters.get_value('tau_0')
      print(f"Yield stress: {tau_0:.2f} Pa")

      # Good for pastes, suspensions, food products

Bingham
~~~~~~~

.. autoclass:: rheo.models.Bingham
   :members:
   :undoc-members:
   :show-inheritance:

   **Description**: Linear viscoplastic (yield stress + Newtonian).

   **Equation**: τ = τ_0 + η_pl·γ̇

   **Parameters**:
      - tau_0 (Pa): Yield stress
      - eta_pl (Pa·s): Plastic viscosity

   **Example**:

   .. code-block:: python

      from rheo.models import Bingham

      bingham = Bingham()
      bingham.fit(shear_rate, stress)

      # Simpler than Herschel-Bulkley when n ≈ 1

Model Comparison
----------------

Comparing Multiple Models
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheo.models import Maxwell, Zener, FractionalMaxwellGel
   import numpy as np

   models = [Maxwell(), Zener(), FractionalMaxwellGel()]
   model_names = ['Maxwell', 'Zener', 'FractionalMaxwellGel']

   for name, model in zip(model_names, models):
       model.fit(omega, G_star)
       r2 = model.score(omega, G_star)
       n_params = len(model.parameters)

       print(f"{name:25} R² = {r2:.4f}  ({n_params} parameters)")

   # Select best model by AIC
   aic_values = []
   for model in models:
       y_pred = model.predict(omega)
       rss = np.sum((np.abs(G_star) - np.abs(y_pred))**2)
       n = len(omega)
       k = len(model.parameters)
       aic = n * np.log(rss/n) + 2 * k
       aic_values.append(aic)

   best_idx = np.argmin(aic_values)
   print(f"\nBest model (AIC): {model_names[best_idx]}")

See Also
--------

- :doc:`/user_guide/model_selection` - Decision tree for model selection
- :doc:`/user_guide/modular_api` - Direct model usage examples
- :doc:`/user_guide/multi_technique_fitting` - Fitting across multiple test modes
- :class:`rheo.core.base.BaseModel` - Base class documentation
