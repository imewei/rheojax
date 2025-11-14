.. _model_usage:

Section 2: Model Usage (Weeks 3-6)
===================================

**Practical model fitting and selection strategies**

.. admonition:: Section Overview
   :class: note

   This section teaches you how to apply rheological models to experimental data.
   You will learn to choose appropriate models, initialize parameters, fit data,
   validate results, and troubleshoot common problems.

   **Timeline**: Weeks 3-6 (12-16 hours)

   **Prerequisites**: Section 1 (Fundamentals) — Core rheology concepts

Learning Objectives
-------------------

By completing this section, you will be able to:

1. Fit basic rheological models (Maxwell, Zener, PowerLaw) to experimental data
2. Navigate the model landscape (classical, fractional, flow models)
3. Select appropriate models using decision flowcharts and compatibility checking
4. Initialize parameters intelligently for faster convergence
5. Validate fitted models and identify poor fits
6. Troubleshoot common fitting problems (convergence, unphysical parameters)

Section Contents
----------------

.. toctree::
   :maxdepth: 2

   getting_started
   model_families
   model_selection
   fitting_strategies

Section Roadmap
---------------

**Week 3: Getting Started**

- :doc:`getting_started` — Your first model fit in 10 lines of code

**Week 4: Model Landscape**

- :doc:`model_families` — Understanding classical, fractional, and flow model families

**Week 5: Model Selection**

- :doc:`model_selection` — Decision flowcharts and compatibility checking

**Week 6: Fitting Strategies**

- :doc:`fitting_strategies` — Initialization, validation, and troubleshooting

Key Skills You'll Develop
--------------------------

**1. Basic Fitting Workflow**

.. code-block:: python

   from rheojax.models.maxwell import Maxwell
   import numpy as np

   # Load data (time, stress)
   t = np.loadtxt('relaxation.csv', delimiter=',', usecols=0)
   G_t = np.loadtxt('relaxation.csv', delimiter=',', usecols=1)

   # Create and fit model
   model = Maxwell()
   model.fit(t, G_t, test_mode='relaxation')

   # Inspect results
   print(f"G0 = {model.parameters.get_value('G0'):.3e} Pa")
   print(f"eta = {model.parameters.get_value('eta'):.3e} Pa·s")
   print(f"tau = {model.parameters.get_value('G0') / model.parameters.get_value('eta'):.2f} s")

**2. Model Selection Decision Tree**

.. code-block:: text

   [What type of data do you have?]
      │
      ├─→ Oscillation (SAOS)
      │     │
      │     ├─→ Liquid-like (G" > G' at low ω) → Maxwell, FML
      │     ├─→ Solid-like (G' plateaus) → Zener, FZSS
      │     └─→ Gel-like (G' ~ ω^α) → FMG, SpringPot
      │
      ├─→ Relaxation
      │     │
      │     ├─→ Exponential decay → Maxwell, Zener
      │     ├─→ Power-law decay → Fractional models
      │     └─→ Plateau at long times → Zener, FZSS
      │
      └─→ Steady shear flow
            │
            ├─→ Newtonian → PowerLaw (n=1)
            ├─→ Shear thinning → PowerLaw, Carreau
            └─→ Yield stress → Herschel-Bulkley, Bingham

**3. Validation Checklist**

.. code-block:: python

   # After fitting, always validate:

   # 1. Visual inspection
   model.plot(t, G_t)  # Predicted vs. actual

   # 2. Parameter reasonableness
   assert model.parameters.get_value('G0') > 0, "Modulus must be positive"
   assert 1e-6 < model.parameters.get_value('tau') < 1e6, "Relaxation time unrealistic"

   # 3. Residual analysis
   predictions = model.predict(t)
   residuals = G_t - predictions
   relative_error = np.mean(np.abs(residuals / G_t)) * 100
   print(f"Mean absolute error: {relative_error:.2f}%")

   # 4. Compatibility checking (NEW in v0.2.0)
   from rheojax.utils.compatibility import check_model_compatibility
   compat = check_model_compatibility(model, t=t, G_t=G_t, test_mode='relaxation')
   if not compat['compatible']:
       print(f"Warning: {compat['message']}")

**4. Intelligent Initialization** (NEW in v0.2.0)

.. code-block:: python

   # Smart initialization for fractional models in oscillation mode
   from rheojax.models.fractional_zener_ss import FractionalZenerSolidSolid

   omega = np.logspace(-2, 2, 50)
   G_star = ...  # Complex modulus [G', G"]

   # Automatic smart initialization (no user action needed)
   model = FractionalZenerSolidSolid()
   model.fit(omega, G_star, test_mode='oscillation')
   # Parameters automatically initialized from frequency features

   # Manual initialization (optional)
   model.parameters.set_value('Ge', 1e4)  # Equilibrium modulus
   model.parameters.set_value('alpha', 0.5)  # Fractional order
   model.fit(omega, G_star, test_mode='oscillation')

Learning Pathway Variations
----------------------------

**Fast Track (Experienced Users)**

If you're already familiar with rheology and just need RheoJAX syntax:

1. :doc:`getting_started` — Basic API (15 minutes)
2. :doc:`model_selection` — Choose your model (15 minutes)
3. Skip to :doc:`../04_practical_guides/pipeline_api` — Production workflows

**Deep Dive (Research Focus)**

For comprehensive model understanding:

1. :doc:`getting_started` — Basics
2. :doc:`model_families` — All 20 models
3. :doc:`model_selection` — Decision framework
4. :doc:`fitting_strategies` — Advanced techniques
5. :doc:`../03_advanced_topics/fractional_viscoelasticity_reference` — Fractional calculus

**Application-Focused (Industry)**

For solving specific material problems:

1. :doc:`getting_started` — Quick start
2. :doc:`model_selection` — Flowchart to your model
3. :doc:`../04_practical_guides/batch_processing` — High-throughput analysis
4. :doc:`../05_appendices/troubleshooting` — Common issues

Common Pitfalls and How to Avoid Them
--------------------------------------

**Pitfall 1: Wrong Test Mode**

.. code-block:: python

   # WRONG: Fitting flow data with viscoelastic model
   from rheojax.models.maxwell import Maxwell
   model = Maxwell()
   model.fit(shear_rate, viscosity, test_mode='rotation')  # Will fail!

   # CORRECT: Use flow model for steady shear data
   from rheojax.models.power_law import PowerLaw
   model = PowerLaw()
   model.fit(shear_rate, viscosity, test_mode='rotation')  # ✓

**Pitfall 2: Ignoring Compatibility Warnings** (NEW)

.. code-block:: python

   # Enable compatibility checking
   model.fit(t, G_t, test_mode='relaxation', check_compatibility=True)

   # If fitting fails, check the error message for physics-based guidance
   # Example: "FZSS expects power-law decay but detected exponential.
   #           Try: Maxwell, Zener, FractionalMaxwellLiquid instead."

**Pitfall 3: Poor Initialization**

.. code-block:: python

   # WRONG: Default bounds may be too wide
   model.fit(omega, G_star)  # May not converge

   # BETTER: Use smart initialization (automatic for fractional models)
   model.fit(omega, G_star, test_mode='oscillation')  # Smart init applied

   # OR: Provide reasonable initial guess
   model.parameters.set_value('G0', 1e5)  # Estimate from data
   model.fit(omega, G_star)

**Pitfall 4: Over-fitting**

.. code-block:: python

   # WRONG: Using complex model for simple data
   from rheojax.models.fractional_burgers import FractionalBurgers  # 7 parameters
   model.fit(omega, G_star)  # Data only has 10 points → overfitting!

   # CORRECT: Start with simplest model
   from rheojax.models.maxwell import Maxwell  # 2 parameters
   model.fit(omega, G_star)

   # Add complexity only if needed

Assessment
----------

.. admonition:: Self-Assessment Checklist
   :class: tip

   After completing this section, you should be able to:

   ☐ Fit a Maxwell model to stress relaxation data

   ☐ Choose between classical, fractional, and flow model families

   ☐ Use the model selection flowchart to pick an appropriate model

   ☐ Initialize fractional model parameters from frequency sweep data

   ☐ Validate fitted parameters for physical reasonableness

   ☐ Troubleshoot convergence problems

   ☐ Interpret compatibility warnings and select alternative models

Resources
---------

**Model Handbook** (detailed equations and theory):

- :doc:`/models/classical/index` — Maxwell, Zener, SpringPot
- :doc:`/models/fractional/index` — All 11 fractional models
- :doc:`/models/flow/index` — PowerLaw, Carreau, Herschel-Bulkley

**Example Notebooks**:

- ``examples/basic/01-maxwell_fitting.ipynb`` — Step-by-step tutorial
- ``examples/basic/02-model_comparison.ipynb`` — Comparing multiple models
- ``examples/advanced/05-model_selection.ipynb`` — Automated model selection

Next Steps
----------

After mastering model usage, proceed to:

**Section 3: Advanced Topics** (:doc:`../03_advanced_topics/index`)

Learn Bayesian inference for uncertainty quantification, fractional viscoelasticity theory,
and multi-technique fitting strategies.
