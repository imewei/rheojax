.. _advanced_topics:

Section 3: Advanced Topics (Weeks 7-12)
========================================

**Deep dives into Bayesian inference, fractional viscoelasticity, multi-technique analysis,
and specialized frameworks (SGR, SPP)**

.. admonition:: Section Overview
   :class: note

   This section covers advanced analysis techniques for uncertainty quantification,
   fractional calculus applications, simultaneous multi-technique fitting, and
   specialized frameworks for soft glassy materials (SGR) and LAOS analysis (SPP).

   **Timeline**: Weeks 7-12 (18-24 hours)

   **Prerequisites**: Sections 1-2 (Fundamentals and Model Usage)

Learning Objectives
-------------------

By completing this section, you will be able to:

1. Perform Bayesian inference to quantify parameter uncertainty
2. Understand fractional viscoelasticity concepts and applications
3. Fit multiple experimental techniques simultaneously
4. Apply time-temperature superposition (TTS) for mastercurves
5. Interpret credible intervals and posterior distributions
6. Classify soft glassy materials using the SGR framework
7. Extract yield stresses and cage moduli from LAOS using SPP

Section Contents
----------------

.. toctree::
   :maxdepth: 2
   :caption: Core Advanced Topics

   bayesian_inference
   fractional_viscoelasticity_reference
   multi_technique_fitting
   time_temperature_superposition

.. toctree::
   :maxdepth: 2
   :caption: Specialized Frameworks

   sgr_analysis
   spp_analysis

Section Roadmap
---------------

**Weeks 7-9: Bayesian Inference**

- :doc:`bayesian_inference` — MCMC sampling and uncertainty quantification

**Weeks 10-11: Fractional Viscoelasticity**

- :doc:`fractional_viscoelasticity_reference` — Fractional calculus theory

**Week 12: Advanced Fitting & Specialized Frameworks**

- :doc:`multi_technique_fitting` — Simultaneous SAOS + relaxation fitting
- :doc:`time_temperature_superposition` — Mastercurve construction
- :doc:`sgr_analysis` — Soft Glassy Rheology for complex fluids
- :doc:`spp_analysis` — Sequence of Physical Processes for LAOS

Key Skills Summary
------------------

**Bayesian Workflow** (NLSQ → NUTS):

.. code-block:: python

   from rheojax.models.maxwell import Maxwell

   # 1. NLSQ point estimate (fast)
   model = Maxwell()
   model.fit(t, G_t, test_mode='relaxation')

   # 2. Bayesian inference (uncertainty quantification)
   result = model.fit_bayesian(t, G_t, num_samples=2000, num_warmup=1000)

   # 3. Credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)

**Multi-Technique Fitting**:

.. code-block:: python

   from rheojax.pipeline import ModelComparisonPipeline

   pipeline = ModelComparisonPipeline()
   pipeline.fit_multi_technique(
       techniques=['relaxation', 'oscillation'],
       data_dict={'relaxation': (t, G_t), 'oscillation': (omega, G_star)}
   )

**SGR Phase Classification**:

.. code-block:: python

   from rheojax.models import SGRConventional

   model = SGRConventional()
   model.fit(omega, G_star, test_mode='oscillation')

   x = model.parameters.get_value('x')
   print(f"x = {x:.2f}: {'Glass' if x < 1 else 'Fluid'}")

**SPP Yield Stress Extraction**:

.. code-block:: python

   from rheojax.transforms import SPPDecomposer

   decomposer = SPPDecomposer(omega=1.0, gamma_0=1.0, n_harmonics=39)
   result = decomposer.transform(laos_data)
   spp_results = decomposer.get_results()

   print(f"Cage modulus: {spp_results['G_cage']:.1f} Pa")
   print(f"Static yield stress: {spp_results['sigma_sy']:.1f} Pa")

Next Steps
----------

After completing advanced topics, proceed to:

**Section 4: Practical Guides** (:doc:`../04_practical_guides/index`)

Learn production workflows, data I/O, visualization, and batch processing.
