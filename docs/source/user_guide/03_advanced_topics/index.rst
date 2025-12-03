.. _advanced_topics:

Section 3: Advanced Topics (Weeks 7-12)
========================================

**Deep dives into Bayesian inference, fractional viscoelasticity, and multi-technique analysis**

.. admonition:: Section Overview
   :class: note

   This section covers advanced analysis techniques for uncertainty quantification,
   fractional calculus applications, and simultaneous multi-technique fitting.

   **Timeline**: Weeks 7-12 (18-24 hours)

   **Prerequisites**: Sections 1-2 (Fundamentals and Model Usage)

Learning Objectives
-------------------

By completing this section, you will be able to:

1. Perform Bayesian inference to quantify parameter uncertainty
2. Understand fractional viscoelasticity conceptsand applications
3. Fit multiple experimental techniques simultaneously
4. Apply time-temperature superposition (TTS) for mastercurves
5. Interpret credible intervals and posterior distributions

Section Contents
----------------

.. toctree::
   :maxdepth: 2

   bayesian_inference
   fractional_viscoelasticity_reference
   multi_technique_fitting
   time_temperature_superposition
   ../spp_analysis

Section Roadmap
---------------

**Weeks 7-9: Bayesian Inference**

- :doc:`bayesian_inference` — MCMC sampling and uncertainty quantification

**Weeks 10-11: Fractional Viscoelasticity**

- :doc:`fractional_viscoelasticity_reference` — Fractional calculus theory

**Week 12: Advanced Fitting**

- :doc:`multi_technique_fitting` — Simultaneous SAOS + relaxation fitting
- :doc:`time_temperature_superposition` — Mastercurve construction

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

Next Steps
----------

After completing advanced topics, proceed to:

**Section 4: Practical Guides** (:doc:`../04_practical_guides/index`)

Learn production workflows, data I/O, visualization, and batch processing.
