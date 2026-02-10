.. _user_guide:

User Guide: Graduate Student Learning Pathway
==============================================

Welcome to the RheoJAX User Guide, structured as a **16-week graduate course** in rheological analysis with JAX acceleration.

.. admonition:: How to Use This Guide
   :class: tip

    This guide is organized progressively from fundamentals to advanced topics.
    Each section includes:

    - **Learning Objectives**: What you'll master in this section
    - **Key Concepts**: Essential takeaways presented conceptually
    - **Worked Examples**: Practical code demonstrations
    - **Self-Check Questions**: Test your understanding
    - **Further Reading**: Deep dives into mathematical details

    **For beginners:** Start with Section 1 (Fundamentals) and progress sequentially

    **For experienced users:** Jump directly to Section 3 (Advanced Topics) or Section 4 (Practical Guides)

    **For quick reference:** Use the :doc:`/models/index` or :doc:`/api_reference`

About This Guide
----------------

This User Guide emphasizes **conceptual understanding** and **practical application** over mathematical derivations.
All equations and detailed theory are available in the Model Handbook and Transform Reference sections.

The guide is designed for:

- Graduate students learning rheology and computational analysis
- Researchers transitioning from commercial software to Python-based workflows
- Scientists seeking to understand rheological model selection and Bayesian inference
- Engineers applying rheology to materials characterization

Learning Pathway Overview
--------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 15 60

   * - Section
     - Timeline
     - Focus
   * - 1. Fundamentals
     - Weeks 1-2
     - Core rheological concepts and terminology
   * - 2. Model Usage
     - Weeks 3-6
     - Practical model fitting and selection
   * - 3. Advanced Topics
     - Weeks 7-12
     - Bayesian inference, fractional calculus, multi-technique
   * - 4. Practical Guides
     - Weeks 13-16
     - Workflows, data I/O, visualization, batch processing
   * - 5. Appendices
     - Reference
     - Experimental design, materials database, troubleshooting

----

Section 1: Fundamentals (Weeks 1-2)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Foundation in rheological concepts and terminology**

Build your understanding of stress, strain, viscoelasticity, and material classification.
Learn to interpret rheological parameters and understand different test modes.

.. toctree::
   :maxdepth: 2

   01_fundamentals/index

**Key Topics:**

- What is rheology and why does it matter?
- Material classification: liquids, solids, gels
- Test modes: SAOS, relaxation, creep, flow
- Physical meaning of rheological parameters

Section 2: Model Usage (Weeks 3-6)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Practical model fitting and selection strategies**

Master the art of choosing appropriate models, fitting experimental data, and validating results.
Learn initialization strategies and troubleshooting techniques.

.. toctree::
   :maxdepth: 2

   02_model_usage/index

**Key Topics:**

- Getting started with model fitting
- Understanding model families (classical, fractional, flow)
- Model selection flowcharts and decision trees
- Fitting strategies and validation

Section 3: Advanced Topics (Weeks 7-12)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Deep dives into Bayesian inference, fractional viscoelasticity, and multi-technique analysis**

Explore uncertainty quantification, fractional calculus applications, and simultaneous fitting
of multiple experimental techniques.

.. toctree::
   :maxdepth: 2

   03_advanced_topics/index

**Key Topics:**

- Bayesian inference for parameter uncertainty
- Fractional viscoelasticity concepts and applications
- Multi-technique fitting strategies
- Time-temperature superposition

Section 4: Practical Guides (Weeks 13-16)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**Workflows, data management, and production-ready analysis**

Learn to build efficient analysis pipelines, manage diverse data formats, create publication-quality
visualizations, and process multiple datasets in batch mode.

.. toctree::
   :maxdepth: 2

   04_practical_guides/index

**Key Topics:**

- Pipeline API for fluent workflows
- Modular API for low-level control
- Data I/O for multiple instrument formats
- Visualization and batch processing

Section 5: Appendices
~~~~~~~~~~~~~~~~~~~~~~

**Reference materials for experimental design, material properties, and troubleshooting**

Quick-reference guides consolidating best practices, material databases, and common solutions
to fitting problems.

.. toctree::
   :maxdepth: 2

   05_appendices/index

**Reference Materials:**

- Experimental design guidelines
- Material property database (100+ materials)
- Troubleshooting guide
- Glossary of rheological terms

Section 6: GUI Reference
~~~~~~~~~~~~~~~~~~~~~~~~

**Graphical user interface for interactive analysis**

Learn to use the optional RheoJAX GUI for visual data exploration, interactive
model fitting, and Bayesian inference with ArviZ diagnostics.

.. toctree::
   :maxdepth: 2

   06_gui/index

**GUI Topics:**

- Getting started with the GUI
- Data loading and visualization
- Interactive model fitting
- Bayesian inference and diagnostics
- Transforms and exporting

----

Alternative Learning Paths
---------------------------

Not following the 16-week course? Choose a path that matches your goals:

.. admonition:: Rapid Onboarding (1 week)
   :class: tip

   For experienced rheologists familiar with commercial software:

   1. :doc:`02_model_usage/getting_started` — RheoJAX basics
   2. :doc:`02_model_usage/model_selection` — Choose the right model
   3. :doc:`04_practical_guides/pipeline_api` — Build workflows
   4. :doc:`04_practical_guides/data_io` — Import your data

.. admonition:: Bayesian Focus (4 weeks)
   :class: note

   For researchers interested in uncertainty quantification:

   1. :doc:`01_fundamentals/index` — Core concepts
   2. :doc:`02_model_usage/getting_started` — Basic fitting
   3. :doc:`03_advanced_topics/bayesian_inference` — Bayesian workflow
   4. :doc:`04_practical_guides/visualization` — Posterior diagnostics

.. admonition:: Fractional Viscoelasticity (6 weeks)
   :class: note

   For advanced users studying complex materials:

   1. :doc:`01_fundamentals/material_classification` — Material types
   2. :doc:`02_model_usage/model_families` — Model overview
   3. :doc:`03_advanced_topics/fractional_viscoelasticity_reference` — Fractional calculus
   4. :doc:`02_model_usage/fitting_strategies` — Initialization and validation

.. admonition:: High-Throughput Analysis (2 weeks)
   :class: tip

   For labs processing many samples:

   1. :doc:`04_practical_guides/data_io` — Auto-detect file formats
   2. :doc:`04_practical_guides/pipeline_api` — Automated workflows
   3. :doc:`04_practical_guides/batch_processing` — Process multiple datasets
   4. :doc:`05_appendices/troubleshooting` — Handle edge cases

.. admonition:: Soft Glassy & LAOS Analysis (4 weeks)
   :class: note

   For researchers working with yield stress fluids, gels, and nonlinear rheology:

   1. :doc:`01_fundamentals/material_classification` — Understanding soft materials
   2. :doc:`03_advanced_topics/sgr_analysis` — SGR framework for soft glassy materials
   3. :doc:`03_advanced_topics/spp_analysis` — SPP for LAOS yield stress extraction
   4. :doc:`03_advanced_topics/bayesian_inference` — Uncertainty in SGR/SPP parameters

.. admonition:: Thixotropy & Yield Stress (4 weeks)
   :class: note

   For researchers working with drilling muds, waxy oils, emulsions, and structured fluids:

   1. :doc:`01_fundamentals/material_classification` — Material behavior fundamentals
   2. :doc:`03_advanced_topics/thixotropy_yielding` — DMT, Fluidity, HL, STZ, EPM models
   3. :doc:`03_advanced_topics/constitutive_ode_models` — IKH/FIKH kinematic hardening
   4. :doc:`03_advanced_topics/bayesian_inference` — Uncertainty quantification

.. admonition:: Polymer Network Theory (4 weeks)
   :class: note

   For researchers studying associative polymers, wormlike micelles, and biological gels:

   1. :doc:`02_model_usage/getting_started` — Basic fitting workflow
   2. :doc:`03_advanced_topics/constitutive_ode_models` — Giesekus nonlinear viscoelasticity
   3. :doc:`03_advanced_topics/polymer_network_models` — TNT and VLB frameworks
   4. :doc:`03_advanced_topics/vitrimer_models` — HVM/HVNM for covalent adaptable networks

.. admonition:: Smart Materials & Nanocomposites (3 weeks)
   :class: note

   For researchers working with vitrimers, nanocomposites, and adaptive materials:

   1. :doc:`03_advanced_topics/polymer_network_models` — VLB distribution tensor foundations
   2. :doc:`03_advanced_topics/vitrimer_models` — HVM and HVNM models
   3. :doc:`03_advanced_topics/bayesian_inference` — Bayesian parameter estimation

.. admonition:: Dense Suspensions & Glasses (4 weeks)
   :class: note

   For researchers studying colloidal glasses, metallic glasses, and amorphous solids:

   1. :doc:`03_advanced_topics/sgr_analysis` — SGR soft glassy framework
   2. :doc:`03_advanced_topics/dense_suspensions_glasses` — ITT-MCT mode-coupling theory
   3. :doc:`03_advanced_topics/thixotropy_yielding` — STZ and EPM for amorphous solids
   4. :doc:`03_advanced_topics/bayesian_inference` — Phase classification with uncertainty

----

Related Documentation
---------------------

- :doc:`/quickstart` — 5-minute installation and first fit
- :doc:`/models/index` — Comprehensive Model Handbook (53 models, all equations and theory)
- :doc:`/transforms/index` — Transform Reference (7 transforms, mathematical details)
- :doc:`/api_reference` — Complete API documentation
- :doc:`/examples/index` — 170+ example notebooks across all model families

Getting Help
------------

- **Troubleshooting Guide**: :doc:`05_appendices/troubleshooting`
- **Glossary**: :doc:`05_appendices/glossary`
- **GitHub Issues**: Report bugs or request features
- **Examples**: :doc:`/examples/index` — 170+ worked notebooks across all model families

Next Steps
----------

**Begin your learning journey:**

- **New to rheology?** Start with :doc:`01_fundamentals/what_is_rheology`
- **Ready to fit models?** Jump to :doc:`02_model_usage/getting_started`
- **Need quick reference?** Check :doc:`05_appendices/index`
