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

The guide progresses through five sections over 16 weeks:

.. raw:: html

   <div style="margin: 20px 0;">
   <table style="width: 100%; border-collapse: collapse;">
   <tr style="background-color: #f0f0f0;">
     <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Section</th>
     <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Timeline</th>
     <th style="padding: 10px; text-align: left; border: 1px solid #ddd;">Focus</th>
   </tr>
   <tr>
     <td style="padding: 10px; border: 1px solid #ddd;">1. Fundamentals</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Weeks 1-2</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Core rheological concepts and terminology</td>
   </tr>
   <tr style="background-color: #f9f9f9;">
     <td style="padding: 10px; border: 1px solid #ddd;">2. Model Usage</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Weeks 3-6</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Practical model fitting and selection</td>
   </tr>
   <tr>
     <td style="padding: 10px; border: 1px solid #ddd;">3. Advanced Topics</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Weeks 7-12</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Bayesian inference, fractional calculus, multi-technique</td>
   </tr>
   <tr style="background-color: #f9f9f9;">
     <td style="padding: 10px; border: 1px solid #ddd;">4. Practical Guides</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Weeks 13-16</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Workflows, data I/O, visualization, batch processing</td>
   </tr>
   <tr>
     <td style="padding: 10px; border: 1px solid #ddd;">5. Appendices</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Reference</td>
     <td style="padding: 10px; border: 1px solid #ddd;">Experimental design, materials database, troubleshooting</td>
   </tr>
   </table>
   </div>

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

Alternative Learning Paths
---------------------------

**Path 1: Rapid Onboarding (1 week)**

For experienced rheologists familiar with commercial software:

1. :doc:`02_model_usage/getting_started` — RheoJAX basics
2. :doc:`02_model_usage/model_selection` — Choose the right model
3. :doc:`04_practical_guides/pipeline_api` — Build workflows
4. :doc:`04_practical_guides/data_io` — Import your data

**Path 2: Bayesian Focus (4 weeks)**

For researchers interested in uncertainty quantification:

1. :doc:`01_fundamentals/index` — Core concepts
2. :doc:`02_model_usage/getting_started` — Basic fitting
3. :doc:`03_advanced_topics/bayesian_inference` — Bayesian workflow
4. :doc:`04_practical_guides/visualization` — Posterior diagnostics

**Path 3: Fractional Viscoelasticity Deep Dive (6 weeks)**

For advanced users studying complex materials:

1. :doc:`01_fundamentals/material_classification` — Material types
2. :doc:`02_model_usage/model_families` — Model overview
3. :doc:`03_advanced_topics/fractional_viscoelasticity_reference` — Fractional calculus
4. :doc:`02_model_usage/fitting_strategies` — Initialization and validation

**Path 4: High-Throughput Analysis (2 weeks)**

For labs processing many samples:

1. :doc:`04_practical_guides/data_io` — Auto-detect file formats
2. :doc:`04_practical_guides/pipeline_api` — Automated workflows
3. :doc:`04_practical_guides/batch_processing` — Process multiple datasets
4. :doc:`05_appendices/troubleshooting` — Handle edge cases

**Path 5: Soft Glassy & LAOS Analysis (4 weeks)**

For researchers working with yield stress fluids, gels, and nonlinear rheology:

1. :doc:`01_fundamentals/material_classification` — Understanding soft materials
2. :doc:`03_advanced_topics/sgr_analysis` — SGR framework for soft glassy materials
3. :doc:`03_advanced_topics/spp_analysis` — SPP for LAOS yield stress extraction
4. :doc:`03_advanced_topics/bayesian_inference` — Uncertainty in SGR/SPP parameters

Related Documentation
---------------------

- :doc:`/quickstart` — 5-minute installation and first fit
- :doc:`/models/index` — Comprehensive Model Handbook (all equations and theory)
- :doc:`/transforms/index` — Transform Reference (mathematical details)
- :doc:`/api_reference` — Complete API documentation
- :doc:`/examples/index` — 24 example notebooks

Getting Help
------------

- **Self-Check Questions**: Test understanding throughout each section
- **Troubleshooting Guide**: :doc:`05_appendices/troubleshooting`
- **Glossary**: :doc:`05_appendices/glossary`
- **GitHub Issues**: Report bugs or request features
- **Examples**: :doc:`/examples/index` — 24 worked notebooks

Next Steps
----------

**Begin your learning journey:**

- **New to rheology?** Start with :doc:`01_fundamentals/what_is_rheology`
- **Ready to fit models?** Jump to :doc:`02_model_usage/getting_started`
- **Need quick reference?** Check :doc:`05_appendices/index`
