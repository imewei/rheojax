.. _model-{model_slug}:

{title_underline}
{model_name} — Handbook
{title_underline}

.. note:: Documentation Template

   This is a template for RheoJAX model documentation. Replace placeholders with
   actual content following the guidelines in ``_guides/model_documentation_style.rst``.

   Target length: 500-1200 lines depending on model complexity.

Quick Reference
---------------

- **Use when:** {one-line summary of when to use this model}
- **Parameters:** {N} ({list of parameter names})
- **Key equation:** :math:`{primary equation in LaTeX}`
- **Test modes:** {comma-separated list: Oscillation, relaxation, creep, steady shear, etc.}
- **Material examples:** {comma-separated list of 4-6 representative materials}

Notation Guide
--------------

.. list-table::
   :widths: 15 85
   :header-rows: 1

   * - Symbol
     - Meaning
   * - :math:`{symbol_1}`
     - {Description of symbol_1}
   * - :math:`{symbol_2}`
     - {Description of symbol_2}

Overview
--------

{2-3 paragraphs introducing the model, its purpose, and physical motivation.
Include key distinguishing features and why this model matters.}

Historical Context
~~~~~~~~~~~~~~~~~~

{1-2 paragraphs on the development of this model. Who created it, when, and why.
What problems does it solve that earlier models couldn't?}

----

Physical Foundations
--------------------

Mechanical/Microstructural Analogue
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

{Description of the physical picture underlying the model. For classical models,
this is a mechanical analogue (springs, dashpots). For microstructural models,
describe the mesoscopic elements, their interactions, and dynamics.}

.. code-block:: text

   {ASCII diagram of mechanical analogue or schematic}

Material Examples with Typical Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Representative parameters for common materials
   :header-rows: 1
   :widths: 30 15 15 15 15 10

   * - Material
     - {Param1}
     - {Param2}
     - {Param3}
     - {Derived}
     - Ref
   * - {Material 1}
     - {value}
     - {value}
     - {value}
     - {value}
     - [{n}]_
   * - {Material 2}
     - {value}
     - {value}
     - {value}
     - {value}
     - [{n}]_

Connection to Physics
~~~~~~~~~~~~~~~~~~~~~

{Explain the molecular or microstructural origin of each parameter. For polymer
models, connect to molecular weight, entanglement, etc. For colloidal/soft
matter models, connect to particle interactions, jamming, etc.}

**Scaling laws** (if applicable):
   - {Parameter} :math:`\sim` {molecular property}
   - {Parameter} :math:`\sim` {physical observable}

----

Governing Equations
-------------------

Mathematical Derivation
~~~~~~~~~~~~~~~~~~~~~~~

{Step-by-step derivation of the constitutive equation from physical principles.
Number each step clearly.}

**Step 1**: {Starting point}
   {Equation or relation}

**Step 2**: {Next step}
   {Equation or relation}

**Step N**: Final constitutive form

.. math::

   {Constitutive equation in differential or integral form}

Predictions for Test Modes
~~~~~~~~~~~~~~~~~~~~~~~~~~

**Oscillatory Shear (SAOS)**

.. math::

   G^*(\omega) = {complex modulus expression}

   G'(\omega) = {storage modulus}

   G''(\omega) = {loss modulus}

**Stress Relaxation**

.. math::

   G(t) = {relaxation modulus expression}

**Creep Compliance** (if applicable)

.. math::

   J(t) = {creep compliance expression}

**Steady Shear / Flow Curve** (if applicable)

.. math::

   \sigma(\dot{\gamma}) = {flow curve expression}

   \eta(\dot{\gamma}) = {viscosity expression}

Limiting Cases and Asymptotic Behavior
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Asymptotic behavior
   :header-rows: 1
   :widths: 20 25 25 30

   * - Regime
     - :math:`G'(\omega)`
     - :math:`G''(\omega)`
     - Physical interpretation
   * - Low :math:`\omega`
     - :math:`\sim {exponent}`
     - :math:`\sim {exponent}`
     - {interpretation}
   * - High :math:`\omega`
     - :math:`\to {limit}`
     - :math:`\to {limit}`
     - {interpretation}

----

Parameters
----------

.. list-table:: Parameters
   :header-rows: 1
   :widths: 15 12 12 18 43

   * - Name
     - Symbol
     - Units
     - Bounds
     - Notes
   * - ``{param_name}``
     - :math:`{symbol}`
     - {units}
     - :math:`{bounds}`
     - {Physical meaning and typical values}

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**{Parameter 1} ({Symbol})**:
   - **Physical meaning**: {What this parameter represents physically}
   - **Molecular origin**: {Connection to microstructure}
   - **Typical ranges**:
      - {Material class 1}: {range with units}
      - {Material class 2}: {range with units}
   - **Scaling**: {Relationship to molecular/physical properties}

{Repeat for each parameter}

----

Validity and Assumptions
------------------------

Model Assumptions
~~~~~~~~~~~~~~~~~

1. {Assumption 1}
2. {Assumption 2}
3. {Assumption 3}

Data Requirements
~~~~~~~~~~~~~~~~~

- **Required data**: {What experimental data is needed}
- **Frequency/time range**: {Recommended experimental window}
- **Quality criteria**: {Data quality requirements}

Limitations
~~~~~~~~~~~

**{Limitation 1}**:
   {Description of when/why this limitation matters}

**{Limitation 2}**:
   {Description of when/why this limitation matters}

----

Regimes and Behavior
--------------------

{Describe the different behavioral regimes of the model and what they represent
physically. Include phase diagrams if applicable.}

.. list-table:: Behavioral regimes
   :header-rows: 1
   :widths: 15 25 60

   * - Regime
     - Condition
     - Physical interpretation
   * - {Regime 1}
     - :math:`{condition}`
     - {interpretation}

Diagnostic Signatures
~~~~~~~~~~~~~~~~~~~~~

- **{Observable 1}**: {What it indicates}
- **{Observable 2}**: {What it indicates}

----

What You Can Learn
------------------

This section explains what insights you can extract from fitting this model
to your experimental data and how to translate results into actionable knowledge.

Parameter Interpretation
~~~~~~~~~~~~~~~~~~~~~~~~

**{Parameter 1}:**
   Fitted value tells you {physical meaning}. Typical ranges:

   - **Low values (< X)**: {interpretation, what it suggests about microstructure}
   - **High values (> Y)**: {interpretation, what it suggests about microstructure}

   *For graduate students*: {Connection to theory, scaling laws, molecular interpretation}

   *For practitioners*: {What this means for processing, product performance}

{Repeat for each parameter}

Material Classification
~~~~~~~~~~~~~~~~~~~~~~~

Based on fitted parameters, classify your material:

.. list-table:: Material Classification from Fitted Parameters
   :header-rows: 1
   :widths: 20 20 30 30

   * - Parameter Range
     - Material Type
     - Typical Examples
     - Processing Implications
   * - {Range A}
     - {Behavior type}
     - {Real materials}
     - {Processing guidance}

Microstructural Insights
~~~~~~~~~~~~~~~~~~~~~~~~

Connect parameters to microstructure:

- **{Parameter}** correlates with {molecular feature}
- **{Scaling law}** suggests {structural property}
- Compare with literature values: {reference ranges}

Diagnostic Indicators
~~~~~~~~~~~~~~~~~~~~~

Warning signs in fitted parameters:

- **If {parameter} hits bounds**: {what it means, likely causes}
- **If {ratio} is unusual**: {possible issues, recommended checks}
- **If R² is low despite good visual fit**: {interpretation}

Application Examples
~~~~~~~~~~~~~~~~~~~~

**Quality Control**: {How to use fitted parameters for QC decisions}

**Process Optimization**: {How parameters guide processing adjustments}

**Material Development**: {What parameter changes indicate during formulation}

----

Experimental Design
-------------------

When to Use This Model
~~~~~~~~~~~~~~~~~~~~~~

{Describe the scenarios where this model is the appropriate choice.}

**Use this model when**:
   - {Condition 1}
   - {Condition 2}
   - {Condition 3}

**Consider alternatives when**:
   - {Condition A} → use {Alternative model}
   - {Condition B} → use {Alternative model}

Recommended Test Protocols
~~~~~~~~~~~~~~~~~~~~~~~~~~

**1. {Protocol Name} (Primary)**

**Optimal for {model}**:
   - {Why this is the best test}

**Protocol**:
   - {Step-by-step experimental procedure}
   - {Recommended parameters: strain amplitude, frequency range, etc.}
   - {Expected data quality metrics}

**2. {Protocol Name} (Alternative)**

{Repeat structure for alternative protocols}

Sample Preparation Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**{Material type 1}**:
   - {Sample preparation guidance}
   - {Geometry recommendations}
   - {Common artifacts to avoid}

Common Experimental Artifacts
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table:: Troubleshooting experimental issues
   :header-rows: 1
   :widths: 25 35 40

   * - Artifact
     - Symptom
     - Solution
   * - {Artifact 1}
     - {Observable symptom}
     - {How to fix}

----

Computational Implementation
----------------------------

JAX Vectorization
~~~~~~~~~~~~~~~~~

{Describe how the model leverages JAX for performance.}

.. code-block:: python

   # Example of efficient vectorized prediction
   {code snippet showing key implementation pattern}

Numerical Stability
~~~~~~~~~~~~~~~~~~~

{Describe numerical considerations and how they're handled.}

- **{Issue 1}**: {How it's addressed}
- **{Issue 2}**: {How it's addressed}

Performance Considerations
~~~~~~~~~~~~~~~~~~~~~~~~~~

- **JIT compilation**: {First call vs subsequent calls timing}
- **Batch processing**: {How to efficiently process multiple datasets}
- **GPU acceleration**: {When/how GPU provides benefit}

----

Fitting Guidance
----------------

Parameter Initialization
~~~~~~~~~~~~~~~~~~~~~~~~

**Method 1: From {data type}**

**Step 1**: {Procedure}
   {Equation or instruction}

**Step 2**: {Procedure}
   {Equation or instruction}

**Method 2: From {alternative data type}**

{Repeat structure}

Optimization Algorithm Selection
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

**RheoJAX default: NLSQ (GPU-accelerated)**
   - Recommended for {model} ({N} parameters)
   - {Performance notes}

**Bayesian inference (NUTS)**
   - Use when {situation}
   - Warm-start from NLSQ fit for faster convergence

**Bounds**:
   - {param_1}: [{lower}, {upper}] {units}
   - {param_2}: [{lower}, {upper}] {units}

Bayesian Workflow
~~~~~~~~~~~~~~~~~

.. include:: /_includes/bayesian_workflow.rst

Troubleshooting
~~~~~~~~~~~~~~~

.. list-table:: Fitting diagnostics and solutions
   :header-rows: 1
   :widths: 25 35 40

   * - Problem
     - Diagnostic
     - Solution
   * - {Problem 1}
     - {How to identify}
     - {How to fix}

----

Usage
-----

Basic Example
~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from rheojax.models import {ModelClass}

   # Create data
   {x_variable} = np.logspace({range})

   # Create and fit model
   model = {ModelClass}()
   model.fit({x_variable}, {y_data}, test_mode='{mode}')

   # Extract parameters
   {param} = model.parameters.get_value('{param_name}')

   # Predict
   {y_pred} = model.predict({x_variable})

Bayesian Inference
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.models import {ModelClass}

   model = {ModelClass}()
   model.fit({x}, {y}, test_mode='{mode}')

   # Bayesian with warm-start
   result = model.fit_bayesian(
       {x}, {y},
       test_mode='{mode}',
       num_warmup=1000,
       num_samples=2000
   )

   # Credible intervals
   intervals = model.get_credible_intervals(result.posterior_samples)

{Specialized Examples}
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # {Description of specialized use case}
   {code}

----

See Also
--------

- :doc:`{related_model_1}` — {one-line description}
- :doc:`{related_model_2}` — {one-line description}
- :doc:`{related_transform}` — {one-line description}
- :doc:`../../examples/{example_path}` — {notebook description}

----

API References
--------------

- Module: :mod:`rheojax.models`
- Class: :class:`rheojax.models.{ModelClass}`

----

References
----------

{Minimum 10 references with proper formatting}

.. [{n}] {Author}, "{Title}." *{Journal}*, **{Volume}**, {Pages} ({Year}).
   {DOI or URL}

.. [{n}] {Author}, *{Book Title}*, {Edition}. {Publisher}, {Year}.

Further Reading
~~~~~~~~~~~~~~~

- {Additional reference with annotation}
- {Additional reference with annotation}
