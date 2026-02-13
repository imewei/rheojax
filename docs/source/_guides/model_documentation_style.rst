.. _model-documentation-style:

====================================
Model Documentation Style Guide
====================================

This guide establishes standards for RheoJAX model documentation to ensure
consistency, completeness, and usefulness for both graduate students and
industry practitioners.

Overview
--------

Every model documentation page should serve as a **handbook** that enables readers to:

1. **Understand** the physical and mathematical basis of the model
2. **Decide** whether the model is appropriate for their material/data
3. **Apply** the model correctly with proper experimental protocols
4. **Interpret** fitted parameters in terms of material properties
5. **Troubleshoot** common issues during fitting and analysis

Target Audience
---------------

Write for **two primary audiences**:

**Graduate Students**
   - Learning rheology and viscoelasticity
   - Need theoretical foundations and derivations
   - Want to connect models to molecular physics
   - Benefit from worked examples and educational context

**Industry Practitioners**
   - Experienced rheologists with practical needs
   - Want actionable guidance for real materials
   - Need quick reference for parameter interpretation
   - Value troubleshooting tables and checklists

**Balance**: Each section should serve both audiences. Use subsections to
separate theoretical depth (students) from practical guidance (practitioners)
when needed.

Document Structure
------------------

Use the template in ``_templates/model_handbook_template.rst`` as starting point.

Required Sections
~~~~~~~~~~~~~~~~~

Every model documentation **must** include these sections:

1. **Quick Reference** — 5-line summary at top
2. **Notation Guide** — Symbol table
3. **Overview** — Introduction with historical context
4. **Physical Foundations** — Mechanical/microstructural picture
5. **Governing Equations** — Mathematical formulation
6. **Parameters** — Table with interpretation
7. **Validity and Assumptions** — When the model applies
8. **What You Can Learn** — Knowledge extraction guidance
9. **Fitting Guidance** — Practical optimization advice
10. **Usage** — Code examples
11. **See Also** — Related models and transforms
12. **References** — Minimum 10 citations

Optional Sections (model-dependent)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- **Regimes and Behavior** — For models with phase transitions or distinct regimes
- **Experimental Design** — Detailed protocols for complex models
- **Computational Implementation** — JAX-specific optimizations
- **Model Comparison** — When multiple similar models exist
- **Extended Features** — Shear banding, thixotropy extensions

Target Length
~~~~~~~~~~~~~

.. list-table:: Documentation length guidelines
   :header-rows: 1
   :widths: 40 30 30

   * - Model Type
     - Target Lines
     - Example
   * - Simple classical (2-3 params)
     - 500-700
     - Maxwell, Power Law
   * - Intermediate (3-5 params)
     - 600-900
     - Carreau, Zener, FML
   * - Complex/advanced (4+ params)
     - 800-1200
     - SGR, ITT-MCT, GMM

RST Formatting Conventions
--------------------------

Headings
~~~~~~~~

Use consistent heading hierarchy::

   ===================
   Document Title
   ===================

   Section 1
   ---------

   Subsection 1.1
   ~~~~~~~~~~~~~~

   Subsubsection 1.1.1
   ^^^^^^^^^^^^^^^^^^^

Math Notation
~~~~~~~~~~~~~

- Use ``:math:`` for inline math: ``:math:`G'(\omega)```
- Use ``.. math::`` blocks for displayed equations
- Number only equations that are referenced elsewhere
- Use consistent symbol conventions (see Notation Guide)

**Symbol conventions**:

- **Greek letters**: :math:`\alpha, \beta, \gamma` for fractional orders
- **Subscripts**: :math:`G_0` (plateau), :math:`G_e` (equilibrium), :math:`G_\infty` (high-freq)
- **Time/frequency**: :math:`t` (time), :math:`\omega` (angular frequency), :math:`\tau` (relaxation time)
- **Moduli**: :math:`G'` (storage), :math:`G''` (loss), :math:`G^*` (complex)
- **Viscosity**: :math:`\eta` (dynamic), :math:`\eta_0` (zero-shear), :math:`\eta_\infty` (infinite-shear)

Code Examples
~~~~~~~~~~~~~

- Use ``.. code-block:: python`` for all code
- Keep examples minimal and focused
- Include imports in first example only
- Use realistic but simple data ranges
- Comment key lines

Tables
~~~~~~

Use ``.. list-table::`` directive for all tables::

   .. list-table:: Table caption
      :header-rows: 1
      :widths: 20 30 50

      * - Column 1
        - Column 2
        - Column 3
      * - Data 1
        - Data 2
        - Data 3

Cross-References
~~~~~~~~~~~~~~~~

- Link to related models: ``:doc:`related_model```
- Link to transforms: ``:doc:`../../transforms/transform_name```
- Link to API: ``:class:`rheojax.models.ModelClass```
- Use labels for internal links: ``.. _label-name:`` and ``:ref:`label-name```

Writing Style
-------------

General Guidelines
~~~~~~~~~~~~~~~~~~

1. **Be precise**: Use exact terminology consistently
2. **Be concise**: Remove filler words, get to the point
3. **Be practical**: Every section should have actionable content
4. **Be educational**: Explain *why*, not just *what*
5. **No emojis**: Professional technical documentation only

Tone
~~~~

- **Factual and objective** — No marketing language
- **Direct and instructive** — "Use when..." not "You might consider..."
- **Technically rigorous** — Correct physics and mathematics

Active Voice
~~~~~~~~~~~~

Prefer active voice for instructions:

- Good: "Fit the model to frequency sweep data"
- Avoid: "The model should be fitted to frequency sweep data"

Equations and Derivations
~~~~~~~~~~~~~~~~~~~~~~~~~

For derivations:

1. Number each step
2. State what you're doing, then show the math
3. Highlight key insights with ``.. note::`` or ``.. tip::``
4. Don't skip steps that aren't obvious

"What You Can Learn" Section
----------------------------

This section is **critical** for addressing user needs. Structure it as:

1. **Parameter Interpretation**
   - What each fitted parameter tells you about the material
   - Typical ranges with physical interpretations
   - Dual-audience format: theory (students) + practice (industry)

2. **Material Classification**
   - Table mapping parameter ranges to material types
   - Include real-world examples and processing implications

3. **Microstructural Insights**
   - How to connect parameters to molecular/particle properties
   - Scaling laws and literature comparisons

4. **Diagnostic Indicators**
   - What warning signs in fitted parameters mean
   - When to suspect model inadequacy

5. **Application Examples**
   - Quality control applications
   - Process optimization guidance
   - Material development use cases

Example "What You Can Learn" Entry
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

::

   **x (Effective Noise Temperature)**:
      Fitted value reveals the material's position relative to the glass transition:

      - **Low values** (:math:`x < 1`): Glass phase. Material exhibits aging, yield stress,
        and cannot reach equilibrium. Indicates strong interparticle attraction or
        high crowding.

      - **Near critical** (:math:`x \approx 1`): Glass transition. Maximum sensitivity to
        processing conditions. Small formulation changes have large rheological effects.

      - **High values** (:math:`x > 1.5`): Fluid phase. Material flows easily and reaches
        equilibrium. Indicates weak attractions or low volume fraction.

      *For graduate students*: The SGR model predicts :math:`x - 1` as the power-law
      exponent in :math:`G' \sim \omega^{x-1}`. Connect to MCT predictions via
      :math:`x_c = 1 + f_c` where :math:`f_c` is the non-ergodicity parameter.

      *For practitioners*: Target :math:`x \approx 1.2-1.5` for good shelf stability
      with acceptable pourability. If :math:`x < 1`, product will age visibly.

Reference Standards
-------------------

Citation Format
~~~~~~~~~~~~~~~

Use numbered references::

   .. [1] Author, A. B., Author, C. D. "Title of Paper."
      *Journal Name*, **Volume**, Pages (Year).
      https://doi.org/xxxxx

Minimum Reference Count
~~~~~~~~~~~~~~~~~~~~~~~

- Simple models (2 params): 8-10 references
- Intermediate models: 10-15 references
- Advanced models: 15-25 references

Include categories:

1. Original model paper
2. Major textbook references (Ferry, Macosko, etc.)
3. Application examples
4. Recent reviews (2020+)

Quality Checklist
-----------------

Before submitting model documentation, verify:

**Completeness**:
   - [ ] Quick Reference has all 5 fields
   - [ ] Notation Guide with symbol table
   - [ ] Physical Foundations with mechanical analogue
   - [ ] Governing Equations with step-by-step derivation
   - [ ] Parameters table with all parameters
   - [ ] What You Can Learn section complete
   - [ ] At least 3 usage examples
   - [ ] 10+ references

**Accuracy**:
   - [ ] Equations match model implementation
   - [ ] Parameter bounds match code
   - [ ] Test modes listed match supported modes
   - [ ] Code examples run without error

**Usefulness**:
   - [ ] Graduate student can understand physics
   - [ ] Practitioner can apply model immediately
   - [ ] Troubleshooting addresses common issues
   - [ ] Cross-references work correctly

**Style**:
   - [ ] Consistent heading levels
   - [ ] Tables properly formatted
   - [ ] Code blocks syntax highlighted
   - [ ] No orphaned references
   - [ ] Builds without Sphinx warnings

Validation
----------

Run the validation script before committing::

   python scripts/validate_model_docs.py docs/source/models/

This checks:
   - Required sections present
   - Parameter consistency with model code
   - Reference count
   - Internal link validity
   - RST syntax
