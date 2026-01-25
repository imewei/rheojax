.. admonition:: Thixotropy Fundamentals
   :class: note

   **Thixotropy** is the reversible, time-dependent decrease in viscosity under
   constant shear rate, with subsequent recovery at rest. It arises from
   competition between microstructural breakdown (shear) and buildup (aging).

   **Physical Mechanisms:**

   - **Breakdown**: Shear disrupts network bonds, aggregates, or particle structures
   - **Buildup (aging)**: Brownian motion, attractive forces, or reaction kinetics rebuild structure
   - **Structure parameter** (λ): Dimensionless variable tracking microstructural state (0-1)

   **Characteristic Experimental Signatures:**

   1. **Hysteresis loops**: Different stress-strain rate curves for increasing vs decreasing shear
   2. **Stress overshoot**: Peak stress in startup flow before steady-state
   3. **Delayed yielding**: Time-dependent creep response, viscosity bifurcation
   4. **Recovery kinetics**: Gradual viscosity increase after shear cessation

   **Common Kinetic Equation:**

   .. math::

      \\frac{d\\lambda}{dt} = \\underbrace{\\frac{1-\\lambda}{t_{eq}}}_{\\text{aging}} - \\underbrace{a\\lambda|\\dot{\\gamma}|^c/t_{eq}}_{\\text{rejuvenation}}

   where :math:`t_{eq}` is equilibration time, :math:`a` is breakdown rate, and :math:`c` is
   shear-rate exponent.

   **Model Selection Guide:**

   .. list-table::
      :widths: 25 25 50
      :header-rows: 1

      * - Model Family
        - Best For
        - Key Features
      * - :doc:`/models/dmt/index`
        - Industrial fluids
        - Simple kinetics, exponential/HB closures
      * - :doc:`/models/ikh/index`
        - Metal plasticity
        - Hardening/softening, yield surface evolution
      * - :doc:`/models/fluidity/index`
        - Yield stress fluids
        - Fluidity evolution, Saramito viscoelasticity

   **Experimental Protocols for Thixotropic Materials:**

   - **Three-interval test**: Low rate → high rate → low rate to measure breakdown/recovery
   - **Step-rate tests**: Instantaneous rate changes to probe kinetics
   - **Startup flow**: Constant rate from rest to observe overshoot
   - **Creep**: Constant stress to observe delayed yielding
