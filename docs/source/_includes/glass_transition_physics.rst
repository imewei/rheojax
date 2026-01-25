.. admonition:: Glass Transition Physics
   :class: note

   **Common Physical Framework**

   Models in this category describe materials near or below the glass transition—where
   thermal fluctuations become insufficient for structural relaxation on experimental
   timescales. These materials exhibit:

   **Characteristic Signatures:**

   - **Cage effect**: Particles trapped by neighbors, requiring cooperative rearrangements
   - **Aging**: Properties evolve with waiting time (time since preparation)
   - **Yield stress**: Finite stress required for macroscopic flow
   - **Power-law rheology**: G'(ω) ~ G''(ω) ~ ω^n with weak frequency dependence
   - **Structural relaxation**: α-relaxation timescale diverges at glass transition

   **Key Control Parameters:**

   .. list-table::
      :widths: 30 20 50
      :header-rows: 1

      * - Model
        - Parameter
        - Physical meaning
      * - SGR
        - x (noise temperature)
        - Ratio of activation energy to trap depth
      * - ITT-MCT
        - ε (separation parameter)
        - Distance from ideal glass transition
      * - STZ
        - χ (effective temperature)
        - Configurational disorder
      * - EPM
        - σ/σ_y (stress ratio)
        - Proximity to yield

   **Glass Transition Regimes:**

   - **Liquid regime** (above Tg or critical point): Equilibrium relaxation, aging absent
   - **Glass regime** (below Tg): Frozen structure, aging, yield stress emerges
   - **Critical point**: Power-law divergences, scale-free avalanches

   **Related Concepts:**

   - :doc:`/user_guide/soft_glassy_materials` — Introduction to SGMs
   - :doc:`/transforms/mastercurve` — Time-temperature superposition near Tg
   - :doc:`/models/sgr/index` — SGR model family
   - :doc:`/models/itt_mct/index` — Mode-coupling theory approach
