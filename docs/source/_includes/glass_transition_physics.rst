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
   - **Power-law rheology**: :math:`G'(\omega) \sim G''(\omega) \sim \omega^n` with weak frequency dependence
   - **Structural relaxation**: :math:`\alpha`-relaxation timescale diverges at glass transition

   **Key Control Parameters:**

   .. list-table::
      :widths: 30 20 50
      :header-rows: 1

      * - Model
        - Parameter
        - Physical meaning
      * - SGR
        - :math:`x` (noise temperature)
        - Ratio of activation energy to trap depth
      * - ITT-MCT
        - :math:`\varepsilon` (separation parameter)
        - Distance from ideal glass transition
      * - STZ
        - :math:`\chi` (effective temperature)
        - Configurational disorder
      * - EPM
        - :math:`\sigma/\sigma_y` (stress ratio)
        - Proximity to yield

   **Glass Transition Regimes:**

   - **Liquid regime** (above :math:`T_g` or critical point): Equilibrium relaxation, aging absent
   - **Glass regime** (below :math:`T_g`): Frozen structure, aging, yield stress emerges
   - **Critical point**: Power-law divergences, scale-free avalanches

   **Related Concepts:**

   - :doc:`/user_guide/soft_glassy_materials` — Introduction to SGMs
   - :doc:`/transforms/mastercurve` — Time-temperature superposition near :math:`T_g`
   - :doc:`/models/sgr/index` — SGR model family
   - :doc:`/models/itt_mct/index` — Mode-coupling theory approach
