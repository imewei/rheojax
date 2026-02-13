.. admonition:: Linear Viscoelastic Assumptions
   :class: important

   **This model assumes linear viscoelasticity:**

   1. **Small strains**: Deformations remain in the linear regime (typically :math:`\gamma < 1`-5%)
   2. **Time-invariant properties**: Material parameters constant throughout measurement
   3. **Isothermal conditions**: Temperature held constant (±0.1°C for precision)
   4. **Boltzmann superposition**: Stress response to sequential deformations is additive
   5. **No structural changes**: Sample microstructure remains unchanged during testing

   **Validity checks:**

   - Perform strain amplitude sweep to identify linear viscoelastic region (LVR)
   - Verify :math:`G'`, :math:`G''` independence of strain amplitude within LVR
   - Check time-reproducibility by repeating measurements

   **When assumptions break down:**

   - Large amplitude oscillatory shear (LAOS) → use :doc:`/models/spp/index`
   - Thixotropic materials → use :doc:`/models/dmt/index` or :doc:`/models/ikh/index`
   - Yielding behavior → check :doc:`/models/epm/index` or :doc:`/models/stz/index`
