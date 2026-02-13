.. admonition:: Transient Network Theory Foundations
   :class: note

   **Transient Network Theory (TNT)** describes materials with reversible crosslinks
   that continuously break and reform on characteristic timescales, creating dynamic
   networks with time-dependent mechanical properties.

   **Physical Basis:**

   - **Network strands**: Polymer chains or other elements spanning crosslink junctions
   - **Crosslink lifetime** (:math:`\tau_b`): Mean bond survival time before detachment
   - **Creation rate**: New bonds form to maintain equilibrium network density
   - **Conformation tensor** (:math:`\mathbf{S}`): Tracks average chain stretch and orientation

   **Characteristic Experimental Signatures:**

   1. **Single relaxation mode**: Maxwell-like behavior with relaxation time :math:`\tau = \tau_b`
   2. **Shear thinning**: Viscosity decreases as flow disrupts network structure
   3. **Strain softening**: Modulus reduction under large deformations (chain stretch)
   4. **Transient overshoot**: Stress peaks during startup as network orientation saturates
   5. **Normal stress differences**: :math:`N_1 > 0` from chain anisotropy (extensional resistance)

   **Fundamental Constitutive Equation:**

   .. math::

      \frac{D\mathbf{S}}{Dt} = \mathbf{\kappa} \cdot \mathbf{S} + \mathbf{S} \cdot \mathbf{\kappa}^T
      - \frac{1}{\tau_b(\mathbf{S})}(\mathbf{S} - \mathbf{I})

   where :math:`\mathbf{\kappa}` is the velocity gradient and :math:`\mathbf{I}` is the identity.
   The stress is given by :math:`\boldsymbol{\sigma} = G \cdot f(\mathbf{S})` where
   :math:`f(\mathbf{S})` depends on the chain model:

   - **Hookean**: :math:`f(\mathbf{S}) = \mathbf{S} - \mathbf{I}` (linear)
   - **FENE-P**: :math:`f(\mathbf{S}) = \frac{L^2_{max}}{L^2_{max} - \text{tr}(\mathbf{S}) + 3}(\mathbf{S} - \mathbf{I})`
     (finite extensibility)

   **Bond Kinetics Models:**

   .. list-table::
      :widths: 25 50 25
      :header-rows: 1

      * - Kinetics
        - Breakage Rate :math:`1/\tau_b`
        - Use Case
      * - Constant (Tanaka-Edwards)
        - :math:`1/\tau_0`
        - Baseline Maxwell-like
      * - Bell Model
        - :math:`(1/\tau_0)\exp(\nu F/k_BT)`
        - Force-activated unbinding
      * - Power-law
        - :math:`(1/\tau_0)(F/F_0)^m`
        - Empirical force-weakening
      * - Stretch-enhanced creation
        - Creation :math:`\propto \text{tr}(\mathbf{S})`
        - Strain-induced crosslinking

   **Advanced Extensions:**

   - **Loop-bridge equilibrium**: Two-species kinetics (:math:`f_B` equilibrium bridge fraction)
   - **Sticky Rouse**: Multi-mode relaxation with sticker-limited dynamics
   - **Cates model**: Living polymers with scission/recombination
   - **Non-affine slip**: Gordon-Schowalter parameter :math:`\xi` for partial coupling
   - **Multi-species networks**: Multiple bond types with different lifetimes and moduli

   **Model Selection Guide:**

   .. list-table::
      :widths: 30 70
      :header-rows: 1

      * - Model
        - Best For
      * - :doc:`TNTSingleMode <tnt_tanaka_edwards>` (constant)
        - Simple physical gels, baseline characterization
      * - :doc:`TNTSingleMode <tnt_bell>` (Bell)
        - Bio-networks with force-sensitive bonds (fibrin, collagen)
      * - :doc:`TNTSingleMode <tnt_fene_p>` (FENE-P)
        - Polymeric gels near maximum extensibility
      * - :doc:`TNTLoopBridge <tnt_loop_bridge>`
        - Telechelic polymers with two junction types
      * - :doc:`TNTStickyRouse <tnt_sticky_rouse>`
        - Multi-sticker associating polymers (broad relaxation)
      * - :doc:`TNTCates <tnt_cates>`
        - Wormlike micelles, living polymers
      * - :doc:`TNTSingleMode <tnt_non_affine>` (Non-Affine)
        - Networks with imperfect chain-flow coupling (:math:`N_2 \neq 0`)
      * - :doc:`TNTSingleMode <tnt_stretch_creation>` (Stretch-Creation)
        - Strain-crystallizing or mechanophore-activated networks
      * - :doc:`TNTMultiSpecies <tnt_multi_species>`
        - Dual-crosslinked hydrogels, multi-strength assemblies

   **Dual Formulation:**

   TNT models admit two mathematically equivalent formulations:

   - **Differential (conformation tensor ODE):** Evolve :math:`\mathbf{S}(t)` via the constitutive ODE above — efficient for steady-state and simple histories
   - **Integral (cohort/history):** Track chain cohorts born at each time :math:`t'` and integrate their stress contributions — natural for complex deformation histories

   Both yield identical predictions; the choice is computational convenience.
   See :doc:`tnt_protocols` for details.

   **Typical Parameter Ranges:**

   - Network modulus :math:`G`: 1--:math:`10^6` Pa (depends on crosslink density)
   - Bond lifetime :math:`\tau_b`: :math:`10^{-6}`--:math:`10^4` s (wide range across materials)
   - Bell parameter :math:`\nu`: 0.01--20 (bond sensitivity to force)
   - FENE extensibility :math:`L_{\max}`: 2--100 (chain contour length ratio)
   - Slip parameter :math:`\xi`: 0 (affine) to 1 (full slip)
