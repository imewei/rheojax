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
   5. **Normal stress differences**: N₁ > 0 from chain anisotropy (extensional resistance)

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

   - **Loop-bridge equilibrium**: Two-species kinetics (f_B equilibrium bridge fraction)
   - **Sticky Rouse**: Multi-mode relaxation with sticker-limited dynamics
   - **Cates model**: Living polymers with scission/recombination
   - **Non-affine slip**: Gordon-Schowalter parameter :math:`\xi` for partial coupling

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

   **Typical Parameter Ranges:**

   - Network modulus **G**: 1-10⁶ Pa (depends on crosslink density)
   - Bond lifetime **τ_b**: 10⁻⁶-10⁴ s (wide range across materials)
   - Bell parameter **ν**: 0.01-20 (bond sensitivity to force)
   - FENE extensibility **L_max**: 2-100 (chain contour length ratio)
   - Slip parameter **ξ**: 0 (affine) to 1 (full slip)
