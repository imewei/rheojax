Model Capabilities
==================

RheoJAX provides a wide range of rheological models, each suited for specific material behaviors and experimental protocols. This guide categorizes the available models to help you select the appropriate one for your analysis.

Protocol Support Matrix
-----------------------

The following table summarizes which rheological test protocols are supported by each model family.

.. list-table:: Model Protocol Support
   :widths: 15 20 10 10 10 10 10 15
   :header-rows: 1

   * - Model Type
     - Model Name
     - Flow Curve (Steady Shear)
     - Creep
     - Relaxation
     - Start-up
     - SAOS (Oscillation)
     - LAOS (Large Amplitude)
   * - **Classical**
     - Maxwell
     - ✅ (Newtonian)
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * -
     - Zener (SLS)
     - ✅ (Newtonian)
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * -
     - SpringPot
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * - **Flow**
     - Carreau
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Power Law
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Herschel-Bulkley
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Bingham
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * -
     - Cross
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ❌
   * - **Fractional**
     - Fractional Maxwell
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * -
     - Fractional Kelvin-Voigt
     - ❌
     - ✅
     - ✅
     - ❌
     - ✅
     - ❌
   * - **Multi-mode**
     - Generalized Maxwell
     - ✅ (Newtonian)
     - ✅
     - ✅
     - ✅ (Linear)
     - ✅
     - ✅ (Linear Only)
   * - **SGR**
     - SGR Conventional
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * -
     - SGR Generic
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **STZ**
     - STZ Conventional
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **SPP**
     - SPP Yield Stress
     - ✅
     - ❌
     - ❌
     - ❌
     - ❌
     - ✅ (Amp. Sweep)
   * - **Giesekus**
     - Single-Mode / Multi-Mode
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **Fluidity**
     - Local / Nonlocal
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **Saramito**
     - Local / Nonlocal
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **IKH**
     - MIKH / MLIKH
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **FIKH**
     - FIKH / FMLIKH
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **DMT**
     - Local / Nonlocal
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **HL**
     - Hébraud-Lequeux
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **EPM**
     - Lattice / Tensorial
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **ITT-MCT**
     - Schematic / Isotropic
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **TNT**
     - 5 variants
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **VLB**
     - 4 variants
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **HVM**
     - HVM Local
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
   * - **HVNM**
     - HVNM Local
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅
     - ✅

DMTA / DMA Support (Deformation Modes)
---------------------------------------

All **41 oscillation-capable models** support DMTA/DMA data through automatic :math:`E^* \leftrightarrow G^*` modulus conversion at the ``BaseModel`` boundary. Pass ``deformation_mode='tension'`` and ``poisson_ratio`` to ``fit()`` and ``predict()``:

.. code-block:: python

   model = Maxwell()
   model.fit(omega, E_star, test_mode='oscillation',
             deformation_mode='tension', poisson_ratio=0.5)

**Supported deformation modes**: ``shear`` (default), ``tension``, ``bending``, ``compression``.

**Models that do NOT support DMTA** (shear-only): Flow models (Carreau, PowerLaw, Bingham, HB, Cross, Carreau-Yasuda), SPP Yield Stress, Fluidity Nonlocal PDE variants, and EPM lattice models — these are either flow-curve-only or require explicit shear geometry.

**Poisson ratio presets**: rubber :math:`\approx 0.5`, glassy polymer :math:`\approx 0.35`, semicrystalline :math:`\approx 0.40`.

For the full DMTA reference including conversion theory and workflows, see :doc:`/models/dmta/index`.

Detailed Capabilities
---------------------

1. Advanced Physics Models (SGR & STZ)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These models are the most versatile, capable of simulating complex non-linear transient behaviors using JAX-accelerated ODE solvers (``Diffrax``).

*   **SGR (Soft Glassy Rheology):** Supports all protocols. Includes thixotropy, aging, and rejuvenation. Can simulate full LAOS cycles using Monte Carlo methods or SAOS approximations.
*   **STZ (Shear Transformation Zone):** Supports all protocols. Uses internal state variables (effective temperature) to capture plasticity, yield stress, and transient responses like stress overshoot in startup flow.

2. Generalized Maxwell (Multi-mode)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **Primary Use:** Linear Viscoelasticity (LVE) master curves.
*   **Limitations:** While it supports "Flow" and "LAOS" modes technically, it predicts **linear** responses only (constant viscosity, sinusoidal stress without harmonics). It is excellent for Relaxation, Creep, and SAOS spectra but cannot model shear-thinning or non-linearities.

3. Flow Models
^^^^^^^^^^^^^^

*   **Focus:** Strictly for steady-state shear viscosity (:math:`\eta` vs :math:`\dot{\gamma}`).
*   **Models:** Carreau, Power Law, Herschel-Bulkley, Cross, Bingham.
*   **Behavior:** Purely inelastic; they do not simulate time-dependent storage/loss moduli (:math:`G', G''`).

4. Fractional Models
^^^^^^^^^^^^^^^^^^^^

*   **Focus:** Modeling power-law relaxation spectra with fewer parameters than multi-mode models.
*   **Capabilities:** Excellent for fitting LVE data (:math:`G', G'', G(t)`) of gels and biological tissues over wide frequency ranges.

5. SPP (Sequence of Physical Processes)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **Specialization:** Specifically designed to analyze LAOS **Amplitude Sweeps** and extract yield stress parameters (Static vs. Dynamic yield stress). It bridges the gap between oscillatory data and flow curves.

6. Nonlinear Constitutive Models (Giesekus, Saramito)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **Giesekus:** Full tensorial ODE for polymer solutions/melts. Predicts shear thinning, first and second normal stress differences (N₁, N₂), stress overshoot in startup, and LAOS harmonics. Mobility parameter α controls nonlinearity.
*   **Saramito-Fluidity:** Tensorial elastoviscoplastic model with Von Mises yield criterion and thixotropic fluidity evolution. Predicts yield stress, normal stresses, viscosity bifurcation in creep, and shear banding (nonlocal variant).

7. Thixotropic Models (DMT, IKH, FIKH, Fluidity, HL)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **DMT:** Structure parameter λ ∈ [0,1] with exponential or Herschel-Bulkley viscosity closure. Models time-dependent viscosity, stress overshoot, and delayed yielding.
*   **IKH/FIKH:** Isotropic-kinematic hardening with optional fractional derivatives for power-law memory. Backstress tensor evolves with deformation history.
*   **Fluidity:** Cooperative flow models where fluidity f = 1/η evolves under shear. Nonlocal variant includes spatial cooperativity for shear banding.
*   **HL:** Mean-field model for dense suspensions. Stress distribution P(σ,t) evolves via Fokker-Planck equation with yielding events creating noise for neighbors.

8. Amorphous Solid Models (EPM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **EPM (Lattice):** Mesoscopic lattice of elasto-plastic blocks with Eshelby stress redistribution after plastic events. FFT-accelerated spatial correlations for avalanche dynamics.

9. Dense Suspension Models (ITT-MCT)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **Schematic (F₁₂):** Memory kernel from mode-coupling theory with strain decorrelation. Captures glass transition, cage effect, and yield stress from first principles. Semi-quantitative.
*   **Isotropic:** Uses Percus-Yevick structure factor S(k) for quantitative hard-sphere predictions.

10. Transient Network Models (TNT, VLB)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **TNT:** Five variants for different network architectures: single-mode, wormlike micelles (Cates), telechelic loop-bridge, multi-species, and sticky Rouse chains.
*   **VLB:** Distribution tensor formulation tracking chain end-to-end vector statistics. Variants include Bell force-sensitivity, FENE chain extensibility, and nonlocal PDE for shear banding.

11. Vitrimer Models (HVM, HVNM)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

*   **HVM:** Three-subnetwork model (Permanent + Exchangeable + Dissociative) with transition state theory kinetics for bond exchange reactions. Captures topology-freezing transition, temperature-dependent relaxation, and the vitrimer hallmark: σ_E → 0 at steady state.
*   **HVNM:** Extends HVM with a fourth interphase subnetwork for nanoparticle-filled vitrimers. Guth-Gold strain amplification and dual TST kinetics for matrix and interphase exchange.

Protocol-Driven Architecture
----------------------------

RheoJAX uses a ``TestMode`` enum (e.g., ``ROTATION``, ``OSCILLATION``) to dispatch valid predictions.

*   **Universal Models**: Models like **SGR**, **STZ**, **Giesekus**, **DMT**, **TNT**, **VLB**, **HVM**, **HVNM**, and **ITT-MCT** are constitutive equations that can predict responses for any flow history (all 6 protocols).
*   **Empirical Models**: **Flow** models are empirical curve fits restricted to steady-state conditions.
*   **Linear Models**: **Classical** and **Fractional** models describe linear viscoelastic response (SAOS, relaxation, creep) but not nonlinear behavior.
