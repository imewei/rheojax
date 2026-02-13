.. _hvm_advanced:

==============================================
HVM Advanced Theory & Numerical Methods
==============================================

This page documents the thermodynamic foundations, kinematics, and
numerical methods underlying the HVM.  For the constitutive equations,
see :doc:`hvm`.  For protocol derivations, see :doc:`hvm_protocols`.


.. _hvm-thermodynamics:

Thermodynamic Framework
========================

Helmholtz Free Energy
---------------------

The total Helmholtz free energy density is the sum of contributions from
each subnetwork, with damage coupling on the permanent network:

.. math::

   \Psi_{tot} = (1-D)\,\Psi_P(\mathbf{F})
   + \Psi_E[\boldsymbol{\mu}^E, \boldsymbol{\mu}^E_{nat}]
   + \Psi_D[\boldsymbol{\mu}^D]
   + p(\det\mathbf{F} - 1)

**Permanent network** (Neo-Hookean, Gaussian chains):

.. math::

   \Psi_P(\mathbf{F}) = \frac{G_P}{2}\left(\text{tr}(\mathbf{B}) - 3\right)

where :math:`\mathbf{B} = \mathbf{F}\mathbf{F}^T` and :math:`G_P = c_P k_B T`.

**Exchangeable (vitrimer) network:**

.. math::

   \Psi_E = \frac{G_E}{2}\,\text{tr}\!\left(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat}\right)

The stress vanishes when :math:`\boldsymbol{\mu}^E = \boldsymbol{\mu}^E_{nat}`,
not when :math:`\boldsymbol{\mu}^E = \mathbf{I}`.  This distinction is
the hallmark of associative exchange.

**Dissociative (physical) network:**

.. math::

   \Psi_D = \frac{G_D}{2}\,\text{tr}(\boldsymbol{\mu}^D - \mathbf{I})

Natural state is always :math:`\mathbf{I}` (bonds reform stress-free).


Clausius-Duhem Derivation
--------------------------

The second law requires non-negative dissipation:

.. math::

   \mathcal{D} = \boldsymbol{\sigma}:\mathbf{D} - \dot{\Psi}_{tot} \geq 0

Expanding :math:`\dot{\Psi}_{tot}` and collecting terms linear in
:math:`\mathbf{L}` identifies the Cauchy stress:

.. math::

   \boldsymbol{\sigma}_{tot} = (1-D) G_P (\mathbf{B} - \mathbf{I})
   + G_E (\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})
   + G_D (\boldsymbol{\mu}^D - \mathbf{I}) - p\mathbf{I}

The remaining terms yield the **dissipation from kinetic processes**,
each of which must be individually non-negative:

**Exchangeable network dissipation:**

.. math::

   \mathcal{D}_{exch} = \frac{G_E}{2} k_{BER}
   \text{tr}\!\left[(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})^2
   \cdot (\boldsymbol{\mu}^E_{nat})^{-1}\right] \geq 0

Guaranteed non-negative because
:math:`(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})^2` is positive
semi-definite.

**Dissociative network dissipation:**

.. math::

   \mathcal{D}_{diss} = \frac{G_D}{2} k_d^D \text{tr}(\boldsymbol{\mu}^D - \mathbf{I})^2 \geq 0

**Damage dissipation:**

.. math::

   \mathcal{D}_{dam} = \Psi_P \dot{D} \geq 0

Satisfied because :math:`\Psi_P \geq 0` and :math:`\dot{D} \geq 0` (damage
is irreversible).


.. _hvm-upper-convected:

Upper-Convected Kinematics
===========================

The evolution equations use the full velocity gradient :math:`\mathbf{L}`,
not the symmetric part :math:`\mathbf{D}` alone:

.. math::

   \dot{\boldsymbol{\mu}}^E = \mathbf{L}\boldsymbol{\mu}^E + \boldsymbol{\mu}^E\mathbf{L}^T
   + k_{BER}(\boldsymbol{\mu}^E_{nat} - \boldsymbol{\mu}^E)

The decomposition :math:`\mathbf{L} = \mathbf{D} + \mathbf{W}` means:

.. math::

   \mathbf{L}\boldsymbol{\mu} + \boldsymbol{\mu}\mathbf{L}^T
   = \mathbf{D}\boldsymbol{\mu} + \boldsymbol{\mu}\mathbf{D}
   + \mathbf{W}\boldsymbol{\mu} - \boldsymbol{\mu}\mathbf{W}

The vorticity terms :math:`\mathbf{W}\boldsymbol{\mu} - \boldsymbol{\mu}\mathbf{W}`
provide the Jaumann co-rotational correction for rigid-body rotation.

The upper-convected derivative form is:

.. math::

   \overset{\nabla}{\boldsymbol{\mu}}^E \equiv \dot{\boldsymbol{\mu}}^E
   - \mathbf{L}\boldsymbol{\mu}^E - \boldsymbol{\mu}^E\mathbf{L}^T
   = k_{BER}(\boldsymbol{\mu}^E_{nat} - \boldsymbol{\mu}^E)

**Simple shear as a special case:** For
:math:`\mathbf{L} = \dot{\gamma} \mathbf{e}_1 \otimes \mathbf{e}_2` with
isotropic initial conditions, :math:`\mu_{22} = \mu_{33}`, which means
:math:`\boldsymbol{\mu}` commutes with :math:`\mathbf{W}`, and the vorticity
terms vanish.  The :math:`\mathbf{L}` and :math:`\mathbf{D}` formulations
then coincide -- this is why all protocol derivations in
:doc:`hvm_protocols` use scalar ODEs.


TST Kinetics Deep Dive
========================

Stress-Coupled vs Stretch-Coupled
-----------------------------------

**Option A -- Stress-based (von Mises invariant, ``kinetics="stress"``):**

.. math::

   f(\boldsymbol{\sigma}^E) = \sqrt{\tfrac{3}{2} \boldsymbol{\sigma}^E:\boldsymbol{\sigma}^E}

Appropriate when the exchange barrier is reduced by total stress magnitude.
Isotropic and simple to evaluate.

**Option B -- Stretch-based (chain stretch invariant, ``kinetics="stretch"``):**

.. math::

   f(\boldsymbol{\mu}^E) = G_E \sqrt{\text{tr}(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})}

Measures elastic stretch of exchangeable chains relative to their current
natural state.  More appropriate when force along the chain backbone directly
lowers the barrier (Bell model picture).

In the zero-stress limit, both reduce to the thermal rate
:math:`k_{BER,0}(T) = \nu_0 \exp(-E_a / k_B T)`.


Von Mises Computation (Simple Shear)
--------------------------------------

With :math:`\sigma^E_{ij} = G_E(\mu^E_{ij} - \mu^{E,nat}_{ij})`, the
von Mises equivalent stress is:

.. math::

   \sigma_{VM}^E = G_E \sqrt{(\Delta_{xx})^2 + (\Delta_{yy})^2
   - \Delta_{xx}\Delta_{yy} + 3(\Delta_{xy})^2}

where :math:`\Delta_{ij} = \mu^E_{ij} - \mu^{E,nat}_{ij}`.

**Square root singularity:** At :math:`\sigma_{VM} = 0`, the gradient of
:math:`\cosh(\cdot)` can produce numerical issues.  The implementation uses
:math:`\sqrt{\max(x, 0) + 10^{-30}}` to guard against infinite gradients
(see :ref:`hvm-numerical` below).


.. _hvm-phenomenological-mode:

Phenomenological Fast Mode
===========================

For computational efficiency (avoiding ODE stiffness from the exponential
stress-coupling), a linearized rate is available:

.. math::

   k_{BER}^{phen} = k_{BER,0}(T) \cdot \left[1 + \alpha\,(\text{tr}(\boldsymbol{\mu}^E) - 3)\right]

where :math:`\alpha = V_{act} G_E / (2 k_B T)` maps TST parameters to the
phenomenological enhancement coefficient.  This is a first-order Taylor
expansion valid for small deformations.

See :doc:`/models/vlb/vlb_advanced` for the analogous VLB Bell breakage
mechanism.


.. _hvm-temperature:

Temperature & Topological Freezing
=====================================

Arrhenius Structure
-------------------

The TST rate directly produces the topological freezing temperature
:math:`T_v`.  Defining :math:`T_v` as the temperature where
:math:`\tau_{BER} = 1/k_{BER,0}` exceeds :math:`10^3` s:

.. math::

   T_v = \frac{E_a}{k_B \ln(\nu_0 \cdot 10^3)}

**Temperature regimes:**

.. list-table::
   :widths: 20 30 50
   :header-rows: 1

   * - Regime
     - Condition
     - Behavior
   * - Below :math:`T_v`
     - :math:`T < T_v`
     - Vitrimer behaves as thermoset (no exchange)
   * - Above :math:`T_v`
     - :math:`T > T_v`
     - Active BER, material flows
   * - Well above :math:`T_v`
     - :math:`T \gg T_v`
     - Fast exchange, approaches viscous liquid

**Arrhenius shift factor** for time-temperature superposition:

.. math::

   \ln a_T = \frac{E_a}{k_B}\left(\frac{1}{T} - \frac{1}{T_{ref}}\right)

If :math:`G_P` and :math:`G_E` have the entropic :math:`T`-scaling
(:math:`G \propto T`), a vertical shift :math:`b_T = T_{ref}/T` is also needed.


Dissociative Bond Temperature Dependence
-----------------------------------------

The D-network rate can follow Arrhenius:

.. math::

   k_d^D(T) = k_{d,0}^D \exp\!\left(-\frac{E_a^D}{k_B T}\right)

For force-dependent dissociation (Bell-Evans model):

.. math::

   k_d^D(\boldsymbol{\mu}^D) = k_{d,0}^D \exp\!\left(-\frac{E_a^D - V_{act}^D \|\boldsymbol{\sigma}^D\|}{k_B T}\right)


.. _hvm-numerical:

Numerical Implementation
=========================

**ODE solver:** diffrax ``Tsit5`` (explicit 5th-order Runge-Kutta) with
``PIDController`` adaptive stepping (``rtol=1e-8``, ``atol=1e-10``).

**Stiffness handling:** TST stress-coupling can make the ODEs stiff at high
shear rates (large :math:`k_{BER}` variations within a timestep).  The
explicit Tsit5 solver handles moderate stiffness; for extreme cases, reduce
shear rate or switch to ``kinetics="stretch"`` (smoother coupling).

.. note::

   Implicit solvers (e.g., Kvaerno5) were tested but produce
   ``TracerBoolConversionError`` due to lineax LU transpose checks during JAX
   tracing.  Tsit5 is the recommended solver.

**Square-root guard:** The BER rate computation involves
:math:`\sqrt{\text{tr}(\boldsymbol{\mu}^E - \boldsymbol{\mu}^E_{nat})}`,
which has infinite gradient at zero.  The implementation uses:

.. code-block:: python

   safe_stretch = jnp.sqrt(jnp.maximum(stretch_invariant, 0.0) + 1e-30)

**Initial conditions:** All tensors at identity
(:math:`\mu_{xx} = \mu_{yy} = 1`, :math:`\mu_{xy} = 0`),
:math:`\gamma = 0`, :math:`D = 0`.


References
===========

1. Vernerey, F.J., Long, R. & Brighenti, R. (2017). "A statistically-based
   continuum theory for polymers with transient networks." *J. Mech. Phys.
   Solids*, 107, 1--20.
   https://doi.org/10.1016/j.jmps.2017.05.016

2. Vernerey, F.J. (2018). "Transient response of nonlinear polymer networks:
   A kinetic theory." *J. Mech. Phys. Solids*, 115, 230--247.
   https://doi.org/10.1016/j.jmps.2018.02.018
   :download:`PDF <../../../reference/vernerey_2018_tnt_kinetic_theory.pdf>`

3. Vernerey, F.J., Brighenti, R., Long, R. & Shen, T. (2018). "Statistical
   Damage Mechanics of Polymer Networks." *Macromolecules*, 51(17), 6609--6622.
   https://doi.org/10.1021/acs.macromol.8b01052

4. Meng, F., Saed, M.O. & Terentjev, E.M. (2019). "Elasticity and Relaxation
   in Full and Partial Vitrimer Networks." *Macromolecules*, 52(19), 7423--7429.
   https://doi.org/10.1021/acs.macromol.9b01123

5. Shen, T., Song, Z., Cai, S. & Vernerey, F.J. (2021). "Nonsteady fracture
   of transient networks: The case of vitrimer." *PNAS*, 118(29), e2105974118.
   https://doi.org/10.1073/pnas.2105974118

6. Wagner, R.J., Hobbs, E. & Vernerey, F.J. (2021). "A network model of
   transient polymers: exploring the micromechanics of nonlinear
   viscoelasticity." *Soft Matter*, 17, 8742.
   https://doi.org/10.1039/D1SM00753J
   :download:`PDF <../../../reference/wagner_2021_tnt_network_model.pdf>`

7. Lamont, S.C., Mulderrig, J., Bouklas, N. & Vernerey, F.J. (2021).
   "Rate-Dependent Damage Mechanics of Polymer Networks with Reversible
   Bonds." *Macromolecules*, 54(23), 10801--10813.
   https://doi.org/10.1021/acs.macromol.1c01943

8. Meng, F., Saed, M.O. & Terentjev, E.M. (2022). "Rheology of vitrimers."
   *Nature Communications*, 13, 5753.
   https://doi.org/10.1038/s41467-022-33321-w

9. Vernerey, F.J., Rezaei, B. & Lamont, S.C. (2024). "A kinetic theory for
   the mechanics and remodeling of transient anisotropic networks." *J. Mech.
   Phys. Solids*, 190, 105713.
   https://doi.org/10.1016/j.jmps.2024.105713

10. Wagner, R.J. & Silberstein, M.N. (2025). "A foundational framework for
    the mesoscale modeling of dynamic elastomers and gels." *J. Mech. Phys.
    Solids*, 194, 105914.
    https://doi.org/10.1016/j.jmps.2024.105914

11. Karim, M.R., Vernerey, F. & Sain, T. (2025). "Constitutive Modeling of
    Vitrimers and Their Nanocomposites Based on Transient Network Theory."
    *Macromolecules*, 58(10), 4899--4912.
    https://doi.org/10.1021/acs.macromol.4c02872
    :download:`PDF <../../../reference/karim_2025_vitrimer_nanocomposites.pdf>`

12. Alkhoury, K. & Chester, S.A. (2025). "A chemo-thermo-mechanically coupled
    theory of photo-reacting polymers: Application to modeling photo-degradation
    with irradiation-driven heat transfer." *J. Mech. Phys. Solids*, 197, 106050.
    https://doi.org/10.1016/j.jmps.2025.106050

13. White, Z.T., Smith, A.M. & Vernerey, F.J. (2025). "Mechanical cooperation
    between time-dependent and covalent bonds in molecular damage of polymer
    networks." *Communications Physics*, 8, 265.
    https://doi.org/10.1038/s42005-025-02192-0
    :download:`PDF <../../../reference/white_2025_molecular_damage.pdf>`
