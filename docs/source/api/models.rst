Models API
==========

Concise parameter reference for every rheological model. For derivations, limits,
and usage recipes see the :doc:`/models/index` handbook.

Classical Models
----------------

Maxwell
~~~~~~~

:class:`rheojax.models.classical.maxwell.Maxwell` | Handbook: :doc:`/models/classical/maxwell`
Spring and dashpot in series for single-mode relaxation.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G0``
     - Pa
     - [0.001, 1e+09]
     - Elastic modulus
   * - ``eta``
     - Pa·s
     - [1e-06, 1e+12]
     - Viscosity

.. autoclass:: rheojax.models.classical.maxwell.Maxwell
   :members:
   :undoc-members:
   :show-inheritance:

Zener (Standard Linear Solid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.classical.zener.Zener` | Handbook: :doc:`/models/classical/zener`
Adds an equilibrium spring to capture solid plateaus.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Ge``
     - Pa
     - [0.001, 1e+09]
     - Equilibrium modulus
   * - ``Gm``
     - Pa
     - [0.001, 1e+09]
     - Maxwell modulus
   * - ``eta``
     - Pa·s
     - [1e-06, 1e+12]
     - Viscosity

.. autoclass:: rheojax.models.classical.zener.Zener
   :members:
   :undoc-members:
   :show-inheritance:

SpringPot
~~~~~~~~~

:class:`rheojax.models.classical.springpot.SpringPot` | Handbook: :doc:`/models/classical/springpot`
Fractional element interpolating between elastic and viscous limits.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c_alpha``
     - Pa·s\ :sup:`α`
     - [0.001, 1e+09]
     - Material constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Power-law exponent (0=fluid, 1=solid)

.. autoclass:: rheojax.models.classical.springpot.SpringPot
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Mode Models
-----------------

GeneralizedMaxwell (Prony Series)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.multimode.generalized_maxwell.GeneralizedMaxwell` | Handbook: :doc:`/models/multi_mode/generalized_maxwell`
N-mode Maxwell elements in parallel for complex relaxation spectra.

**v0.3.0+**: Transparent element minimization with R²-based auto-optimization
**v0.4.0+**: Warm-start optimization for 2-5x speedup in element search workflows

.. list-table:: Parameters (Dynamic: 2N+1 total)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_inf`` (or ``E_inf``)
     - Pa
     - [0.001, 1e+09]
     - Equilibrium modulus (rubbery plateau)
   * - ``G_i`` (or ``E_i``)
     - Pa
     - [0.001, 1e+09]
     - Mode i strength (i=1...N)
   * - ``tau_i``
     - s
     - [1e-09, 1e+09]
     - Mode i relaxation time (i=1...N)

**Note**: For ``n_modes=3``, generates 7 parameters: ``G_inf``, ``G_1``, ``G_2``, ``G_3``, ``tau_1``, ``tau_2``, ``tau_3``

**Element Minimization**: Set ``optimization_factor=1.5`` (default) to auto-reduce N until R² degrades

.. autoclass:: rheojax.models.multimode.generalized_maxwell.GeneralizedMaxwell
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Models
-----------------

Fractional Maxwell Gel
~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_maxwell_gel.FractionalMaxwellGel` | Handbook: :doc:`/models/fractional/fractional_maxwell_gel`
Spring in series with SpringPot for gel-like plateaus.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c_alpha``
     - Pa·s\ :sup:`α`
     - [0.001, 1e+09]
     - SpringPot material constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Power-law exponent
   * - ``eta``
     - Pa·s
     - [1e-06, 1e+12]
     - Dashpot viscosity

.. autoclass:: rheojax.models.fractional.fractional_maxwell_gel.FractionalMaxwellGel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Maxwell Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_maxwell_liquid.FractionalMaxwellLiquid` | Handbook: :doc:`/models/fractional/fractional_maxwell_liquid`
SpringPot plus dashpot for liquid-like systems with memory.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Gm``
     - Pa
     - [0.001, 1e+09]
     - Maxwell modulus
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Power-law exponent
   * - ``tau_alpha``
     - s^alpha
     - [1e-06, 1e+06]
     - Relaxation time

.. autoclass:: rheojax.models.fractional.fractional_maxwell_liquid.FractionalMaxwellLiquid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Maxwell (General)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_maxwell_model.FractionalMaxwellModel` | Handbook: :doc:`/models/fractional/fractional_maxwell_model`
Two SpringPots in series for broad relaxation spectra.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c1``
     - Pa·s\ :sup:`α`
     - [0.001, 1e+09]
     - Material constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - First fractional order
   * - ``beta``
     - dimensionless
     - [0, 1]
     - Second fractional order
   * - ``tau``
     - s
     - [1e-06, 1e+06]
     - Relaxation time

.. autoclass:: rheojax.models.fractional.fractional_maxwell_model.FractionalMaxwellModel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Kelvin-Voigt
~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_kelvin_voigt.FractionalKelvinVoigt` | Handbook: :doc:`/models/fractional/fractional_kelvin_voigt`
Spring plus SpringPot in parallel for solid-like gels.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Ge``
     - Pa
     - [0.001, 1e+09]
     - Equilibrium modulus
   * - ``c_alpha``
     - Pa·s\ :sup:`α`
     - [0.001, 1e+09]
     - SpringPot constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order

.. autoclass:: rheojax.models.fractional.fractional_kelvin_voigt.FractionalKelvinVoigt
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Zener Solid-Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_zener_sl.FractionalZenerSolidLiquid` | Handbook: :doc:`/models/fractional/fractional_zener_sl`
Equilibrium spring parallel to fractional Maxwell arm.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Ge``
     - Pa
     - [0.001, 1e+09]
     - Equilibrium modulus
   * - ``c_alpha``
     - Pa·s\ :sup:`α`
     - [0.001, 1e+09]
     - SpringPot constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - s
     - [1e-06, 1e+06]
     - Relaxation time

.. autoclass:: rheojax.models.fractional.fractional_zener_sl.FractionalZenerSolidLiquid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Zener Solid-Solid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_zener_ss.FractionalZenerSolidSolid` | Handbook: :doc:`/models/fractional/fractional_zener_ss`
Two springs plus SpringPot for stiff gels and glasses.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Ge``
     - Pa
     - [0.001, 1e+09]
     - Equilibrium modulus
   * - ``Gm``
     - Pa
     - [0.001, 1e+09]
     - Maxwell arm modulus
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau_alpha``
     - s^alpha
     - [1e-06, 1e+06]
     - Relaxation time

.. autoclass:: rheojax.models.fractional.fractional_zener_ss.FractionalZenerSolidSolid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Zener Liquid-Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_zener_ll.FractionalZenerLiquidLiquid` | Handbook: :doc:`/models/fractional/fractional_zener_ll`
Most general fractional Zener with three fractional orders.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c1``
     - Pa·s\ :sup:`α`
     - [0.001, 1e+09]
     - First SpringPot constant
   * - ``c2``
     - Pa·s\ :sup:`γ`
     - [0.001, 1e+09]
     - Second SpringPot constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - First fractional order
   * - ``beta``
     - dimensionless
     - [0, 1]
     - Second fractional order
   * - ``gamma``
     - dimensionless
     - [0, 1]
     - Third fractional order
   * - ``tau``
     - s
     - [1e-06, 1e+06]
     - Relaxation time

.. autoclass:: rheojax.models.fractional.fractional_zener_ll.FractionalZenerLiquidLiquid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Kelvin-Voigt Zener
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_kv_zener.FractionalKelvinVoigtZener` | Handbook: :doc:`/models/fractional/fractional_kv_zener`
Series spring plus fractional Kelvin-Voigt block.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Ge``
     - Pa
     - [0.001, 1e+09]
     - Series spring modulus
   * - ``Gk``
     - Pa
     - [0.001, 1e+09]
     - KV element modulus
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - s
     - [1e-06, 1e+06]
     - Retardation time

.. autoclass:: rheojax.models.fractional.fractional_kv_zener.FractionalKelvinVoigtZener
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Burgers
~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_burgers.FractionalBurgersModel` | Handbook: :doc:`/models/fractional/fractional_burgers`
Combines Maxwell and Kelvin-Voigt elements for asphalt-like flows.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Jg``
     - 1/Pa
     - [1e-09, 1000]
     - Glassy compliance
   * - ``eta1``
     - Pa·s
     - [1e-06, 1e+12]
     - Viscosity (Maxwell arm)
   * - ``Jk``
     - 1/Pa
     - [1e-09, 1000]
     - Kelvin compliance
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau_k``
     - s
     - [1e-06, 1e+06]
     - Retardation time

.. autoclass:: rheojax.models.fractional.fractional_burgers.FractionalBurgersModel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Poynting-Thomson
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_poynting_thomson.FractionalPoyntingThomson` | Handbook: :doc:`/models/fractional/fractional_poynting_thomson`
Fractional parallel branch with retardation time.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``Ge``
     - Pa
     - [0.001, 1e+09]
     - Instantaneous modulus
   * - ``Gk``
     - Pa
     - [0.001, 1e+09]
     - Retarded modulus
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau``
     - s
     - [1e-06, 1e+06]
     - Retardation time

.. autoclass:: rheojax.models.fractional.fractional_poynting_thomson.FractionalPoyntingThomson
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Jeffreys
~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional.fractional_jeffreys.FractionalJeffreysModel` | Handbook: :doc:`/models/fractional/fractional_jeffreys`
Two dashpots plus SpringPot for thixotropic-like liquids.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta1``
     - Pa·s
     - [1e-06, 1e+12]
     - First viscosity
   * - ``eta2``
     - Pa·s
     - [1e-06, 1e+12]
     - Second viscosity
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order
   * - ``tau1``
     - s
     - [1e-06, 1e+06]
     - Relaxation time

.. autoclass:: rheojax.models.fractional.fractional_jeffreys.FractionalJeffreysModel
   :members:
   :undoc-members:
   :show-inheritance:

Non-Newtonian Flow Models
-------------------------

Power Law
~~~~~~~~~

:class:`rheojax.models.flow.power_law.PowerLaw` | Handbook: :doc:`/models/flow/power_law`
Two-parameter viscosity curve tau = K*gamma^n.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``K``
     - Pa·s\ :sup:`n`
     - [1e-06, 1e+06]
     - Consistency index
   * - ``n``
     - dimensionless
     - [0.01, 2]
     - Flow behavior index

.. autoclass:: rheojax.models.flow.power_law.PowerLaw
   :members:
   :undoc-members:
   :show-inheritance:

Carreau
~~~~~~~

:class:`rheojax.models.flow.carreau.Carreau` | Handbook: :doc:`/models/flow/carreau`
Smooth transition from Newtonian plateau to shear thinning.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta0``
     - Pa·s
     - [0.001, 1e+12]
     - Zero-shear viscosity
   * - ``eta_inf``
     - Pa·s
     - [1e-06, 1e+06]
     - Infinite-shear viscosity
   * - ``lambda_``
     - s
     - [1e-06, 1e+06]
     - Time constant
   * - ``n``
     - dimensionless
     - [0.01, 1]
     - Power-law index

.. autoclass:: rheojax.models.flow.carreau.Carreau
   :members:
   :undoc-members:
   :show-inheritance:

Carreau-Yasuda
~~~~~~~~~~~~~~

:class:`rheojax.models.flow.carreau_yasuda.CarreauYasuda` | Handbook: :doc:`/models/flow/carreau_yasuda`
Adds the Yasuda exponent to tune transition sharpness.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta0``
     - Pa·s
     - [0.001, 1e+12]
     - Zero-shear viscosity
   * - ``eta_inf``
     - Pa·s
     - [1e-06, 1e+06]
     - Infinite-shear viscosity
   * - ``lambda_``
     - s
     - [1e-06, 1e+06]
     - Time constant
   * - ``n``
     - dimensionless
     - [0.01, 1]
     - Power-law index
   * - ``a``
     - dimensionless
     - [0.1, 2]
     - Transition parameter

.. autoclass:: rheojax.models.flow.carreau_yasuda.CarreauYasuda
   :members:
   :undoc-members:
   :show-inheritance:

Cross
~~~~~

:class:`rheojax.models.flow.cross.Cross` | Handbook: :doc:`/models/flow/cross`
Alternative viscosity curve with rate exponent m.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta0``
     - Pa·s
     - [0.001, 1e+12]
     - Zero-shear viscosity
   * - ``eta_inf``
     - Pa·s
     - [1e-06, 1e+06]
     - Infinite-shear viscosity
   * - ``lambda_``
     - s
     - [1e-06, 1e+06]
     - Time constant
   * - ``m``
     - dimensionless
     - [0.1, 2]
     - Rate constant

.. autoclass:: rheojax.models.flow.cross.Cross
   :members:
   :undoc-members:
   :show-inheritance:

Herschel-Bulkley
~~~~~~~~~~~~~~~~

:class:`rheojax.models.flow.herschel_bulkley.HerschelBulkley` | Handbook: :doc:`/models/flow/herschel_bulkley`
Yield stress plus power-law viscosity.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``sigma_y``
     - Pa
     - [0, 1e+06]
     - Yield stress
   * - ``K``
     - Pa·s\ :sup:`n`
     - [1e-06, 1e+06]
     - Consistency index
   * - ``n``
     - dimensionless
     - [0.01, 2]
     - Flow behavior index

.. autoclass:: rheojax.models.flow.herschel_bulkley.HerschelBulkley
   :members:
   :undoc-members:
   :show-inheritance:

Bingham
~~~~~~~

:class:`rheojax.models.flow.bingham.Bingham` | Handbook: :doc:`/models/flow/bingham`
Yield stress and constant plastic viscosity.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``sigma_y``
     - Pa
     - [0, 1e+06]
     - Yield stress
   * - ``eta_p``
     - Pa·s
     - [1e-06, 1e+12]
     - Plastic viscosity

.. autoclass:: rheojax.models.flow.bingham.Bingham
   :members:
   :undoc-members:
   :show-inheritance:

Soft Glassy Rheology (SGR) Models
---------------------------------

SGR Conventional
~~~~~~~~~~~~~~~~

:class:`rheojax.models.sgr.sgr_conventional.SGRConventional` | Handbook: :doc:`/models/sgr/sgr_conventional`
Statistical mechanics model for soft glassy materials (Sollich 1998).

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``x``
     - dimensionless
     - [0.5, 3.0]
     - Noise temperature (:math:`x < 1`: glass, :math:`1 < x < 2`: SGM, :math:`x \geq 2`: Newtonian)
   * - ``G0``
     - Pa
     - [0.001, 1e+09]
     - Characteristic modulus
   * - ``tau0``
     - s
     - [1e-09, 1e+06]
     - Microscopic attempt time

**Material Classification by Noise Temperature:**

- :math:`x < 1`: Glass (aging, non-ergodic, yield stress)
- :math:`1 < x < 2`: Soft Glassy Material (power-law rheology)
- :math:`x \geq 2`: Newtonian liquid (no memory effects)

.. autoclass:: rheojax.models.sgr.sgr_conventional.SGRConventional
   :members:
   :undoc-members:
   :show-inheritance:

SGR Generic (GENERIC Framework)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.sgr.sgr_generic.SGRGeneric` | Handbook: :doc:`/models/sgr/sgr_generic`
Thermodynamically consistent SGR using GENERIC framework (Fuereder & Ilg 2013).

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``x``
     - dimensionless
     - [0.5, 3.0]
     - Noise temperature
   * - ``G0``
     - Pa
     - [0.001, 1e+09]
     - Characteristic modulus
   * - ``tau0``
     - s
     - [1e-09, 1e+06]
     - Microscopic attempt time

**Advantages over Conventional SGR:**

- Satisfies Onsager reciprocal relations
- Enhanced numerical stability near glass transition (x → 1)
- Consistent thermodynamic framework

.. autoclass:: rheojax.models.sgr.sgr_generic.SGRGeneric
   :members:
   :undoc-members:
   :show-inheritance:

STZ Models
----------

STZ Conventional
~~~~~~~~~~~~~~~~

:class:`rheojax.models.stz.conventional.STZConventional` | Handbook: :doc:`/models/stz/stz_conventional`
Shear Transformation Zone model for amorphous plasticity (Langer 2008).

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G0``
     - Pa
     - [1e6, 1e12]
     - Elastic shear modulus
   * - ``sigma_y``
     - Pa
     - [1e3, 1e9]
     - Yield stress scale
   * - ``chi_inf``
     - -
     - [0.01, 0.5]
     - Steady-state effective temperature
   * - ``tau0``
     - s
     - [1e-14, 1e-9]
     - Molecular attempt time
   * - ``epsilon0``
     - -
     - [0.01, 1.0]
     - Strain increment per flip
   * - ``c0``
     - -
     - [0.1, 100]
     - Specific heat
   * - ``ez``
     - -
     - [0.1, 5.0]
     - STZ formation energy

.. autoclass:: rheojax.models.stz.conventional.STZConventional
   :members:
   :undoc-members:
   :show-inheritance:

Elasto-Plastic Models (EPM)
---------------------------

Lattice EPM
~~~~~~~~~~~

:class:`rheojax.models.epm.lattice.LatticeEPM` | Handbook: :doc:`/models/epm/lattice_epm`
Mesoscopic lattice model for amorphous solids with spatial heterogeneity and avalanches.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``mu``
     - Pa
     - [0.1, 100.0]
     - Shear modulus
   * - ``tau_pl``
     - s
     - [0.01, 100.0]
     - Plastic relaxation timescale
   * - ``sigma_c_mean``
     - Pa
     - [0.1, 10.0]
     - Mean yield threshold
   * - ``sigma_c_std``
     - Pa
     - [0.0, 1.0]
     - Disorder strength (std dev of thresholds)
   * - ``smoothing_width``
     - Pa
     - [0.01, 1.0]
     - Width for smooth yielding approx (inference)

.. autoclass:: rheojax.models.epm.lattice.LatticeEPM
   :members:
   :undoc-members:
   :show-inheritance:

Tensorial EPM
~~~~~~~~~~~~~

:class:`rheojax.models.epm.tensor.TensorialEPM` | Handbook: :doc:`/models/epm/tensorial_epm`
Full stress tensor implementation with normal stress difference predictions.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``mu``
     - Pa
     - [0.1, 100.0]
     - Shear modulus
   * - ``nu``
     - dimensionless
     - [0.3, 0.5]
     - Poisson's ratio (plane strain)
   * - ``tau_pl_shear``
     - s
     - [0.01, 100.0]
     - Plastic relaxation time for shear
   * - ``tau_pl_normal``
     - s
     - [0.01, 100.0]
     - Plastic relaxation time for normal stresses
   * - ``sigma_c_mean``
     - Pa
     - [0.1, 10.0]
     - Mean yield threshold
   * - ``sigma_c_std``
     - Pa
     - [0.0, 1.0]
     - Disorder strength (std dev)
   * - ``w_N1``
     - dimensionless
     - [0.1, 10.0]
     - Weight for N₁ in combined fitting
   * - ``hill_H``
     - dimensionless
     - [0.1, 5.0]
     - Hill anisotropy parameter H
   * - ``hill_N``
     - dimensionless
     - [0.1, 5.0]
     - Hill anisotropy parameter N

**Configuration**: ``L=64`` (lattice size), ``dt=0.01`` (timestep), ``yield_criterion="von_mises"`` or ``"hill"``

**Key Features**:
- Tracks full stress tensor [σ_xx, σ_yy, σ_xy]
- Predicts normal stress differences N₁, N₂
- Von Mises (isotropic) or Hill (anisotropic) yield criteria
- Flexible fitting: shear-only or combined [σ_xy, N₁]

.. autoclass:: rheojax.models.epm.tensor.TensorialEPM
   :members:
   :undoc-members:
   :show-inheritance:

SPP LAOS Models
---------------

SPP Yield Stress
~~~~~~~~~~~~~~~~

:class:`rheojax.models.spp.spp_yield_stress.SPPYieldStress` | Handbook: :doc:`/models/spp/spp_yield_stress`
Yield stress model for Sequence of Physical Processes (SPP) LAOS analysis.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``sigma_y``
     - Pa
     - [0, 1e+06]
     - Static yield stress
   * - ``G_cage``
     - Pa
     - [0.001, 1e+09]
     - Cage modulus
   * - ``n``
     - dimensionless
     - [0.01, 2]
     - Power-law exponent

**Note**: Use with :class:`rheojax.transforms.spp_decomposer.SPPDecomposer` for
complete LAOS analysis. See :doc:`/api/spp_models` for the full SPP API reference.

.. autoclass:: rheojax.models.spp.spp_yield_stress.SPPYieldStress
   :members:
   :undoc-members:
   :show-inheritance:

Fluidity Models
---------------

FluidityLocal
~~~~~~~~~~~~~

:class:`rheojax.models.fluidity.local.FluidityLocal` | Handbook: :doc:`/models/fluidity/fluidity_local`
Local (0D) fluidity model with aging/rejuvenation kinetics for thixotropic yield-stress fluids.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [1e+03, 1e+09]
     - Elastic modulus
   * - ``tau_y``
     - Pa
     - [10, 1e+06]
     - Yield stress
   * - ``K``
     - Pa·s\ :sup:`n`
     - [1, 1e+06]
     - Flow consistency (HB K parameter)
   * - ``n_flow``
     - --
     - [0.1, 2]
     - Flow exponent (HB n parameter)
   * - ``f_eq``
     - 1/(Pa·s)
     - [1e-12, 0.001]
     - Equilibrium fluidity (aging limit)
   * - ``f_inf``
     - 1/(Pa·s)
     - [1e-06, 1]
     - High-shear fluidity (rejuvenation limit)
   * - ``theta``
     - s
     - [0.1, 1e+04]
     - Structural relaxation time (aging timescale)
   * - ``a``
     - --
     - [0, 100]
     - Rejuvenation amplitude
   * - ``n_rejuv``
     - --
     - [0, 2]
     - Rejuvenation exponent

.. autoclass:: rheojax.models.fluidity.local.FluidityLocal
   :members:
   :undoc-members:
   :show-inheritance:

FluidityNonlocal
~~~~~~~~~~~~~~~~

:class:`rheojax.models.fluidity.nonlocal_model.FluidityNonlocal` | Handbook: :doc:`/models/fluidity/fluidity_nonlocal`
Nonlocal (1D) fluidity model with cooperativity length for shear banding prediction.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [1e+03, 1e+09]
     - Elastic modulus
   * - ``tau_y``
     - Pa
     - [10, 1e+06]
     - Yield stress
   * - ``K``
     - Pa·s\ :sup:`n`
     - [1, 1e+06]
     - Flow consistency (HB K parameter)
   * - ``n_flow``
     - --
     - [0.1, 2]
     - Flow exponent (HB n parameter)
   * - ``f_eq``
     - 1/(Pa·s)
     - [1e-12, 0.001]
     - Equilibrium fluidity (aging limit)
   * - ``f_inf``
     - 1/(Pa·s)
     - [1e-06, 1]
     - High-shear fluidity (rejuvenation limit)
   * - ``theta``
     - s
     - [0.1, 1e+04]
     - Structural relaxation time
   * - ``a``
     - --
     - [0, 100]
     - Rejuvenation amplitude
   * - ``n_rejuv``
     - --
     - [0, 2]
     - Rejuvenation exponent
   * - ``xi``
     - m
     - [1e-09, 0.001]
     - Cooperativity length (non-local diffusion scale)

.. autoclass:: rheojax.models.fluidity.nonlocal_model.FluidityNonlocal
   :members:
   :undoc-members:
   :show-inheritance:

Fluidity-Saramito EVP Models
-----------------------------

FluiditySaramitoLocal
~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fluidity.saramito.local.FluiditySaramitoLocal` | Handbook: :doc:`/models/fluidity/saramito_evp`
Local (0D) elastoviscoplastic model combining Saramito tensorial viscoelasticity
with thixotropic fluidity evolution and Von Mises yielding.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [10, 1e+08]
     - Elastic modulus
   * - ``eta_s``
     - Pa·s
     - [0, 1e+03]
     - Solvent viscosity
   * - ``tau_y0``
     - Pa
     - [0.1, 1e+05]
     - Base yield stress (Von Mises threshold)
   * - ``K_HB``
     - Pa·s\ :sup:`n`
     - [0.01, 1e+05]
     - Herschel-Bulkley consistency index
   * - ``n_HB``
     - --
     - [0.1, 1.5]
     - Herschel-Bulkley flow exponent
   * - ``f_age``
     - 1/(Pa·s)
     - [1e-12, 0.01]
     - Aging fluidity limit
   * - ``f_flow``
     - 1/(Pa·s)
     - [1e-06, 1]
     - Flow fluidity limit (rejuvenation)
   * - ``t_a``
     - s
     - [0.01, 1e+05]
     - Aging timescale (structural build-up)
   * - ``b``
     - --
     - [0, 1e+03]
     - Rejuvenation amplitude
   * - ``n_rej``
     - --
     - [0.1, 3]
     - Rejuvenation rate exponent

.. autoclass:: rheojax.models.fluidity.saramito.local.FluiditySaramitoLocal
   :members:
   :undoc-members:
   :show-inheritance:

FluiditySaramitoNonlocal
~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fluidity.saramito.nonlocal_model.FluiditySaramitoNonlocal` | Handbook: :doc:`/models/fluidity/saramito_evp`
Nonlocal (1D) Saramito EVP model with spatial cooperativity for shear banding.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [10, 1e+08]
     - Elastic modulus
   * - ``eta_s``
     - Pa·s
     - [0, 1e+03]
     - Solvent viscosity
   * - ``tau_y0``
     - Pa
     - [0.1, 1e+05]
     - Base yield stress (Von Mises threshold)
   * - ``K_HB``
     - Pa·s\ :sup:`n`
     - [0.01, 1e+05]
     - Herschel-Bulkley consistency index
   * - ``n_HB``
     - --
     - [0.1, 1.5]
     - Herschel-Bulkley flow exponent
   * - ``f_age``
     - 1/(Pa·s)
     - [1e-12, 0.01]
     - Aging fluidity limit
   * - ``f_flow``
     - 1/(Pa·s)
     - [1e-06, 1]
     - Flow fluidity limit
   * - ``t_a``
     - s
     - [0.01, 1e+05]
     - Aging timescale
   * - ``b``
     - --
     - [0, 1e+03]
     - Rejuvenation amplitude
   * - ``n_rej``
     - --
     - [0.1, 3]
     - Rejuvenation rate exponent
   * - ``xi``
     - m
     - [1e-07, 0.01]
     - Cooperativity length (interface width)

.. autoclass:: rheojax.models.fluidity.saramito.nonlocal_model.FluiditySaramitoNonlocal
   :members:
   :undoc-members:
   :show-inheritance:

DMT Thixotropic Models
----------------------

DMTLocal
~~~~~~~~

:class:`rheojax.models.dmt.local.DMTLocal` | Handbook: :doc:`/models/dmt/dmt`
Local (0D) de Souza Mendes-Thompson thixotropic model with scalar structure parameter
and exponential or Herschel-Bulkley viscosity closure.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_0``
     - Pa·s
     - [100, 1e+08]
     - Zero-shear viscosity (fully structured, :math:`\lambda=1`)
   * - ``eta_inf``
     - Pa·s
     - [0.001, 100]
     - Infinite-shear viscosity (fully broken, :math:`\lambda=0`)
   * - ``t_eq``
     - s
     - [0.1, 1e+04]
     - Structural equilibrium (buildup) timescale
   * - ``a``
     - --
     - [0.001, 100]
     - Breakdown rate coefficient
   * - ``c``
     - --
     - [0.1, 2]
     - Breakdown rate exponent (shear rate sensitivity)

.. autoclass:: rheojax.models.dmt.local.DMTLocal
   :members:
   :undoc-members:
   :show-inheritance:

DMTNonlocal
~~~~~~~~~~~

:class:`rheojax.models.dmt.nonlocal_model.DMTNonlocal` | Handbook: :doc:`/models/dmt/dmt`
Nonlocal (1D) DMT model with structure diffusion for shear banding prediction.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_0``
     - Pa·s
     - [100, 1e+08]
     - Zero-shear viscosity (fully structured)
   * - ``eta_inf``
     - Pa·s
     - [0.001, 100]
     - Infinite-shear viscosity (fully broken)
   * - ``t_eq``
     - s
     - [0.1, 1e+04]
     - Structural equilibrium timescale
   * - ``a``
     - --
     - [0.001, 100]
     - Breakdown rate coefficient
   * - ``c``
     - --
     - [0.1, 2]
     - Breakdown rate exponent
   * - ``D_lambda``
     - m\ :sup:`2`/s
     - [1e-10, 0.01]
     - Structure diffusion coefficient (cooperativity)

.. autoclass:: rheojax.models.dmt.nonlocal_model.DMTNonlocal
   :members:
   :undoc-members:
   :show-inheritance:

IKH Models (Isotropic-Kinematic Hardening)
------------------------------------------

MIKH
~~~~

:class:`rheojax.models.ikh.mikh.MIKH` | Handbook: :doc:`/models/ikh/mikh`
Single-mode thixotropic elasto-viscoplastic model with Maxwell viscoelasticity,
Armstrong-Frederick kinematic hardening, and thixotropic structure evolution.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [0.1, 1e+09]
     - Shear modulus
   * - ``eta``
     - Pa·s
     - [0.001, 1e+12]
     - Maxwell viscosity (relaxation time :math:`= \eta/G`)
   * - ``C``
     - Pa
     - [0, 1e+09]
     - Kinematic hardening modulus
   * - ``gamma_dyn``
     - --
     - [0, 1e+04]
     - Dynamic recovery parameter
   * - ``m``
     - --
     - [0.5, 3]
     - AF recovery exponent
   * - ``sigma_y0``
     - Pa
     - [0, 1e+09]
     - Minimal yield stress (destructured)
   * - ``delta_sigma_y``
     - Pa
     - [0, 1e+09]
     - Structural yield stress contribution
   * - ``tau_thix``
     - s
     - [1e-06, 1e+12]
     - Thixotropic rebuilding timescale
   * - ``Gamma``
     - --
     - [0, 1e+04]
     - Structural breakdown coefficient
   * - ``eta_inf``
     - Pa·s
     - [0, 1e+09]
     - High-shear viscosity
   * - ``mu_p``
     - Pa·s
     - [0.001, 1e+12]
     - Plastic viscosity (Perzyna regularization)

.. autoclass:: rheojax.models.ikh.mikh.MIKH
   :members:
   :undoc-members:
   :show-inheritance:

MLIKH
~~~~~

:class:`rheojax.models.ikh.ml_ikh.MLIKH` | Handbook: :doc:`/models/ikh/ml_ikh`
Multi-layer IKH model with multiple viscoelastic-thixotropic modes for
broad relaxation spectra in complex yield-stress fluids.

.. list-table:: Parameters (per mode ``i``)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_i``
     - Pa
     - [0, 1e+09]
     - Mode *i* shear modulus
   * - ``eta_i``
     - Pa·s
     - [0.001, 1e+12]
     - Mode *i* Maxwell viscosity
   * - ``C_i``
     - Pa
     - [0, 1e+09]
     - Mode *i* kinematic hardening modulus
   * - ``gamma_dyn_i``
     - --
     - [0, 1e+04]
     - Mode *i* dynamic recovery parameter
   * - ``sigma_y0_i``
     - Pa
     - [0, 1e+09]
     - Mode *i* minimal yield stress
   * - ``delta_sigma_y_i``
     - Pa
     - [0, 1e+09]
     - Mode *i* structural yield stress
   * - ``tau_thix_i``
     - s
     - [1e-06, 1e+12]
     - Mode *i* thixotropic timescale
   * - ``Gamma_i``
     - --
     - [0, 1e+04]
     - Mode *i* breakdown coefficient

.. list-table:: Global Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_inf``
     - Pa·s
     - [0, 1e+09]
     - High-shear viscosity (shared)

.. autoclass:: rheojax.models.ikh.ml_ikh.MLIKH
   :members:
   :undoc-members:
   :show-inheritance:

FIKH Models (Fractional IKH)
-----------------------------

FIKH
~~~~

:class:`rheojax.models.fikh.fikh.FIKH` | Handbook: :doc:`/models/fikh/fikh`
Fractional IKH model with Caputo fractional derivative for power-law memory
in structure evolution (single-mode).

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [0.1, 1e+09]
     - Shear modulus
   * - ``eta``
     - Pa·s
     - [0.001, 1e+12]
     - Maxwell viscosity
   * - ``C``
     - Pa
     - [0, 1e+09]
     - Kinematic hardening modulus
   * - ``gamma_dyn``
     - --
     - [0, 1e+04]
     - Dynamic recovery parameter
   * - ``m``
     - --
     - [0.5, 3]
     - AF recovery exponent
   * - ``sigma_y0``
     - Pa
     - [0, 1e+09]
     - Minimal yield stress (destructured)
   * - ``delta_sigma_y``
     - Pa
     - [0, 1e+09]
     - Structural yield stress contribution
   * - ``tau_thix``
     - s
     - [1e-06, 1e+12]
     - Thixotropic rebuilding timescale
   * - ``Gamma``
     - --
     - [0, 1e+04]
     - Structural breakdown coefficient
   * - ``alpha_structure``
     - --
     - [0.01, 0.99]
     - Fractional order for structure evolution
   * - ``eta_inf``
     - Pa·s
     - [0, 1e+09]
     - High-shear viscosity
   * - ``mu_p``
     - Pa·s
     - [0.001, 1e+12]
     - Plastic viscosity (Perzyna regularization)

.. autoclass:: rheojax.models.fikh.fikh.FIKH
   :members:
   :undoc-members:
   :show-inheritance:

FMLIKH
~~~~~~

:class:`rheojax.models.fikh.fmlikh.FMLIKH` | Handbook: :doc:`/models/fikh/fmlikh`
Fractional multi-layer IKH model combining multi-mode viscoelasticity with
Caputo fractional structure kinetics.

.. list-table:: Parameters (per mode ``i``)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_i``
     - Pa
     - [0.1, 1e+09]
     - Mode *i* shear modulus
   * - ``eta_i``
     - Pa·s
     - [0.001, 1e+12]
     - Mode *i* Maxwell viscosity
   * - ``C_i``
     - Pa
     - [0, 1e+09]
     - Mode *i* kinematic hardening modulus
   * - ``gamma_dyn_i``
     - --
     - [0, 1e+04]
     - Mode *i* dynamic recovery parameter

.. list-table:: Shared Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``sigma_y0``
     - Pa
     - [0, 1e+09]
     - Minimal yield stress
   * - ``delta_sigma_y``
     - Pa
     - [0, 1e+09]
     - Structural yield stress contribution
   * - ``tau_thix``
     - s
     - [1e-06, 1e+12]
     - Thixotropic rebuilding timescale
   * - ``Gamma``
     - --
     - [0, 1e+04]
     - Structural breakdown coefficient
   * - ``alpha_structure``
     - --
     - [0.01, 0.99]
     - Fractional order for structure evolution
   * - ``eta_inf``
     - Pa·s
     - [0, 1e+09]
     - High-shear viscosity
   * - ``mu_p``
     - Pa·s
     - [0.001, 1e+12]
     - Plastic viscosity

.. autoclass:: rheojax.models.fikh.fmlikh.FMLIKH
   :members:
   :undoc-members:
   :show-inheritance:

Hébraud-Lequeux Model
---------------------

HebraudLequeux
~~~~~~~~~~~~~~

:class:`rheojax.models.hl.hebraud_lequeux.HebraudLequeux` | Handbook: :doc:`/models/hl/hebraud_lequeux`
Mean-field stochastic model for soft glassy materials with stress-induced yielding
and mechanical noise coupling.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``alpha``
     - --
     - [0, 1]
     - Coupling parameter (:math:`\alpha < 0.5` implies yield stress)
   * - ``tau``
     - s
     - [1e-06, 1e+04]
     - Microscopic yield timescale
   * - ``sigma_c``
     - Pa
     - [0.001, 1e+06]
     - Critical yield stress threshold

.. autoclass:: rheojax.models.hl.hebraud_lequeux.HebraudLequeux
   :members:
   :undoc-members:
   :show-inheritance:

Giesekus Models
---------------

GiesekusSingleMode
~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.giesekus.single_mode.GiesekusSingleMode` | Handbook: :doc:`/models/giesekus/giesekus`
Nonlinear differential constitutive model with anisotropic mobility for shear-thinning
polymer solutions and melts.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_p``
     - Pa·s
     - [0.001, 1e+06]
     - Polymer viscosity (zero-shear polymer contribution)
   * - ``lambda_1``
     - s
     - [1e-06, 1e+04]
     - Relaxation time
   * - ``alpha``
     - --
     - [0, 0.5]
     - Mobility factor (0 = UCM, 0.5 = max anisotropy)
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity (Newtonian contribution)

.. autoclass:: rheojax.models.giesekus.single_mode.GiesekusSingleMode
   :members:
   :undoc-members:
   :show-inheritance:

GiesekusMultiMode
~~~~~~~~~~~~~~~~~

:class:`rheojax.models.giesekus.multi_mode.GiesekusMultiMode` | Handbook: :doc:`/models/giesekus/giesekus`
Multi-mode Giesekus model for broad relaxation spectra in polymer systems.

.. list-table:: Parameters (per mode ``i``)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_p_i``
     - Pa·s
     - [0.001, 1e+06]
     - Mode *i* polymer viscosity
   * - ``lambda_1_i``
     - s
     - [1e-06, 1e+04]
     - Mode *i* relaxation time
   * - ``alpha_i``
     - --
     - [0, 0.5]
     - Mode *i* mobility factor

.. list-table:: Global Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity (shared)

.. autoclass:: rheojax.models.giesekus.multi_mode.GiesekusMultiMode
   :members:
   :undoc-members:
   :show-inheritance:

ITT-MCT Models (Integration Through Transients MCT)
----------------------------------------------------

ITTMCTSchematic
~~~~~~~~~~~~~~~

:class:`rheojax.models.itt_mct.schematic.ITTMCTSchematic` | Handbook: :doc:`/models/itt_mct/itt_mct_schematic`
F\ :sub:`12` schematic MCT model for dense colloidal suspensions and glassy materials
with memory kernel :math:`m(\Phi) = v_1\Phi + v_2\Phi^2`.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``v1``
     - --
     - [0, 5]
     - Linear vertex coefficient (typically 0 for F\ :sub:`12`)
   * - ``v2``
     - --
     - [0.5, 10]
     - Quadratic vertex coefficient (glass at :math:`v_2 > 4`)
   * - ``Gamma``
     - 1/s
     - [1e-06, 1e+06]
     - Bare relaxation rate
   * - ``gamma_c``
     - --
     - [0.01, 0.5]
     - Critical strain for cage breaking
   * - ``G_inf``
     - Pa
     - [1, 1e+12]
     - High-frequency elastic modulus

.. autoclass:: rheojax.models.itt_mct.schematic.ITTMCTSchematic
   :members:
   :undoc-members:
   :show-inheritance:

ITTMCTIsotropic
~~~~~~~~~~~~~~~

:class:`rheojax.models.itt_mct.isotropic.ITTMCTIsotropic` | Handbook: :doc:`/models/itt_mct/itt_mct_isotropic`
Isotropic Schematic Model (ISM) with Percus-Yevick structure factor :math:`S(k)`
for quantitative predictions from microscopic parameters.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``phi``
     - --
     - [0.1, 0.64]
     - Volume fraction (glass at :math:`\phi \approx 0.516`)
   * - ``sigma_d``
     - m
     - [1e-09, 0.001]
     - Particle diameter
   * - ``D0``
     - m\ :sup:`2`/s
     - [1e-18, 1e-06]
     - Bare short-time diffusion coefficient
   * - ``kBT``
     - J
     - [1e-24, 1e-18]
     - Thermal energy :math:`k_B T`
   * - ``gamma_c``
     - --
     - [0.01, 0.5]
     - Critical strain for cage breaking

.. autoclass:: rheojax.models.itt_mct.isotropic.ITTMCTIsotropic
   :members:
   :undoc-members:
   :show-inheritance:

TNT Models (Transient Network Theory)
--------------------------------------

TNTSingleMode
~~~~~~~~~~~~~

:class:`rheojax.models.tnt.single_mode.TNTSingleMode` | Handbook: :doc:`/models/tnt/tnt_bell`
Single-mode transient network model with configurable breakage kinetics
(Bell, power-law, stretch-creation) and optional FENE-P springs.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [1, 1e+08]
     - Network modulus (active chain density)
   * - ``tau_b``
     - s
     - [1e-06, 1e+04]
     - Bond lifetime (mean detachment time)
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity (Newtonian background)
   * - ``nu``
     - --
     - [0.01, 20]
     - Bell force sensitivity (higher = more shear-thinning)
   * - ``L_max``
     - --
     - [2, 100]
     - Maximum chain extensibility (FENE-P, if enabled)

.. autoclass:: rheojax.models.tnt.single_mode.TNTSingleMode
   :members:
   :undoc-members:
   :show-inheritance:

TNTLoopBridge
~~~~~~~~~~~~~

:class:`rheojax.models.tnt.loop_bridge.TNTLoopBridge` | Handbook: :doc:`/models/tnt/tnt_loop_bridge`
Loop-bridge transient network model with stress-dependent loop-to-bridge conversion.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G``
     - Pa
     - [1, 1e+08]
     - Network modulus (fully bridged state)
   * - ``tau_b``
     - s
     - [1e-06, 1e+04]
     - Bridge detachment time
   * - ``tau_a``
     - s
     - [1e-06, 1e+04]
     - Loop attachment time
   * - ``nu``
     - --
     - [0.01, 20]
     - Bell force sensitivity
   * - ``f_B_eq``
     - --
     - [0.01, 0.99]
     - Equilibrium bridge fraction at rest
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity

.. autoclass:: rheojax.models.tnt.loop_bridge.TNTLoopBridge
   :members:
   :undoc-members:
   :show-inheritance:

TNTStickyRouse
~~~~~~~~~~~~~~

:class:`rheojax.models.tnt.sticky_rouse.TNTStickyRouse` | Handbook: :doc:`/models/tnt/tnt_sticky_rouse`
Sticky Rouse model combining Rouse chain dynamics with reversible sticker groups.

.. list-table:: Parameters (per mode ``k``)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_k``
     - Pa
     - [1, 1e+08]
     - Mode *k* modulus
   * - ``tau_R_k``
     - s
     - [1e-06, 1e+04]
     - Mode *k* Rouse relaxation time

.. list-table:: Global Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``tau_s``
     - s
     - [1e-06, 1e+04]
     - Sticker lifetime
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity

.. autoclass:: rheojax.models.tnt.sticky_rouse.TNTStickyRouse
   :members:
   :undoc-members:
   :show-inheritance:

TNTCates
~~~~~~~~

:class:`rheojax.models.tnt.cates.TNTCates` | Handbook: :doc:`/models/tnt/tnt_cates`
Cates living polymer model with coupled reptation and reversible scission
for worm-like micelle solutions.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_0``
     - Pa
     - [1, 1e+08]
     - Plateau modulus
   * - ``tau_rep``
     - s
     - [1e-04, 1e+06]
     - Reptation time (curvilinear diffusion)
   * - ``tau_break``
     - s
     - [1e-06, 1e+04]
     - Average breaking time (scission events)
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity

.. autoclass:: rheojax.models.tnt.cates.TNTCates
   :members:
   :undoc-members:
   :show-inheritance:

TNTMultiSpecies
~~~~~~~~~~~~~~~

:class:`rheojax.models.tnt.multi_species.TNTMultiSpecies` | Handbook: :doc:`/models/tnt/tnt_multi_species`
Multi-species transient network model with multiple independent bond types
of distinct lifetimes and moduli.

.. list-table:: Parameters (per species ``i``)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_i``
     - Pa
     - [1, 1e+08]
     - Network modulus for bond species *i*
   * - ``tau_b_i``
     - s
     - [1e-06, 1e+04]
     - Bond lifetime for species *i*

.. list-table:: Global Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta_s``
     - Pa·s
     - [0, 1e+04]
     - Solvent viscosity

.. autoclass:: rheojax.models.tnt.multi_species.TNTMultiSpecies
   :members:
   :undoc-members:
   :show-inheritance:

VLB Models (Vernerey-Long-Brighenti)
-------------------------------------

VLBLocal
~~~~~~~~

:class:`rheojax.models.vlb.local.VLBLocal` | Handbook: :doc:`/models/vlb/vlb`
Local VLB transient network model with chain-length-resolved bond kinetics
and non-affine deformation.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G0``
     - Pa
     - [1, 1e+08]
     - Network modulus (:math:`n k_B T`)
   * - ``k_d``
     - 1/s
     - [1e-06, 1e+06]
     - Dissociation rate (inverse relaxation time)

.. autoclass:: rheojax.models.vlb.local.VLBLocal
   :members:
   :undoc-members:
   :show-inheritance:

VLBMultiNetwork
~~~~~~~~~~~~~~~

:class:`rheojax.models.vlb.multi_network.VLBMultiNetwork` | Handbook: :doc:`/models/vlb/vlb`
Multi-network VLB model with multiple independent transient sub-networks
of distinct moduli and kinetics.

.. list-table:: Parameters (per network ``i``)
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G0_i``
     - Pa
     - [1, 1e+08]
     - Network *i* modulus
   * - ``k_d_i``
     - 1/s
     - [1e-06, 1e+06]
     - Network *i* dissociation rate

.. autoclass:: rheojax.models.vlb.multi_network.VLBMultiNetwork
   :members:
   :undoc-members:
   :show-inheritance:

VLBVariant
~~~~~~~~~~

:class:`rheojax.models.vlb.variant.VLBVariant` | Handbook: :doc:`/models/vlb/vlb_advanced`
VLB model variant with modified kinetics (Bell, FENE-P, stretch-creation)
for enhanced nonlinear rheology.

.. autoclass:: rheojax.models.vlb.variant.VLBVariant
   :members:
   :undoc-members:
   :show-inheritance:

VLBNonlocal
~~~~~~~~~~~

:class:`rheojax.models.vlb.nonlocal_model.VLBNonlocal` | Handbook: :doc:`/models/vlb/vlb_advanced`
Nonlocal VLB model with spatial diffusion for shear banding and
inhomogeneous deformation prediction.

.. autoclass:: rheojax.models.vlb.nonlocal_model.VLBNonlocal
   :members:
   :undoc-members:
   :show-inheritance:

HVM Model (Hybrid Vitrimer)
----------------------------

HVMLocal
~~~~~~~~

:class:`rheojax.models.hvm.local.HVMLocal` | Handbook: :doc:`/models/hvm/hvm_advanced`
Hybrid vitrimer constitutive model with permanent covalent crosslinks,
associative exchangeable bonds (BER/TST kinetics), and optional dissociative
physical bonds.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_P``
     - Pa
     - [1, 1e+08]
     - Permanent network modulus (covalent crosslinks)
   * - ``G_E``
     - Pa
     - [1, 1e+08]
     - Exchangeable network modulus (vitrimer bonds)
   * - ``G_D``
     - Pa
     - [0, 1e+08]
     - Dissociative network modulus (physical bonds, optional)
   * - ``nu_0``
     - 1/s
     - [1e+06, 1e+14]
     - BER attempt frequency (Eyring pre-factor)
   * - ``E_a``
     - J/mol
     - [4e+04, 2.5e+05]
     - BER activation energy
   * - ``V_act``
     - m\ :sup:`3`
     - [1e-08, 0.01]
     - BER activation volume
   * - ``k_d_D``
     - 1/s
     - [1e-06, 1e+06]
     - Dissociation rate for D-network (optional)
   * - ``T``
     - K
     - [250, 350]
     - Temperature

.. autoclass:: rheojax.models.hvm.local.HVMLocal
   :members:
   :undoc-members:
   :show-inheritance:

HVNM Model (Hybrid Vitrimer Nanocomposite)
-------------------------------------------

HVNMLocal
~~~~~~~~~

:class:`rheojax.models.hvnm.local.HVNMLocal` | Handbook: :doc:`/models/hvnm/hvnm_advanced`
Hybrid vitrimer nanocomposite model extending HVM with a 4th interphase
sub-network around nanoparticles. Guth-Gold strain amplification :math:`X(\phi)`.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``G_P``
     - Pa
     - [1, 1e+08]
     - Permanent network modulus (amplified by :math:`X(\phi)`)
   * - ``G_E``
     - Pa
     - [1, 1e+08]
     - Exchangeable network modulus (matrix TST kinetics)
   * - ``G_D``
     - Pa
     - [0, 1e+08]
     - Dissociative network modulus (optional)
   * - ``G_I``
     - Pa
     - [1, 1e+08]
     - Interphase network modulus (optional)
   * - ``nu_0``
     - 1/s
     - [1e+06, 1e+14]
     - Matrix BER attempt frequency
   * - ``E_a``
     - J/mol
     - [4e+04, 2.5e+05]
     - Matrix BER activation energy
   * - ``nu_0_int``
     - 1/s
     - [1e+06, 1e+14]
     - Interfacial BER attempt frequency (optional)
   * - ``E_a_int``
     - J/mol
     - [4e+04, 2.5e+05]
     - Interfacial BER activation energy (optional)
   * - ``phi``
     - --
     - [0, 0.4]
     - NP volume fraction
   * - ``R_NP``
     - m
     - [1e-09, 1e-06]
     - NP radius
   * - ``beta_I``
     - --
     - [1, 10]
     - Interphase reinforcement factor
   * - ``T``
     - K
     - [250, 350]
     - Temperature

.. autoclass:: rheojax.models.hvnm.local.HVNMLocal
   :members:
   :undoc-members:
   :show-inheritance:
