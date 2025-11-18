Models API
==========

Concise parameter reference for every rheological model. For derivations, limits,
and usage recipes see the :doc:`/models/index` handbook.

.. contents::
   :local:
   :depth: 2

Classical Models
----------------

Maxwell
~~~~~~~

:class:`rheojax.models.maxwell.Maxwell` | Handbook: :doc:`/models/classical/maxwell`
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
     - Pa*s
     - [1e-06, 1e+12]
     - Viscosity

.. autoclass:: rheojax.models.maxwell.Maxwell
   :members:
   :undoc-members:
   :show-inheritance:

Zener (Standard Linear Solid)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.zener.Zener` | Handbook: :doc:`/models/classical/zener`
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
     - Pa*s
     - [1e-06, 1e+12]
     - Viscosity

.. autoclass:: rheojax.models.zener.Zener
   :members:
   :undoc-members:
   :show-inheritance:

SpringPot
~~~~~~~~~

:class:`rheojax.models.springpot.SpringPot` | Handbook: :doc:`/models/classical/springpot`
Fractional element interpolating between elastic and viscous limits.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c_alpha``
     - Pa*s^alpha
     - [0.001, 1e+09]
     - Material constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Power-law exponent (0=fluid, 1=solid)

.. autoclass:: rheojax.models.springpot.SpringPot
   :members:
   :undoc-members:
   :show-inheritance:

Multi-Mode Models
-----------------

GeneralizedMaxwell (Prony Series)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.generalized_maxwell.GeneralizedMaxwell` | Handbook: :doc:`/models/multi_mode/generalized_maxwell`
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

.. autoclass:: rheojax.models.generalized_maxwell.GeneralizedMaxwell
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Models
-----------------

Fractional Maxwell Gel
~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_maxwell_gel.FractionalMaxwellGel` | Handbook: :doc:`/models/fractional/fractional_maxwell_gel`
Spring in series with SpringPot for gel-like plateaus.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c_alpha``
     - Pa*s^alpha
     - [0.001, 1e+09]
     - SpringPot material constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Power-law exponent
   * - ``eta``
     - Pa*s
     - [1e-06, 1e+12]
     - Dashpot viscosity

.. autoclass:: rheojax.models.fractional_maxwell_gel.FractionalMaxwellGel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Maxwell Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_maxwell_liquid.FractionalMaxwellLiquid` | Handbook: :doc:`/models/fractional/fractional_maxwell_liquid`
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

.. autoclass:: rheojax.models.fractional_maxwell_liquid.FractionalMaxwellLiquid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Maxwell (General)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_maxwell_model.FractionalMaxwellModel` | Handbook: :doc:`/models/fractional/fractional_maxwell_model`
Two SpringPots in series for broad relaxation spectra.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c1``
     - Pa*s^alpha
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

.. autoclass:: rheojax.models.fractional_maxwell_model.FractionalMaxwellModel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Kelvin-Voigt
~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_kelvin_voigt.FractionalKelvinVoigt` | Handbook: :doc:`/models/fractional/fractional_kelvin_voigt`
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
     - Pa*s^alpha
     - [0.001, 1e+09]
     - SpringPot constant
   * - ``alpha``
     - dimensionless
     - [0, 1]
     - Fractional order

.. autoclass:: rheojax.models.fractional_kelvin_voigt.FractionalKelvinVoigt
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Zener Solid-Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_zener_sl.FractionalZenerSolidLiquid` | Handbook: :doc:`/models/fractional/fractional_zener_sl`
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
     - Pa*s^alpha
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

.. autoclass:: rheojax.models.fractional_zener_sl.FractionalZenerSolidLiquid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Zener Solid-Solid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_zener_ss.FractionalZenerSolidSolid` | Handbook: :doc:`/models/fractional/fractional_zener_ss`
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

.. autoclass:: rheojax.models.fractional_zener_ss.FractionalZenerSolidSolid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Zener Liquid-Liquid
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_zener_ll.FractionalZenerLiquidLiquid` | Handbook: :doc:`/models/fractional/fractional_zener_ll`
Most general fractional Zener with three fractional orders.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``c1``
     - Pa*s^alpha
     - [0.001, 1e+09]
     - First SpringPot constant
   * - ``c2``
     - Pa*s^gamma
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

.. autoclass:: rheojax.models.fractional_zener_ll.FractionalZenerLiquidLiquid
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Kelvin-Voigt Zener
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_kv_zener.FractionalKelvinVoigtZener` | Handbook: :doc:`/models/fractional/fractional_kv_zener`
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

.. autoclass:: rheojax.models.fractional_kv_zener.FractionalKelvinVoigtZener
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Burgers
~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_burgers.FractionalBurgersModel` | Handbook: :doc:`/models/fractional/fractional_burgers`
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
     - Pa*s
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

.. autoclass:: rheojax.models.fractional_burgers.FractionalBurgersModel
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Poynting-Thomson
~~~~~~~~~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_poynting_thomson.FractionalPoyntingThomson` | Handbook: :doc:`/models/fractional/fractional_poynting_thomson`
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

.. autoclass:: rheojax.models.fractional_poynting_thomson.FractionalPoyntingThomson
   :members:
   :undoc-members:
   :show-inheritance:

Fractional Jeffreys
~~~~~~~~~~~~~~~~~~~

:class:`rheojax.models.fractional_jeffreys.FractionalJeffreysModel` | Handbook: :doc:`/models/fractional/fractional_jeffreys`
Two dashpots plus SpringPot for thixotropic-like liquids.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta1``
     - Pa*s
     - [1e-06, 1e+12]
     - First viscosity
   * - ``eta2``
     - Pa*s
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

.. autoclass:: rheojax.models.fractional_jeffreys.FractionalJeffreysModel
   :members:
   :undoc-members:
   :show-inheritance:

Non-Newtonian Flow Models
-------------------------

Power Law
~~~~~~~~~

:class:`rheojax.models.power_law.PowerLaw` | Handbook: :doc:`/models/flow/power_law`
Two-parameter viscosity curve tau = K*gamma^n.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``K``
     - Pa*s^n
     - [1e-06, 1e+06]
     - Consistency index
   * - ``n``
     - dimensionless
     - [0.01, 2]
     - Flow behavior index

.. autoclass:: rheojax.models.power_law.PowerLaw
   :members:
   :undoc-members:
   :show-inheritance:

Carreau
~~~~~~~

:class:`rheojax.models.carreau.Carreau` | Handbook: :doc:`/models/flow/carreau`
Smooth transition from Newtonian plateau to shear thinning.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta0``
     - Pa*s
     - [0.001, 1e+12]
     - Zero-shear viscosity
   * - ``eta_inf``
     - Pa*s
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

.. autoclass:: rheojax.models.carreau.Carreau
   :members:
   :undoc-members:
   :show-inheritance:

Carreau-Yasuda
~~~~~~~~~~~~~~

:class:`rheojax.models.carreau_yasuda.CarreauYasuda` | Handbook: :doc:`/models/flow/carreau_yasuda`
Adds the Yasuda exponent to tune transition sharpness.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta0``
     - Pa*s
     - [0.001, 1e+12]
     - Zero-shear viscosity
   * - ``eta_inf``
     - Pa*s
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

.. autoclass:: rheojax.models.carreau_yasuda.CarreauYasuda
   :members:
   :undoc-members:
   :show-inheritance:

Cross
~~~~~

:class:`rheojax.models.cross.Cross` | Handbook: :doc:`/models/flow/cross`
Alternative viscosity curve with rate exponent m.

.. list-table:: Parameters
   :header-rows: 1
   :widths: 18 14 18 50

   * - Parameter
     - Units
     - Bounds
     - Description
   * - ``eta0``
     - Pa*s
     - [0.001, 1e+12]
     - Zero-shear viscosity
   * - ``eta_inf``
     - Pa*s
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

.. autoclass:: rheojax.models.cross.Cross
   :members:
   :undoc-members:
   :show-inheritance:

Herschel-Bulkley
~~~~~~~~~~~~~~~~

:class:`rheojax.models.herschel_bulkley.HerschelBulkley` | Handbook: :doc:`/models/flow/herschel_bulkley`
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
     - Pa*s^n
     - [1e-06, 1e+06]
     - Consistency index
   * - ``n``
     - dimensionless
     - [0.01, 2]
     - Flow behavior index

.. autoclass:: rheojax.models.herschel_bulkley.HerschelBulkley
   :members:
   :undoc-members:
   :show-inheritance:

Bingham
~~~~~~~

:class:`rheojax.models.bingham.Bingham` | Handbook: :doc:`/models/flow/bingham`
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
     - Pa*s
     - [1e-06, 1e+12]
     - Plastic viscosity

.. autoclass:: rheojax.models.bingham.Bingham
   :members:
   :undoc-members:
   :show-inheritance:
