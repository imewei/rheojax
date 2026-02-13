.. _transform-srfs:

Strain-Rate Frequency Superposition (SRFS)
==========================================

Overview
--------

The Strain-Rate Frequency Superposition (SRFS) transform is a data analysis technique for
soft glassy materials that extends time-temperature superposition (TTS) principles to
strain-rate-dependent rheology. It enables construction of master curves from nonlinear
rheological data measured at different strain rates.

SRFS is particularly useful for:

- **Soft glassy materials**: Concentrated emulsions, pastes, colloidal gels
- **Yield stress fluids**: Materials with strain-rate-dependent microstructure
- **Thixotropic systems**: Materials where structure evolves with deformation history

The transform is closely related to the SGR model's prediction that for power-law materials,
:math:`G'(\omega, \dot{\gamma})` and :math:`G''(\omega, \dot{\gamma})` can be collapsed
onto master curves by appropriate rescaling.

Theory
------

For soft glassy materials described by the SGR model, linear viscoelastic moduli at different
strain amplitudes (or strain rates in flow) can be superposed via:

.. math::

   G'(\omega, \dot{\gamma}) = b(\dot{\gamma}) \, G'_{\text{master}}(a(\dot{\gamma}) \cdot \omega)

   G''(\omega, \dot{\gamma}) = b(\dot{\gamma}) \, G''_{\text{master}}(a(\dot{\gamma}) \cdot \omega)

where:
   - :math:`a(\dot{\gamma})` is the horizontal shift factor (time/frequency rescaling)
   - :math:`b(\dot{\gamma})` is the vertical shift factor (modulus rescaling)

For ideal SGR materials:
   - :math:`a(\dot{\gamma}) \sim \dot{\gamma}^{-1/(x-1)}`
   - :math:`b(\dot{\gamma}) \sim \dot{\gamma}^{-1}`

The shift factors reveal information about the underlying microstructural dynamics:

- **Horizontal shift**: Related to acceleration of relaxation by flow
- **Vertical shift**: Related to flow-induced softening of the elastic network

Extended Features
-----------------

Shear Banding Detection
~~~~~~~~~~~~~~~~~~~~~~~

The SRFS transform includes tools to detect and characterize shear banding—the
spatial coexistence of regions with different local shear rates:

.. code-block:: python

   from rheojax.transforms import SRFS
   from rheojax.models import SGRConventional

   srfs = SRFS()
   model = SGRConventional(x=0.8, G0=100.0, tau0=0.01)

   # Detect shear banding from flow curve
   gamma_dot_range = (1e-3, 1e2)
   is_banding, critical_rates = srfs.detect_shear_banding(
       model, gamma_dot_range
   )

   if is_banding:
       print(f"Shear banding detected between {critical_rates}")

       # Compute band coexistence via lever rule
       low_band, high_band, fraction = srfs.compute_shear_band_coexistence(
           model,
           applied_rate=1.0  # Global shear rate
       )
       print(f"Low band: {low_band:.3f} 1/s ({fraction:.1%})")
       print(f"High band: {high_band:.3f} 1/s ({1-fraction:.1%})")

**Physical interpretation**:
   Shear banding occurs when the constitutive curve :math:`\sigma(\dot{\gamma})` is
   non-monotonic. The material splits into coexisting bands at different local shear
   rates, connected by a stress plateau (the "banding stress").

Thixotropy Kinetics
~~~~~~~~~~~~~~~~~~~

For thixotropic soft glasses, the SRFS transform includes structural kinetics:

.. code-block:: python

   from rheojax.transforms import SRFS
   import numpy as np

   srfs = SRFS()

   # Define structural evolution parameters
   thixo_params = {
       'tau_buildup': 100.0,    # Structure recovery time (s)
       'k_destruction': 0.1,    # Destruction rate coefficient
       'alpha': 1.0,            # Strain rate exponent
       'beta': 1.0              # Structure exponent
   }

   # Compute structure parameter evolution under step shear
   t = np.linspace(0, 100, 500)
   gamma_dot = np.where(t < 50, 10.0, 0.0)  # Shear then rest

   lambda_t = srfs.evolve_thixotropy_lambda(
       t, gamma_dot,
       lambda_0=1.0,  # Initial fully structured
       **thixo_params
   )

   # lambda_t shows breakdown during shear, recovery during rest

The structural parameter :math:`\lambda \in [0, 1]` evolves according to:

.. math::

   \frac{d\lambda}{dt} = \frac{1 - \lambda}{\tau_b} - k_d |\dot{\gamma}|^\alpha \lambda^\beta

where the first term describes recovery (buildup) and the second describes shear-induced
breakdown (destruction).

Parameters
----------

.. list-table:: SRFS Transform Parameters
   :header-rows: 1
   :widths: 20 15 65

   * - Parameter
     - Type
     - Description
   * - ``reference_rate``
     - float
     - Reference strain rate for master curve construction
   * - ``method``
     - str
     - Shift factor estimation method ('auto', 'power_law', 'empirical')
   * - ``vertical_shift``
     - bool
     - Whether to apply vertical (modulus) shifting

Usage
-----

Master Curve Construction
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SRFS
   import numpy as np

   # Multi-rate oscillatory data
   rates = [0.01, 0.1, 1.0, 10.0]  # strain rates
   omega = np.logspace(-2, 2, 50)
   G_star_data = {rate: measure_modulus(omega, rate) for rate in rates}

   # Create master curve
   srfs = SRFS(reference_rate=1.0)
   master_curve, shift_factors = srfs.transform(omega, G_star_data)

   # shift_factors contains a_gamma_dot and b_gamma_dot for each rate
   print("Horizontal shifts:", shift_factors['a'])
   print("Vertical shifts:", shift_factors['b'])

Extracting x from Shift Factors
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   from scipy.stats import linregress

   # If a(gamma_dot) ~ gamma_dot^(-1/(x-1)), then
   # log(a) = -1/(x-1) * log(gamma_dot)

   log_rates = np.log10(rates)
   log_a = np.log10(shift_factors['a'])

   slope, intercept, r_value, _, _ = linregress(log_rates, log_a)

   x_estimated = 1 - 1/slope
   print(f"Estimated x = {x_estimated:.3f} (R² = {r_value**2:.4f})")

Integration with SGR Model
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.transforms import SRFS
   from rheojax.models import SGRConventional

   # Fit SGR model to master curve
   srfs = SRFS(reference_rate=1.0)
   master_omega, master_G = srfs.transform(omega, G_star_data)

   model = SGRConventional()
   model.fit(master_omega, master_G, test_mode='oscillation')

   # Use x from SGR to validate shift factor scaling
   x_fitted = model.parameters.get_value('x')
   expected_slope = -1 / (x_fitted - 1)

   print(f"Expected shift slope: {expected_slope:.3f}")
   print(f"Measured shift slope: {slope:.3f}")

See also
--------

- :doc:`../models/sgr/sgr_conventional` — SGR model underlying SRFS theory
- :doc:`../models/sgr/sgr_generic` — Thermodynamically consistent SGR
- :doc:`mastercurve` — Time-temperature superposition (analogous technique)
- ``examples/transforms/07-srfs-strain-rate-superposition.ipynb`` — Tutorial notebook for SRFS
  master curve construction and SGR integration

API References
--------------

- Module: :mod:`rheojax.transforms`
- Class: :class:`rheojax.transforms.SRFS`

References
----------

1. Wyss, H. M., et al. "Strain-Rate Frequency Superposition: A Rheological Probe of
   Structural Relaxation in Soft Materials." *Physical Review Letters*, **98**, 238303 (2007).

2. Sollich, P. "Rheological constitutive equation for a model of soft glassy materials."
   *Physical Review E*, **58**, 738 (1998).

3. Fielding, S. M., et al. "Aging and rheology in soft materials."
   *Journal of Rheology*, **44**, 323 (2000).
