Fluidity Model Identifiability
==============================

.. note::

   **Key result:** Different protocols constrain different subsets of the
   9 Fluidity parameters. Fitting all 9 parameters to a single-protocol
   dataset typically produces high :math:`R^2` with physically wrong
   values. Use
   :py:meth:`~rheojax.models.fluidity._base.FluidityBase.identifiability_check`
   to check before fitting.


Why Identifiability Matters
---------------------------

The FluidityLocal model has 9 parameters:
(``G``, ``tau_y``, ``K``, ``n_flow``, ``f_eq``, ``f_inf``, ``theta``,
``a``, ``n_rejuv``), but not every parameter appears in every protocol's
residual. Fitting an "inactive" parameter (zero Jacobian column) leaves
it to drift unconstrained. Fitting a "product-degenerate" set
(e.g. :math:`G` and :math:`f` during pure relaxation) produces a
one-parameter family of indistinguishable solutions — the optimizer
picks one based on initial seed and bounds.

A fit with :math:`R^2 > 0.999` but parameters 100× off from the known
physical values is diagnostic of this problem, not a bug.


Protocol-by-Protocol Identifiability
------------------------------------

For each protocol, the ``identifiable`` column lists parameters that
appear non-degenerately in the residual. The ``product-degenerate``
column lists parameters that only enter through a product with another
parameter (a one-parameter scale degeneracy). The ``inactive`` column
lists parameters absent from the residual entirely.

.. list-table::
   :widths: 15 28 25 32
   :header-rows: 1

   * - Protocol
     - Identifiable
     - Product-degenerate
     - Inactive
   * - ``flow_curve``
     - ``tau_y``, ``K``, ``n_flow``
     - —
     - ``G``, ``f_eq``, ``f_inf``, ``theta``, ``a``, ``n_rejuv``
   * - ``oscillation`` (SAOS)
     - ``G``, ``f_eq``
     - —
     - ``tau_y``, ``K``, ``n_flow``, ``f_inf``, ``theta``, ``a``, ``n_rejuv``
   * - ``relaxation``
     - ``theta``
     - ``G``, ``f_eq``, ``f_inf`` *(only* :math:`G \cdot f` *identifiable)*
     - ``tau_y``, ``K``, ``n_flow``, ``a``, ``n_rejuv``
   * - ``startup``
     - ``G``, ``f_eq``, ``f_inf``, ``theta``, ``a``, ``n_rejuv``
     - —
     - ``tau_y``, ``K``, ``n_flow``
   * - ``creep``
     - ``f_eq``, ``f_inf``, ``theta``, ``a``, ``n_rejuv``
     - —
     - ``G`` (quasi-inactive), ``tau_y``, ``K``, ``n_flow``
   * - ``laos``
     - ``G``, ``f_eq``, ``f_inf``, ``theta``, ``a``, ``n_rejuv``
     - —
     - ``tau_y``, ``K``, ``n_flow``


The Relaxation Scale Degeneracy
-------------------------------

At :math:`\dot{\gamma}=0` the stress ODE reduces to

.. math::

   \frac{d\sigma}{dt} = -G \cdot \sigma \cdot f(t).

Only the product :math:`G \cdot f` enters. The transformation

.. math::

   G \to \lambda G, \qquad f_{\rm eq} \to f_{\rm eq}/\lambda,
   \qquad f_\infty \to f_\infty/\lambda

leaves :math:`\sigma(t)` invariant for any positive :math:`\lambda`,
so G and f are not separately identifiable from relaxation alone.

Additionally the rejuvenation term :math:`a \cdot |\dot{\gamma}|^{n_{\rm rejuv}}
\cdot (f_\infty - f)` is gated by :math:`\dot{\gamma}^n`. At
:math:`\dot{\gamma}=0` the small-:math:`\dot{\gamma}` regulariser
(:math:`|\dot{\gamma}|+10^{-6})^n` contributes only :math:`\sim 10^{-6}`
relative to aging, so ``a`` and ``n_rejuv`` are effectively inactive.

And ``tau_y``, ``K``, ``n_flow`` live in the Herschel–Bulkley steady-state
flow curve only — they never appear in the transient ODE.

**Net: 3 of 9 parameters are identifiable from relaxation data alone**
(one identifiable parameter :math:`\theta` plus two identifiable products
:math:`G f_{\rm eq}` and :math:`G f_\infty`).


Programmatic Check
------------------

.. code-block:: python

   from rheojax.models.fluidity import FluidityLocal

   report = FluidityLocal.identifiability_check("relaxation")
   # WARNING: Fluidity identifiability for test_mode='relaxation':
   #   Identifiable (1): theta
   #   Product-degenerate (3): G, f_eq, f_inf
   #     -- only their product(s) with G are identifiable
   #   Inactive (5): tau_y, K, n_flow, a, n_rejuv
   #     -- absent from this protocol's residual

   print(report)
   # {'identifiable': ('theta',),
   #  'product_degenerate': ('G', 'f_eq', 'f_inf'),
   #  'inactive': ('tau_y', 'K', 'n_flow', 'a', 'n_rejuv')}


Practical Recipes
-----------------

Option 1: Single-protocol fit with frozen auxiliaries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Freeze the non-identifiable parameters to values from an independent
measurement, then fit only the identifiable ones:

.. code-block:: python

   model = FluidityLocal()
   # Seed from prior flow-curve / SAOS measurements
   model.parameters.set_value("tau_y", tau_y_from_flow_curve)
   model.parameters.set_value("K",     K_from_flow_curve)
   model.parameters.set_value("n_flow", n_from_flow_curve)
   model.parameters.set_value("f_inf",  f_inf_from_SAOS_plateau)
   model.parameters.set_value("a",      1.0)       # literature value
   model.parameters.set_value("n_rejuv", 1.0)      # literature value

   # Freeze via tight bounds (± 0.01%)
   for name in ("tau_y", "K", "n_flow", "f_inf", "a", "n_rejuv"):
       v = model.parameters.get_value(name)
       model.parameters[name].bounds = (v * 0.9999, v * 1.0001)

   # Fit the 3 identifiable parameters
   model.fit(t_data, stress, test_mode="relaxation", sigma_0=sigma_0)


Option 2: Joint multi-protocol fit
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For full 9-parameter recovery, fit multiple protocols simultaneously so
each parameter is constrained by at least one protocol's residual:

.. code-block:: python

   # 1. Flow curve → pins tau_y, K, n_flow
   m1 = FluidityLocal()
   m1.fit(gamma_dot, sigma_ss, test_mode="flow_curve")
   for p in ("tau_y", "K", "n_flow"):
       v = m1.parameters.get_value(p)
       m1.parameters[p].bounds = (v * 0.9999, v * 1.0001)

   # 2. Startup (gamma_dot != 0) → pins G, f_eq, f_inf, theta, a, n_rejuv
   m1.fit(t_startup, sigma_startup, test_mode="startup",
          gamma_dot=gamma_dot_applied)

   # 3. Validation: relaxation should now match without further fitting
   sigma_pred = m1.predict(t_relax, test_mode="relaxation", sigma_0=sigma_0)


Why :math:`R^2` alone is misleading
-----------------------------------

For every protocol where a scale degeneracy exists, there is a
one-parameter family of :math:`(G, f_{\rm eq}, f_\infty)` triples that
reproduce the data to within numerical precision. All of them give
:math:`R^2 \approx 1` — the residual surface is flat along the
degenerate direction. Only independent information (another protocol,
an informative prior, or a frozen auxiliary) can break the tie.

Always inspect:

1. The **identifiable products** (``G*f_eq``, ``G*f_inf``) against
   prior physics or literature values.
2. **Posterior correlations** in Bayesian fits — :math:`|\mathrm{corr}(G, f_{\rm eq})|
   > 0.95` is a red flag for unresolved degeneracy.
3. **Parameter recovery** on synthetic data before trusting the fit
   on real measurements.


FluidityNonlocal: a Different Identifiability Map
-------------------------------------------------

The **FluidityNonlocal** model does **not** inherit the Local
identifiability map. Its PDE kernels
(``fluidity_nonlocal_pde_rhs``, ``fluidity_nonlocal_creep_pde_rhs``)
use HB-aging only:

.. math::

   \frac{\partial f}{\partial t} = \frac{f_{\rm loc}(\sigma;\, \tau_y, K, n_{\rm flow}) - f}{\theta}
   + \xi^2 \nabla^2 f,

and do **not** include the rejuvenation term
:math:`a |\dot\gamma|^n (f_\infty - f)` that the Local ODE carries.
Consequently:

- ``a``, ``n_rejuv``, ``f_inf`` are **inert in every transient protocol**
  — they never enter the nonlocal residual.
- ``f_eq`` only sets the initial f-field (decays in ~:math:`\theta`), so
  it is weakly identifiable at best.
- ``xi`` is inert for uniform initial conditions with Neumann BCs
  (:math:`\nabla^2 f \equiv 0`) — it only becomes identifiable with
  non-uniform initial f-fields or curved geometries.
- The HB parameters (``tau_y``, ``K``, ``n_flow``) enter through
  :math:`f_{\rm loc}` and are active in every transient protocol.

.. list-table:: FluidityNonlocal protocol × parameter identifiability
   :widths: 15 35 50
   :header-rows: 1

   * - Protocol
     - Identifiable
     - Inactive
   * - ``flow_curve``
     - ``tau_y``, ``K``, ``n_flow``
     - ``G``, ``f_eq``, ``f_inf``, ``theta``, ``a``, ``n_rejuv``, ``xi``
   * - ``startup``
     - ``G``, ``tau_y``, ``K``, ``n_flow``, ``theta``
     - ``f_eq``, ``f_inf``, ``a``, ``n_rejuv``, ``xi``
   * - ``relaxation``
     - ``G``, ``tau_y``, ``K``, ``n_flow``, ``theta``
     - ``f_eq``, ``f_inf``, ``a``, ``n_rejuv``, ``xi``
   * - ``creep``
     - ``tau_y``, ``K``, ``n_flow``, ``theta``
     - ``G``, ``f_eq``, ``f_inf``, ``a``, ``n_rejuv``, ``xi``
   * - ``oscillation``
     - ``G``, ``f_eq``, ``theta``
     - ``tau_y``, ``K``, ``n_flow``, ``f_inf``, ``a``, ``n_rejuv``, ``xi``
   * - ``laos``
     - ``G``, ``tau_y``, ``K``, ``n_flow``, ``theta``
     - ``f_eq``, ``f_inf``, ``a``, ``n_rejuv``, ``xi``

Note the two differences vs FluidityLocal:

1. **No product-degenerate column** — HB aging couples :math:`f_{\rm loc}`
   directly to :math:`\sigma`, breaking the :math:`G \cdot f` scale
   degeneracy that Local shows in relaxation.
2. **Creep constrains only 4 parameters** in nonlocal vs 5 in local
   (the Local rejuvenation path identifies ``f_eq``, ``f_inf``, ``a``,
   ``n_rejuv``; the Nonlocal path identifies the HB triple + ``theta``).

Worked example: ``examples/fluidity/09_fluidity_nonlocal_creep.ipynb``
demonstrates the recipe — pin inert params to plausible-prior values,
warm-start the 4 identifiable params 10–30% off truth, and watch the
post-fit stuck-parameter warning confirm the pin.


NLSQ Stuck-Parameter Detection
------------------------------

After every ``nlsq_optimize`` run, RheoJAX checks whether any fitted
parameter moved less than **1% of its bound span AND 1% of its initial
value**. When parameters fail to move they are usually:

1. **Pinned intentionally** (tight bounds around a plausible-prior value)
   — informational;
2. **Non-identifiable for this protocol** — the kind of silent failure
   this page warns against;
3. **Already converged at the start** — rare but possible with excellent
   warm-starts.

The warning surfaces all three so you can distinguish them by context.
Example output when fitting nonlocal creep without pinning inert params:

.. code-block:: text

   WARNING rheojax.utils.optimization: NLSQ identifiability check:
     5 parameter(s) did not move (< 1.0% of bound span and < 1.0% of
     initial value): G, f_inf, a, n_rejuv, xi. This usually means these
     parameters are pinned, non-identifiable for the current protocol,
     or the objective is flat in their direction. Consult the model's
     identifiability_check() if unexpected.


See Also
--------

- ``examples/fluidity/04_fluidity_local_relaxation.ipynb`` — worked example
  of the Local relaxation identifiability fix.
- ``examples/fluidity/09_fluidity_nonlocal_creep.ipynb`` — worked example
  of the Nonlocal creep identifiability recipe with plausible-prior pins.
- :py:meth:`~rheojax.models.fluidity._base.FluidityBase.identifiability_check`
  — programmatic API. FluidityNonlocal overrides the map at
  :py:class:`~rheojax.models.fluidity.nonlocal_model.FluidityNonlocal._IDENTIFIABILITY`.
