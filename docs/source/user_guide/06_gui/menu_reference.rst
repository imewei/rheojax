.. _gui-menu-reference:

==============
Menu Reference
==============

Complete inventory of the RheoJAX menu bar, listing every ``QAction`` with its
label, keyboard shortcut, status-bar tooltip, handler method, and notes on
special behaviour (disabled, checkable, etc.).

The menu bar is implemented by :class:`rheojax.gui.app.menu_bar.MenuBar` and
wired to handlers in :class:`rheojax.gui.app.main_window.RheoJAXMainWindow`.

.. contents:: Menu Groups
   :local:
   :depth: 2


File Menu
=========

.. list-table::
   :widths: 18 22 14 26 20
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``new_file_action``
     - &New
     - Ctrl+N
     - Create a new project
     - ``_on_new_file``
   * - ``open_file_action``
     - &Open...
     - Ctrl+O
     - Open an existing project
     - ``_on_open_file``
   * - ``save_file_action``
     - &Save
     - Ctrl+S
     - Save the current project
     - ``_on_save_file``
   * - ``save_as_action``
     - Save &As...
     - Ctrl+Shift+S
     - Save the current project with a new name
     - ``_on_save_as``
   * - ``import_action``
     - &Import Data...
     - Ctrl+I
     - Import rheological data
     - ``_on_import``
   * - ``export_action``
     - &Export Results...
     - Ctrl+E
     - Export analysis results
     - ``_on_export``
   * - ``exit_action``
     - E&xit
     - Ctrl+Q
     - Exit the application
     - ``close``

The File menu also contains a **Recent Files** submenu (initially empty).


Edit Menu
=========

.. list-table::
   :widths: 18 22 14 26 20
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``undo_action``
     - &Undo
     - Ctrl+Z
     - Undo last action
     - ``_on_undo``
   * - ``redo_action``
     - &Redo
     - Ctrl+Shift+Z
     - Redo last undone action
     - ``_on_redo``
   * - ``cut_action``
     - Cu&t
     - Ctrl+X
     - Cut selection to clipboard
     - ``_on_cut``
   * - ``copy_action``
     - &Copy
     - Ctrl+C
     - Copy selection to clipboard
     - ``_on_copy``
   * - ``paste_action``
     - &Paste
     - Ctrl+V
     - Paste from clipboard
     - ``_on_paste``
   * - ``preferences_action``
     - &Preferences...
     - Ctrl+,
     - Open preferences dialog
     - ``_on_preferences``

.. note::

   ``undo_action`` and ``redo_action`` are created **disabled** and are
   enabled dynamically when undo/redo history is available.


View Menu
=========

.. list-table::
   :widths: 18 22 14 26 20
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``zoom_in_action``
     - Zoom &In
     - Ctrl+=
     - Zoom in on plot
     - ``_on_zoom_in``
   * - ``zoom_out_action``
     - Zoom &Out
     - Ctrl+-
     - Zoom out on plot
     - ``_on_zoom_out``
   * - ``reset_zoom_action``
     - &Reset Zoom
     - Ctrl+0
     - Reset plot zoom to default
     - ``_on_reset_zoom``
   * - ``view_data_dock_action``
     - &Data Panel
     - —
     - Toggle data panel visibility
     - *toggle sidebar visibility*
   * - ``view_log_dock_action``
     - &Log Panel
     - —
     - Toggle log panel visibility
     - *toggle log dock visibility*

**Theme submenu** (View > Theme):

.. list-table::
   :widths: 18 22 14 26 20
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``theme_light_action``
     - &Light
     - —
     - —
     - ``_on_theme_changed("light")``
   * - ``theme_dark_action``
     - &Dark
     - —
     - —
     - ``_on_theme_changed("dark")``
   * - ``theme_auto_action``
     - &Auto (System)
     - —
     - —
     - ``_on_theme_changed("auto")``

.. note::

   ``view_data_dock_action``, ``view_log_dock_action``, and all three theme
   actions are **checkable** toggles.  ``theme_auto_action`` is checked by
   default.


Data Menu
=========

.. list-table::
   :widths: 22 22 16 22 18
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``new_dataset_action``
     - &New Dataset...
     - Ctrl+Shift+N
     - Create a new dataset
     - ``_on_new_dataset``
   * - ``delete_dataset_action``
     - &Delete Dataset
     - Delete
     - Delete selected dataset
     - ``_on_delete_dataset``

**Set Test Mode submenu** (Data > Set Test Mode):

.. list-table::
   :widths: 22 22 16 22 18
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``test_mode_oscillation``
     - &Oscillation
     - —
     - —
     - ``_on_set_test_mode("oscillation")``
   * - ``test_mode_relaxation``
     - &Relaxation
     - —
     - —
     - ``_on_set_test_mode("relaxation")``
   * - ``test_mode_creep``
     - &Creep
     - —
     - —
     - ``_on_set_test_mode("creep")``
   * - ``test_mode_rotation``
     - R&otation
     - —
     - —
     - ``_on_set_test_mode("rotation")``
   * - ``test_mode_flow_curve``
     - &Flow Curve
     - —
     - —
     - ``_on_set_test_mode("flow_curve")``
   * - ``test_mode_startup``
     - &Startup
     - —
     - —
     - ``_on_set_test_mode("startup")``
   * - ``test_mode_laos``
     - &LAOS
     - —
     - —
     - ``_on_set_test_mode("laos")``

.. list-table::
   :widths: 22 22 16 22 18
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``auto_detect_mode_action``
     - &Auto-detect Test Mode
     - Ctrl+Shift+D
     - Automatically detect test mode from data
     - ``_on_auto_detect_mode``


.. _gui-menu-models:

Models Menu
===========

All model actions dispatch ``SET_ACTIVE_MODEL`` to the state store and navigate
to the Fit page.  Handler: ``_on_select_model(model_id)``.

Classical
---------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_maxwell``
     - Maxwell
     - ``maxwell``
   * - ``model_zener``
     - Zener (SLS)
     - ``zener``
   * - ``model_springpot``
     - SpringPot
     - ``springpot``

Flow (Non-Newtonian)
--------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_power_law``
     - Power Law
     - ``power_law``
   * - ``model_carreau``
     - Carreau
     - ``carreau``
   * - ``model_carreau_yasuda``
     - Carreau-Yasuda
     - ``carreau_yasuda``
   * - ``model_cross``
     - Cross
     - ``cross``
   * - ``model_herschel_bulkley``
     - Herschel-Bulkley
     - ``herschel_bulkley``
   * - ``model_bingham``
     - Bingham
     - ``bingham``

Fractional > Maxwell Family
---------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_fmaxwell_gel``
     - Maxwell Gel
     - ``fractional_maxwell_gel``
   * - ``model_fmaxwell_liquid``
     - Maxwell Liquid
     - ``fractional_maxwell_liquid``
   * - ``model_fmaxwell_model``
     - Maxwell Model
     - ``fractional_maxwell_model``
   * - ``model_fkelvin_voigt``
     - Kelvin-Voigt
     - ``fractional_kelvin_voigt``

Fractional > Zener Family
-------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_fzener_sl``
     - Solid-Liquid (FZSL)
     - ``fractional_zener_sl``
   * - ``model_fzener_ss``
     - Solid-Solid (FZSS)
     - ``fractional_zener_ss``
   * - ``model_fzener_ll``
     - Liquid-Liquid (FZLL)
     - ``fractional_zener_ll``
   * - ``model_fkv_zener``
     - KV-Zener (FKVZ)
     - ``fractional_kv_zener``

Fractional > Advanced
---------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_fburgers``
     - Burgers (FBM)
     - ``fractional_burgers``
   * - ``model_fpoynting``
     - Poynting-Thomson (FPT)
     - ``fractional_poynting_thomson``
   * - ``model_fjeffreys``
     - Jeffreys (FJM)
     - ``fractional_jeffreys``

Multi-Mode
----------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_gmaxwell``
     - Generalized Maxwell
     - ``generalized_maxwell``

Soft Glassy Rheology (SGR)
--------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_sgr_conventional``
     - SGR Conventional
     - ``sgr_conventional``
   * - ``model_sgr_generic``
     - SGR GENERIC
     - ``sgr_generic``

SPP (LAOS)
----------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_spp_yield_stress``
     - SPP Yield Stress
     - ``spp_yield_stress``

STZ (Shear Transformation Zone)
-------------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_stz_conventional``
     - STZ Conventional
     - ``stz_conventional``

EPM (Elasto-Plastic)
--------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_lattice_epm``
     - Lattice EPM
     - ``lattice_epm``
   * - ``model_tensorial_epm``
     - Tensorial EPM
     - ``tensorial_epm``

Fluidity
--------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_fluidity_local``
     - Fluidity Local
     - ``fluidity_local``
   * - ``model_fluidity_nonlocal``
     - Fluidity Nonlocal
     - ``fluidity_nonlocal``

Saramito (EVP)
--------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_saramito_local``
     - Saramito Local
     - ``fluidity_saramito_local``
   * - ``model_saramito_nonlocal``
     - Saramito Nonlocal
     - ``fluidity_saramito_nonlocal``

IKH (Isotropic Kinematic Hardening)
------------------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_mikh``
     - MIKH
     - ``mikh``
   * - ``model_mlikh``
     - MLIKH
     - ``ml_ikh``

FIKH (Fractional IKH)
---------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_fikh``
     - FIKH
     - ``fikh``
   * - ``model_fmlikh``
     - FMLIKH
     - ``fmlikh``

Hébraud-Lequeux
---------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_hebraud_lequeux``
     - Hébraud-Lequeux
     - ``hebraud_lequeux``

ITT-MCT
-------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_itt_mct_schematic``
     - Schematic (F₁₂)
     - ``itt_mct_schematic``
   * - ``model_itt_mct_isotropic``
     - Isotropic (ISM)
     - ``itt_mct_isotropic``

DMT (Thixotropic)
-----------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_dmt_local``
     - DMT Local
     - ``dmt_local``
   * - ``model_dmt_nonlocal``
     - DMT Nonlocal
     - ``dmt_nonlocal``

Giesekus
--------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_giesekus_single``
     - Single Mode
     - ``giesekus_single``
   * - ``model_giesekus_multi``
     - Multi Mode
     - ``giesekus_multi``

TNT (Transient Network)
-----------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_tnt_single_mode``
     - Single Mode
     - ``tnt_single_mode``
   * - ``model_tnt_cates``
     - Cates
     - ``tnt_cates``
   * - ``model_tnt_loop_bridge``
     - Loop-Bridge
     - ``tnt_loop_bridge``
   * - ``model_tnt_multi_species``
     - Multi-Species
     - ``tnt_multi_species``
   * - ``model_tnt_sticky_rouse``
     - Sticky Rouse
     - ``tnt_sticky_rouse``

VLB (Viscoelastic Liquid-Bridge)
---------------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_vlb_local``
     - Local
     - ``vlb_local``
   * - ``model_vlb_multi_network``
     - Multi-Network
     - ``vlb_multi_network``
   * - ``model_vlb_variant``
     - Variant (Bell/FENE)
     - ``vlb_variant``
   * - ``model_vlb_nonlocal``
     - Nonlocal
     - ``vlb_nonlocal``

HVM (Hybrid Vitrimer)
---------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_hvm_local``
     - HVM Local
     - ``hvm_local``

HVNM (Vitrimer Nanocomposite)
------------------------------

.. list-table::
   :widths: 26 24 50
   :header-rows: 1

   * - Attribute
     - Label
     - Model ID
   * - ``model_hvnm_local``
     - HVNM Local
     - ``hvnm_local``


.. _gui-menu-transforms:

Transforms Menu
===============

All transform actions dispatch ``APPLY_TRANSFORM`` and run the transform via
:class:`~rheojax.gui.services.transform_service.TransformService`, either
asynchronously through ``WorkerPool`` or synchronously as a fallback.
Handler: ``_on_apply_transform(transform_id)``.

.. list-table::
   :widths: 22 24 30 24
   :header-rows: 1

   * - Attribute
     - Label
     - Status Tip
     - Transform ID
   * - ``transform_fft``
     - &FFT (Fourier Transform)
     - Apply Fast Fourier Transform
     - ``fft``
   * - ``transform_mastercurve``
     - &Mastercurve (TTS)
     - Time-Temperature Superposition
     - ``mastercurve``
   * - ``transform_srfs``
     - &SRFS (Strain-Rate Frequency Superposition)
     - Strain-Rate Frequency Superposition
     - ``srfs``
   * - ``transform_mutation``
     - Mutation &Number
     - Calculate mutation number
     - ``mutation_number``
   * - ``transform_owchirp``
     - &OWChirp
     - Optimally Windowed Chirp transform
     - ``owchirp``
   * - ``transform_derivatives``
     - &Derivatives
     - Calculate numerical derivatives
     - ``derivative``
   * - ``transform_spp``
     - S&PP (LAOS Analysis)
     - Sequence of Physical Processes yield stress extraction
     - ``spp``
   * - ``transform_cox_merz``
     - Cox-&Merz Rule
     - Validate Cox-Merz rule (|η*| vs η)
     - ``cox_merz``
   * - ``transform_lve_envelope``
     - LVE &Envelope
     - Linear viscoelastic startup stress envelope
     - ``lve_envelope``
   * - ``transform_prony``
     - &Prony Conversion
     - Time ↔ frequency domain via Prony series decomposition
     - ``prony_conversion``
   * - ``transform_spectrum``
     - &Spectrum Inversion
     - Recover relaxation spectrum H(τ) from G(t) or G*(ω)
     - ``spectrum_inversion``

.. note::

   ``mastercurve``, ``srfs``, and ``cox_merz`` are **multi-dataset
   transforms** that require ≥ 2 loaded datasets.  If fewer are available the
   handler shows a status-bar warning and returns without executing.


Analysis Menu
=============

.. list-table::
   :widths: 22 22 14 22 20
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``analysis_fit_nlsq``
     - &Fit (NLSQ)
     - Ctrl+F
     - Fit model using non-linear least squares
     - ``_on_fit``
   * - ``analysis_fit_bayesian``
     - Fit &Bayesian (NUTS)
     - Ctrl+B
     - Fit model using Bayesian inference with NUTS
     - ``_on_bayesian``
   * - ``analysis_batch_fit``
     - &Batch Fit...
     - —
     - Fit multiple datasets in parallel
     - ``_on_batch_fit``
   * - ``analysis_compare``
     - &Compare Models...
     - —
     - Compare multiple model fits
     - ``_on_compare_models``
   * - ``analysis_compatibility``
     - Check &Compatibility
     - —
     - Check model-data compatibility
     - ``_on_check_compatibility``


Pipeline Menu
=============

.. list-table::
   :widths: 22 22 14 24 18
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``pipeline_new_action``
     - &New Pipeline
     - Ctrl+Alt+N
     - Create a new empty pipeline
     - ``_on_new_pipeline``
   * - ``pipeline_open_action``
     - &Open Pipeline...
     - —
     - Open a pipeline from a YAML file
     - ``_on_open_pipeline``
   * - ``pipeline_save_action``
     - &Save Pipeline...
     - Ctrl+Shift+S
     - Save current pipeline to a YAML file
     - ``_on_save_pipeline``

**From Template submenu** (Pipeline > From Template):

The template actions are stored in ``self.pipeline_template_actions``
(``dict[str, QAction]``) and each maps to
``_on_load_pipeline_template(key)``.

.. list-table::
   :widths: 30 24 46
   :header-rows: 1

   * - Dict Key
     - Label
     - Description
   * - ``nlsq_fitting``
     - NLSQ Fitting
     - Load → Fit (NLSQ) → Export workflow
   * - ``bayesian_inference``
     - Bayesian Inference
     - Load → Fit (NLSQ) → Bayesian (NUTS) → Export workflow
   * - ``transform_fit``
     - Transform + Fit
     - Load → Transform → Fit → Export workflow

.. note::

   ``pipeline_save_action`` shares ``Ctrl+Shift+S`` with ``save_as_action``
   (File menu).  Qt resolves the ambiguity based on which menu is active.
   The ``Ctrl+Alt+N`` shortcut for New Pipeline was chosen to avoid colliding
   with ``Ctrl+Shift+N`` (New Dataset in Data menu).


Tools Menu
==========

.. list-table::
   :widths: 22 22 14 24 18
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``tools_console``
     - &Python Console
     - Ctrl+Shift+P
     - Python console (coming soon)
     - ``_on_python_console``
   * - ``tools_jax_profiler``
     - &JAX Profiler
     - —
     - JAX profiler (coming soon)
     - ``_on_jax_profiler``
   * - ``tools_memory_monitor``
     - &Memory Monitor
     - —
     - Memory monitor (coming soon)
     - ``_on_memory_monitor``
   * - ``tools_preferences``
     - &Preferences...
     - Ctrl+,
     - Open preferences dialog
     - ``_on_preferences``

.. note::

   ``tools_console``, ``tools_jax_profiler``, and ``tools_memory_monitor``
   are created **disabled** — these features are planned but not yet
   implemented.


Help Menu
=========

.. list-table::
   :widths: 18 22 14 24 22
   :header-rows: 1

   * - Attribute
     - Label
     - Shortcut
     - Status Tip
     - Handler
   * - ``help_docs``
     - &Documentation
     - F1
     - Open online documentation
     - ``_on_open_docs``
   * - ``help_tutorials``
     - &Tutorials
     - —
     - View tutorials
     - ``_on_open_tutorials``
   * - ``help_shortcuts``
     - &Keyboard Shortcuts
     - —
     - View keyboard shortcuts
     - ``_on_show_shortcuts``
   * - ``help_about``
     - &About RheoJAX
     - —
     - About RheoJAX
     - ``_on_about``


Summary
=======

.. list-table::
   :widths: 40 20 40
   :header-rows: 1

   * - Menu
     - Actions
     - Notes
   * - File
     - 7
     - + Recent Files submenu
   * - Edit
     - 6
     - Undo/Redo start disabled
   * - View
     - 8
     - 5 checkable toggles
   * - Data
     - 10
     - 7 test-mode submenu items
   * - Models
     - 53
     - 20 family submenus
   * - Transforms
     - 11
     - 3 multi-dataset transforms
   * - Analysis
     - 5
     -
   * - Pipeline
     - 6
     - 3 template actions in dict
   * - Tools
     - 4
     - 3 disabled (future)
   * - Help
     - 4
     -
   * - **Total**
     - **114**
     -
