.. _io-architecture:

=======================================
RheoJAX Data Loader Architecture
=======================================

.. contents:: Table of Contents
   :depth: 3
   :local:


Overview
========

The RheoJAX I/O subsystem (``rheojax.io``) provides a unified, instrument-agnostic
data loading and export pipeline for rheological data. It is designed around four
architectural principles:

1. **Format-agnostic API** — A single ``auto_load()`` entry point that cascades
   through instrument-specific readers based on extension heuristics and content
   inspection.
2. **Canonical column mapping** — A centralized registry (``CanonicalField``) that
   normalizes vendor-specific column names (e.g., TRIOS "Angular Frequency", Anton
   Paar "ω") into canonical RheoJAX names.
3. **Protocol-aware validation** — Post-load quality checks (``validate_protocol``)
   that are test-mode–specific (relaxation monotonicity, oscillation frequency range,
   creep stress metadata, etc.).
4. **Lossless round-trip** — Writers (HDF5, NPZ, Excel) that preserve all metadata,
   units, complex modulus dtype, and test_mode for full reconstruction.

The subsystem spans **~10,000 lines** across 25 Python files and handles data from
TA Instruments (TRIOS), Anton Paar (RheoCompass), generic CSV/Excel, and HDF5/NPZ
archives.


Module Map
==========

.. code-block:: text

   rheojax/io/
   ├── __init__.py              (72L)  — Public API re-exports
   ├── _exceptions.py           (15L)  — RheoJaxFormatError, RheoJaxValidationWarning
   ├── spp_export.py           (983L)  — SPP analysis export (TXT/HDF5/CSV/MATLAB)
   │
   ├── readers/
   │   ├── __init__.py          (72L)  — Reader re-exports
   │   ├── _column_mapping.py  (343L)  — CanonicalField registry + match_column()
   │   ├── _utils.py           (719L)  — Shared: unit extraction, domain/mode detection,
   │   │                                  transform validation, complex modulus, unit conversion
   │   ├── _validation.py      (330L)  — Protocol-aware quality checks (LoaderReport)
   │   ├── auto.py             (802L)  — auto_load() dispatcher + format cascade
   │   ├── csv_reader.py       (613L)  — Generic CSV/TSV loader with EU decimal support
   │   ├── excel_reader.py     (387L)  — Generic Excel loader (.xlsx/.xls)
   │   ├── anton_paar.py      (1719L)  — RheoCompass CSV parser (interval blocks)
   │   ├── multi_file.py       (271L)  — load_tts(), load_srfs(), load_series()
   │   │
   │   └── trios/                      — TA Instruments TRIOS multi-format package
   │       ├── __init__.py     (269L)  — load_trios() unified dispatcher
   │       ├── common.py       (832L)  — Shared: TRIOSFile/Table/Segment, column mappings,
   │       │                             unit conversions, test type detection, segment→RheoData
   │       ├── txt.py         (1741L)  — TRIOS TXT (LIMS format) parser + chunked reader
   │       ├── csv.py         (1053L)  — TRIOS CSV export parser
   │       ├── excel.py        (869L)  — TRIOS Excel export parser
   │       ├── json.py         (491L)  — TRIOS JSON export parser (schema-validated)
   │       └── schema/                 — JSON schema validation
   │           ├── __init__.py  (22L)  — TRIOSExperiment dataclass
   │           ├── experiment.py(364L) — Experiment/Result/Step schema
   │           └── dataset.py  (151L)  — Dataset schema + validation
   │
   └── writers/
       ├── __init__.py          (19L)  — Writer re-exports
       ├── hdf5_writer.py      (452L)  — HDF5 save/load (atomic write, gzip compression)
       ├── npz_writer.py       (161L)  — NumPy NPZ save/load (no unsafe deserialization)
       └── excel_writer.py     (375L)  — Excel report writer (parameters, fit quality, plots)


Data Flow
=========

The canonical data flow from file to model fitting:

.. code-block:: text

   ┌──────────────┐     ┌───────────────┐     ┌──────────────┐     ┌──────────────┐
   │   Raw File   │────▶│   auto_load() │────▶│   RheoData   │────▶│ model.fit()  │
   │  .csv/.xlsx/ │     │  Format       │     │  JAX-native  │     │  NLSQ/NUTS   │
   │  .txt/.json  │     │  Detection    │     │  arrays      │     │              │
   └──────────────┘     └───────┬───────┘     └──────────────┘     └──────────────┘
                                │
                    ┌───────────┼───────────┐
                    ▼           ▼           ▼
              ┌──────────┐ ┌────────┐ ┌──────────┐
              │  TRIOS   │ │ Anton  │ │ Generic  │
              │ TXT/CSV/ │ │ Paar   │ │ CSV/     │
              │ Excel/   │ │ Rheo-  │ │ Excel    │
              │ JSON     │ │Compass │ │          │
              └──────────┘ └────────┘ └──────────┘

**Pipeline integration:**

.. code-block:: python

   # Fluent API
   Pipeline().load("data.csv", format="csv").fit(Maxwell()).plot().save("results.h5")

   # Direct loading
   data = auto_load("frequency_sweep.csv")  # → RheoData
   model.fit(data.x, data.y, test_mode=data.test_mode)

   # Multi-file TTS workflow
   datasets = load_tts(["T1.csv", "T2.csv", "T3.csv"], T_ref=298.15,
                        temperatures=[25, 50, 75], temperature_unit="C")

   # NPZ format via Pipeline
   Pipeline().load("archive.npz")  # Dispatches to load_npz()


Format Detection Cascade (``auto_load``)
=========================================

``auto_load()`` is the primary entry point. It accepts an optional ``format=`` hint
to skip auto-detection, or cascades through readers based on file extension:

.. list-table:: Extension-Based Dispatch
   :header-rows: 1
   :widths: 15 55 30

   * - Extension
     - Reader Cascade
     - Metadata Guard
   * - ``.txt``
     - TRIOS → Anton Paar → CSV
     - TRIOS metadata check
   * - ``.csv``
     - TRIOS → CSV
     - ``_has_trios_metadata()``
   * - ``.xlsx/.xls``
     - TRIOS → Excel
     - ``_has_trios_metadata()``
   * - ``.json``
     - TRIOS JSON only
     - —
   * - ``.tsv``
     - CSV (delimiter=``\t``)
     - —
   * - Other
     - TRIOS → Anton Paar → CSV → Excel
     - —

**Key design decisions:**

- **Fatal exceptions bypass cascade**: ``KeyboardInterrupt``, ``SystemExit``,
  ``MemoryError``, ``PermissionError``, ``OSError`` are never caught during reader
  attempts.
- **Warning suppression during speculation**: Captured warnings are only re-emitted
  if the speculative reader succeeds; failed readers' warnings are discarded.
- **TRIOS metadata guard**: ``_has_trios_metadata()`` checks for TRIOS-specific keys
  (``filename``, ``instrument_serial_number``, ``geometry``, etc.) to prevent
  misclassifying generic files as TRIOS.
- **Format provenance injection**: Every loaded ``RheoData`` receives
  ``metadata["format_detected"]`` and ``metadata["readers_attempted"]``.
- **GUI bridge**: ``_translate_y2_col()`` converts GUI-layer ``y2_col`` to
  reader-layer ``y_cols=[storage, loss]`` before dispatch.
- **Modulus pair detection**: ``_detect_modulus_pair()`` scans for E'/E'' or G'/G''
  column patterns using regex with loss-first priority to handle E'' matching E'.
- **Large file warning**: Files > 100 MB emit a ``ResourceWarning``.

**Direct format dispatch (``format=`` parameter):**

.. code-block:: python

   auto_load("data.csv", format="trios")       # Skips cascade, calls load_trios()
   auto_load("data.csv", format="anton_paar")   # Calls load_anton_paar()
   auto_load("data.csv", format="csv")          # Calls load_csv()
   auto_load("data.xlsx", format="excel")       # Calls load_excel()


Canonical Column Mapping
========================

``_column_mapping.py`` defines a registry of 20 ``CanonicalField`` entries that
normalize vendor-specific column names to canonical RheoJAX names:

.. code-block:: python

   @dataclass
   class CanonicalField:
       canonical_name: str        # e.g., "angular_frequency"
       patterns: list[str]        # Regex patterns: [r"^omega$", r"^frequency$", ...]
       si_unit: str               # Target SI unit: "rad/s"
       applicable_modes: list[str] # ["oscillation"]
       is_x_candidate: bool       # True if this can be x-axis
       is_y_candidate: bool       # True if this can be y-axis
       priority: int              # Lower = higher priority (5 beats 100)

**Registered fields (20 total):**

.. list-table::
   :header-rows: 1
   :widths: 25 15 15 10

   * - Canonical Name
     - SI Unit
     - Modes
     - Priority
   * - ``angular_frequency``
     - rad/s
     - oscillation
     - 5
   * - ``shear_rate``
     - 1/s
     - rotation
     - 5
   * - ``time``
     - s
     - creep, relaxation, oscillation, rotation
     - 10
   * - ``storage_modulus``
     - Pa
     - oscillation
     - 5
   * - ``loss_modulus``
     - Pa
     - oscillation
     - 5
   * - ``tensile_storage_modulus``
     - Pa
     - oscillation
     - 5
   * - ``tensile_loss_modulus``
     - Pa
     - oscillation
     - 5
   * - ``complex_modulus``
     - Pa
     - oscillation
     - 10
   * - ``compliance``
     - 1/Pa
     - creep
     - 5
   * - ``relaxation_modulus``
     - Pa
     - relaxation
     - 5
   * - ``viscosity``
     - Pa.s
     - rotation
     - 5
   * - ``complex_viscosity``
     - Pa.s
     - oscillation
     - 10
   * - ``shear_stress``
     - Pa
     - creep, relaxation, rotation
     - 20
   * - ``shear_strain``
     - dimensionless
     - creep, relaxation
     - 20
   * - ``phase_angle``
     - deg
     - oscillation
     - 100
   * - ``temperature``
     - °C
     - all
     - 100
   * - ``normal_force``
     - N
     - all
     - 100
   * - ``torque``
     - N.m
     - rotation
     - 100
   * - ``strain_amplitude``
     - dimensionless
     - oscillation
     - 100
   * - ``stress_amplitude``
     - Pa
     - oscillation
     - 100

**Matching algorithm:**

1. ``extract_unit_from_header()`` strips ``"(unit)"`` suffix: ``"omega (rad/s)"`` → ``"omega"``
2. Patterns are pre-compiled and sorted by priority (ascending)
3. First matching pattern wins (``match_column()``)
4. ``match_columns()`` processes all headers in batch


Unit Normalization
==================

Two unit conversion systems exist (unified in ``_utils.py``):

**``UNIFIED_UNIT_CONVERSIONS``** — Used by generic CSV/Excel readers:

.. list-table::
   :header-rows: 1
   :widths: 20 20 20

   * - Source Unit
     - Target SI Unit
     - Factor
   * - Hz
     - rad/s
     - 2π
   * - ms
     - s
     - 0.001
   * - min
     - s
     - 60.0
   * - kPa
     - Pa
     - 1000.0
   * - MPa
     - Pa
     - 1e6
   * - GPa
     - Pa
     - 1e9
   * - mPa·s
     - Pa.s
     - 0.001
   * - rpm
     - 1/s
     - 1/60
   * - %
     - dimensionless
     - 0.01
   * - °C
     - K
     - additive (+273.15)
   * - °F
     - K
     - additive

**``TRIOS_UNIT_CONVERSIONS``** — Used by TRIOS readers (case-variant keys):

Identical conversion factors but with case-variant keys (``"Hz"``, ``"HZ"``,
``"Rad/s"``, etc.) for TRIOS-specific column header conventions.

**Temperature normalization:**

``normalize_temperature(value, unit)`` converts °C/°F/K to Kelvin. Used by:

- ``load_tts()`` for TTS temperature metadata
- ``segment_to_rheodata()`` for TRIOS temperature metadata
- ``load_anton_paar()`` for RheoCompass temperature columns


Auto-Detection Systems
======================

Domain Detection
----------------

``detect_domain()`` determines whether data is time-domain or frequency-domain:

1. **Unit-based** (highest priority): ``rad/s``, ``Hz`` → frequency; ``s``, ``min`` → time
2. **Header keyword**: ``omega``, ``frequency`` → frequency; ``time``, ``t`` → time
3. **Y-header patterns**: G'/G'' presence → frequency
4. **Default**: time

Test Mode Detection
-------------------

``detect_test_mode_from_columns()`` infers the rheological protocol:

.. list-table::
   :header-rows: 1

   * - Test Mode
     - Pattern Triggers
   * - ``oscillation``
     - G', G'', G*, E', E'', omega, frequency
   * - ``relaxation``
     - G(t), relaxation
   * - ``creep``
     - J(t), compliance, creep
   * - ``rotation``
     - shear rate, viscosity, eta, η, γ̇

Falls back to unit-based detection (``rad/s`` → oscillation, ``1/s`` + ``Pa·s`` → rotation).

Deformation Mode Detection
---------------------------

``detect_deformation_mode_from_columns()`` distinguishes DMTA from shear:

- **Tensile**: E', E'', E*, tensile, Young's
- **Shear**: G', G'', G*, shear modulus
- **Bending**: E_bend, flexural (overrides tensile)
- **Compression**: E_comp, compressive (overrides tensile)

Geometry-specific modes (bending/compression) override generic tensile when both match.


Instrument-Specific Readers
============================

TRIOS Reader (TA Instruments)
-----------------------------

``rheojax.io.readers.trios`` handles 4 export formats via a unified ``load_trios()``
dispatcher:

**Architecture:**

.. code-block:: text

   load_trios(filepath)
       │
       ├── detect_trios_format()  → "txt" / "csv" / "excel" / "json"
       │
       ├── .txt  → load_trios_txt()     (1741L) — LIMS format, chunked reading
       ├── .csv  → load_trios_csv()     (1053L) — Tabular CSV with step columns
       ├── .xlsx → load_trios_excel()    (869L) — Multi-sheet workbooks
       └── .json → load_trios_json()     (491L) — Schema-validated JSON

**Shared infrastructure (``common.py``, 832L):**

- ``TRIOSFile``: Complete parsed file (metadata + list of ``TRIOSTable``)
- ``TRIOSTable``: Single data table (header, units, DataFrame, step values)
- ``DataSegment``: Processed segment ready for ``RheoData`` conversion
- ``ColumnMapping``: TRIOS-specific column mapping (17 entries)
- ``detect_test_type()``: DataFrame-based test type detection
- ``select_xy_columns()``: Priority-based x/y column selection with complex modulus
  pair detection (G'/G'' or E'/E'')
- ``segment_to_rheodata()``: Final conversion including temperature normalization
  and deformation mode inference

**Multi-step handling:**

TRIOS files may contain multiple experimental steps. The reader:

1. Detects step columns via ``detect_step_column()`` (candidates: "step", "segment",
   "step_number", etc.)
2. Splits into per-step DataFrames via ``split_by_step()``
3. Returns ``list[RheoData]`` when ``return_all_segments=True``

**JSON schema validation:**

``trios/schema/`` provides ``TRIOSExperiment`` dataclass with nested ``Result``,
``Step``, and ``Dataset`` structures for validating TRIOS JSON exports.

Anton Paar Reader (RheoCompass)
-------------------------------

``anton_paar.py`` (1719L) handles RheoCompass CSV exports with:

- **Interval block parsing**: "Interval and data points:" markers
- **Encoding cascade**: UTF-16 → UTF-8 → Latin-1
- **Locale-aware decimals**: European comma (``1,234``) and US dot (``1.234``)
- **Derived quantities**: J(t) from strain/stress, G(t), G* construction
- **Test type auto-detection**: From interval names and column patterns
- **``IntervalBlock`` dataclass**: Per-interval data + units + metadata

.. code-block:: python

   # Direct use
   data = load_anton_paar("rheocompass_export.csv", interval=2, test_mode="oscillation")

   # Interval inspection
   blocks = parse_rheocompass_intervals("export.csv")
   for block in blocks:
       print(f"Interval {block.interval_index}: {block.n_points} points")

Generic CSV Reader
------------------

``csv_reader.py`` (613L) handles arbitrary CSV/TSV files:

- **Delimiter auto-detection**: ``csv.Sniffer`` → heuristic fallback → whitespace regex
- **Encoding cascade**: UTF-8-sig (strict) → UTF-8 (replace) → UTF-16le
- **European decimal support**: Samples 20 values to distinguish US (``1,234.56``)
  from EU (``1.234,56``) formats; handles comma-only EU (``1,56``)
- **Encoding corruption detection**: Checks for ``\ufffd`` replacement characters in
  numeric columns → raises ``ValueError`` if detected
- **Column mapping**: Optional ``column_mapping`` dict applied before any lookup
- **Comment preamble**: Auto-detects ``#`` comment lines
- **Complex modulus**: ``y_cols=[storage, loss]`` constructs G* = G' + iG''
- **Protocol metadata kwargs**: ``strain_amplitude``, ``angular_frequency``,
  ``applied_stress``, ``shear_rate``, ``reference_gamma_dot``

Generic Excel Reader
--------------------

``excel_reader.py`` (387L) mirrors CSV reader features for ``.xlsx/.xls`` files:

- Sheet selection via ``sheet`` parameter
- Same auto-detection systems (domain, test mode, deformation mode)
- ``usecols`` optimization for wide files (only loads needed columns)
- Column mapping support
- Protocol metadata kwargs


Multi-File Loaders
==================

``multi_file.py`` provides three batch loading functions that wrap ``auto_load()``:

load_tts()
----------

Time-Temperature Superposition workflow:

.. code-block:: python

   datasets = load_tts(
       files=["T25.csv", "T50.csv", "T75.csv"],  # or "data/T*.csv" glob
       T_ref=298.15,                               # Reference temperature (K)
       temperatures=[25, 50, 75],                  # Per-file temperatures
       temperature_unit="C",                       # Auto-converts to Kelvin
   )

- Tags each ``RheoData`` with ``metadata["temperature"]`` (Kelvin) and ``metadata["T_ref"]``
- If ``temperatures`` is ``None``, reads from file metadata
- Returns sorted by temperature ascending
- Supports glob patterns (``"data/T*.csv"``)

load_srfs()
-----------

Superposition of Rate-Frequency Sweeps:

.. code-block:: python

   datasets = load_srfs(
       files=["rate0.01.csv", "rate0.1.csv", "rate1.csv"],
       reference_gamma_dots=[0.01, 0.1, 1.0],
   )

- Tags with ``metadata["reference_gamma_dot"]``
- Returns sorted by reference shear rate ascending

load_series()
-------------

Generic multi-file loader with custom metadata:

.. code-block:: python

   datasets = load_series(
       files="data/creep_*.csv",
       protocol="creep",
       sort_by="applied_stress",
       metadata_key="applied_stress",
       metadata_values=[100, 500, 1000],
   )


Protocol Validation
===================

``validate_protocol()`` runs post-load quality checks and returns a ``LoaderReport``:

.. code-block:: python

   @dataclass
   class LoaderReport:
       warnings: list[str]              # Non-fatal quality issues
       errors: list[str]                # Fatal issues
       skipped_rows: int                # NaN rows dropped during loading
       protocol_inferred: bool          # True if test_mode was auto-detected
       units_converted: dict[str, str]  # field → original unit
       quality_flags: dict[str, bool]   # Named boolean flags

**Per-protocol checks:**

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Protocol
     - Checks
   * - ``relaxation``
     - Monotonic decay (>30% increasing → warning); early transient presence
       (t_start/t_range > 0.5 → warning)
   * - ``creep``
     - Applied stress metadata (``sigma_applied`` or ``sigma_0``)
   * - ``oscillation``
     - Frequency range ≥ 2 decades; ≥ 2 positive frequency points
   * - ``rotation``
     - Shear rate metadata (``gamma_dot`` or ``shear_rate``)
   * - ``startup``
     - Shear rate metadata

**Transform validation:**

``validate_transform()`` checks compatibility between ``intended_transform`` and data:

- Required metadata fields (e.g., ``temperature`` for mastercurve)
- Domain compatibility (e.g., ``owchirp`` requires time domain)
- test_mode consistency (e.g., ``srfs`` expects rotation)


Writers
=======

HDF5 Writer/Reader
-------------------

``hdf5_writer.py`` (452L) provides lossless archiving:

**save_hdf5():**

- Atomic write via temp file + ``os.replace()``
- gzip compression (configurable level 0–9, default 4)
- Preserves complex dtype (``complex128`` for G*)
- Stores metadata recursively (dicts → subgroups, lists → attributes/datasets)
- Large arrays (> 60 KB) stored as datasets rather than attributes
- None values → sentinel string ``"__rheojax_None__"``
- Enum values → underlying Python type
- rheojax version tag

**load_hdf5():**

- Bytes → str decoding for cross-platform compatibility
- Recursive metadata reconstruction
- Belt-and-suspenders: test_mode/deformation_mode from both top-level attrs and
  metadata dict

NPZ Writer/Reader
------------------

``npz_writer.py`` (161L) provides lightweight, safe archiving:

- **Safe serialization only**: All data stored as NumPy arrays; strings encoded as
  uint8 bytes. No unsafe deserialization is used — ``allow_pickle=False`` is enforced.
- Metadata serialized as UTF-8 JSON (custom encoder for numpy types)
- ``save_npz()`` / ``load_npz()`` with compressed option
- Shape validation on load (x/y length mismatch → ``ValueError``)
- Auto-appends ``.npz`` extension if missing

Excel Writer
------------

``excel_writer.py`` (375L) for reporting:

- Atomic write via temp file + ``os.replace()``
- Multi-sheet workbook: Parameters, Fit Quality, Predictions, Residuals
- Complex array handling (G* split into G'/G'' columns)
- Deformation-mode–aware labels (E'/E'' for tensile, G'/G'' for shear)
- Embedded matplotlib plots via ``openpyxl.drawing.image``
- JAX/numpy scalar → Python native type conversion

SPP Export
----------

``spp_export.py`` (983L) for Sequence of Physical Processes analysis:

- **TXT**: MATLAB SPPplus_print_v2.m compatible (15-column + 9-column FSF)
- **HDF5**: Hierarchical (``/spp_data``, ``/waveforms``, ``/frenet_serret``, ``/metadata``)
- **CSV**: Comma-separated with optional Frenet-Serret frame columns
- **MATLAB dict**: ``to_matlab_dict()`` for ``scipy.io.savemat()``
- All writers: atomic file write, FSF vector guard (all 3 vectors required)


Error Handling
==============

Custom Exceptions
-----------------

- ``RheoJaxFormatError(ValueError)``: No reader can parse the file
- ``RheoJaxValidationWarning(UserWarning)``: Data quality issues

Error Strategy
--------------

The I/O subsystem uses a layered error strategy:

1. **Fatal exceptions** (``KeyboardInterrupt``, ``MemoryError``, ``PermissionError``,
   ``OSError``) are **never caught** during reader cascade — they propagate immediately.
2. **Format errors** during speculative parsing are caught and the cascade continues.
3. **All readers failed** → ``ValueError`` with concatenated error messages from
   each attempted reader.
4. **Encoding errors** → progressive fallback (UTF-8 strict → UTF-8 replace → UTF-16)
   with warnings for replacement character corruption.
5. **Atomic writes** — All writers use temp file + ``os.replace()`` to prevent
   corrupt files from interrupted writes.


Structured Logging
==================

All I/O operations use ``rheojax.logging`` structured logging:

- ``log_io()`` context manager decorates read/write operations with file path,
  record count, and timing
- Debug-level logging for every detection step (domain, test_mode, unit conversion)
- Warning-level for encoding fallbacks, NaN row drops, large files
- Error-level for parse failures with ``exc_info=True``


Integration Points
==================

Pipeline
--------

``Pipeline.load()`` delegates to ``auto_load()`` with format dispatch:

.. code-block:: python

   Pipeline().load("data.csv")                 # → auto_load()
   Pipeline().load("archive.npz")              # → load_npz() (NPZ format)
   Pipeline().load("archive.h5", format="hdf5") # → load_hdf5()

BaseModel
---------

``RheoData`` metadata flows to ``BaseModel.fit()``:

- ``test_mode`` → protocol-specific kernel dispatch
- ``deformation_mode`` → E* ↔ G* conversion at model boundary
- ``metadata["gamma_dot"]`` → flow curve protocol kwargs
- ``metadata["temperature"]`` → TTS shift factor

GUI
---

The GUI layer passes ``y2_col`` for loss modulus selection, which ``auto_load()``
translates to ``y_cols=[storage, loss]`` via ``_translate_y2_col()``.


Design Patterns Summary
========================

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Pattern
     - Usage
   * - **Strategy (Reader Cascade)**
     - ``auto_load()`` tries readers in sequence; first success wins
   * - **Registry (Column Mapping)**
     - ``CANONICAL_FIELDS`` and ``TRIOS_COLUMN_MAPPINGS`` centralize column name patterns
   * - **Dataclass DTOs**
     - ``TRIOSFile``, ``TRIOSTable``, ``DataSegment``, ``IntervalBlock``,
       ``CanonicalField``, ``LoaderReport``
   * - **Builder (Metadata)**
     - Progressive metadata assembly: source → user → protocol → transform
   * - **Template Method**
     - CSV and Excel readers share detection logic from ``_utils.py``
   * - **Atomic Write**
     - All writers use temp file + ``os.replace()``
   * - **Provenance Tracking**
     - ``format_detected`` and ``readers_attempted`` metadata injection
   * - **Speculative Parsing**
     - Warnings captured/discarded during failed reader attempts
