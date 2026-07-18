<!-- Generated: 2026-07-18 | Files scanned: ~30 (core/data + io) | Token estimate: ~500 -->

# Data Model & I/O (no relational database)

RheoJAX has no DB. "Data" = the `RheoData` container + file formats it round-trips through.

## RheoData (`core/data.py`)

Core container: `x`/`y` arrays (protocol-typed, e.g. time/G(t), freq/G'+G''), `protocol_type`,
`metadata` dict, provenance/lineage tracking. Runtime-validated at I/O boundaries (shape, dtype,
NaN, monotonicity — per root CLAUDE.md §3). Flows unchanged through both the fit path and the
transform path (see architecture.md).

## Readers (`io/readers/`)

| Format | Reader |
|---|---|
| TRIOS (TA Instruments) | `trios.py` |
| Anton Paar | `anton_paar.py` |
| CSV / Excel | `csv.py`, `excel.py` |
| Auto-detect | `auto.py` → `auto_load()` |

## Writers (`io/writers/`)

| Format | Writer | Used by |
|---|---|---|
| HDF5 | `hdf5_writer.py` | `cli export`, batch pipeline |
| Excel / NPZ | `excel_writer.py`, `npz_writer.py` | `cli export` |
| SPP MATLAB-compat | (SPP export module) | `spp` transform export |
| CSV/NetCDF/JSON bundle | `gui/services/export_service.py` | GUI wizard "Export Bundle..." (Fit/Transform step) |

## GUI Project File (v2 schema)

`gui/foundation/project_codec.py` — `save_project_v2()`/`load_project_v2()`. Serializes
`AppState` (fit/transform/pipeline state, dataset library refs, UI prefs) to a single project
file for GUI "Save"/"Open". Distinct from the CLI's JSON "envelope" format (backend.md).

## Provenance

Every fit/transform result carries a `provenance` dict (model_key, config, protocol, data_ref,
revision, convergence_verdict when Bayesian) — written into export bundles' `provenance.json` and
into the project file, so results are traceable back to the exact inputs/config that produced
them (research-reproducibility requirement, root CLAUDE.md §Principles).
