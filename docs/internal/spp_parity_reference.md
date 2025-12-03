# SPP Parity Reference (MATLAB SPPplus v2.1, R oreo, RheoJAX)

## Scope
Concise reference to drive SPP parity work: MATLAB implementation summary, gap matrix (feature parity across MATLAB SPPplus v2.1, R oreo, RheoJAX), and golden-data test harness plan. No code changes yet.

## MATLAB SPPplus v2.1 (key points)
- Entry point: `RunSPPplus_v2.m` — sets user inputs (`fname`, `ftype`, `var_loc`, `var_conv`, `data_trunc`, `an_use`, `omega`, `M`, `p`, `k`, `num_mode`, `out_type`, `is_fsf`, `save_figs`) and dispatches Fourier and/or numerical paths.
- Reader: `SPPplus_read_v2.m` — loads .txt/.csv, applies unit conversions; optional truncation; if rate column missing, infers rate via 8-point 4th-order derivative of strain on wrapped data; returns `time_wave` shifted to start at 0, `resp_wave` (strain, rate, stress), length `L`.
- Fourier analysis: `SPPplus_fourier_v2.m` — enforces scalar `omega`, divides rate by `omega`; FFT, select harmonics spaced by cycles `p`, normalize stress harmonics to the first harmonic; phase offset `Delta` from strain fundamental, rotates coefficients, shifts time by `Delta/omega`; reconstructs strain/rate from fundamental, stress from odd harmonics up to `M`; computes T/N/B, G′/G″/|G*|, tan δ, δ, derivatives, displacement stress, equilibrium strain. Outputs `spp_params=[omega,M,p,W,NaN,NaN]`, `spp_data_out` 15 cols, `fsf_data_out` (T/N/B), `ft_out` harmonic spectrum. Figures via `SPPplus_figures_v2`.
- Numerical analysis: `SPPplus_numerical_v2.m` — supports scalar or per-sample `omega`; average `dt`; two derivative modes: `num_mode=1` edge-aware forward/backward + centered 4th-order, `num_mode=2` fully periodic centered with wrap. Same Frenet/moduli workflow; `spp_params=[mean(omega),NaN,NaN,NaN,k,num_mode]`; outputs share 15-col schema (uses original time grid and measured waveforms).
- Export: `SPPplus_print_v2.m` — writes `.txt` or `.mat`. `spp_data_out` columns: time, strain, rate, stress, G′_t, G″_t, |G*_t|, tan δ, δ, displacement stress, eq_strain_est, dG′/dt, dG″/dt, G_speed, δ̇. Optional FSF file (9 cols T/N/B). Method note and params recorded (including `num_mode`).
- Defaults: `M=39`, `p=1`, `k=8`, `omega` required; requires integer cycles and prefers even samples per cycle for FFT; rate always divided by `omega` before analysis.

## Gap Matrix (parity notes)
Legend: ✅ parity, 🟡 partial/behavior differs, ❌ missing, ➕ RheoJAX-only.

| Feature / Behavior | MATLAB SPPplus v2.1 | R oreo | RheoJAX |
| --- | --- | --- | --- |
| Input reader, unit conv | ✅ (`var_loc`, `var_conv`) | ✅ | 🟡 (expects pre-scaled; limited conv) |
| Rate inference if missing | ✅ 8-pt 4th-order wrap | ✅ same | ❌ (requires rate or numerical diff on given rate) |
| Cycle selection | ✅ via integer `p` | ✅ `p` | 🟡 uses `start_cycle/end_cycle`, auto-detect cycles |
| Frequency handling | ✅ scalar `omega` (FFT); vector allowed numeric | ✅ same | 🟡 allows scalar; vector partly supported; rate not auto-divided by `omega` |
| Harmonic selection | ✅ strain/rate fundamental; stress odd harmonics to `M` | ✅ same | 🟡 configurable `n_harmonics`; may include strain/rate harmonics |
| Phase alignment/time shift | ✅ `Delta` from strain fundamental, rotate coeffs, shift t | ✅ | 🟡 claims compat; needs verification |
| Numerical differentiation modes | ✅ `num_mode` 1 (edge) / 2 (looped) | ✅ | 🟡 single path; no explicit looped toggle |
| Derivative order/stencils | ✅ 4th-order (and 8-pt for inferred rate) | ✅ | ✅ 4th-order JAX; no 8-pt rate inference |
| Moduli + derivatives formulas | ✅ G′, G″, |G*|, tan δ, δ, G′̇, G″̇, G_speed, δ̇ | ✅ | ✅ implemented; needs tolerance check |
| Frenet-Serret outputs | ✅ T/N/B 9 cols | ✅ | ✅ (kernels + export module) |
| Output schema (15 cols) | ✅ fixed order | ✅ fixed order | 🟡 extra metrics present; need strict 15-col parity export |
| Harmonic spectrum (`ft_out`) | ✅ | ✅ | ❌ not emitted by default |
| FSF export toggle | ✅ via `is_fsf` | ✅ | 🟡 export module exists, not wired in transform pipeline |
| Figures/plots | ✅ standard + recon + harmonics | ✅ | ❌ not replicated |
| Defaults | M=39, k=8, p=1 | similar | 🟡 n_harmonics~5–15, step_size=1 |
| Yield stress extraction | ❌ | ❌ | ➕ static/dynamic yield + power-law model |
| Lissajous metrics (G_L, η_L, S/T) | ❌ | ❌ | ➕ computed |
| Export formats | .txt/.mat | .csv/.xls | 🟡 .csv/.h5/.mat support in `io/spp_export`, not fully integrated |

## Golden-Data Harness (actionable plan)
- Datasets (synthetic, deterministic seed):
  - `sin_fundamental`: γ=Γ·sin(ωt)+h3·sin(3ωt), σ=A·sin(ωt); ω=2π rad/s; Γ=1.0; h3=0.15; 3 cycles, 256 pts/cycle.
  - `sin_noisy`: same plus Gaussian noise (σ_noise=0.01·A, γ_noise=0.01·Γ), seed=0 for reproducibility.
  - Optional `amp_sweep`: Γ ∈ {0.5,1.0,2.0}, fixed ω, same h3.
- Scripts (under `scripts/`):
  - `gen_inputs.py` → `scripts/golden_data/input/<dataset>.csv` with `t,gamma,sigma`.
  - `run_sppplus_v2p1.m` → `scripts/golden_data/outputs/matlab/<dataset>_*.txt` (Fourier and numerical).
  - `run_oreo.R` → `scripts/golden_data/outputs/r/<dataset>_*.csv` (Fourier and numerical).
  - `run_rheojax.py` → `scripts/golden_data/outputs/rheojax/<dataset>_*.csv` (spp_data_out, fsf_data_out, ft_out).
- Standard output columns for comparison: `t, gamma, sigma, Gp_t, Gpp_t, G_star_t, delta_t, G_speed, yield_stress, yield_strain, frenet_t_x, frenet_t_y, frenet_n_x, frenet_n_y, meta_tool, meta_dataset, meta_version` (allow NA for unavailable fields).
- Pytest harness (`tests/integration/test_spp_golden_parity.py`):
  - Compare each tool vs MATLAB reference with `rtol=1e-2/atol=1e-4` for core columns; Frenet components unit-length within 1e-3; treat NaN==NaN.
  - Mark slow/integration; provide `GOLDEN_FAST=1` to run only `sin_fundamental` smoke (fewer points) and skip others; allow `GOLDEN_DATA_DIR` override.
- Directory layout: `scripts/golden_data/input/`, `scripts/golden_data/outputs/{matlab,r,rheojax}/`, plus scripts in `scripts/`.

### How to generate goldens
1) Generate inputs: `python scripts/gen_inputs.py` (writes to `scripts/golden_data/input/`).
2) MATLAB goldens: in MATLAB from repo root, run `run('scripts/run_sppplus_v2p1.m')` (writes to `scripts/golden_data/outputs/matlab/`).
3) R goldens: `Rscript scripts/run_oreo.R` (writes to `scripts/golden_data/outputs/r/`).
4) RheoJAX goldens: `python scripts/run_rheojax.py` (writes to `scripts/golden_data/outputs/rheojax/`).

**Important:** The `p` parameter in MATLAB/R scripts must match `n_cycles` in `gen_inputs.py`. Currently both are set to 3. If you change the number of cycles in the input data, update `p` accordingly in `run_sppplus_v2p1.m` and `run_oreo.R`.

Note: `tests/integration/test_spp_golden_parity.py` will skip parity comparisons until MATLAB/R goldens exist. After running steps 2–3, rerun pytest to exercise the comparisons.

## Immediate use
- Use this doc as the single source for closing parity gaps and standing up golden-data regression tests before code changes.
