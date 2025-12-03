# SPP Parity Status (MATLAB SPPplus v2.1 / R oreo / RheoJAX)

## Current status
- Core parity fixes landed: omega-based rate scaling, strain/rate fundamental-only in Fourier, stress odd harmonics (default M=39), phase shift (Delta/time shift), numerical `num_mode` (edge vs periodic), MATLAB-format exports (15-col `spp_data_out`, 9-col `fsf_data_out`, `ft_out`), and strict column-shape assertions in tests.
- JIT disabled on Fourier helpers to avoid tracer concretization; tests pass and performance is acceptable for now.
- Golden harness implemented (inputs, runners for MATLAB/RheoJAX/R, pytest parity check). Parity test currently skips when MATLAB/R goldens are absent.

## What’s done
- Scripts: `scripts/gen_inputs.py`, `scripts/run_rheojax.py`, `scripts/run_sppplus_v2p1.m`, `scripts/run_oreo.R`.
- Tests: `tests/integration/test_spp_golden_parity.py` (compares RheoJAX vs MATLAB/R goldens when present; skips otherwise). Transform/unit suites updated to assert export shapes.
- Docs: parity reference and this status note.

## Remaining actions (require MATLAB/R availability)
1) Generate goldens:
   - MATLAB: run `run('scripts/run_sppplus_v2p1.m')` from repo root.
   - R: `Rscript scripts/run_oreo.R`.
2) Rerun parity check: `pytest tests/integration/test_spp_golden_parity.py -q`.
3) If any comparisons fail, inspect diffs and minimally adjust tolerances (current: rtol=1e-2, atol=1e-4 on core columns) or reconcile remaining gaps.
4) (Optional) Re-enable JIT for Fourier helpers with static shapes if performance is needed.

## Open risks / follow-ups
- No verified MATLAB/R outputs yet; parity claim pending those runs.
- Tolerances may need tuning after first MATLAB/R comparison.
- If upstream MATLAB/R scripts change, regenerate goldens and rerun parity tests.
