# CLAUDE.md

RheoJAX: JAX-accelerated rheological analysis (2-10x vs scipy).

## Stack & Commands

**Stack:** Python 3.12+, JAX >=0.8.3, NLSQ >=0.6.10, NumPyro, ArviZ >=0.23.4

```bash
# Testing (tiered)
pytest -n auto -m "smoke"           # ~1838 tests, CI gate
pytest -n auto -m "not slow..."     # ~4714 tests
pytest -n auto                      # Full suite (~4963 tests)

# Quality
make format && make quick           # Black + Ruff + smoke
make clean                          # Preserves .venv/, .claude/, .specify/, agent-os/

# GPU (Linux + CUDA 12+/13+)
make install-jax-gpu                # Auto-detect CUDA
make gpu-check                      # Verify backend
```

## Architecture

```
rheojax/
├── core/          # BaseModel, RheoData, Parameter, BayesianMixin
├── models/        # 53 models across 22 families (see inventory below)
├── transforms/    # 11: FFT, mastercurve, mutation_number, owchirp, derivatives, SRFS, SPP, Cox-Merz, Prony, spectrum_inversion, LVE_envelope
├── pipeline/      # Fluent API: Pipeline, BayesianPipeline, batch
├── io/            # TRIOS, CSV, Excel, Anton Paar readers; HDF5/Excel writers
├── visualization/ # Auto plot selection, 3 styles
├── logging/       # Structured logging (config, formatters, JAX-safe)
└── utils/         # optimization, prony, mct_kernels, structure_factor, compatibility, modulus_conversion, initialization/
```

**Model Families (53 total):** Classical (3), Flow (6), Fractional Maxwell (4), Fractional Zener (4), Fractional Advanced (3), Multi-Mode/GMM (1), SGR (2), STZ (1), EPM (2), Fluidity (2), Fluidity-Saramito (2), IKH (2), FIKH (2), HL (1), SPP (1), Giesekus (2), DMT (2), ITT-MCT (2), TNT (5), VLB (4), HVM (1), HVNM (1)

## Key Patterns

1. **BaseModel**: scikit-learn API (`.fit()/.predict()`), JAX-compatible, BayesianMixin for all 53 models
2. **RheoData**: JAX-native arrays, `.to_jax()`/`.to_numpy()`, auto test_mode detection
3. **Parameter/ParameterSet**: Bounds + priors, JAX array conversion
4. **Pipeline**: Fluent `Pipeline().load().fit().plot().save()`

## Critical Invariants

### Float64 (MANDATORY)
```python
# CORRECT — always use this in every module
from rheojax.core.jax_config import safe_import_jax
jax, jnp = safe_import_jax()

# NEVER import jax directly — float64 won't be configured
```

### Oscillation Data (G* = G' + iG'')
Models must handle three input formats for oscillation protocol:
- **Complex `G*`** (N,): Preserve with `jnp.complex128` — `create_least_squares_objective` fits G' and G'' independently
- **Real `(N, 2)`** [G', G'']: Element-wise residuals on both columns
- **Real `(N,)`** |G*|: Magnitude-only fallback (acceptable when user only has |G*|)

The dispatch happens in `create_least_squares_objective` (`rheojax/utils/optimization.py`). Model `_fit` methods must NOT cast complex y to `float64` before passing to the objective.

### Bayesian Mode-Awareness
```python
# Always pass test_mode — closure captures it for correct posteriors
result = model.fit_bayesian(omega, G_star, test_mode='oscillation')
# Default: num_chains=4, num_warmup=1000, num_samples=2000
# Diagnostics: R-hat <1.01, ESS >400
```

### DMTA / DMA Support
All 45 oscillation-capable models auto-convert E* <-> G* at BaseModel boundary:
```python
model.fit(omega, E_star, test_mode="oscillation", deformation_mode="tension", poisson_ratio=0.5)
```

### EPM Constitutive Forms (`fluidity_form`)
Both `LatticeEPM` and `TensorialEPM` accept a `fluidity_form` constructor argument that selects the plastic-strain-rate constitutive law. **Default is `"overstress"` (Herschel-Bulkley)** — the classical Bingham form lives under `"linear"` for backwards compatibility.

- `"linear"`: `plastic_rate = activation · σ · fluidity`. Bingham at high γ̇; cannot fit HB data.
- `"power"`: `plastic_rate ∝ |σ/σ_c|^n_fluid · σ_c / τ_pl`. Shear-thinning but no additive yield baseline.
- `"overstress"` *(default)*: `plastic_rate ∝ (|σ|−σ_c)_+^n_fluid / σ_c^(n_fluid−1) / τ_pl`. Gives σ = σ_y + K·γ̇^(1/n_fluid), the full HB form, and reaches R² > 0.99 on the emulsion HB benchmark.

The `n_fluid` fitted parameter is the reciprocal of the HB shear-thinning exponent (`n_HB = 1/n_fluid`). `n_fluid ≈ 2` → `n_HB ≈ 0.5`, typical for concentrated emulsions and foams.

**Tensorial gotcha**: the `TensorialEPM` pure-shear yield plateau is at `σ_c_mean / √3` (not `σ_c_mean`), due to the von Mises effective stress `σ_eff = √3·|σ_xy|`. To match a measured yield stress `σ_y` with tensorial, set `sigma_c_mean = σ_y · √3`. The scalar `LatticeEPM` has no such factor — there `sigma_c_mean` equals `σ_y` directly. Similarly, the tensorial stress-rate convention `dσ_ij/dt = 2μ·ε̇` introduces a factor of 2 in the high-rate asymptote, so the scalar `tau_pl` maps to `tau_pl_shear ≈ 2·tau_pl` on the tensorial side. See `docs/source/models/epm/tensorial_epm.rst` for the full mapping recipe.

**`TensorialEPM.fit()`** is now implemented (was `NotImplementedError` before) and supports 1D shear-only `σ_xy` targets across all five protocols. Combined `[σ_xy, N₁]` fitting is still a future feature.

## Workflow: NLSQ -> Bayesian

```python
model = Maxwell()
model.fit(t, G_data)                                    # NLSQ point estimate
result = model.fit_bayesian(t, G_data, seed=42)         # NUTS with warm-start
intervals = model.get_credible_intervals(result.posterior_samples, credibility=0.95)
```

NLSQ features: `workflow="auto"/"auto_global"`, `auto_bounds=True`, `stability='auto'`, `fallback=True`, `compute_diagnostics=True`, `result.prediction_interval(x, alpha=0.95)`.

## Adding Models

1. Create `rheojax/models/<family>/my_model.py` with `safe_import_jax()`
2. Inherit `BaseModel`, implement `_fit()`, `_predict()`
3. Handle oscillation complex/real dispatch in `_fit()` (see "Oscillation Data" above)
4. Register: `@ModelRegistry.register("my_model")`
5. Tests: `tests/models/test_my_model.py`

## Adding Transforms

1. Create `rheojax/transforms/my_transform.py`
2. Inherit `BaseTransform`, implement `transform()`
3. Register: `@TransformRegistry.register("my_transform")`
4. Tests: `tests/transforms/test_my_transform.py`

## Test Organization

- `tests/core/`: base, data, parameters, bayesian, float64, deformation_mode
- `tests/models/`: 26 model test suites (including hvm/, hvnm/)
- `tests/transforms/`: 11 transforms
- `tests/pipeline/`: Pipeline + BayesianPipeline
- `tests/utils/`: optimization, compatibility, initialization, modulus_conversion
- `tests/io/`: readers + DMTA + writers
- `tests/integration/`: End-to-end (NLSQ -> NUTS)
- `tests/validation/`: pyrheo, hermes-rheo (local only)

## Troubleshooting

- **Float64 Error:** Use `safe_import_jax()`, verify with `verify_float64()`
- **NLSQ Convergence:** `max_iter=5000`, `ftol/xtol=1e-6`, check data quality
- **Bayesian (R-hat >1.1, low ESS, divergences):** NLSQ warm-start (critical), increase `num_warmup/num_samples`
- **Debug Logging:** `RHEOJAX_LOG_LEVEL=DEBUG` or `configure_logging(level="DEBUG")`
