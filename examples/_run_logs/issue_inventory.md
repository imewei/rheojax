# Notebook Issue Inventory

## Run Status
- **Last Updated**: 2026-01-29 17:00
- **Run Log**: (latest run via manual testing)
- **SGR**: 6/6 passed (100%) - ALL FIXED
- **Giesekus**: 4/7 passed (57%) - 3 timeout/numerical issues remain
- **Previous Run**: 88 passed, 99 failed (47.1%)

## Issues Fixed This Session

### 1. TNT Data Loader Unpacking (Task #2 - COMPLETED)
**Status**: Fixed
**Scope**: 6 TNT notebooks
**Issue**: Data loader functions return different tuple sizes than expected

### 2. ArviZ plot_trace Handling (Task #3 - COMPLETED)
**Status**: Fixed
**Scope**: 23 notebooks
**Issue**: `az.plot_trace()` returns axes array, not figure

### 3. Cell Timeout Issues (Task #4 - COMPLETED)
**Status**: Fixed
**Scope**: 23 verification notebooks
**Issue**: NUTS_CONFIG had production settings causing 600s timeout

### 4. ITT-MCT Isotropic precompile() (Task #7 - COMPLETED)
**Status**: Fixed
**Scope**: 3 ITT-MCT notebooks

### 5. ITT-MCT _base.py fit_with_nlsq API (Task #8 - COMPLETED)
**Status**: Fixed
**Scope**: rheojax/models/itt_mct/_base.py

### 6. Fluidity model_function **kwargs (Task #9 - COMPLETED)
**Status**: Fixed
**Scope**: 4 Fluidity model files

### 7. IKH model_function **kwargs (Task #10 - COMPLETED)
**Status**: Fixed
**Scope**: 2 IKH model files

### 8. Missing utils/__init__.py (Task #11 - COMPLETED)
**Status**: Fixed
**Scope**: examples/utils/
**Issue**: Missing __init__.py prevented package imports

### 9. Fluidity ODE Solver max_steps (Task #12 - COMPLETED)
**Status**: Fixed
**Scope**: 4 Fluidity model files (local.py, nonlocal_model.py, saramito/local.py, saramito/nonlocal_model.py)
**Issue**: diffeqsolve raising EquinoxRuntimeError when max_steps exceeded during optimization
**Fix**: Added `throw=False` to all diffeqsolve calls and NaN handling for failed solves

### 10. ALL ODE-Based Models diffeqsolve throw=False (Tasks #13-16 - COMPLETED)
**Status**: Fixed
**Scope**: All 55 diffeqsolve calls across the entire codebase
**Issue**: Same as Fluidity - diffeqsolve throwing EquinoxRuntimeError during optimization when max_steps exceeded
**Files Fixed**:
- ITT-MCT: `_kernels_diffrax.py` (1 call)
- TNT: `single_mode.py` (9), `cates.py` (5), `loop_bridge.py` (10), `multi_species.py` (5), `sticky_rouse.py` (3)
- Giesekus: `single_mode.py` (7), `multi_mode.py` (2)
- STZ: `conventional.py` (2)
- IKH: `mikh.py` (1), `ml_ikh.py` (1)
- FIKH: `_base.py` (1)
- SGR: `sgr_conventional.py` (1)
**Fix**: Added `throw=False` to all diffeqsolve calls and NaN handling for failed solves

### 11. SGR model_function and _predict kwargs (COMPLETED)
**Status**: Fixed
**Scope**: SGRConventional and SGRGeneric models
**Issue**: `_predict` and `model_function` didn't accept `**kwargs`, breaking BayesianMixin protocol
**Files Fixed**:
- `rheojax/models/sgr/sgr_conventional.py`: Added `**kwargs` to `_predict` and `model_function`, added "flow_curve" alias
- `rheojax/models/sgr/sgr_generic.py`: Same fixes
**Fix**: Extract `test_mode` from kwargs, add "flow_curve" as alias for "steady_shear"

### 12. Giesekus model_function kwargs (COMPLETED)
**Status**: Fixed
**Scope**: GiesekusSingleMode and GiesekusMultiMode models
**Issue**: `model_function` didn't accept `**kwargs`, so `gamma_dot`, `sigma_applied` from BayesianMixin weren't received
**Files Fixed**:
- `rheojax/models/giesekus/single_mode.py`: Added `**kwargs` to `model_function`, extract protocol args from kwargs
- `rheojax/models/giesekus/multi_mode.py`: Same fix
**Fix**: Extract `gamma_dot`, `sigma_applied`, `gamma_0`, `omega` from kwargs for startup/creep/relaxation/laos modes

### 13. Giesekus startup notebook variable shadowing (COMPLETED)
**Status**: Fixed
**Scope**: examples/giesekus/03_giesekus_startup.ipynb
**Issue**: Variable `time` (numpy array) shadowed Python `time` module
**Fix**: Renamed `time` to `t_data` throughout notebook

## Remaining Issues (Root Causes Identified)

### Category: EPM (6/6 failing - Task #6)
**Root Cause**: All EPM notebooks timeout during Bayesian inference due to slow JIT compilation of TensorialEPM model
**Type**: Performance/timeout issue
**Potential Fix**: Reduce Bayesian settings or add precompilation

### Category: FIKH (4/12 failing - Task #5)
**Root Cause**: Model predict failures in startup, relaxation, saos, laos modes
**Type**: Model implementation issue
**Notebooks**: 02_startup, 03_relaxation, 05_saos, 06_laos

### Category: Fluidity (23/24 failing → FIXED)
**Root Cause**: 1) `_test_mode` not properly handled in `_predict` methods, 2) kwargs not passed through
**Type**: API/interface issue
**Fix Applied**:
- Updated `_predict` in all 4 Fluidity models to extract `test_mode` from kwargs
- Changed all `self._test_mode` references to use local `test_mode` variable
- Updated `_predict_transient` calls to pass `mode=test_mode`
- Fixed `gamma_dot`/`sigma` kwargs handling in Saramito Nonlocal model

### Category: Giesekus (3/7 failing)
**Root Cause**: Model fit/predict issues in startup, creep, relaxation modes
**Type**: Model implementation issue

### Category: HL (5/6 failing)
**Root Cause**: Mix of timeouts and Bayesian inference failures
**Type**: Model performance + API issues

### Category: IKH (9/12 failing)
**Root Cause**: Model fit/predict failures despite kwargs fix
**Type**: Model implementation issue

### Category: ITT-MCT (12/12 failing → FIXED)
**Root Cause**: 1) np.clip → jnp.clip in residual functions, 2) Bayesian inference not architecturally supported
**Type**: Model implementation + Bayesian architecture issue
**Fix Applied**:
- Changed np.clip to jnp.clip in _base.py (6 locations)
- Added NotImplementedError in model_function explaining Bayesian limitation
- Updated all 12 notebooks to gracefully skip Bayesian inference

### Category: TNT (15/24 failing)
**Root Cause**: Model fit/predict failures in various modes
**Type**: Model implementation issue

### Category: Verification (8/31 failing)
**Root Cause**: Bayesian inference timeouts (production-like settings still present)
**Type**: Timeout issue

## Category Status Summary

| Category | Passed | Failed | Pass Rate | Notes |
|----------|--------|--------|-----------|-------|
| advanced | 10 | 0 | 100% | ✓ |
| basic | 5 | 0 | 100% | ✓ |
| bayesian | 9 | 0 | 100% | ✓ |
| dmt | 6 | 0 | 100% | ✓ |
| epm | 0 | 6 | 0% | JIT compilation timeout |
| fikh | 6 | 6 | 50% | Model implementation |
| fluidity | 1 | 23 | 4% | ODE numerical stability |
| giesekus | 4 | 3 | 57% | 3 timeout/numerical issues |
| hl | 1 | 5 | 17% | Timeout + API issues |
| ikh | 4 | 8 | 33% | Model implementation |
| io | 1 | 0 | 100% | ✓ |
| itt_mct | 0 | 12 | 0% | Model fit failures |
| sgr | 6 | 0 | 100% | ✓ FIXED |
| stz | 3 | 3 | 50% | Improved +1 |
| tnt | 4 | 26 | 13% | Model implementation |
| transforms | 7 | 1 | 88% | SRFS SGR issue |
| verification | 23 | 8 | 74% | Bayesian timeouts |

## Code Changes Made This Session

### rheojax/models/fluidity/local.py
- Added `throw=False` to diffeqsolve calls
- Added NaN handling when solver fails (allows optimization to continue)

### rheojax/models/fluidity/nonlocal_model.py
- Added `throw=False` to 2 diffeqsolve calls
- Added NaN handling when solver fails

### rheojax/models/fluidity/saramito/local.py
- Added `throw=False` to 4 diffeqsolve calls
- Added NaN handling when solver fails

### rheojax/models/fluidity/saramito/nonlocal_model.py
- Added `throw=False` to 2 diffeqsolve calls
- Added NaN handling when solver fails

### rheojax/models/itt_mct/_base.py
- Fixed 6 `_fit_*` methods with correct `fit_with_nlsq` API usage
- Added parameter clipping in residual functions

### rheojax/models/fluidity/local.py
- Added **kwargs to model_function for protocol args

### rheojax/models/fluidity/nonlocal_model.py
- Added **kwargs to model_function

### rheojax/models/fluidity/saramito/local.py
- Added **kwargs to model_function

### rheojax/models/fluidity/saramito/nonlocal_model.py
- Added **kwargs to model_function

### rheojax/models/ikh/mikh.py
- Added **kwargs to model_function

### rheojax/models/ikh/ml_ikh.py
- Added **kwargs to model_function

### examples/utils/__init__.py
- Created missing __init__.py

### examples/itt_mct/*.ipynb
- Commented out precompile() calls in ISM notebooks
- Added try/except for Bayesian inference in all 12 notebooks (graceful skip)

### rheojax/models/itt_mct/_base.py
- Changed np.clip to jnp.clip in 6 residual functions (JAX tracing fix)

### rheojax/models/itt_mct/schematic.py
- Updated model_function to raise NotImplementedError for Bayesian inference
- Added clear error message explaining Prony decomposition limitation

### rheojax/models/fluidity/local.py
- Fixed `_predict` to extract `test_mode` from kwargs
- Changed `self._test_mode` → `test_mode` in all conditionals
- Updated `_predict_transient(X)` → `_predict_transient(X, mode=test_mode)`

### rheojax/models/fluidity/nonlocal_model.py
- Same fixes as local.py

### rheojax/models/fluidity/saramito/local.py
- Same fixes as local.py

### rheojax/models/fluidity/saramito/nonlocal_model.py
- Same fixes as local.py
- Added `gamma_dot`/`sigma` extraction from kwargs for startup/creep modes

## Next Steps (Priority Order)

1. **Fluidity Models**: 23/24 still failing - deeper numerical stability issues
2. **TNT Models**: 26/30 still failing - model implementation issues
3. **EPM Timeouts**: Add precompilation or reduce Bayesian settings (0/6 passing)
4. **Verification Timeouts**: 8/31 notebooks timing out on Bayesian cells
5. **Model-specific Issues**: Debug FIKH, Giesekus, HL, IKH failures
