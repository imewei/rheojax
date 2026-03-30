# FIKH (Fractional Isotropic-Kinematic Hardening) Comprehensive Validation Report

**Date:** 2026-03-29
**Mode:** Enterprise (--deep --security --performance)
**Scope:** `rheojax/models/fikh/`, `docs/source/models/fikh/`, `examples/fikh/`, `tests/models/fikh/`

---

## Summary

- **Assessment:** ⚠️ Needs Minor Work (2 documentation issues)
- **Confidence:** High
- **Mode:** Enterprise with --deep --security --performance

---

## 1. Automated Checks

| Dimension | Tool | Result |
|-----------|------|--------|
| Unit Tests | pytest | 90/90 PASSED |
| Linting | ruff | All checks passed |
| Type Checking | mypy | No FIKH-specific errors |
| Security (S rules) | ruff --select S | All checks passed |
| Assert statements | grep | None found (clean) |

---

## 2. Physics Verification Against Literature

### 2.1 Caputo Fractional Structure Evolution — CORRECT

**Implementation:** `D^α_C λ = (1-λ)/τ_thix - Γ·λ·|γ̇^p|`

**Literature:** The RHS is the classical Moore/Coussot build-up/breakdown form, confirmed in Dimitriou & McKinley (2014) *Soft Matter* 10:6619. The promotion to Caputo derivative on the LHS is a synthesis extending Jaishankar & McKinley (2013, 2014) fractional viscoelasticity to thixotropic structure. No single paper proposes this exact combination — it is a well-motivated novel synthesis. ✓

### 2.2 L1 Scheme for Caputo Derivative — CORRECT

**Implementation:** `b_k = (k+1)^(1-α) - k^(1-α)`, normalization `1/(Γ(2-α)·dt^α)`

**Literature:** Standard L1 scheme from Lin & Xu (2007), textbook treatment in Li & Zeng (2015). Convergence order O(Δt^{2-α}), which for α ∈ (0,1) lies between O(Δt) and O(Δt²). Implementation matches exactly. ✓

### 2.3 Armstrong-Frederick Backstress — CORRECT

**Implementation:** `dA/dt = C·γ̇^p·sign(ξ) - γ_dyn·|A|^(m-1)·A·|γ̇^p|`

**Literature:** Standard AF rule (Armstrong & Frederick 1966, Chaboche 1991). Dimitriou & McKinley (2014) first adapted AF for thixotropic yield stress fluids within IKH. The exponent generalization m ≠ 1 is the Chaboche multi-surface extension. Implementation with |A|^(m-1) regularized as `max(|A|, 1e-10)^(m-1)` is numerically sound. ✓

### 2.4 Perzyna Plastic Flow Rule — CORRECT

**Implementation:** `γ̇^p = ⟨|σ-C·A| - σ_y⟩/μ_p · sign(σ-C·A)`

**Literature:** Standard Perzyna (1963) overstress model. Reduces to Bingham when σ_y=const, C=0. Macaulay bracket and sign function correctly implemented with sign-safety regularization (F-001 fix). ✓

### 2.5 Mittag-Leffler Relaxation — CORRECT

**Implementation:** Solution of `D^α λ = -λ/τ` gives `λ(t) = E_α(-(t/τ)^α)`

**Literature:** Textbook result (Mainardi 1996, Gorenflo & Mainardi 1997). Asymptotics correct:
- Short time: `E_α(z) ≈ 1 - z/Γ(1+α)` (stretched onset)
- Long time: `E_α(z) ≈ (-z)^{-1}/Γ(1-α)` (power-law tail)
- α → 1 limit: `E_1(-t/τ) = exp(-t/τ)` (exponential recovery) ✓

### 2.6 Cole-Cole Depression Angle — CORRECT

**Implementation:** `θ = (1-α)π/2` for fractional elements

**Literature:** Confirmed in Friedrich (1992) *Rheol. Acta* 31:309 for the fractional spring-pot/Scott Blair element. Applies to fractional Maxwell model; more complex models produce distorted arcs. ✓

### 2.7 Arrhenius Thermal Coupling — CORRECT

**Implementation:** `η(T) = η_ref · exp(E_a/R · (1/T - 1/T_ref))` with R = 8.314462618 J/(mol·K)

**Literature:** Standard Arrhenius form. Exponent clipped to [-50, 50] for overflow protection. ✓

### 2.8 Taylor-Quinney Coefficient — CORRECT (with caveat)

**Implementation:** `ρc_p·dT/dt = χ·σ·γ̇^p - h·(T-T_env)` with χ default 0.9

**Literature:** Taylor & Quinney (1934) measured χ ≈ 0.9 for metals. For structured fluids (no dislocation storage), χ should be closer to 1.0. Default of 0.9 is conservative and acceptable. ✓

---

## 3. Documentation vs. Code Consistency

### 3.1 Issues Found

#### ISSUE 1 (Important): Fractional Weissenberg Number Not Dimensionally Consistent

- **Doc claim** (`fikh.rst:355-362`): `Wi_α = γ̇ · τ_thix^{1/α}` described as a "dimensionless group"
- **Literature:** Not found in any published reference. For α ≠ 1, this quantity has dimensions s^{1/α - 1}, which is NOT dimensionless. The standard Weissenberg number Wi = γ̇·τ is dimensionless. A dimensionally consistent fractional form would require: `Wi_α = (γ̇·τ_ref)·(τ_thix/τ_ref)^{1/α}` where τ_ref absorbs the dimensional mismatch.
- **Impact:** Misleading for users trying to compute dimensionless groups. The quantity is a useful heuristic scaling but should not be called a "dimensionless group."

#### ISSUE 2 (Minor): alpha_structure Bounds Inconsistency

- **Code** (`fractional_mixin.py:23`): `FRACTIONAL_ORDER_BOUNDS = (0.0, 1.0)`
- **Docs** (`fikh.rst:817`, `fmlikh.rst:624,664`): `(0.05, 0.99)`
- The docs are more conservative (avoiding degenerate limits α=0, α=1), which is better practical guidance. The code allows the full range. Either align docs to code or tighten code bounds.

### 3.2 No Issues

- All 6 protocols documented and implemented consistently
- Parameter names, defaults, and units match between code and docs
- Thermal coupling documented and implemented correctly
- FMLIKH multi-mode architecture matches docs
- All references are real and correctly cited
- Mittag-Leffler asymptotics correctly described

---

## 4. Examples vs. Code Consistency

### 4.1 No Significant Issues

All 12 example notebooks (6 FIKH + 6 FMLIKH) demonstrate:
- Correct protocol usage with appropriate parameters
- Real data (waxy crude oil from Wei et al. 2018) and synthetic data
- Physically reasonable parameter ranges
- Correct claims about power-law relaxation, Bauschinger effect, Cole-Cole depression

### 4.2 Minor Notes

- NB01 flow curve fitting pushes τ_thix to upper bounds (1.2 years) — acknowledged in notebook as a limitation of steady-state data for transient parameter identification
- NB05 SAOS warns about Bayesian inference cost (>10 min) — appropriate caveat
- FAST_MODE uses only 1 chain — correctly noted as not production-quality

---

## 5. Test Coverage Assessment

| Area | Tests | Coverage |
|------|-------|----------|
| Initialization & params | 9 | All configs tested |
| Predictions (6 protocols) | 15 | All protocols + edge cases |
| Limiting behavior | 2 | α→1 exponential, α small slow recovery |
| Model function | 3 | Dict and array interfaces |
| Caputo derivative | 12 | L1 coefficients, history buffer, constant/linear functions |
| Thermal coupling | 6 | Arrhenius, yield stress, heating/cooling |
| Integration (fit→predict) | 2 | Startup and flow curve round-trips |
| Sign safety (F-001) | 2 | Bug fix validation |

**Edge cases covered:** α→1 integer limit, α=0.1 slow recovery, zero strain, zero shear, E_a=0, constant function derivative=0, sign-safe regularization.

**Missing tests (suggestions):**
- α→0 limit behavior (extremely strong memory)
- FMLIKH shared vs per-mode α comparison
- Short-memory truncation error quantification

---

## 6. Security Analysis

| Check | Result |
|-------|--------|
| Secrets in code | None found |
| `assert` usage | None (clean) |
| Code injection | No eval/exec |
| Dependency vulnerabilities | N/A |

---

## 7. Performance Notes

- **L1 scheme:** O(n_history) per step via JAX scan — efficient fixed-window approximation
- **JIT compilation:** All kernels within `jax.lax.scan` loops
- **Multi-mode (FMLIKH):** `jax.vmap` parallelization over N modes
- **Precompilation:** Explicit `precompile()` method for reducing first-call latency
- **History buffer:** Fixed-size (n_history=100 default) prevents memory growth

---

## Recommendations

### Must Fix (Documentation)
1. **Fix Fractional Weissenberg Number** in `fikh.rst:355-362`: Either (a) add a note that Wi_α is not dimensionless for α≠1, or (b) replace with a dimensionally consistent definition using a reference timescale, or (c) rename to "fractional scaling parameter" instead of "dimensionless group"

### Should Fix
2. **Align alpha_structure bounds** between code and docs: Either update `FRACTIONAL_ORDER_BOUNDS` to `(0.05, 0.99)` to match docs, or update docs to `(0.0, 1.0)` to match code. The docs' conservative range is better practical guidance.

### Nice to Fix
3. Add tests for α→0 extreme memory regime
4. Add FMLIKH shared-α vs per-mode-α comparison test

---

## Evidence

- 90/90 tests passing
- Ruff lint: clean
- Mypy: no FIKH-specific errors
- Security: clean (no asserts, no S101)
- 8/8 physics claims verified against published literature
- 12 example notebooks reviewed
