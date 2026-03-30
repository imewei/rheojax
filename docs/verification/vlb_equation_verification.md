# VLB Model Equation Verification

**Date:** 2026-03-30
**Sources:** Vernerey, Long & Brighenti (2017) JMPS 107:1-20; Vernerey (2018) JMPS 115:217-247 (PMC6824477); Vernerey et al. (2018) Polymers 10(8):848 (PMC6403683); Bell (1978) Science 200:618-627.

---

## 1. Distribution Tensor Evolution Equation

### Paper (Eq. 3b in Polymers 2018; Eq. 17 in JMPS 2018)

The full evolution equation from Vernerey (2018 JMPS, Eq. 17) is:

```
D(c·μ)/Dt = k_a·(C - c)·I - k_d·(c·μ) + c·(L·μ) + c·(L·μ)^T
```

where `c` = attached chain concentration, `C` = total concentration, `k_a` = association rate, `k_d` = dissociation rate, `L` = velocity gradient tensor.

For incompressible material with constant concentration (c = const, C = const), and using the steady-state relation `c = C·k_a/(k_a + k_d)`, this simplifies. In the simplest case where `k_a = k_d` (equal rates) and `c = C/2`, or more generally absorbing concentration into the effective rates:

```
dμ/dt = k_a·((C-c)/c)·I - k_d·μ + L·μ + μ·L^T
```

At dynamic equilibrium of concentrations, `k_a·(C - c)/c = k_d`, giving:

```
dμ/dt = k_d·(I - μ) + L·μ + μ·L^T
```

### RheoJAX Implementation

```
dmu/dt = k_d*(I - mu) + L·mu + mu·L^T
```

**VERDICT: CORRECT.** The implementation uses the full velocity gradient `L` (not just symmetric `D`), matching the paper. The docstring at the top of `_kernels.py` incorrectly states `D·mu + mu·D` but the actual code uses `L·mu + mu·L^T` which is correct.

> **Note:** The docstring header says `D·mu + mu·D` (symmetric part only) but the derivation in the comments and the actual code at lines 127-131 correctly use the full velocity gradient. The docstring should be corrected to say `L·mu + mu·L^T`.

---

## 2. Cauchy Stress

### Paper (Eq. 28 in JMPS 2018, Gaussian limit N→∞)

For Gaussian chains (N → ∞), the stress from Eq. 28 reduces to:

```
σ = c·k_B·T·(μ - I) + p·I
```

where `G₀ = c·k_B·T` is the network modulus.

### RheoJAX Implementation

```python
sigma_xy = G0 * mu_xy           # (since I_xy = 0)
N1 = G0 * (mu_xx - mu_yy)       # = sigma_xx - sigma_yy
```

**VERDICT: CORRECT.** The shear stress `σ_xy = G₀·μ_xy` and N1 = G₀·(μ_xx - μ_yy) are correct for the Gaussian limit.

---

## 3. Free Energy

### Paper (Eq. 5 in Polymers 2018; Eq. 26/27 in JMPS 2018)

For Gaussian chains, the stored elastic energy (from the leading term of Eq. 5 in Polymers 2018):

```
ΔΨ_e = (1/2)·c·k_B·T·[tr(μ) - 3 - ln det(μ)]
     = (1/2)·G₀·[tr(μ) - 3 - ln det(μ)]
```

This is the Neo-Hookean form. The quantity `[tr(μ) - 3 - ln det(μ)] ≥ 0` with equality only at `μ = I`.

### RheoJAX Implementation

**NOT IMPLEMENTED.** No `vlb_free_energy` function exists in `_kernels.py`.

**VERDICT: MISSING (not a bug — the free energy is not needed for stress/strain predictions, only for thermodynamic consistency checks).**

---

## 4. Dissipation

### Paper (from Eqs. 19-21 in JMPS 2018)

For Gaussian chains, the dissipation is:

```
D = G₀·k_d·[tr(μ) - 3 - ln det(μ)] ≥ 0
```

This is `2·k_d·ΔΨ_e`, confirming non-negative dissipation.

### RheoJAX Implementation

**NOT IMPLEMENTED.** No `vlb_dissipation` function exists.

**VERDICT: MISSING (same rationale as free energy — not needed for predictions).**

---

## 5. Simple Shear Flow Components

### Paper derivation

With `L = [[0, γ̇, 0], [0, 0, 0], [0, 0, 0]]` and `L^T = [[0, 0, 0], [γ̇, 0, 0], [0, 0, 0]]`:

```
(L·μ)_xx = γ̇·μ_xy,     (μ·L^T)_xx = μ_xy·γ̇     → 2γ̇·μ_xy
(L·μ)_yy = 0,           (μ·L^T)_yy = 0           → 0
(L·μ)_zz = 0,           (μ·L^T)_zz = 0           → 0
(L·μ)_xy = γ̇·μ_yy,     (μ·L^T)_xy = 0           → γ̇·μ_yy
```

Therefore:
```
dμ_xx/dt = k_d·(1 - μ_xx) + 2·γ̇·μ_xy
dμ_yy/dt = k_d·(1 - μ_yy)
dμ_zz/dt = k_d·(1 - μ_zz)
dμ_xy/dt = -k_d·μ_xy + γ̇·μ_yy
```

### RheoJAX Implementation (lines 127-131)

```python
dmu_xx = k_d * (1.0 - mu_xx) + 2.0 * gamma_dot * mu_xy
dmu_yy = k_d * (1.0 - mu_yy)
dmu_zz = k_d * (1.0 - mu_zz)
dmu_xy = -k_d * mu_xy + gamma_dot * mu_yy
```

**VERDICT: CORRECT.** Matches exactly. This is equivalent to the upper-convected Maxwell model.

---

## 6. Shear Stress and N1

### Paper (Gaussian limit)

```
σ_xy = G₀·(μ_xy - I_xy) = G₀·μ_xy      (since I_xy = 0)
N1 = G₀·(μ_xx - μ_yy)
```

### RheoJAX Implementation

```python
def vlb_shear_stress(mu_xy, G0):
    return G0 * mu_xy

def vlb_normal_stress_1(mu_xx, mu_yy, G0):
    return G0 * (mu_xx - mu_yy)
```

**VERDICT: CORRECT.**

---

## 7. Steady-State Shear

### Paper derivation

At steady state (`dμ/dt = 0`):
- `μ_yy = 1` (from `k_d·(1-μ_yy) = 0`)
- `μ_xy = γ̇·μ_yy/k_d = γ̇/k_d`
- `σ = G₀·γ̇/k_d = η₀·γ̇` where `η₀ = G₀/k_d`

This is Newtonian with zero-shear viscosity `η₀ = G₀/k_d`.

### RheoJAX Implementation

```python
def vlb_steady_shear(gamma_dot, G0, k_d):
    return G0 * gamma_dot / k_d
```

**VERDICT: CORRECT.** Newtonian response for constant `k_d`.

---

## 8. SAOS (Small Amplitude Oscillatory Shear)

### Paper derivation

The VLB model in the Gaussian limit reduces to the Maxwell model. For a Maxwell element with modulus G₀ and relaxation time τ = 1/k_d:

```
G'(ω) = G₀·ω²τ²/(1 + ω²τ²) = G₀·ω²/(ω² + k_d²)
G''(ω) = G₀·ωτ/(1 + ω²τ²) = G₀·ω·k_d/(ω² + k_d²)
```

### RheoJAX Implementation

```python
t_R = 1.0 / k_d
wt = omega * t_R
wt2 = wt * wt
denom = 1.0 + wt2
G_prime = G0 * wt2 / denom        # = G0·ω²/(ω² + k_d²)
G_double_prime = G0 * wt / denom   # = G0·ω·k_d/(ω² + k_d²)  [since wt/denom = ωτ/(1+ω²τ²)]
```

Wait — let me verify: `wt/denom = (ω/k_d)/(1 + ω²/k_d²) = ω·k_d/(k_d² + ω²)`.
So `G'' = G₀·ω·k_d/(ω² + k_d²)`. But we can also write this as `G₀·ω/(k_d + ω²/k_d)`.

The formula `G''(ω) = G₀·ω·k_d/(ω² + k_d²)` matches the standard Maxwell form.

**VERDICT: CORRECT.**

---

## 9. Stress Relaxation

### Paper

Single exponential decay (Maxwell element):

```
G(t) = G₀·exp(-k_d·t) = G₀·exp(-t/τ)
```

where `τ = 1/k_d`.

### RheoJAX Implementation

```python
def vlb_relaxation_modulus(t, G0, k_d):
    return G0 * jnp.exp(-k_d * t)
```

**VERDICT: CORRECT.**

---

## 10. Creep Compliance

### Analysis

For a Maxwell element (spring G₀ in series with dashpot η = G₀/k_d), the creep compliance under constant stress σ₀ is:

```
J(t) = 1/G₀ + t/η = 1/G₀ + k_d·t/G₀ = (1 + k_d·t)/G₀
```

This is **not** `J(t) = (1/G₀)(1 - exp(-k_d·t)) + t·k_d/G₀` — that would be the Kelvin-Voigt or SLS form.

The Maxwell creep: instantaneous elastic strain `1/G₀` followed by linear viscous flow `t/η`.

### RheoJAX Implementation

```python
def vlb_creep_compliance_single(t, G0, k_d):
    return (1.0 + k_d * t) / G0
```

**VERDICT: CORRECT.** This is the standard Maxwell creep compliance. The question mark in the user's query is resolved: `J(t) = (1 + k_d·t)/G₀` is correct for Maxwell. The exponential form would apply to a Kelvin-Voigt element or Standard Linear Solid, which is implemented separately as `vlb_creep_compliance_dual` for the case with a permanent elastic network (G_e > 0).

---

## 11. Bell Model Force-Dependent Dissociation

### Paper (Bell 1978, applied in Vernerey framework)

Bell (1978) proposed for bond dissociation under force:

```
k_off(f) = k_off_0 · exp(f·γ / k_B·T)
```

In the Vernerey VLB context, this is adapted using chain stretch rather than force directly. The mean chain stretch is `λ̄ = sqrt(tr(μ)/3)`, and the Bell-type relation becomes:

```
k_d(μ) = k_d_0 · exp(ν · (λ̄ - 1))
```

where `ν` absorbs the force sensitivity. At equilibrium (`tr(μ) = 3`, `λ̄ = 1`), `k_d = k_d_0`.

### RheoJAX Implementation

```python
def vlb_breakage_bell(mu_xx, mu_yy, mu_zz, k_d_0, nu):
    tr_mu = mu_xx + mu_yy + mu_zz
    stretch = jnp.sqrt(jnp.maximum(tr_mu / 3.0, 1e-30))
    return k_d_0 * jnp.exp(nu * (stretch - 1.0))
```

**VERDICT: CORRECT.** The stretch definition `√(tr(μ)/3)` and the Bell-type exponential are consistent with the VLB framework. The `1e-30` floor prevents sqrt(0) gradient issues.

---

## 12. FENE-P Finite Extensibility

### Paper context

The FENE-P (Peterlin) closure introduces a nonlinear spring factor that diverges as chain extension approaches its limit. For a distribution tensor with equilibrium `tr(μ) = 3`:

```
f(μ) = L²/(L² - (tr(μ) - 3))  = L²/(L² - tr(μ) + 3)
```

The stress becomes: `σ = G₀·f(μ)·(μ - I) + p·I`

At equilibrium: `f(tr=3) = L²/L² = 1` (recovers linear stress).
As `tr(μ) → L² + 3`: `f → ∞` (chains reach maximum extensibility).

### RheoJAX Implementation

```python
def vlb_fene_factor(mu_xx, mu_yy, mu_zz, L_max):
    tr_mu = mu_xx + mu_yy + mu_zz
    L2 = L_max * L_max
    return L2 / jnp.maximum(L2 - tr_mu + 3.0, 1e-10)

def vlb_stress_fene_xy(mu_xx, mu_yy, mu_zz, mu_xy, G0, L_max):
    f = vlb_fene_factor(mu_xx, mu_yy, mu_zz, L_max)
    return G0 * f * mu_xy
```

**VERDICT: CORRECT.** The offset of `+3` in the denominator correctly accounts for `tr(I) = 3`.

---

## Summary Table

| # | Equation | Paper Reference | RheoJAX | Status |
|---|----------|----------------|---------|--------|
| 1 | Distribution tensor evolution | JMPS 2018 Eq.17, Polymers 2018 Eq.3b | `_kernels.py:127-131` | CORRECT |
| 2 | Cauchy stress | JMPS 2018 Eq.28 (N→∞) | `_kernels.py:176-216` | CORRECT |
| 3 | Free energy | Polymers 2018 Eq.5 | — | NOT IMPLEMENTED |
| 4 | Dissipation | JMPS 2018 Eqs.19-21 | — | NOT IMPLEMENTED |
| 5 | Simple shear components | Derived from Eq.17 | `_kernels.py:127-131` | CORRECT |
| 6 | σ_xy and N1 | Gaussian limit of Eq.28 | `_kernels.py:176-216` | CORRECT |
| 7 | Steady-state shear | Polymers 2018 §3.2 | `_kernels.py:225-250` | CORRECT |
| 8 | SAOS G'(ω), G''(ω) | Maxwell limit | `_kernels.py:387-415` | CORRECT |
| 9 | Stress relaxation G(t) | Maxwell limit | `_kernels.py:357-378` | CORRECT |
| 10 | Creep compliance J(t) | Maxwell element | `_kernels.py:462-483` | CORRECT |
| 11 | Bell breakage | Bell (1978) + VLB | `_kernels.py:1055-1081` | CORRECT |
| 12 | FENE-P factor | FENE-P closure | `_kernels.py:1090-1115` | CORRECT |

## Issues Found

1. **Docstring inconsistency (minor):** Line 17 of `_kernels.py` states `dmu/dt = k_d*(I - mu) + D·mu + mu·D` using the symmetric part `D`, but the actual model uses the full velocity gradient `L·mu + mu·L^T`. The code is correct; only the docstring needs updating.

2. **Free energy and dissipation not implemented:** These are thermodynamic quantities useful for consistency checks but not required for mechanical predictions. Consider adding for completeness.

---

## Key Clarification: L vs D in the Evolution Equation

The VLB paper uses the **full velocity gradient** `L = ∇v`, not just its symmetric part `D = (L + L^T)/2`. This is critical because:

- With `L·μ + μ·L^T`: the model is equivalent to the **upper-convected Maxwell** model
- With `D·μ + μ·D`: the model would be the **corotational** form

The implementation correctly uses `L·μ + μ·L^T` (verified by the simple shear components where `dμ_yy/dt` has NO deformation contribution, which is only true with the full `L`).
