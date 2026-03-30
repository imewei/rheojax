# IKH Model: Literature Verification Report

**Date:** 2026-03-29
**Purpose:** Verify RheoJAX IKH implementation against original literature

---

## 1. Primary References

### 1A. Dimitriou & McKinley (2014) — The Original IKH Model
- **Title:** "A comprehensive constitutive law for waxy crude oil: a thixotropic yield stress fluid"
- **Journal:** Soft Matter, 2014, **10**, 6619-6644
- **DOI:** 10.1039/C4SM00578C (Open Access, CC-BY 3.0)
- **URL:** https://pubs.rsc.org/en/content/articlehtml/2014/sm/c4sm00578c
- **PDF:** https://dspace.mit.edu/bitstream/handle/1721.1/96895/Dimitriou-2014-Comprehensive%20constitutive.pdf

### 1B. Dimitriou PhD Thesis (2013)
- **Title:** "The rheological complexity of waxy crude oils: Yielding, thixotropy and shear heterogeneities"
- **Institution:** MIT, Department of Mechanical Engineering, 2013
- **URL:** https://dspace.mit.edu/handle/1721.1/81696

### 1C. Wei, Solomon & Larson (2018) — The ML-IKH Extension
- **Title:** "A multimode structural kinetics constitutive equation for the transient rheology of thixotropic elasto-viscoplastic fluids"
- **Journal:** Journal of Rheology, 2018, **62**(1), 321-342
- **URL:** https://pubs.aip.org/sor/jor/article/62/1/321/241840/

### 1D. Armstrong & Frederick (1966) — Back-Stress Evolution Origin
- **Title:** "A mathematical representation of the multiaxial Bauschinger effect"
- **Report:** CEGB Report RD/B/N731, Berkeley Nuclear Laboratories (1966)
- **Note:** Originally a technical report, later widely cited. The key equation for nonlinear kinematic hardening.

### 1E. Fraggedakis, Dimakopoulos & Tsamopoulos (2016)
- **Title:** "Yielding the yield stress analysis: A thorough comparison of recently proposed elasto-visco-plastic (EVP) fluid models"
- **Journal:** J. Non-Newtonian Fluid Mechanics, 2016, **236**, 104-122

### 1F. Larson & Wei (2019) — Thixotropic Review
- **Title:** "Modeling the rheology of thixotropic elasto-visco-plastic materials"
- **Journal:** Journal of Rheology, 2019, **63**(4), 5049136
- **URL:** https://sor.scitation.org/doi/10.1122/1.5049136

---

## 2. Original IKH Equations (Dimitriou & McKinley 2014)

### 2.1 Strain Decomposition — Eq. (13)
```
gamma = gamma^ve + gamma^p
```
where `gamma^ve = gamma^v + gamma^e` (viscoelastic = viscous + elastic for Maxwell sub-element).

**RheoJAX:** Implicit in the return mapping algorithm. The elastic predictor step assumes `d_gamma = d_gamma_e + d_gamma_p`, consistent with the paper.

### 2.2 Flow Rule — Eq. (14)-(15)
The original Bingham-like equation (Eq. 14):
```
gamma_dot^p = (1/mu_p) * (|sigma| - sigma_y) * n_p,  when |sigma| > sigma_y
gamma_dot^p = 0,                                       when |sigma| <= sigma_y
```
where `n_p = sigma/|sigma|` is the direction (codirectionality hypothesis).

Modified with backstress (Eq. 15-17):
```
gamma_dot^p = (1/mu_p) * (|sigma_eff| - sigma_y(lambda)) * n_eff,  when |sigma_eff| > sigma_y(lambda)   [Eq. 15]
sigma_eff = sigma - sigma_back                                                                           [Eq. 16]
sigma_y = sigma_y(lambda)  (function of structure)                                                        [Eq. 17]
```
where `n_eff = sigma_eff/|sigma_eff|` is the direction of the effective stress.

**RheoJAX (_kernels.py line 126-137):**
```python
xi = sigma - alpha          # xi = sigma_eff (Eq. 16)
f_yield = xi_abs - sigma_y  # yield function f = |sigma_eff| - sigma_y
gamma_dot_p = macaulay(f_yield) / mu_p * sign_xi  # Eq. 15 with Macaulay brackets
```
**MATCH:** Correct. Uses `xi = sigma - alpha` where alpha = sigma_back, and Macaulay brackets `<f>` enforce the yield condition.

### 2.3 Back Stress — Eq. (18)
```
sigma_back = C * A                   [Eq. 18]
```
where C is the back stress modulus (Pa) and A is the back strain (dimensionless).

### 2.4 Back Strain Evolution (Armstrong-Frederick) — Eq. (19)-(20)
General form (Eq. 19):
```
dA/dt = gamma_dot^p - f(A) * |gamma_dot^p|
```
where `f(A)` is the dynamic recovery function.

Generalized form (Eq. 20):
```
f(A) = (q|A|)^m * sign(A)
```
so the full equation becomes:
```
dA/dt = gamma_dot^p - (q|A|)^m * sign(A) * |gamma_dot^p|      [Eq. 19+20]
```

When m=1 (classical Armstrong-Frederick):
```
dA/dt = gamma_dot^p - q*A*|gamma_dot^p|
```

**Converting to backstress (sigma_back = C*A, so A = sigma_back/C):**
```
d(sigma_back)/dt = C * dA/dt
                 = C * gamma_dot^p - C * (q|A|)^m * sign(A) * |gamma_dot^p|
                 = C * gamma_dot^p - C * (q|sigma_back/C|)^m * sign(sigma_back) * |gamma_dot^p|
```

For m=1 this simplifies to:
```
d(sigma_back)/dt = C * gamma_dot^p - q * sigma_back * |gamma_dot^p|
```

**RheoJAX (_kernels.py lines 147-152):**
```python
# dα/dt = C·γ̇ᵖ - γ_dyn·|α|^(m-1)·α·|γ̇ᵖ|
recovery_term = gamma_dyn * jnp.power(alpha_abs + 1e-20, m - 1) * alpha * gamma_dot_p_abs
d_alpha = C * gamma_dot_p - recovery_term
```

**DISCREPANCY FOUND:** The paper uses `f(A) = (q|A|)^m * sign(A)`, which expands to:
```
q^m * |A|^m * sign(A) * |gamma_dot^p|
```

RheoJAX implements:
```
gamma_dyn * |alpha|^(m-1) * alpha * |gamma_dot^p|
```
Note that `|alpha|^(m-1) * alpha = |alpha|^(m-1) * |alpha| * sign(alpha) = |alpha|^m * sign(alpha)`.

So RheoJAX computes: `gamma_dyn * |alpha|^m * sign(alpha) * |gamma_dot^p|`

**But** `alpha = sigma_back` in RheoJAX (not `A = sigma_back/C`). So this is:
```
gamma_dyn * |sigma_back|^m * sign(sigma_back) * |gamma_dot^p|
```

The paper's version in terms of sigma_back (where A = sigma_back/C):
```
C * (q|sigma_back/C|)^m * sign(sigma_back) * |gamma_dot^p|
= C * (q/C)^m * |sigma_back|^m * sign(sigma_back) * |gamma_dot^p|
```

So the mapping is: `gamma_dyn = C * (q/C)^m = q^m * C^(1-m)`.

For m=1: `gamma_dyn = q` (exact match).
For m!=1: `gamma_dyn = q^m * C^(1-m)` (different parameterization but equivalent if parameters are fit accordingly).

**VERDICT:** The implementation is mathematically equivalent for m=1 (the classical case). For m!=1, the parameterization differs from the paper but is equivalent if `gamma_dyn` is interpreted as `q^m * C^(1-m)` rather than `q` directly. This is a valid reparameterization but should be documented.

### 2.5 Yield Stress (Isotropic Hardening) — Eq. (21)
```
sigma_y(lambda) = sigma_y,0 + k3 * lambda        [Eq. 21]
```
where:
- `sigma_y,0`: minimum yield stress (fully destructured, lambda=0)
- `k3`: structure-dependent yield stress increment (Pa)
- `lambda`: structural parameter (0 to 1)

**RheoJAX (_kernels.py line 124):**
```python
sigma_y = sigma_y0 + delta_sigma_y * lam
```
where `delta_sigma_y` = k3 in the paper.

**MATCH:** Correct. `delta_sigma_y` is equivalent to k3.

### 2.6 Structure Evolution — Eq. (22)
```
d(lambda)/dt = k1*(1 - lambda) - k2*lambda*|gamma_dot^p|     [Eq. 22]
```
where:
- k1 (s^-1): buildup rate constant (sets aging timescale tw ~ 1/k1)
- k2 (dimensionless): breakdown coefficient
- Only **plastic** strain rate drives breakdown (key distinction from other thixotropy models)

**RheoJAX (_kernels.py lines 56-73):**
```python
# d(lambda)/dt = (1 - lambda)/tau_thix - Gamma*lambda*|gamma_dot_p|
build_up = k1 * (1.0 - lam)
break_down = k2 * lam * gamma_dot_p_abs
return build_up - break_down
```
with `k1 = 1/tau_thix` and `k2 = Gamma`.

**MATCH:** Correct. The reparameterization `k1 = 1/tau_thix`, `k2 = Gamma` is documented and equivalent.

### 2.7 MIKH Stress Evolution — Eq. (26)
For the Maxwell variant (MIKH), with `gamma^ve = gamma^v + gamma^e`:
```
d(sigma)/dt = G*(gamma_dot - gamma_dot^p) - (G/eta)*sigma     [Eq. 26]
```
This combines elastic loading `G*gamma_dot_e` with Maxwell relaxation `-(G/eta)*sigma`.

**RheoJAX (_kernels.py line 144):**
```python
d_sigma = G * gamma_dot_e - (G / eta) * sigma
```

**MATCH:** Exact match.

### 2.8 Steady-State Flow Curve — Eq. (28)
At steady state (dA/dt=0, dlambda/dt=0):
```
lambda_ss = k1/(k1 + k2*|gamma_dot|)                         [from Eq. 22 at steady state]
sigma_y_ss = sigma_y,0 + k3*lambda_ss                         [Eq. 21]
sigma = sigma_back_ss + sigma_y_ss + mu_p*|gamma_dot|          [from Eq. 15 at SS]
```
where `sigma_back_ss = C/q * sign(gamma_dot)` (back stress saturates at C/q).

**RheoJAX (_kernels.py lines 1068-1109):**
```python
lambda_ss = k1 / (k1 + k2 * gamma_dot_abs + 1e-20)
sigma_y_ss = sigma_y0 + delta_sigma_y * lambda_ss
sigma = sigma_y_ss + eta_inf * gamma_dot_abs
```

**POTENTIAL ISSUE:** The steady-state flow curve in RheoJAX does NOT include the backstress saturation term `C/q`. The original paper's Eq. (28) includes:
```
sigma = C/q + sigma_y,0 + k3*lambda_ss + mu_p*|gamma_dot|
```
RheoJAX uses `eta_inf` instead of `mu_p` and omits `C/q`. This may be intentional if the model assumes backstress is absorbed into sigma_y0 at steady state, but it should be verified.

### 2.9 Nine Model Parameters
The MIKH variant has 9 fitting parameters (from Sec 4.1):
```
G, eta, mu_p, k1, k2, k3, C, q, m
```

Representative values from the paper (for model waxy crude):
- C/q = 0.85 Pa
- mu_p = 0.42 Pa s
- k3 = 1.5 Pa
- k1/k2 = 0.033 s^-1
- eta = 500 Pa s
- G = 250 Pa
- k1 = 0.1 s^-1
- C = 70 Pa
- m = 0.25

**RheoJAX (mikh.py):** Parameters are: G, eta, C, gamma_dyn (=q for m=1), m, sigma_y0, delta_sigma_y (=k3), tau_thix (=1/k1), Gamma (=k2), eta_inf, mu_p — **11 parameters** vs 9 in the paper. The extra parameters are eta_inf (background viscosity, distinct from mu_p) and sigma_y0 (the paper implicitly includes this as part of the steady-state balance).

---

## 3. ML-IKH Extension (Wei, Solomon & Larson 2018)

Key additions over the base IKH:
1. **Multiple structure parameters** lambda_1, ..., lambda_N with different timescales
2. **Stretched exponential** thixotropic relaxation from multi-lambda superposition
3. **Per-mode** tau_thix_i and Gamma_i for each structure mode
4. **Two architectures:**
   - Per-mode yield surfaces (each mode has its own sigma_y_i)
   - Weighted-sum yield surface: `sigma_y = sigma_y0 + k3 * sum(w_i * lambda_i)`

**RheoJAX (ml_ikh.py, _kernels.py):** Implements BOTH architectures:
- `ml_ikh_scan_kernel` / `ml_ikh_maxwell_ode_rhs_per_mode` — per-mode yield surfaces
- `ml_ikh_weighted_sum_kernel` / `ml_ikh_maxwell_ode_rhs_weighted_sum` — weighted-sum yield surface

**MATCH:** Both architectures match the Wei et al. (2018) formulation.

---

## 4. Return Mapping Algorithm

The paper does not specify a particular numerical algorithm; it presents the continuous-time ODEs. The implementation in RheoJAX uses a **radial return mapping** algorithm (standard in computational plasticity) which is an implicit integration scheme:

1. **Elastic predictor:** `sigma_trial = sigma_n + G * d_gamma`
2. **Yield check:** `f = |sigma_trial - alpha_n| - sigma_y(lambda_n)`
3. **Plastic corrector:** If f > 0, compute `d_gamma_p = f / (G + C - AF_correction)` (includes Armstrong-Frederick correction in denominator)
4. **Update** sigma, alpha, lambda

The AF correction in the denominator (`G + C - gamma_dyn*|alpha|^(m-1)*sign(xi)*alpha`) is a linearization of the backstress evolution, which is standard for return mapping with nonlinear kinematic hardening.

**Key implementation detail (line 774-778):** Lambda is updated AFTER the stress calculation, using the plastic strain rate. The code comments note this as a "timing fix" — this is consistent with the operator-splitting approach where structure updates lag by one timestep.

---

## 5. Summary of Verification

| Equation | Paper Reference | RheoJAX | Status |
|---|---|---|---|
| Strain decomposition | Eq. (13) | Implicit in return mapping | MATCH |
| Flow rule (Perzyna) | Eq. (15) | `_kernels.py:136` | MATCH |
| Effective stress | Eq. (16) | `_kernels.py:128` | MATCH |
| Yield stress | Eq. (21) | `_kernels.py:124` | MATCH (k3 = delta_sigma_y) |
| Back stress | Eq. (18) | Absorbed into alpha directly | REPARAMETERIZED |
| Back strain evolution | Eq. (19)+(20) | `_kernels.py:147-152` | MATCH for m=1; reparameterized for m!=1 |
| Structure evolution | Eq. (22) | `_kernels.py:56-73` | MATCH (k1=1/tau_thix, k2=Gamma) |
| MIKH stress rate | Eq. (26) | `_kernels.py:144` | MATCH |
| Steady-state flow curve | Eq. (28) | `_kernels.py:1068-1109` | MISSING backstress saturation C/q |
| ML-IKH per-mode | Wei et al. (2018) | `_kernels.py:246-347` | MATCH |
| ML-IKH weighted-sum | Wei et al. (2018) | `_kernels.py:376-441` | MATCH |

### Issues Found

1. **Backstress parameterization for m!=1:** RheoJAX uses `gamma_dyn*|alpha|^m*sign(alpha)` where alpha IS the backstress. The paper uses `(q|A|)^m*sign(A)` where A is the back STRAIN and `sigma_back = C*A`. For m=1 these are equivalent with `gamma_dyn=q`. For m!=1, the relationship is `gamma_dyn = q^m * C^(1-m)`. This should be documented.

2. **Steady-state flow curve missing C/q term:** The function `ikh_flow_curve_steady_state` does not include the backstress saturation contribution `C/q` (or `C/gamma_dyn` in RheoJAX parameterization for m=1). At steady state under unidirectional flow, the backstress saturates and contributes to the total stress.

3. **Parameter naming:** The paper uses `mu_p` for plastic viscosity in the flow rule AND as the high-shear viscosity. RheoJAX separates these into `mu_p` (regularization) and `eta_inf` (background viscosity), which is physically cleaner but differs from the original parameterization.

---

## 6. Armstrong-Frederick Equation (Original)

The original Armstrong & Frederick (1966) equation in metals plasticity:
```
d(chi)/dt = (2/3)*C_1*d(epsilon^p)/dt - gamma_1*chi*d(p)/dt
```
where chi is the backstress tensor, epsilon^p is plastic strain, p is accumulated plastic strain, C_1 and gamma_1 are material constants.

In 1D scalar form:
```
d(alpha)/dt = C*gamma_dot^p - gamma*alpha*|gamma_dot^p|
```

This is EXACTLY what the IKH model uses when m=1 and the back-stress form is adopted directly (i.e., working with sigma_back rather than back-strain A).

**RheoJAX:** For m=1, `d_alpha = C*gamma_dot_p - gamma_dyn*alpha*|gamma_dot_p|` — this is the classical Armstrong-Frederick equation.

---

## 7. Fraggedakis et al. Context

Fraggedakis, Dimakopoulos & Tsamopoulos (2016, 2019) extended EVP models with kinematic hardening in a tensorial framework (Saramito-type models). Their work is complementary but uses a different base model (Oldroyd-B/Saramito rather than Maxwell+Bingham). The IKH model in RheoJAX follows the Dimitriou-McKinley formulation, not the Fraggedakis formulation.
