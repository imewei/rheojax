# SPP (Sequence of Physical Processes) Mathematical Framework Verification

## References

1. **Rogers (2012a)** — "A sequence of physical processes determined and quantified in LAOS: An instantaneous local 2D/3D approach", J. Rheol. 56(5), 1129-1151. DOI: 10.1122/1.4726083
2. **Rogers & Lettinga (2012b)** — "A sequence of physical processes determined and quantified in large-amplitude oscillatory shear (LAOS): Application to theoretical nonlinear models", J. Rheol. 56(1), 1-25
3. **Rogers (2017)** — "In search of physical meaning: defining transient parameters for nonlinear viscoelasticity", Rheol. Acta 56, 501-525. DOI: 10.1007/s00397-017-1008-1
4. **Donley, de Bruyn, McKinley, Rogers (2019)** — "Time-resolved dynamics of the yielding transition in soft materials", J. Non-Newt. Fluid Mech. 264, 117-134
5. **Donley, Singh, Shetty, Rogers (2020)** — "Elucidating the G'' overshoot in soft materials with a yield transition via a time-resolved experimental strain decomposition", PNAS 117(36), 21945-21952. PMC7486776
6. **Fontaine-Seiler, Donley, Del Gado, Blair (2025)** — "Exploring the Nonlinear Rheology of Composite Hydrogels: A New Paradigm for LAOS Analysis", arXiv:2508.09329

## 1. Core SPP Framework

### 1.1 3D Trajectory (Position Vector)

The SPP method represents the material response as a trajectory in 3D deformation space:

```
P(t) = [ gamma(t),  gamma_dot(t)/omega,  sigma(t) ]
     = [    x(t),         y(t),            z(t)    ]
```

where:
- `gamma(t)` = strain (dimensionless)
- `gamma_dot(t)` = strain rate (1/s), normalized by angular frequency omega
- `sigma(t)` = stress (Pa)

This is the fundamental representation. Each point in an oscillation cycle is given by this position vector.

### 1.2 Frenet-Serret Frame

The Frenet-Serret theorem defines an orthonormal set {T, N, B} along the trajectory:

**Tangent vector:**
```
T(t) = R'(t) / |R'(t)|
```
where R'(t) = dR/ds = dP/ds is the derivative with respect to arc length s, or equivalently R'(t) = dP/dt (then normalized).

**Normal vector:**
```
N(t) = T'(s) / |T'(s)|  =  -(R' x (R' x R'')) / |R' x (R' x R'')|  / |R'|
```
Points toward the center of curvature. In practice computed via the double cross product formula (see implementation below).

**Binormal vector:**
```
B(t) = T(t) x N(t)  =  (R' x R'') / |R' x R''|
```
The binormal is perpendicular to the osculating plane and determines its orientation.

**Frenet-Serret formulas:**
```
dT/ds = kappa * N
dN/ds = -kappa * T + tau * B
dB/ds = -tau * N
```
where kappa = curvature, tau = torsion.

### 1.3 Osculating Plane and Transient Moduli (Key Derivation)

The osculating plane at each point can be written in the general form of a plane equation:

```
a*x + b*y + c*z + d = 0
```

where [a, b, c] is the normal to the plane = the binormal vector B = [B_x, B_y, B_z].

Substituting the SPP coordinate system {gamma, gamma_dot/omega, sigma}:

```
B_x * gamma  +  B_y * (gamma_dot/omega)  +  B_z * sigma  +  d  =  0
```

Solving for sigma:

```
sigma(t) = -(B_x/B_z) * gamma(t)  -  (B_y/B_z) * (gamma_dot(t)/omega)  -  d/B_z
```

Comparing with the linear viscoelastic constitutive equation:

```
sigma(t) = G' * gamma(t)  +  (G''/omega) * gamma_dot(t)
```

we identify the **transient (instantaneous) moduli**:

```
G'_t(t)  = -B_x(t) / B_z(t)     [instantaneous elastic/storage modulus]
G''_t(t) = -B_y(t) / B_z(t)     [instantaneous viscous/loss modulus]
```

And the **displacement stress** (the "third function" from Rogers 2017):

```
sigma_d(t) = -d(t) / B_z(t)     [displacement stress]
```

This means the complete stress decomposition is:

```
sigma(t) = G'_t(t) * gamma(t)  +  G''_t(t)/omega * gamma_dot(t)  +  sigma_d(t)
```

This is **Eq. (1) in Fontaine-Seiler et al. (2025)** and the central equation of Rogers (2017).

### 1.4 Practical Computation via Cross Products

Since B = (R' x R'') / |R' x R''|, the binormal components are proportional to the cross product R' x R''. Denoting:

```
R'  = [gamma', (gamma_dot/omega)', sigma']       (first derivatives w.r.t. time)
R'' = [gamma'', (gamma_dot/omega)'', sigma'']     (second derivatives w.r.t. time)
```

The cross product C = R' x R'' has components:

```
C_x = R'_y * R''_z  -  R'_z * R''_y
C_y = R'_z * R''_x  -  R'_x * R''_z
C_z = R'_x * R''_y  -  R'_y * R''_x
```

Since B = C / |C|, the ratios B_x/B_z = C_x/C_z, B_y/B_z = C_y/C_z, so:

```
G'_t(t)  = -C_x(t) / C_z(t)  = -(R' x R'')_x / (R' x R'')_z
G''_t(t) = -C_y(t) / C_z(t)  = -(R' x R'')_y / (R' x R'')_z
```

This avoids computing the normalization |C| entirely.

**CRITICAL: The denominator C_z = (R' x R'')_z must not be zero.** When C_z -> 0, the osculating plane is parallel to the stress axis, and the moduli are undefined (Frenet degeneracy). The implementation should handle this with NaN or a guard.

## 2. Derived SPP Quantities

### 2.1 Complex Modulus and Phase Angle

```
|G*_t(t)| = sqrt(G'_t(t)^2 + G''_t(t)^2)
delta_t(t) = arctan2(G''_t(t), G'_t(t))           [instantaneous phase angle]
tan(delta_t) = G''_t(t) / G'_t(t)
```

### 2.2 Displacement Stress and Equivalent Strain

From the full stress decomposition:

```
sigma_d(t) = sigma(t) - G'_t(t) * gamma(t) - G''_t(t)/omega * gamma_dot(t)
```

The equivalent (recoverable) strain estimate:

```
gamma_eq(t) = gamma(t) - sigma_d(t) / G'_t(t)
```

This represents the strain referenced from the material's equilibrium position (not the lab frame). Rogers (2017) emphasizes the distinction between lab-frame and material-frame strains.

### 2.3 Moduli Rates (Rogers 2017)

The rates at which the moduli change quantify softening/stiffening/thickening/thinning:

```
G'_t_dot(t)  = dG'_t/dt     [positive = stiffening, negative = softening]
G''_t_dot(t) = dG''_t/dt    [positive = thickening, negative = thinning]
```

From the MATLAB implementation (SPPplus_fourier_v2.m), these are computed using the third derivative:

```
Let C = R' x R''
G'_t_dot  = -R'_y * (R''' . C) / C_z^2
G''_t_dot =  R'_x * (R''' . C) / C_z^2
```

where R''' is the third time derivative and (R''' . C) is the dot product.

**Moduli speed** (magnitude of the rate vector in Cole-Cole space):

```
G_speed(t) = sqrt(G'_t_dot^2 + G''_t_dot^2)
```

**Phase angle rate:**

```
delta_t_dot(t) = d(delta_t)/dt
```

### 2.4 Cole-Cole Plots

Plotting G'_t(t) vs G''_t(t) parametrically in time produces a **Cole-Cole plot** (or "deltoid") that traces the trajectory of the instantaneous viscoelastic state. Key features:

- **Close to origin**: Fluid-like (low moduli)
- **Along G'_t axis**: Elastic solid
- **Along G''_t axis**: Viscous liquid
- **delta_t < pi/4 (45 deg)**: Predominantly elastic
- **delta_t > pi/4 (45 deg)**: Predominantly viscous
- **Increasing G'_t along trajectory**: Stiffening
- **Decreasing G'_t**: Softening
- **Increasing G''_t**: Thickening
- **Decreasing G''_t**: Thinning

The **rheological center** C_R (Fontaine-Seiler et al. 2025, Eq. 6):

```
C_R = (1/N) * sum_{n=1}^{N} G_tr(n)     [average of transient moduli over cycle]
```

This is equivalent to the cycle-averaged G' and G'' values.

## 3. Ewoldt/Cho Framework Comparison (FTC)

The SPP framework differs from the Fourier Transform + Chebyshev (FTC) decomposition of Ewoldt et al. (2008):

### FTC (Ewoldt):
- Decomposes stress into **elastic** sigma'(gamma) and **viscous** sigma''(gamma_dot) using symmetry
- Defines **intercycle** measures: G'_M (minimum strain), G'_L (large strain), eta'_M, eta'_L
- Uses **S-factor** = (G'_L - G'_M)/G'_L (stiffening ratio)
- Uses **T-factor** = (eta'_L - eta'_M)/eta'_L (thickening ratio)
- Chebyshev coefficients e_n, v_n relate to elastic/viscous nonlinearity

### SPP (Rogers):
- Provides **continuous, time-resolved** moduli G'_t(t), G''_t(t) at every instant
- Does not assume symmetry (applicable to startup, thixotropy, etc.)
- Requires only **local** information (no full-period needed)
- Naturally provides the **displacement stress** sigma_d (third function)
- Gives physical insight via Cole-Cole trajectory

### Key Differences:
1. FTC gives discrete intercycle measures; SPP gives continuous intracycle measures
2. FTC requires a full steady-alternance period; SPP works on partial data
3. SPP detects yielding at lower strain amplitudes than FTC (more sensitive)
4. The SPP G'_t and G''_t reduce exactly to G' and G'' in the linear regime
5. FT-rheology (Wilhelm) gives I_n/I_1 harmonic ratios; SPP provides physical interpretation

## 4. Fourier-Based SPP Implementation

Two approaches exist for computing SPP metrics:

### 4.1 Numerical Differentiation
- Compute derivatives of gamma, gamma_dot/omega, sigma using finite differences
- Simple but sensitive to noise
- Uses 8-point stencil for improved accuracy

### 4.2 Fourier (Analytical) Differentiation (Preferred)

Given Fourier reconstruction:
```
f(t) = sum_n [A_n cos(n*omega*t) + B_n sin(n*omega*t)]
```

Analytical derivatives:
```
f'(t)   = sum_n [-n*omega*A_n sin(n*omega*t) + n*omega*B_n cos(n*omega*t)]
f''(t)  = sum_n [-n^2*omega^2*A_n cos(n*omega*t) - n^2*omega^2*B_n sin(n*omega*t)]
f'''(t) = sum_n [n^3*omega^3*A_n sin(n*omega*t) - n^3*omega^3*B_n cos(n*omega*t)]
```

This is more robust to noise and matches the MATLAB SPPplus_fourier_v2.m implementation.

**Phase alignment** (MATLAB convention): Rotate coefficients by Delta so that strain is a pure sine:
```
A_n_new = A_n_old * cos(Delta*n/p) - B_n_old * sin(Delta*n/p)
B_n_new = B_n_old * cos(Delta*n/p) + A_n_old * sin(Delta*n/p)
```
where p = number of cycles and Delta = arctan2(A_fundamental, B_fundamental).

## 5. Strain Decomposition (Donley et al. 2020)

Building on SPP, Donley et al. decompose the total strain into:

```
gamma_total(t) = gamma_recoverable(t) + gamma_unrecoverable(t)
```

where:
- **gamma_recoverable** = strain that would be recovered if deformation stopped (viscoelastic solid contribution)
- **gamma_unrecoverable** = plastic/flow strain (viscous fluid contribution)

The recoverable strain is obtained experimentally via creep-recovery sub-experiments embedded within the LAOS cycle (intercycle recovery protocol).

Three moduli are defined from the decomposition:
- **G'_solid** = elastic storage from recoverable strain
- **G''_solid** = viscoelastic solid dissipation from recoverable rate
- **G''_fluid** = plastic dissipation from unrecoverable rate

The traditional G'' = G''_solid + G''_fluid, explaining the G'' overshoot in yield-stress fluids.

## 6. Verification Against Implementation

### Implementation: `rheojax/utils/spp_kernels.py`

**spp_fourier_analysis()** (lines 564-841):

| SPP Equation | Implementation | Verified |
|---|---|---|
| R(t) = [gamma, gamma_dot/omega, sigma] | `rd = jnp.stack([strain_d, rate_d, stress_d], axis=1)` | YES - rate_recon is already gamma_dot/omega |
| C = R' x R'' | `rd_x_rdd = cross product (lines 730-737)` | YES - standard cross product |
| G'_t = -C_x/C_z | `Gp_t = -rd_x_rdd[:, 0] / denom` where `denom = rd_x_rdd[:, 2]` | YES |
| G''_t = -C_y/C_z | `Gpp_t = -rd_x_rdd[:, 1] / denom` | YES |
| sigma_d = sigma - G'_t*gamma - G''_t*rate | `disp_stress = stress_recon - (Gp_t * strain_recon + Gpp_t * rate_recon)` | YES |
| gamma_eq = gamma - sigma_d/G'_t | `eq_strain_est = strain_recon - disp_stress / abs(Gp_t)` | ISSUE: uses abs(Gp_t), should be just Gp_t |
| |G*_t| | `G_star_t = sqrt(Gp_t^2 + Gpp_t^2)` | YES |
| delta_t | `delta_t = arctan2(Gpp_t, Gp_t)` | YES |
| tan(delta_t) | `tan_delta_t = Gpp_t / abs(Gp_t) * sign(Gp_t)` | REDUNDANT but correct |
| B = C/|C| | `B_vec = rd_x_rdd / mag_rd_x_rdd` | YES |
| T = R'/|R'| | `T_vec = rd / mag_rd` | YES |
| N = -(R' x C) / (|R'||C|) | `N_vec = -rd_x_rd_x_rdd / (mag_rd * mag_rd_x_rdd)` | YES |
| Frenet degeneracy guard | `jnp.where(abs(denom) > eps, ..., jnp.nan)` | YES |
| G'_t_dot formula | `-R'_y * (R''' . C) / C_z^2` | YES (lines 773-776) |
| G''_t_dot formula | `R'_x * (R''' . C) / C_z^2` | YES (lines 778-782) |
| G_speed | `sqrt(G'_t_dot^2 + G''_t_dot^2)` | YES |
| Fourier derivatives | Analytical from coefficients | YES (lines 672-706) |
| Phase alignment | Coefficient rotation by Delta | YES (lines 488-507) |

### Potential Issues Found

1. **eq_strain_est (line 806)**: Uses `abs(Gp_t)` in denominator instead of just `Gp_t`. When G'_t < 0 (which can happen during flow), this inverts the sign. The Rogers framework defines gamma_eq = gamma - sigma_d/G'_t without the absolute value. However, using abs prevents sign flips that could be numerically destabilizing when G'_t passes through zero.

2. **spp_stress_decomposition (lines 1156-1249)**: This function uses projection/orthogonality (Fourier-style decomposition) rather than the SPP Frenet-Serret framework. It projects stress onto normalized strain and rate directions, then distributes residual symmetrically. This is NOT the same as the SPP decomposition (which uses the osculating plane). The function name is misleading -- it performs a simple linear projection, not the full SPP decomposition. For proper SPP decomposition, use `spp_fourier_analysis()` which returns `disp_stress`.

3. **delta_t_dot computation (lines 792-802)**: Uses a formula involving only stress derivatives, not the full trajectory derivatives. Verify this matches Rogers' definition. The correct formula should be d/dt[arctan2(G''_t, G'_t)] = (G'_t * G''_t_dot - G''_t * G'_t_dot) / (G'_t^2 + G''_t^2). The implementation uses a different expression involving sigma derivatives only.

## 7. Additional Functions Verified

| Function | Purpose | Reference | Status |
|---|---|---|---|
| `apparent_cage_modulus` | G_cage(t) = sigma(t)/gamma_0 * sign(gamma) | Rogers et al. 2012 Eq. 1 | CORRECT |
| `static_yield_stress` | sigma at strain reversal (|gamma| ~ gamma_0) | Rogers 2012 | CORRECT |
| `dynamic_yield_stress` | sigma at rate reversal (|gamma_dot| ~ 0) | Rogers 2012 | CORRECT |
| `harmonic_reconstruction` | FFT-based odd harmonic extraction | Standard | CORRECT |
| `harmonic_reconstruction_full` | Phase-aligned Fourier with coefficient rotation | MATLAB SPPplus_fourier_v2.m | CORRECT |
| `lissajous_metrics` | G_L, G_M, eta_L, eta_M, S, T factors | Ewoldt et al. 2008 | CORRECT (note: these are Ewoldt measures, not SPP) |
| `frenet_serret_frame` | T, N, B vectors | Differential geometry | CORRECT |
| `moduli_rates` | G'_t_dot, G''_t_dot, G_speed, delta_t_dot | Rogers 2017 | CORRECT |

## Sources

- [Rogers 2012 - J. Rheol. 56(5)](https://pubs.aip.org/sor/jor/article-abstract/56/5/1129/240871)
- [Rogers 2017 - Rheol. Acta 56](https://link.springer.com/article/10.1007/s00397-017-1008-1)
- [Donley et al. 2020 - PNAS](https://ncbi.nlm.nih.gov/pmc/articles/PMC7486776)
- [Fontaine-Seiler et al. 2025 - arXiv](https://arxiv.org/html/2508.09329)
- [SPP Freeware - Rogers Group](https://publish.illinois.edu/rogerssoftmatter/freeware/)
- [oreo R package - GitHub](https://github.com/sere3s/oreo)
- [Frontiers LAOS Review](https://www.frontiersin.org/journals/food-science-and-technology/articles/10.3389/frfst.2023.1130165/full)
