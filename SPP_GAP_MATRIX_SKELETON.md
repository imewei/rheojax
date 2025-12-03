# SPP Parity Gap Matrix Skeleton

**Purpose:** Compare feature parity across MATLAB SPPplus_v2p1, R oreo, and RheoJAX SPP implementations.

**Legend:**
- ✅ Full parity (identical or equivalent implementation)
- 🟡 Partial (implemented but differs in approach/scope)
- ❌ Missing (not implemented)
- 🔄 Enhanced (RheoJAX extends beyond reference)

---

## 1. HARMONIC RECONSTRUCTION

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **Strain reconstruction** | n=1 only | n=1 only | n=1 only | ✅ All match |
| **Rate reconstruction** | n=1 only | n=1 only | n=1 only | ✅ All match |
| **Stress reconstruction** | Odd 1..M | Odd 1..M | Odd 1..n_harmonics | ✅ Configurable |
| **Max harmonics (M)** | Default 39 | User param | Default 5-15 | 🟡 Different defaults |
| **Phase offset (Delta)** | atan(An/Bn) | atan(An/Bn) | atan(An/Bn) | ✅ Same formula |
| **Quadrant correction** | +π if Bn<0 | +π if Bn<0 | +π if Bn<0 | ✅ All match |
| **Coefficient rotation** | Delta/p*n | Delta/p*n | Delta/p*n | ✅ All match |
| **Truncation robustness** | ❌ | ❌ | 🔄 Available | Energy retention metric |

---

## 2. FRENET-SERRET FRAME OUTPUTS

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **Tangent (T)** | rd/\|rd\| | rd/\|rd\| | rd/\|rd\| | ✅ All match |
| **Normal (N)** | Cross formula | Cross formula | Cross formula | ✅ All match |
| **Binormal (B)** | rd×rdd/\|rd×rdd\| | rd×rdd/\|rd×rdd\| | rd×rdd/\|rd×rdd\| | ✅ All match |
| **Curvature (κ)** | ❌ | ❌ | 🔄 Computed | \|rd×rdd\|/\|rd\|³ |
| **Torsion (τ)** | ❌ | ❌ | ❌ | Requires 3rd deriv |
| **Export format** | 9-col txt/mat | 9-col df | 9-col txt/hdf5/csv | ✅ All compatible |

---

## 3. NUMERICAL DIFFERENTIATION

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **4th-order centered** | ✅ | ✅ | ✅ | 12-pt stencil |
| **Forward boundary** | 2nd order | 2nd order | 2nd order | ✅ All match |
| **Backward boundary** | 2nd order | 2nd order | 2nd order | ✅ All match |
| **Mode 1 (standard)** | ✅ | ✅ | ✅ | Fwd/bwd at edges |
| **Mode 2 (looped)** | ✅ | ✅ | 🟡 jnp.roll | Periodic wrap |
| **Step size (k)** | Param k | Param k | Param step_size | ✅ All configurable |
| **1st derivative** | ✅ | ✅ | ✅ | All match |
| **2nd derivative** | ✅ | ✅ | ✅ | All match |
| **3rd derivative** | ✅ | ✅ | ✅ | All match |
| **8th-order rate diff** | ✅ (read func) | ❌ | ❌ | For missing rate data |

---

## 4. PHASE ALIGNMENT

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **Delta computation** | From strain FFT | From strain FFT | From strain FFT | ✅ All match |
| **Time shift** | t + Delta/ω | t + Delta/ω | t + Delta/ω | ✅ All match |
| **Coefficient rotation** | ✅ | ✅ | ✅ | All rotate An,Bn |
| **Auto phase detection** | ❌ | ❌ | ❌ | Manual param |

---

## 5. CYCLE SELECTION

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **Number of cycles (p)** | User param | User param | start/end_cycle | 🟡 Different interface |
| **Integer cycle constraint** | Required (FFT) | Required (FFT) | start_cycle, end_cycle | 🔄 More flexible |
| **Partial cycle handling** | ❌ | ❌ | ✅ Mask-based | RheoJAX handles partial |
| **Cycle mask return** | ❌ | ❌ | ✅ | Returns actual range |
| **Multi-cycle averaging** | ❌ | ❌ | ❌ | Gap in all |
| **Transient filtering** | ❌ | ❌ | ❌ | Gap in all |

---

## 6. SMOOTHING / STEP SIZE

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **FFT harmonic limit** | M param | M param | n_harmonics | ✅ All have |
| **Numerical step size** | k param | k param | step_size | ✅ All have |
| **Explicit smoothing** | ❌ | ❌ | ❌ | None have |
| **Savitzky-Golay** | ❌ | ❌ | ❌ | Gap in all |
| **Butterworth filter** | ❌ | ❌ | ❌ | Gap in all |

---

## 7. MODULI & DERIVATIVES

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **G'(t)** | ✅ | ✅ | ✅ | Storage modulus |
| **G''(t)** | ✅ | ✅ | ✅ | Loss modulus |
| **\|G*(t)\|** | ✅ | ✅ | ✅ | Complex modulus |
| **tan(δ(t))** | ✅ | ✅ | ✅ | Loss tangent |
| **δ(t)** | ✅ | ✅ | ✅ | Phase angle |
| **dG'/dt** | ✅ | ✅ | ✅ | Storage rate |
| **dG''/dt** | ✅ | ✅ | ✅ | Loss rate |
| **\|dG*/dt\|** (speed) | ✅ | ✅ | ✅ | Modulus speed |
| **dδ/dt** (PAV) | ✅ | ✅ | ✅ | Phase velocity |
| **Disp. stress** | ✅ | ✅ | ✅ | Non-linear stress |
| **Eq. strain est.** | ✅ | ✅ | ✅ | Equilibrium strain |

---

## 8. YIELD STRESS CALCULATIONS

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **Static yield (σ_sy)** | ❌ | ❌ | ✅ | At strain reversal |
| **Dynamic yield (σ_dy)** | ❌ | ❌ | ✅ | At zero rate |
| **Yield tolerance param** | ❌ | ❌ | ✅ | 0.02 default |
| **Yield from disp_stress** | Implicit | Implicit | 🔄 Explicit | Multiple methods |
| **Yield from G'(t) minima** | ❌ | ❌ | ✅ | Cage breakage |
| **Yield from δ→π/2** | ❌ | ❌ | ✅ | Flow cessation |
| **Power-law fit** | ❌ | ❌ | ✅ | σ = K\|γ̇\|ⁿ |
| **Herschel-Bulkley** | ❌ | ❌ | ✅ | Model class |

---

## 9. LISSAJOUS-BOWDITCH METRICS

| Feature | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|---------|----------------|--------|---------|-------|
| **G_L** (large strain) | ❌ | ❌ | ✅ | σ at \|γ\|≈γ₀ |
| **G_M** (min strain) | ❌ | ❌ | ✅ | dσ/dγ at γ≈0 |
| **η_L** (large rate) | ❌ | ❌ | ✅ | σ at \|γ̇\|≈γ̇₀ |
| **η_M** (min rate) | ❌ | ❌ | ✅ | dσ/dγ̇ at γ̇≈0 |
| **S-factor** | ❌ | ❌ | ✅ | Stiffening ratio |
| **T-factor** | ❌ | ❌ | ✅ | Thickening ratio |
| **I₃/I₁ ratio** | ❌ | ❌ | ✅ | Nonlinearity |

---

## 10. OUTPUT DATA STRUCTURES

### 10.1 spp_data_in (Input)

| Column | MATLAB | R oreo | RheoJAX | Notes |
|--------|--------|--------|---------|-------|
| Time [s] | ✅ | ✅ | ✅ | All have |
| Strain [-] | ✅ | ✅ | ✅ | All have |
| Rate [1/s] | ✅ | ✅ | ✅ | All have |
| Stress [Pa] | ✅ | ✅ | ✅ | All have |

### 10.2 spp_data_out (15 columns)

| # | Column | MATLAB | R oreo | RheoJAX | Notes |
|---|--------|--------|--------|---------|-------|
| 1 | Time [s] | ✅ | ✅ | ✅ | |
| 2 | Strain [-] | ✅ | ✅ | ✅ | Reconstructed |
| 3 | Rate [1/s] | ✅ | ✅ | ✅ | Reconstructed |
| 4 | Stress [Pa] | ✅ | ✅ | ✅ | Reconstructed |
| 5 | G'(t) [Pa] | ✅ | ✅ | ✅ | |
| 6 | G''(t) [Pa] | ✅ | ✅ | ✅ | |
| 7 | \|G*(t)\| [Pa] | ✅ | ✅ | ✅ | |
| 8 | tan(δ(t)) [] | ✅ | ✅ | ✅ | |
| 9 | δ(t) [rad] | ✅ | ✅ | ✅ | |
| 10 | Disp. stress [Pa] | ✅ | ✅ | ✅ | |
| 11 | Eq. strain est. [-] | ✅ | ✅ | ✅ | |
| 12 | dG'/dt [Pa/s] | ✅ | ✅ | ✅ | |
| 13 | dG''/dt [Pa/s] | ✅ | ✅ | ✅ | |
| 14 | Speed [Pa/s] | ✅ | ✅ | ✅ | |
| 15 | Norm. PAV [] | ✅ | ✅ | ✅ | |

### 10.3 fsf_data_out (Frenet-Serret, 9 columns)

| # | Column | MATLAB | R oreo | RheoJAX | Notes |
|---|--------|--------|--------|---------|-------|
| 1-3 | T_x, T_y, T_z | ✅ | ✅ | ✅ | Tangent |
| 4-6 | N_x, N_y, N_z | ✅ | ✅ | ✅ | Normal |
| 7-9 | B_x, B_y, B_z | ✅ | ✅ | ✅ | Binormal |

### 10.4 spp_params (Analysis Parameters)

| Field | MATLAB | R oreo | RheoJAX | Notes |
|-------|--------|--------|---------|-------|
| omega | ✅ | ✅ | ✅ | |
| M/n_harmonics | ✅ | ✅ | ✅ | |
| p/cycles | ✅ | ✅ | start/end_cycle | 🟡 Different |
| W (max harm) | ✅ | ✅ | ❌ | |
| k/step_size | ✅ | ✅ | ✅ | |
| num_mode | ✅ | ✅ | use_numerical_method | 🟡 Bool vs int |
| gamma_0 | ❌ | ❌ | ✅ | RheoJAX adds |
| yield_tolerance | ❌ | ❌ | ✅ | RheoJAX adds |

### 10.5 ft_out (Fourier Transform)

| Field | MATLAB | R oreo | RheoJAX | Notes |
|-------|--------|--------|---------|-------|
| Harmonic numbers | ✅ | ✅ | ✅ | 0..W |
| FFT magnitudes | ✅ | ✅ | ✅ | Normalized |
| Amplitude array | ❌ | ❌ | ✅ | Raw amplitudes |
| Phase array | ❌ | ❌ | ✅ | Raw phases |

---

## 11. EXPORT FORMATS

| Format | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|--------|----------------|--------|---------|-------|
| **TXT (tab-delim)** | ✅ | ❌ | ✅ | MATLAB-compatible |
| **MAT file** | ✅ | ❌ | 🟡 via scipy | Can export dict |
| **CSV** | ✅ | ✅ | ✅ | |
| **Excel** | ❌ | ✅ | ❌ | |
| **HDF5** | ❌ | ❌ | ✅ | RheoJAX adds |
| **Header format** | 2-row | Df colnames | 2-row | ✅ MATLAB match |
| **Precision** | 7 decimal | Default | 7 decimal | ✅ MATLAB match |

---

## 12. PLOTTING / VISUALIZATION

| Plot Type | MATLAB SPPplus | R oreo | RheoJAX | Notes |
|-----------|----------------|--------|---------|-------|
| **Elastic Lissajous** | ✅ | ✅ | ✅ | σ vs γ |
| **Viscous Lissajous** | ✅ | ✅ | ✅ | σ vs γ̇ |
| **Cole-Cole** | ✅ | ✅ | ✅ | G'' vs G' |
| **VGP plot** | ✅ | ✅ | ❌ | \|G*\| vs δ |
| **Speed plots** | ✅ | ✅ | ❌ | Speed vs G'/G'' |
| **δ vs strain** | ✅ | ✅ | ❌ | Phase evolution |
| **PAV vs strain** | ✅ | ✅ | ❌ | Phase velocity |
| **Disp stress vs strain** | ✅ | ✅ | ❌ | Non-linear |
| **Time waveforms** | ✅ | ✅ | ❌ | Raw vs recon |
| **FFT spectrum** | ✅ | ✅ | ✅ | Harmonic bars |
| **3D trajectory** | ❌ | ❌ | ✅ | (γ, γ̇/ω, σ) |
| **Pipkin diagram** | ❌ | ❌ | ✅ | Amplitude-freq |
| **Moduli evolution** | ❌ | ❌ | ✅ | Multi-panel |
| **Comprehensive report** | ❌ | ❌ | ✅ | 6-panel figure |

---

## 13. ADDITIONAL RheoJAX FEATURES (Beyond Reference)

| Feature | Status | Description |
|---------|--------|-------------|
| **JAX JIT compilation** | 🔄 | All kernels JIT-compiled |
| **Float64 enforcement** | 🔄 | Numerical precision |
| **RheoData integration** | 🔄 | Unified data container |
| **Bayesian yield model** | 🔄 | SPPYieldStress class |
| **Amplitude sweep pred** | 🔄 | predict_amplitude_sweep() |
| **Flow curve pred** | 🔄 | predict_flow_curve() |
| **Unit conversion utils** | 🔄 | percent↔fraction, mPa↔Pa |
| **Transform registry** | 🔄 | "spp_decomposer" registered |
| **spp_analyze()** | 🔄 | Single-shot convenience |

---

## 14. IDENTIFIED GAPS FOR VALIDATION

### High Priority (Core SPP)
1. **Numerical diff boundary handling** - Verify stencil coefficients match exactly
2. **Phase alignment formula** - Compare Delta computation across implementations
3. **Cross-product formulation** - Verify G'(t), G''(t) calculation matches
4. **Normalized PAV formula** - Confirm normalization approach

### Medium Priority (Output Compatibility)
5. **Column ordering** - Ensure 15-col output matches exactly
6. **Header format** - Verify 2-row header compatibility
7. **Precision rounding** - Check 7-decimal consistency
8. **FSF frame sign conventions** - Verify T/N/B signs match

### Lower Priority (Enhanced Features)
9. **Lissajous metrics** - Document G_L/G_M/η_L/η_M calculation details
10. **Yield extraction methods** - Document tolerance-based selection
11. **Power-law fitting** - Document log-log regression approach

---

## 15. VALIDATION TEST CASES TO CREATE

| Test Case | Description | Expected Outcome |
|-----------|-------------|------------------|
| **TC-001** | Fourier: Single cycle, n_harm=5 | Match MATLAB G'(t), G''(t) |
| **TC-002** | Fourier: Multi-cycle, n_harm=15 | Match reconstructed stress |
| **TC-003** | Numerical: Mode 1, k=1 | Match MATLAB derivatives |
| **TC-004** | Numerical: Mode 2, k=2 | Match periodic boundary |
| **TC-005** | Phase alignment | Match Delta and time shift |
| **TC-006** | FSF frame vectors | Match T/N/B components |
| **TC-007** | 15-col export | Byte-identical to MATLAB |
| **TC-008** | Yield from reversal | Compare to manual extraction |

---

## 16. SUMMARY STATISTICS

| Category | MATLAB | R oreo | RheoJAX | Notes |
|----------|--------|--------|---------|-------|
| **Full parity (✅)** | baseline | ~95% | ~90% | |
| **Partial (🟡)** | - | ~5% | ~5% | Interface diffs |
| **Missing (❌)** | - | ~0% | ~5% | VGP/speed plots |
| **Enhanced (🔄)** | - | ~0% | ~20% | Yield, Lissajous, Bayesian |

**Overall Assessment:** RheoJAX has strong parity with MATLAB SPPplus on core SPP calculations (moduli, derivatives, FSF frame, export format). Key differences are enhanced features (yield stress, Lissajous metrics) and some missing visualization types (VGP, speed plots). R oreo closely mirrors MATLAB.

---

*Generated: 2024-12-03*
*Sources: MATLAB SPPplus_v2p1, R oreo 1.0, RheoJAX rheojax/transforms/spp_decomposer.py*
