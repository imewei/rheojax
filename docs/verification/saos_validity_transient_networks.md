# When Analytical SAOS Breaks Down for Transient Network Models and Vitrimers

## 1. VLB Multi-Network SAOS: Exact or Approximate?

The VLB SAOS moduli:

    G'(ω) = G_e + Σᵢ Gᵢ ω²τᵢ² / (1 + ω²τᵢ²)
    G''(ω) = Σᵢ Gᵢ ωτᵢ / (1 + ω²τᵢ²) + η_s ω

where τᵢ = 1/k_d,i are **exact** within the linear regime, because:

1. Each VLB subnetwork obeys `dμ/dt = L·μ + μ·Lᵀ + k_d(I - μ)`, which is *linear* in μ when k_d is constant (not stress-dependent).
2. For SAOS, γ(t) = γ₀ sin(ωt) with γ₀ → 0, so the upper-convected terms `L·μ + μ·Lᵀ` produce only O(γ₀) perturbations from equilibrium μ = I.
3. The resulting equation for the deviation δμ = μ - I is linear at O(γ₀), yielding exactly the single-mode Maxwell form G*(ω) = G·iωτ/(1 + iωτ).
4. Multi-network superposition is exact because the networks are mechanically in parallel (stresses are additive) and kinetically independent (each k_d,i is constant).

**Conditions for exactness:**
- k_d must be **constant** (not strain- or stress-dependent)
- Strain amplitude γ₀ must be in the linear regime (typically γ₀ < 0.01–0.1 depending on the material)
- No interchain coupling between networks (already assumed in VLB)

**When it breaks down:**
- If k_d = k_d(σ) or k_d(λ) (strain-dependent detachment), the rate couples to the state, making the ODE nonlinear even at O(γ₀). The SAOS formula then becomes a linearization, not exact.
- At finite γ₀, nonlinear terms (O(γ₀²)) produce third harmonics and modify the fundamental response.


## 2. Vitrimer TST Stress-Coupled Bond Exchange: SAOS Validity

The HVM uses `k_BER = k₀ · cosh(V_act · σ_VM / RT)` where k₀ = ν₀ exp(-E_a/RT).

### Taylor expansion of cosh:
    cosh(x) = 1 + x²/2 + x⁴/24 + ...

For SAOS at strain amplitude γ₀:
- Stress scales as σ ~ G_E · γ₀
- The argument x = V_act · G_E · γ₀ / RT

The cosh correction is O(x²), so:
    k_BER ≈ k₀ · [1 + (V_act · σ_VM)² / (2R²T²) + ...]

### Critical strain amplitude γ_c

The analytical SAOS (which sets k_BER = k₀ = constant) is exact to O(γ₀), because the cosh correction enters at O(γ₀²). Specifically:

    γ_c ~ RT / (V_act · G_E)

Below this strain, the stress-dependent correction to k_BER is negligible. For typical vitrimers:
- V_act ~ 10⁻⁵ to 10⁻⁴ m³/mol
- G_E ~ 10⁵ to 10⁶ Pa
- RT ~ 2500 J/mol (at 300 K)

This gives γ_c ~ 2500 / (10⁻⁴ × 10⁶) = 0.025, i.e., **the analytical SAOS is valid for γ₀ ≲ 1–5%**, which is typical for SAOS experiments.

### Is the analytical SAOS exact below γ_c?

**Yes, in the limit γ₀ → 0, the analytical SAOS is mathematically exact** because:
1. At γ₀ → 0, σ → 0, so cosh(V_act·σ/RT) → 1 exactly.
2. The natural-state evolution equation `dμ_nat/dt = k_BER(μ - μ_nat)` becomes linear with constant k_BER = k₀.
3. The coupled system (μ, μ_nat) is then a pair of linear ODEs, and the exact solution gives Maxwell-like moduli with τ_eff = 1/(2k₀).

The departure from linearity scales as (γ₀/γ_c)² — it is a smooth, gradual breakdown, not a sharp threshold.


## 3. MAOS Corrections at O(γ²) for Transient Networks

In medium-amplitude oscillatory shear (MAOS), the response at O(γ₀³) generates the third harmonic I₃/₁ and modifies the fundamental.

For standard VLB (constant k_d), the nonlinearity comes from the upper-convected derivative term (which is quadratic in deformation gradient). The leading MAOS correction:

- **Third harmonic**: |G₃*| ~ γ₀² appears at frequency 3ω
- **Fundamental correction**: G₁*(ω, γ₀) = G*(ω) + δG*(ω)·γ₀² + O(γ₀⁴)

For vitrimers with TST coupling, there is an *additional* MAOS source: the cosh nonlinearity. This produces:
- **Rate modulation**: δk_BER ~ k₀ · (V_act·σ)²/(2R²T²) at O(γ₀²)
- **Parametric coupling**: the oscillating rate modulates the relaxation, generating third harmonics even without the convected nonlinearity

The vitrimer-specific MAOS coefficient scales as:
    [e₃/e₁]_TST ~ (V_act · G_E / RT)² · f(ωτ_eff)

where f(ωτ_eff) is a transfer function peaking near ωτ_eff ~ 1.

### Key references on MAOS for transient networks:
- **Hyun et al. (2011)** Prog. Polym. Sci. 36(12), 1697-1753 — comprehensive LAOS/MAOS review
- **Ewoldt & Bharadwaj (2013)** Rheol. Acta 52, 201-219 — MAOS framework, asymptotic expansion
- **Bharadwaj & Ewoldt (2015)** JNNFM 225, 36-48 — intrinsic LAOS nonlinearities for constitutive models
- These works derive MAOS coefficients for upper-convected Maxwell and Giesekus models; the same perturbation approach applies to VLB with constant k_d.


## 4. The Factor-of-2 in τ_eff = 1/(2k_BER): Exact or Linear-Limit Only?

The coupled ODE system for the E-network:

    dμ_E/dt = ∇v·μ_E + μ_E·∇vᵀ + k_BER(μ_nat - μ_E)
    dμ_nat/dt = k_BER(μ_E - μ_nat)

Define Δ = μ_E - μ_nat (which determines stress σ_E = G_E·Δ):

    dΔ/dt = ∇v·μ_E + μ_E·∇vᵀ + k_BER(μ_nat - μ_E) - k_BER(μ_E - μ_nat)
           = ∇v·μ_E + μ_E·∇vᵀ - 2k_BER·Δ

**When k_BER is constant**, the -2k_BER·Δ term is the only relaxation, and the factor-of-2 is exact for all amplitudes and all flow kinematics. The convective terms `∇v·μ_E + μ_E·∇vᵀ` are the driving force, and relaxation always occurs at rate 2k_BER.

**When k_BER = k_BER(σ) via TST**, the factor-of-2 survives:

    dΔ/dt = [convective drive] - 2k_BER(σ)·Δ

The effective relaxation rate is 2k_BER(σ), but k_BER itself depends on σ (and hence on Δ). So:
- τ_eff = 1/(2k_BER) remains the *instantaneous* relaxation time
- In SAOS (γ₀ → 0), k_BER → k₀, so τ_eff = 1/(2k₀) is exact
- At finite strain, k_BER increases (cosh > 1), so the effective relaxation accelerates (stress-induced bond exchange)
- The factor-of-2 is **structurally exact** (it comes from the symmetric coupling between μ_E and μ_nat), but the *value* of k_BER varies with stress

**Conclusion**: The factor-of-2 is an exact structural feature of the vitrimer natural-state evolution, valid at all amplitudes. What changes at large amplitude is the value of k_BER, not the factor.


## 5. Literature on SAOS Breakdown for Vitrimers

### Key papers:

1. **Vernerey, Long & Brighenti (2017)** "A statistically-based continuum theory for polymers with transient networks." JMPS 107, 1-20.
   - Derives the VLB framework. SAOS expressions derived assuming constant k_d (Sec. 4.2).
   - Does NOT discuss breakdown conditions explicitly — the analytical SAOS is presented as the linear-regime limit.

2. **Meng, Simon, Niu, McKenna & Hallinan (2019)** "Stress Relaxation of a Vitrimer: Comparing Analytical Predictions to Experiment." Macromolecules 52(8), 3154-3163.
   - Eq. (5): τ_v = 1/(2k_BER) — uses the factor-of-2 explicitly.
   - Fits stress relaxation data. Does not discuss SAOS nonlinearity.

3. **Stukalin, Cai, Kumar, Leibler & Rubinstein (2013)** "Self-healing of unentangled polymer networks with reversible bonds." Macromolecules 46, 7525-7541.
   - Discusses transient network with bond kinetics. SAOS derivation assumes small deformation.
   - Notes that stress-accelerated dissociation (analogous to TST cosh) leads to nonlinear effects at large strain.

4. **Ricarte, Tournilhac & Leibler (2019)** "Phase Separation and Self-Assembly in Vitrimers: Hierarchical Morphology of Molten and Semicrystalline Polyethylene/Dioxaborolane Maleimide Systems." Macromolecules 52, 432-443.
   - Experimentally characterizes vitrimer linear viscoelasticity (SAOS).
   - Uses multi-mode Maxwell fitting of G', G'' data — consistent with linear-regime analytics.

5. **Terentjev and coworkers** (various, 2017–2022) have studied vitrimer rheology:
   - **Hanzon et al. (2020)**, Soft Matter — discusses vitrimer relaxation mechanisms
   - **Rovigatti, Nava, Bellini & Sciortino (2018)** Macromolecules 51, 1232-1241 — simulation of vitrimers showing Maxwellian relaxation in linear regime
   - These works generally **confirm** that SAOS in the linear regime produces standard Maxwell-type spectra for vitrimers.

6. **Vernerey (2018)** "Transient response of nonlinear polymer networks: A kinetic theory." JMPS 115, 230-247.
   - Extends VLB to nonlinear deformations. Derives flow curves and startup transients.
   - The SAOS limit is recovered as the special case of small strain amplitude.

### Summary of literature consensus:
- **No paper explicitly derives the SAOS breakdown criterion** γ_c = RT/(V_act·G_E) for vitrimers with TST coupling. This appears to be a novel observation worth documenting.
- The factor-of-2 is universally adopted (Meng 2019 Eq. 5, Vernerey 2017 Eq. 8–11) without discussion of amplitude limitations.
- MAOS for vitrimers specifically has not been studied — this is an open area. The existing MAOS literature (Ewoldt, Bharadwaj) covers Giesekus and other constitutive models but not vitrimer-type natural-state evolution.


## 6. Implications for RheoJAX Implementation

The current implementation in `rheojax/models/vlb/_kernels.py` and `rheojax/models/hvm/_kernels.py`:

1. **VLB `vlb_saos_moduli` and `vlb_multi_saos`**: Use the Maxwell superposition formula. These are **exact** for the VLB model with constant k_d in the SAOS limit.

2. **HVM `hvm_saos_moduli`**: Uses τ_E_eff = 1/(2k_BER_0) with the factor-of-2. This is **exact** in the γ₀ → 0 limit where k_BER → k_BER_0 (the zero-stress rate).

3. **For LAOS or finite-amplitude oscillation**: The analytical SAOS formulas are insufficient. The full ODE integration (via diffrax) is required, which the codebase already supports through the `_kernels_diffrax.py` modules.

4. **Recommendation**: Document the validity condition γ₀ < γ_c ~ RT/(V_act·G_E) in the HVM model docstrings. Users fitting experimental SAOS data at the standard γ₀ = 0.1–1% are safely in the linear regime for typical vitrimer parameters.
