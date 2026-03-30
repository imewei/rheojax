# Vitrimer Nanocomposite Constitutive Models: Literature Review

**Date:** 2026-03-30
**Purpose:** Reference for implementing vitrimer nanocomposite rheological models in RheoJAX

---

## 1. Key Paper: Karim, Vernerey & Sain (2025)

**"Constitutive Modeling of Vitrimers and Their Nanocomposites Based on Transient Network Theory"**
- **Journal:** Macromolecules, 2025, Vol. 58, Issue 10, pp. 4899-4912
- **DOI:** [10.1021/acs.macromol.4c02872](https://pubs.acs.org/doi/10.1021/acs.macromol.4c02872)
- **Open access preprint:** [Michigan Tech Digital Commons](https://digitalcommons.mtu.edu/michigantech-p2/1640/)

### Core Framework
- **Transient Network Theory (TNT)** applied to vitrimers and their nanocomposites
- Statistical mechanics of evolving polymer networks
- **Deformation-dependent bond exchange reaction (BER) kinetics** under finite deformation explains nonlinear viscoelasticity in vitrimer short-term mechanical behavior
- Long-term diffusion-driven chain dynamics incorporated via constant-rate kinetic term for slower cross-linked network mobility
- Numerically implemented as a **UMAT subroutine** for commercial FE packages
- Validated against: stress relaxation, loading-unloading, creep at multiple temperatures

---

## 2. Terentjev et al. -- Vitrimer Rheology and Elasticity

### 2a. Elasticity and Relaxation in Full and Partial Vitrimer Networks
- **Authors:** Smallenburg, Leibler, Sciortino (with Terentjev framework)
- **Journal:** Macromolecules, 2019
- **DOI:** [10.1021/acs.macromol.9b01123](https://pubs.acs.org/doi/10.1021/acs.macromol.9b01123)

### 2b. Rheology of Vitrimers
- **Journal:** Nature Communications, 2022
- **DOI:** [10.1038/s41467-022-33321-w](https://www.nature.com/articles/s41467-022-33321-w)
- Continuum model where elastic energy accounts for conserved crosslink number
- Full rheology profile: small deformation (linear) to large deformation (nonlinear) viscoelasticity
- Analytical expressions for experimental data analysis
- **Two relaxation modes identified:**
  1. **Mode I:** Damped elastic motion
  2. **Mode II:** Reshuffling of crosslinks within the network
- Elastic-plastic transition at timescale comparable to exchangeable bond lifetime
- **Topology freezing temperature (Tv):** transition between viscoelastic solid and malleable viscoelastic liquid

### Key Equations (Terentjev Framework)
Stress relaxation in vitrimers follows:
```
sigma(t) = sigma_elastic * exp(-t / tau_BER)
```
where:
- `tau_BER` = characteristic bond exchange reaction time
- `tau_BER(T) = tau_0 * exp(E_a / (R * T))` (Arrhenius temperature dependence)
- `E_a` = activation energy for bond exchange (typically 80-150 kJ/mol)
- `tau_0` = pre-exponential factor
- `R` = gas constant
- `T` = absolute temperature

---

## 3. Other Vitrimer Constitutive Models

### 3a. Meng et al. (2016) -- Dual-Crosslink Networks
- Constitutive model for polymers with dynamic bonds
- Combines physically-based insights with continuum-level constitutive laws
- Applicable to dual-crosslink gels and networks with temperature-sensitive dynamic covalent bonds

### 3b. Mechanics of Vitrimer with Hybrid Networks
- **DOI:** [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167663620307183)
- Models vitrimers containing both permanent and exchangeable crosslinks

### 3c. Nonsteady Fracture of Transient Networks: The Case of Vitrimer
- **Journal:** PNAS, 2021
- **DOI:** [10.1073/pnas.2105974118](https://www.pnas.org/doi/10.1073/pnas.2105974118)
- Rate-dependent fracture mechanics in vitrimer networks

---

## 4. Filler Reinforcement Models

### 4a. Guth-Gold Equation (1945)
Effective modulus of a filled elastomer with spherical particles:
```
G_eff = G_matrix * (1 + 2.5*phi + 14.1*phi^2)
```
**Parameters:**
- `G_matrix` = shear modulus of the unfilled matrix
- `phi` = filler volume fraction
- `2.5*phi` = Einstein viscosity correction (first-order hydrodynamic interaction)
- `14.1*phi^2` = Guth's second-order term for particle-particle interactions

**Modified Guth-Gold (with shape factor f):**
```
G_eff = G_matrix * (1 + 0.67*f*phi + 1.62*f^2*phi^2)
```
where `f` = aspect ratio of non-spherical fillers (f=1 for spheres recovers original).

**Fukahori modification** adds a cubic term:
```
G_eff = G_matrix * (1 + 2.5*phi + 14.1*phi^2 + k*phi^3)
```
for better agreement at high filler loadings.

**References:**
- Guth, E. (1945). "Theory of Filler Reinforcement." J. Appl. Phys., 16, 20-25.
- [Modified Guth-Gold for CB-filled rubbers (ResearchGate)](https://www.researchgate.net/publication/274002275_Modified_guth-gold_equation_for_carbon_black-filled_rubbers)

### 4b. Krieger-Dougherty Equation
Effective modulus (or viscosity) near maximum packing:
```
G_eff = G_matrix * (1 - phi/phi_max)^(-[eta]*phi_max)
```
**Parameters:**
- `phi_max` = maximum packing fraction (random close packing ~0.64 for monodisperse spheres)
- `[eta]` = intrinsic viscosity (= 2.5 for hard spheres)
- Product `[eta]*phi_max` is typically ~1.6 for spheres

**Notes:**
- Diverges as `phi -> phi_max` (captures jamming/gelation)
- Improved Brinkman power-law model using Mooney's crowding factor concept
- Empirical modifications by Blissett add a second term with two constants
- More accurate than Guth-Gold at high volume fractions (phi > 0.3)

**References:**
- Krieger, I.M.; Dougherty, T.J. (1959). Trans. Soc. Rheol., 3, 137-152.

### 4c. Percolation Model for Filler Networks
Above the percolation threshold, filler contribution to modulus:
```
G_filler ~ (phi - phi_c)^t    for phi > phi_c
G_filler = 0                   for phi < phi_c
```
**Full composite modulus:**
```
G_composite = G_matrix(phi) + G_0_filler * (phi - phi_c)^t * H(phi - phi_c)
```
where `H` is the Heaviside step function.

**Parameters:**
- `phi_c` = percolation threshold (critical filler volume fraction)
- `t` = critical exponent
  - **Universal values:** t ~ 1.33 (2D), t ~ 2.0 (3D)
  - **Experimental:** t varies widely (1.3 to ~10) depending on filler geometry, dispersion, orientation
- `G_0_filler` = scaling prefactor depending on filler stiffness and contact mechanics

**Typical percolation thresholds:**
| Filler Type | phi_c (vol%) | Notes |
|---|---|---|
| Carbon black | 10-20% | Depends on structure/grade |
| Silica nanoparticles | 5-15% | Surface treatment dependent |
| CNT | 0.1-2% | Very low due to high aspect ratio |
| Graphene/GO | 0.1-1% | 2D geometry, depends on exfoliation |
| Clay (montmorillonite) | 1-5% | Depends on intercalation/exfoliation |

**References:**
- Stauffer, D.; Aharony, A. "Introduction to Percolation Theory" (1992)
- [Percolation overview (ScienceDirect)](https://www.sciencedirect.com/topics/materials-science/percolation)
- [Explosive percolation in nanocomposites (Nature Comm.)](https://www.nature.com/articles/s41467-022-34631-9)

---

## 5. Payne Effect Models

### 5a. Kraus Model (1984) -- Standard Payne Effect
Storage modulus as a function of strain amplitude:
```
G'(gamma_0) = G'_inf + (G'_0 - G'_inf) / (1 + (gamma_0 / gamma_c)^(2m))
```

Loss modulus (Kraus):
```
G''(gamma_0) = G''_inf + 2*(G''_max - G''_inf) * (gamma_0/gamma_c)^m / (1 + (gamma_0/gamma_c)^(2m))
```

**Parameters:**
- `G'_0` = storage modulus at zero/small strain amplitude (plateau value)
- `G'_inf` = storage modulus at large strain amplitude (asymptotic lower bound)
- `gamma_c` = critical strain amplitude (where G'' is maximum and G' drops most steeply)
- `m` = phenomenological exponent (~0.5, independent of frequency, temperature, and CB content)
- `G''_max` = maximum loss modulus
- `G''_inf` = loss modulus at large strain

**Physical basis:**
- Dynamic equilibrium between breakage and recovery of weak filler-filler contacts
- Rate of breakage `R_b = k_b * N * f_b(gamma_0)` (proportional to existing contacts N)
- Rate of recovery `R_r = k_r * (N_0 - N) * f_r(gamma_0)` (proportional to broken contacts)
- At equilibrium: `R_b = R_r`

**References:**
- Kraus, G. (1984). "Mechanical losses in carbon-black-filled rubbers." J. Appl. Polym. Sci.: Appl. Polym. Symp., 39, 75-92.
- [Payne effect overview (ScienceDirect)](https://www.sciencedirect.com/topics/engineering/payne-effect)
- [Kraus model parameters (MDPI Polymers)](https://www.mdpi.com/2073-4360/15/7/1675)

### 5b. Lion-Kardelky-Haupt Model (Fractional Derivative Approach)
- Constitutive modeling of Payne effect using **fractional derivatives and intrinsic time scales**
- Framework for finite viscoelasticity
- [DOI: ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0749641903001360)

### 5c. Multiple Natural Configurations Framework
- Models Payne effect through evolving natural configurations
- [DOI: ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S002072252030183X)

### 5d. Dynamic Strain-Dependent Relaxation Time Spectrum
- Payne effect modeled via strain-amplitude-dependent relaxation spectrum
- [DOI: ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0167663620305688)

---

## 6. Nanoparticle Effects on Vitrimer Bond Exchange Reactions (BER)

### 6a. Catalytic Acceleration
- Catalysts lower the topology freezing temperature (Tv) by reducing activation energy
- Increasing catalytic loading increases overall exchange rate but does not shift Tv
- **Arrhenius kinetics for BER relaxation time:**
```
tau*(T) = tau_0 * exp(E_a / (R * T))
```
- Typical activation energies: E_a ~ 80-150 kJ/mol
- With catalysts: E_a can drop by 20-50 kJ/mol

### 6b. Nanoparticle Effects on BER
- **Silica nanoparticles:** Slow down stress relaxation compared to unfilled analogues (geometric hindrance to chain mobility), but full relaxation still achieved
- **Surface-functionalized fillers:** Silane-modified silica ensures strong interfacial connections while maintaining dynamic characteristics
- **Fe3O4 nanoparticles:** Enable photothermal triggering of BER via NIR light
- **CNT:** 0-0.5 wt% in epoxy vitrimer; NIR light activates BER via photothermal effect
- **Graphite fillers:** Activation energy 90.0-97.7 kJ/mol measured via stress relaxation analysis

### 6c. Modified BER Rate for Filled Vitrimers
Proposed modification accounting for filler hindrance and catalysis:
```
k_BER(T, phi) = k_0 * exp(-E_a(phi) / (R*T)) * f_hindrance(phi)
```
where:
- `E_a(phi) = E_a0 + Delta_E * phi` (filler may increase or decrease E_a depending on surface chemistry)
- `f_hindrance(phi) = (1 - phi/phi_max)^alpha` (geometric hindrance factor)
- For catalytic fillers: `E_a(phi) = E_a0 - Delta_E_cat * phi` (decreasing activation energy)

---

## 7. Silica-Epoxy Vitrimer Nanocomposites

### Key Experimental Results
- **Legrand & Soulie-Ziakovic (2016):** Pioneering silica-epoxy vitrimer nanocomposites
  - [ResearchGate](https://www.researchgate.net/publication/306379300_Silica-Epoxy_Vitrimer_Nanocomposites)
- **Unmodified silica NPs** (10-15 nm) enhance mechanical properties without preventing topology rearrangements:
  - Tensile stress: 48 -> 60 MPa
  - Elastic modulus: 1.8 -> 2.6 GPa
  - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC8467415/)
- Reinforcing effect mainly in **glassy state**; near Tg, enhanced chain mobility reduces polymer-particle adhesion
- **Interfacial covalent binding** + microphase separation for simultaneous strengthening and toughening
  - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0266353824002549)

### Leibler's Foundational Work
- Ludwik Leibler coined the term "vitrimer" (2011)
- Developed silica-like networks using transesterification of epoxy with fatty dicarboxylic/tricarboxylic acids
- Demonstrated Arrhenius-type stress relaxation (not WLF), distinguishing vitrimers from other dynamic networks

---

## 8. Comprehensive Vitrimer Nanocomposite Model (Proposed)

A full constitutive model for vitrimer nanocomposites combines:

### Total Stress
```
sigma_total = sigma_matrix(phi) + sigma_filler_network
```

### Matrix Contribution (Vitrimer TNT)
```
sigma_matrix = G_eff(phi) * f(B_e) * Phi(t, T, phi)
```
where:
- `G_eff(phi)` = Guth-Gold or Krieger-Dougherty reinforced modulus
- `f(B_e)` = strain energy derivative (e.g., neo-Hookean, Mooney-Rivlin)
- `B_e` = elastic left Cauchy-Green tensor (evolving via BER)
- `Phi(t, T, phi)` = relaxation function from BER kinetics

### BER Evolution (Karim-Vernerey-Sain)
```
dPhi/dt = -k_BER(T, deformation, phi) * Phi
```
with deformation-dependent BER rate capturing nonlinear viscoelasticity.

### Filler Network Contribution (Percolation + Payne)
```
sigma_filler = G_filler(phi, gamma_0) * f_filler(B)
```
where:
```
G_filler(phi, gamma_0) = G_0_filler * (phi - phi_c)^t / (1 + (gamma_0/gamma_c)^(2m))    for phi > phi_c
```
This combines percolation scaling with Kraus-type strain softening.

### Temperature Dependence
```
tau_BER(T) = tau_0 * exp(E_a / (R*T))          [Arrhenius, above Tv]
G_matrix(T) = G_eq + (G_g - G_eq) * g(T/Tg)   [glass transition]
```

---

## 9. Review Articles

1. **Kausar & Ahmad (2024):** "State-of-the-art epoxy vitrimer nanocomposites with graphene, CNT and silica"
   - [SAGE Journals](https://journals.sagepub.com/doi/10.1177/87560879241226504)

2. **"Impact of nanofillers on vitrimerization and recycling strategies: a review"**
   - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC12355401/)

3. **"Functional epoxy vitrimers and composites"**
   - [ScienceDirect](https://www.sciencedirect.com/science/article/abs/pii/S0079642520300748)

4. **"Molecular Simulation of Covalent Adaptable Networks and Vitrimers: A Review"**
   - [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11125108/)

5. **"Vitrimer as a Sustainable Alternative to Traditional Thermoset"**
   - [ACS Polymers Au](https://pubs.acs.org/doi/10.1021/acspolymersau.5c00081)

6. **"Modeling and Simulation of Vitrimers" (Book Chapter)**
   - [Springer](https://link.springer.com/chapter/10.1007/978-981-19-6038-3_8)

7. **"A Review of Computational Modeling of Polymer Composites and Nanocomposites"**
   - [MDPI Polymers](https://www.mdpi.com/2073-4360/18/4/443)

---

## 10. Summary of All Equations

| Model | Equation | Application |
|---|---|---|
| **Guth-Gold** | `G_eff = G_m * (1 + 2.5*phi + 14.1*phi^2)` | Spherical filler reinforcement |
| **Modified Guth-Gold** | `G_eff = G_m * (1 + 0.67*f*phi + 1.62*f^2*phi^2)` | Non-spherical fillers (aspect ratio f) |
| **Krieger-Dougherty** | `G_eff = G_m * (1 - phi/phi_max)^(-[eta]*phi_max)` | High volume fraction, near jamming |
| **Percolation** | `G_filler ~ (phi - phi_c)^t` | Filler network above threshold |
| **Kraus (G')** | `G' = G'_inf + (G'_0 - G'_inf)/(1 + (gamma_0/gamma_c)^(2m))` | Payne effect, storage modulus |
| **Kraus (G'')** | `G'' = G''_inf + 2*(G''_m - G''_inf)*(gamma_0/gamma_c)^m/(1+(gamma_0/gamma_c)^(2m))` | Payne effect, loss modulus |
| **Arrhenius BER** | `tau = tau_0 * exp(E_a/(R*T))` | Vitrimer stress relaxation |
| **BER evolution** | `dPhi/dt = -k(T, F, phi) * Phi` | TNT vitrimer constitutive law |
