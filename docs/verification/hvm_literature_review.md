# Hybrid Vitrimer Model (HVM) -- Literature Review & Equations

## 1. Key References

### Primary papers for the HVM theoretical framework:

1. **Vernerey, Long, & Brighenti (2017)**
   "A statistically-based continuum theory for polymers with transient networks."
   *Journal of the Mechanics and Physics of Solids*, 107, 1-20.
   - Introduces the **distribution tensor** framework for transient networks.
   - Eq. (8): Natural-state evolution (vitrimer hallmark).
   - Eq. (11): Stress as tensor difference sigma_E = G_E * (mu^E - mu^E_nat).
   - https://www.sciencedirect.com/science/article/abs/pii/S0022509617301874

2. **Meng, Saed, & Terentjev (2019)**
   "Elasticity and Relaxation in Full and Partial Vitrimer Networks."
   *Macromolecules*, 52, 7423-7429.
   - Develops continuum model for dynamic-mechanical response of vitrimers.
   - Introduces "partial vitrimer" concept: permanent sub-network + exchangeable sub-network.
   - Partial vitrimer in linear regime behaves as a **Zener (SLS) model**.
   - Eq. (3): k_BER = nu_0 * exp(-E_a / RT) (Arrhenius bond exchange rate).
   - Eq. (5): tau_v = 1/(2*k_BER) (vitrimer relaxation time with factor-of-2).
   - https://pubs.acs.org/doi/10.1021/acs.macromol.9b01123

3. **Meng, Saed, & Terentjev (2022)**
   "Rheology of vitrimers."
   *Nature Communications*, 13, 5753.
   - Full rheological characterization spanning small to large deformation.
   - Analytical expressions for Master Curves across 22 decades of frequency.
   - Treats partial vitrimers (permanent + exchangeable sub-networks).
   - https://www.nature.com/articles/s41467-022-33321-w

4. **Stukalin, Cai, Kumar, Leibler, & Rubinstein (2013)**
   "Self-Healing of Unentangled Polymer Networks with Reversible Bonds."
   *Macromolecules*, 46(18), 7525-7541.
   - Foundational transient network theory for reversible bonds.
   - Scaling theory for self-healing polymers.
   - https://pubs.acs.org/doi/10.1021/ma401111n

5. **Constitutive Modeling of Vitrimers Based on Transient Network Theory (2025)**
   *Macromolecules* (recent).
   - Extends TNT to vitrimers and nanocomposites.
   - https://pubs.acs.org/doi/10.1021/acs.macromol.4c02872

---

## 2. Physical Model: Three-Subnetwork Architecture

The HVM decomposes the polymer into three parallel subnetworks:

| Subnetwork | Symbol | Bond Type | Relaxation | Behavior |
|------------|--------|-----------|------------|----------|
| **Permanent (P)** | G_P | Covalent crosslinks | None (infinite) | Neo-Hookean elastic |
| **Exchangeable (E)** | G_E | Associative vitrimer bonds (BER) | tau_E_eff = 1/(2*k_BER) | Maxwell-like with evolving natural state |
| **Dissociative (D)** | G_D | Physical/reversible bonds | tau_D = 1/k_d^D | Standard Maxwell |

Total stress:
```
sigma = sigma_P + sigma_E + sigma_D
```

---

## 3. Constitutive Equations

### 3.1 Stress Decomposition (General Tensorial Form)

**Permanent network (neo-Hookean):**
```
sigma_P = (1 - D) * G_P * (B - I)
```
where B is the left Cauchy-Green tensor, D is damage [0,1].

**Exchangeable network (vitrimer hallmark):**
```
sigma_E = G_E * (mu^E - mu^E_nat)
```
Stress depends on the *difference* between current distribution and its evolving natural state.

**Dissociative network (standard Maxwell/VLB):**
```
sigma_D = G_D * (mu^D - I)
```

### 3.2 Simple Shear Stress Components

```
sigma_P_xy = (1 - D) * G_P * gamma
sigma_E_xy = G_E * (mu^E_xy - mu^E_nat_xy)
sigma_D_xy = G_D * mu^D_xy
sigma_total = sigma_P_xy + sigma_E_xy + sigma_D_xy
```

### 3.3 Evolution ODEs (Simple Shear)

**E-network distribution tensor (upper-convected + BER relaxation):**
```
d(mu^E_xx)/dt = 2*gamma_dot*mu^E_xy + k_BER*(mu^E_nat_xx - mu^E_xx)
d(mu^E_yy)/dt = k_BER*(mu^E_nat_yy - mu^E_yy)
d(mu^E_xy)/dt = gamma_dot*mu^E_yy + k_BER*(mu^E_nat_xy - mu^E_xy)
```

**E-network natural-state tensor (VITRIMER HALLMARK):**
```
d(mu^E_nat_ij)/dt = k_BER * (mu^E_ij - mu^E_nat_ij)
```
The natural state continuously drifts toward the current state at rate k_BER.
This is the key distinction from conventional transient networks (where natural state = I).

**D-network distribution tensor (standard VLB):**
```
d(mu^D_xx)/dt = 2*gamma_dot*mu^D_xy - k_d^D*(mu^D_xx - 1)
d(mu^D_yy)/dt = -k_d^D*(mu^D_yy - 1)
d(mu^D_xy)/dt = gamma_dot*mu^D_yy - k_d^D*mu^D_xy
```

### 3.4 Factor-of-2 in Relaxation

The stress difference Delta_mu_ij = mu^E_ij - mu^E_nat_ij satisfies:
```
d(Delta_mu_ij)/dt = -2 * k_BER * Delta_mu_ij
```
Both tensors relax toward each other at rate k_BER, so the *difference* (which
determines stress) decays at **2*k_BER**. Therefore:
```
tau_E_eff = 1 / (2 * k_BER_0)
```

---

## 4. Analytical Solutions (Linear Regime, Constant k_BER)

### 4.1 Storage and Loss Moduli (SAOS)

```
G'(omega) = G_P + G_E * (omega*tau_E)^2 / (1 + (omega*tau_E)^2)
                 + G_D * (omega*tau_D)^2 / (1 + (omega*tau_D)^2)

G''(omega) =      G_E * (omega*tau_E)   / (1 + (omega*tau_E)^2)
                 + G_D * (omega*tau_D)   / (1 + (omega*tau_D)^2)
```
where:
- tau_E = tau_E_eff = 1/(2*k_BER_0)  (effective E-network relaxation time)
- tau_D = 1/k_d^D  (D-network relaxation time)

**Physical interpretation:**
- G_P provides a frequency-independent elastic plateau (permanent crosslinks)
- G_E contributes a Maxwell mode with the vitrimer relaxation time
- G_D contributes a second Maxwell mode with the dissociative time

**Limiting behavior:**
- omega -> 0:  G'(0) = G_P,  G''(0) = 0
- omega -> inf: G'(inf) = G_P + G_E + G_D
- Loss peaks at omega = 1/tau_E and omega = 1/tau_D

### 4.2 Relaxation Modulus G(t)

```
G(t) = (1-D)*G_P + G_E*exp(-2*k_BER_0*t) + G_D*exp(-k_d^D*t)
     = (1-D)*G_P + G_E*exp(-t/tau_E_eff) + G_D*exp(-t/tau_D)
```

**Physical interpretation:**
- Permanent plateau (1-D)*G_P survives at all times
- E-network decays exponentially with tau_E_eff = 1/(2*k_BER_0)
- D-network decays exponentially with tau_D = 1/k_d^D
- Initial value: G(0+) = G_P + G_E + G_D
- Long-time value: G(inf) = G_P (or (1-D)*G_P with damage)

### 4.3 Startup Stress (Linear Regime, Constant Rate)

```
sigma(t) = G_P * gamma_dot * t
         + G_E * gamma_dot * tau_E * (1 - exp(-t/tau_E))
         + G_D * gamma_dot * tau_D * (1 - exp(-t/tau_D))
```

### 4.4 Steady-State Shear Stress (Flow Curve)

At steady state, the E-network fully relaxes (mu^E -> mu^E_nat, so sigma_E -> 0):
```
sigma_ss = eta_D * gamma_dot    (viscous contribution only)
```
where eta_D = G_D / k_d^D.

Note: sigma_P = G_P * gamma grows unbounded under continuous shear (elastic storage).

### 4.5 Creep Compliance J(t)

```
J(t) = 1/G_tot + (G_E / (G_P * G_tot)) * (1 - exp(-t/tau_ret_E))
               + (G_D / (G_P * G_tot)) * (1 - exp(-t/tau_ret_D))
```
where:
- G_tot = G_P + G_E + G_D
- tau_ret_E = G_tot / (G_P * 2*k_BER_0)  (E-network retardation time)
- tau_ret_D = G_tot / (G_P * k_d^D)      (D-network retardation time)

Limiting behavior:
- J(0+) = 1/G_tot  (instantaneous elastic compliance)
- J(inf) = 1/G_P   (long-time compliance set by permanent network)

---

## 5. TST Kinetics (Bond Exchange Rate)

### 5.1 Thermal BER Rate (Arrhenius)

```
k_BER_0 = nu_0 * exp(-E_a / (R*T))
```

### 5.2 Stress-Coupled BER Rate (TST)

```
k_BER = k_BER_0 * cosh(V_act * sigma_VM^E / (R*T))
```
where sigma_VM^E is the von Mises equivalent stress on the E-network:
```
sigma_VM = sqrt(sigma_xx^2 + sigma_yy^2 - sigma_xx*sigma_yy + 3*sigma_xy^2)
```

### 5.3 Stretch-Coupled BER Rate

```
k_BER = k_BER_0 * cosh(V_act * G_E * delta_stretch / (R*T))
```
where delta_stretch = sqrt(tr(mu^E - mu^E_nat) / dim).

---

## 6. Parameter Definitions and Physical Meaning

| Parameter | Symbol | Units | Default | Physical Meaning |
|-----------|--------|-------|---------|-----------------|
| G_P | G_P | Pa | 1e4 | Permanent network modulus. Proportional to covalent crosslink density: c_P = G_P/(k_B*T) |
| G_E | G_E | Pa | 1e4 | Exchangeable network modulus. Proportional to exchangeable crosslink density |
| G_D | G_D | Pa | 1e3 | Dissociative network modulus. Proportional to physical bond density |
| nu_0 | nu_0 | 1/s | 1e10 | TST attempt frequency. ~10^8-10^12 for small-molecule bond rearrangements |
| E_a | E_a | J/mol | 80e3 | Activation energy for bond exchange reaction. Typical: 40-150 kJ/mol |
| V_act | V_act | m^3/mol | 1e-5 | Activation volume. Controls mechanochemical coupling (stress-accelerated exchange) |
| T | T | K | 300 | Absolute temperature |
| k_d^D | k_d_D | 1/s | 1.0 | Dissociative bond breakage rate |
| Gamma_0 | Gamma_0 | 1/s | 1e-4 | Damage rate coefficient (optional) |
| lambda_crit | lambda_crit | -- | 2.0 | Critical stretch for damage onset (optional) |

### Derived quantities:
- **k_BER_0** = nu_0 * exp(-E_a/(R*T))  -- thermal BER rate at zero stress
- **tau_E_eff** = 1/(2*k_BER_0)  -- effective vitrimer relaxation time
- **tau_D** = 1/k_d^D  -- dissociative relaxation time
- **eta_D** = G_D/k_d^D  -- dissociative viscosity
- **f_E** = G_E/(G_P + G_E + G_D)  -- exchange fraction
- **T_v** = topology freezing temperature (where k_BER_0 * tau_obs ~ 1)

---

## 7. How Permanent + Dynamic Crosslinks Combine (Vitrimer Behavior)

### Key physics: Evolving natural-state tensor

In conventional transient networks (Maxwell, VLB), broken bonds reform at the equilibrium
(stress-free) configuration, so the natural state is always the identity tensor I.

In vitrimers, **associative bond exchange** rearranges network topology without breaking
the network. Bonds that exchange reform in the *current deformed configuration*, not at
equilibrium. This means the stress-free reference state continuously evolves:

```
d(mu^E_nat)/dt = k_BER * (mu^E - mu^E_nat)
```

**Consequences:**
1. **Stress relaxation to zero**: Under fixed deformation, mu^E and mu^E_nat both converge
   to the same tensor, so sigma_E -> 0. The E-network "forgets" its original shape.

2. **Permanent elastic memory**: The P-network (G_P) never relaxes, providing a permanent
   elastic restoring force. This gives vitrimers their unique combination of reprocessability
   (via BER) and dimensional stability (via permanent crosslinks).

3. **Two-timescale relaxation**: G(t) shows a bi-exponential decay to a permanent plateau:
   - Fast: D-network relaxes at tau_D
   - Intermediate: E-network relaxes at tau_E_eff = 1/(2*k_BER_0)
   - Permanent: G_P plateau

4. **Temperature-controlled reprocessability**: Below T_v, BER is frozen (k_BER ~ 0) and
   the material behaves as a thermoset. Above T_v, BER is active and the material can be
   reprocessed. The transition is controlled by the Arrhenius activation energy E_a.

### Limiting cases:
- **G_P >> G_E, G_D**: Thermoset (permanent network dominates)
- **G_E >> G_P, G_D**: Pure vitrimer (fully exchangeable)
- **G_D >> G_P, G_E**: Maxwell fluid (dissociative bonds dominate)
- **G_P ~ G_E, G_D = 0**: Partial vitrimer (Meng 2019 Zener model)
- **All comparable**: Full HVM with three relaxation mechanisms

---

## 8. Comparison: Vitrimer vs Conventional Transient Network

| Feature | VLB / TNT (Conventional) | HVM (Vitrimer) |
|---------|--------------------------|----------------|
| Natural state | Fixed (I) | Evolving (mu^E_nat) |
| Steady-state stress | sigma = eta * gamma_dot | sigma_E = 0 (BER erases E-stress) |
| Permanent memory | None (fully relaxes) | G_P plateau preserved |
| Relaxation form | Single exponential | Bi-exponential + plateau |
| Bond exchange | Dissociative (network breaks) | Associative (topology changes, network intact) |
| Temperature dependence | k_d ~ T (simple) | Arrhenius k_BER ~ exp(-E_a/RT) (TST) |
