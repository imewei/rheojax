# Fractional Viscoelastic Models: Equation Verification Report

**Date:** 2026-03-30
**Sources:** Schiessel et al. (1995), Jaishankar & McKinley (2013), Bonfanti et al. (2020), Stankiewicz (2018), Eldred et al. (2015/PMC4658031)

---

## 1. Springpot (Scott Blair Element)

### User's equations

- Constitutive: sigma(t) = V * (d^alpha gamma / dt^alpha)
- Complex modulus: G*(omega) = V * (i*omega)^alpha
- V is quasi-property with units [Pa*s^alpha], alpha in [0,1]
- alpha=0 -> spring (sigma = G*gamma), alpha=1 -> dashpot (sigma = eta*gamma_dot)

### Literature verification

**CONFIRMED.** Multiple sources agree exactly.

- Bonfanti et al. (2020, Soft Matter 16, 6002-6020), Eq. (15): The springpot constitutive equation is
  sigma(t) = c_alpha * D^alpha [epsilon(t)], where D^alpha is the Caputo fractional derivative.
  The parameter c_alpha (= V in your notation) has units [Pa*s^alpha].

- Stankiewicz (2018, BIO Web Conf. 10, 02032), Eq. (2): sigma(t) = E*tau^alpha * d^alpha epsilon / dt^alpha.
  Here E*tau^alpha plays the role of V (quasi-property). The Scott Blair element interpolates
  between Hooke spring (alpha=0) and Newton dashpot (alpha=1).

- Jaishankar & McKinley (2013, Proc. R. Soc. A 469, 20120284): Introduced the term "quasi-property"
  for V with dimensions [Pa*s^alpha]. Complex modulus G*(omega) = V*(i*omega)^alpha confirmed.

**Verdict: CORRECT as stated.**

---

## 2. Fractional Maxwell Model (FMM) -- Two Springpots in Series

### User's equation

G*(omega) = [V1*(i*omega)^alpha * V2*(i*omega)^beta] / [V1*(i*omega)^alpha + V2*(i*omega)^beta]

### Literature verification

**CONFIRMED.** This is the standard series combination rule applied to two springpots.

- Stankiewicz (2018), Eq. (6): The FMM constitutive equation with two Scott Blair elements
  (E1, tau1, alpha) and (E2, tau2, beta) in series yields the fractional differential equation:
  tau^(alpha-beta) * D^(alpha-beta)[sigma] + sigma = E*tau^alpha * D^alpha[epsilon]

- The complex modulus follows from series combination: for two elements with individual
  G1*(omega) = V1*(i*omega)^alpha and G2*(omega) = V2*(i*omega)^beta, the series rule gives:
  1/G* = 1/G1* + 1/G2*, hence G* = (G1* * G2*)/(G1* + G2*).

- Bonfanti et al. (2020): Confirms that the FMM connects two springpots in series with
  0 <= beta < alpha <= 1. The more elastic springpot (lower exponent beta) governs
  high-frequency behavior; the more viscous one (higher alpha) governs low-frequency behavior.

- Equivalent parametrization (Stankiewicz 2018): G*(omega) = E*(i*omega*tau)^alpha / [1 + (i*omega*tau)^(alpha-beta)]
  which is algebraically equivalent when V1 = E*tau^alpha, V2 = E*tau^beta, and appropriate substitutions.

**Verdict: CORRECT as stated.**

---

## 3. Fractional Kelvin-Voigt (FKV) -- Two Springpots in Parallel

### User's equation

G*(omega) = V1*(i*omega)^alpha + V2*(i*omega)^beta

### Literature verification

**CONFIRMED.**

- Bonfanti et al. (2020), Eq. (28): G*(omega) = A*(i*omega)^alpha + B*(i*omega)^beta,
  described as "exactly equivalent to the complex modulus of the fractional Kelvin-Voigt model
  consisting of two springpots in parallel."

- RheoJAX docs (FKV model): G*(omega) = G_e + c_alpha*(i*omega)^alpha, which is the special
  case where one element is a pure spring (beta=0, V2=G_e).

- Eldred et al. (PMC4658031): FKVM2 (fractional Kelvin-Voigt with two springpots) uses this
  parallel combination.

**Verdict: CORRECT as stated.**

---

## 4. Fractional Zener (Standard Linear Solid)

### User's description

Various configurations with springpots and springs.

### Literature verification

**CONFIRMED -- multiple variants exist.**

- Bonfanti et al. (2020), Eq. (32): The fractional standard linear solid (Zener) has relaxation
  modulus involving the Mittag-Leffler function. Configuration: spring in parallel with a
  fractional Maxwell arm (spring + springpot in series).

- RheoJAX implements three Zener variants:
  - **FZ-SS** (Solid-Solid): Spring || (Spring -- SpringPot). G(t) = G_e + G_m * E_alpha(-(t/tau)^alpha)
  - **FZ-SL** (Solid-Liquid): Spring || (SpringPot -- Dashpot). Terminal flow behavior.
  - **FZ-LL** (Liquid-Liquid): Dashpot || (SpringPot -- Dashpot). Double flow.

- Schiessel, Metzler, Blumen & Nonnenmacher (1995, J. Phys. A 28, 6567-6584): Systematic
  construction of fractional Zener and Poynting-Thomson models from springpot elements.

**Verdict: CORRECT framework. Multiple valid configurations.**

---

## 5. Relaxation Modulus for FMM

### User's equation

G(t) = G * E_alpha(-(t/tau)^alpha), where E_alpha is the Mittag-Leffler function.

### Literature verification

**PARTIALLY CORRECT -- applies to the special case FMM with one springpot (FMM1), not the general two-springpot FMM.**

- For the **general FMM** (two springpots, orders alpha and beta), from Stankiewicz (2018), Eqs. (7)-(9):
  G(t) = E * (t/tau)^{-beta} * E_{alpha-beta, 1-beta}(-(t/tau)^{alpha-beta})
  where E_{a,b}(z) is the TWO-parameter Mittag-Leffler function.

- For the **special case** where the lower-order element is a spring (beta=0), this reduces to:
  G(t) = G * E_alpha(-(t/tau)^alpha)
  which is exactly the user's formula. This is the Fractional Maxwell Liquid (FML) model.

- Bonfanti et al. (2020): Confirms that the FMM relaxation modulus involves E_{alpha-beta, 1-beta},
  exhibiting stretched-exponential (KWW) at short times and power-law at long times.

**Verdict: CORRECT for FML (beta=0 case). For the general two-springpot FMM, the relaxation modulus is G(t) = E*(t/tau)^{-beta} * E_{alpha-beta, 1-beta}(-(t/tau)^{alpha-beta}).**

---

## 6. Mittag-Leffler Function

### User's equation

E_{alpha,beta}(z) = sum_{k=0}^{inf} z^k / Gamma(alpha*k + beta)
Special case: E_{1,1}(z) = e^z

### Literature verification

**CONFIRMED.**

- Stankiewicz (2018), Eq. (8): E_{phi,mu}(z) = sum_{k=0}^{inf} z^k / Gamma(phi*k + mu)
  Identical to user's definition.

- RheoJAX docs (FML model): E_{alpha,beta}(z) = sum_{k=0}^{inf} z^k / Gamma(alpha*k + beta).
  "This generalization of the exponential function is essential for fractional viscoelasticity."
  E_{1,1}(z) = exp(z) confirmed.

- One-parameter form: E_alpha(z) = E_{alpha,1}(z) = sum_{k=0}^{inf} z^k / Gamma(alpha*k + 1).

**Verdict: CORRECT as stated.**

---

## 7. Creep Compliance for FMM

### User's equation

J(t) = (1/G) * t^alpha / Gamma(1+alpha), or involves Mittag-Leffler.

### Literature verification

**PARTIALLY CORRECT -- applies to the springpot element, not the full FMM.**

- For a **single springpot**: J(t) = (1/V) * t^alpha / Gamma(1+alpha).
  This is the creep compliance of the Scott Blair element alone. CONFIRMED.

- For the **FML** (spring + springpot in series, beta=0):
  J(t) = (1/G) * t^alpha * E_{alpha, 1+alpha}((t/tau)^alpha)
  from RheoJAX docs. This involves the two-parameter Mittag-Leffler function.

- For the **FKV** (spring || springpot):
  J(t) = (1/G_e) * [1 - E_alpha(-(t/tau_eps)^alpha)]
  where tau_eps = (c_alpha/G_e)^{1/alpha}. From RheoJAX docs.

- For the **general FMM** (two springpots):
  J(t) = (1/E) * (t/tau)^alpha * E_{alpha-beta, 1+alpha}((t/tau)^{alpha-beta})

**Verdict: The formula J(t) = (1/G)*t^alpha/Gamma(1+alpha) is the springpot-only creep compliance. The full FMM creep compliance involves Mittag-Leffler as noted.**

---

## Summary Table

| Model/Equation | User's Form | Status | Notes |
|---|---|---|---|
| Springpot constitutive | sigma = V*D^alpha[gamma] | CORRECT | Caputo derivative, V = quasi-property |
| Springpot G*(omega) | V*(i*omega)^alpha | CORRECT | Jaishankar & McKinley (2013) |
| FMM G*(omega) | V1*V2*(iw)^{a+b} / (V1*(iw)^a + V2*(iw)^b) | CORRECT | Series combination rule |
| FKV G*(omega) | V1*(iw)^a + V2*(iw)^b | CORRECT | Parallel combination |
| Fractional Zener | Various springpot+spring configs | CORRECT | Multiple variants (SS, SL, LL) |
| FMM G(t) | G*E_alpha(-(t/tau)^alpha) | PARTIALLY | Only for FML (beta=0). General: involves E_{a-b,1-b} |
| Mittag-Leffler | sum z^k/Gamma(ak+b) | CORRECT | E_{1,1}=exp confirmed |
| FMM J(t) | (1/G)*t^a/Gamma(1+a) | PARTIALLY | Springpot-only. Full FMM uses E_{a-b,1+a} |

---

## Key References

1. Schiessel H, Metzler R, Blumen A, Nonnenmacher TF. "Generalized viscoelastic models: their fractional equations with solutions." J. Phys. A: Math. Gen. 28, 6567-6584 (1995). DOI: 10.1088/0305-4470/28/23/012
2. Jaishankar A, McKinley GH. "Power-law rheology in the bulk and at the interface: quasi-properties and fractional constitutive equations." Proc. R. Soc. A 469, 20120284 (2013). DOI: 10.1098/rspa.2012.0284
3. Bonfanti A, Kaplan JL, Charras G, Kabla A. "Fractional viscoelastic models for power-law materials." Soft Matter 16, 6002-6020 (2020). DOI: 10.1039/D0SM00354A
4. Stankiewicz A. "Fractional Maxwell model of viscoelastic biological materials." BIO Web Conf. 10, 02032 (2018). DOI: 10.1051/bioconf/20181002032
5. Eldred et al. "Fractional Generalizations of Maxwell and Kelvin-Voigt Models for Biopolymer Characterization." PLoS ONE 10(11), e0143090 (2015). DOI: 10.1371/journal.pone.0143090
6. Bagley RL, Torvik PJ. "A theoretical basis for the application of fractional calculus to viscoelasticity." J. Rheol. 27, 201-210 (1983).
7. Scott Blair GW. "The role of psychophysics in rheology." J. Colloid Sci. 2, 21-32 (1947).
