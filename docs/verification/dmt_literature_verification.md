# DMT Model: Literature Verification Report

## CRITICAL FINDING: Naming Discrepancy

**The RheoJAX "DMT" model is the de Souza Mendes-Thompson model, NOT the Dullaert-Mewis
thixotropic model.** These are two distinct models in the thixotropy literature:

| Aspect | Dullaert-Mewis (2006) | de Souza Mendes-Thompson (2012/2013) |
|---|---|---|
| Abbreviation in literature | "DM" or "Dullaert-Mewis" | "SMT" or "DSM-T" |
| RheoJAX abbreviation | -- (not implemented) | "DMT" |
| Paper | JNNFM 139, 21-30 (2006) | JNNFM 187-188, 8-15 (2012); Rheol. Acta 52, 673-694 (2013) |

The RheoJAX `_base.py` correctly references de Souza Mendes & Thompson in its docstrings.
The "DMT" abbreviation stands for **D**e Souza **M**endes-**T**hompson, not Dullaert-Mewis-Thixotropic.

---

## 1. Dullaert-Mewis Model (2006) -- Literature Equations

### 1.1 Primary References

- **Dullaert, K. & Mewis, J. (2006)** "A structural kinetics model for thixotropy."
  *J. Non-Newtonian Fluid Mech.* 139, 21-30. DOI: 10.1016/j.jnnfm.2006.06.002

- **Dullaert, K. & Mewis, J. (2005)** "Thixotropy: Build-up and breakdown curves during flow."
  *J. Rheol.* 49(6), 1213.

- **Dullaert, K. & Mewis, J. (2005)** "A model system for thixotropy studies."
  *Rheol. Acta* 45, 23-32.

- **Dullaert, K. (2005)** PhD thesis, "Constitutive equations for thixotropic dispersions,"
  KU Leuven (supervisor: J. Mewis).

### 1.2 Stress Decomposition

The Dullaert-Mewis model decomposes total stress into **elastic** and **viscous** contributions:

    sigma = sigma_e + sigma_v

where:
- sigma_e: elastic stress from aggregate deformation (floc strain)
- sigma_v: viscous stress from flow through/around the structure

The **viscous contribution** has three sub-terms:

    sigma_v = sigma_y(lambda) + eta_str(lambda) * gamma_dot + eta_medium * gamma_dot

where:
- sigma_y(lambda): structure-dependent yield stress
- eta_str(lambda): structural viscosity (hydrodynamic interaction of aggregates)
- eta_medium: viscosity of the fully de-structured suspending medium

### 1.3 Elastic Stress Component

The elastic stress is modeled via an **aggregate strain** variable gamma_a:

    sigma_e = G_a(lambda) * gamma_a

where G_a(lambda) is the structure-dependent elastic (aggregate) modulus, and gamma_a
evolves according to a relaxation/deformation equation:

    d(gamma_a)/dt = gamma_dot - gamma_a / theta_a(lambda)

where theta_a(lambda) is a structure-dependent relaxation time for the aggregates.

This is formally a Maxwell-type evolution for the elastic stress:

    d(sigma_e)/dt = G_a * gamma_dot - sigma_e / theta_a

### 1.4 Structure Kinetics (dλ/dt)

The structure parameter lambda in [0, 1] evolves via:

    d(lambda)/dt = k_B * (1 - lambda)^b  -  k_D * lambda^d * |gamma_dot|^a

where:
- k_B: Brownian buildup rate constant (thermal restructuring)
- b: buildup exponent (Dullaert & Mewis used b ~ 0.5)
- k_D: shear-induced breakdown rate constant
- d: breakdown structure exponent (Dullaert & Mewis suggested d ~ 1-2)
- a: shear rate exponent for breakdown (Dullaert & Mewis used a ~ 1)

**Steady state** (d(lambda)/dt = 0) gives:

    k_B * (1 - lambda_eq)^b = k_D * lambda_eq^d * |gamma_dot|^a

### 1.5 Distribution of Time Constants

A key distinguishing feature of Dullaert-Mewis vs. simpler models: both the kinetic
equation and the relaxation equation contain a **distribution of time constants** (weighted
sum over multiple relaxation modes), which can be treated as a single effective time
constant for simplified implementations.

### 1.6 Material Function Dependencies on lambda

- **Elastic modulus**: G_a = G_a0 * lambda^alpha_G (power-law in lambda)
- **Structural viscosity**: eta_str = eta_str0 * lambda^alpha_eta
- **Yield stress**: sigma_y = sigma_y0 * lambda^alpha_y
- **Relaxation time**: theta_a = eta_str / G_a (Maxwell relation)

### 1.7 Oscillatory/Viscoelastic Variant

The Dullaert-Mewis model inherently includes viscoelasticity through the elastic stress
contribution sigma_e. For SAOS (small amplitude oscillatory shear):

- The structure is assumed constant at equilibrium: lambda ~ lambda_0
- The elastic modulus G_a(lambda_0) and relaxation time theta_a(lambda_0) are constant
- Standard Maxwell SAOS expressions apply:
  - G'(omega) = G_a * (omega*theta_a)^2 / (1 + (omega*theta_a)^2)
  - G''(omega) = G_a * (omega*theta_a) / (1 + (omega*theta_a)^2) + eta_medium * omega

For LAOS (large amplitude oscillatory shear), lambda varies within each cycle, and the
full coupled ODE system must be integrated (as done by Armstrong et al., 2016).

---

## 2. de Souza Mendes-Thompson Model (2012/2013) -- What RheoJAX Implements

### 2.1 Primary References

- **de Souza Mendes, P.R. & Thompson, R.L. (2012)** "A critical overview of
  elasto-viscoplastic thixotropic modeling." JNNFM 187-188, 8-15.

- **de Souza Mendes, P.R. & Thompson, R.L. (2013)** "A unified approach to model
  elasto-viscoplastic thixotropic yield-stress materials and apparent yield-stress fluids."
  Rheol. Acta 52, 673-694.

### 2.2 Stress Equation

Maxwell-type stress evolution:

    d(sigma)/dt = G(lambda) * gamma_dot - sigma / theta_1(lambda)

where theta_1 = eta(lambda) / G(lambda) is the relaxation time.

### 2.3 Viscosity Closure

Two options (both in RheoJAX):

**Exponential closure (original DSM-T 2013):**

    eta(lambda) = eta_inf * (eta_0 / eta_inf)^lambda

**Herschel-Bulkley closure:**

    eta_eff = tau_y(lambda) * (1 - exp(-m*|gamma_dot|))/|gamma_dot|
              + K(lambda) * |gamma_dot|^(n-1) + eta_inf

with tau_y(lambda) = tau_y0 * lambda^m1, K(lambda) = K0 * lambda^m2

### 2.4 Structure Kinetics

    d(lambda)/dt = (1 - lambda)/t_eq  -  a * lambda * |gamma_dot|^c / t_eq

This is a simplified form compared to Dullaert-Mewis (b=1, d=1 fixed; single timescale
t_eq governs both buildup and breakdown).

Equilibrium structure:

    lambda_eq = 1 / (1 + a * |gamma_dot|^c)

---

## 3. Key Differences Between the Two Models

| Feature | Dullaert-Mewis (2006) | de Souza Mendes-Thompson (2013) |
|---|---|---|
| Stress decomposition | sigma_e + sigma_v (separate elastic/viscous) | Single Maxwell equation |
| Buildup exponent b | Free parameter (~0.5) | Fixed at 1 |
| Breakdown exponent d | Free parameter (~1-2) | Fixed at 1 (linear in lambda) |
| Kinetics timescale | Separate k_B, k_D rate constants | Single t_eq timescale |
| Time constant distribution | Multiple modes possible | Single mode |
| Viscous sub-contributions | Yield + structural + medium | Single viscosity function |
| Parameter count | Higher (more physics) | Lower (more parsimonious) |

---

## 4. Comparison with RheoJAX Implementation

The RheoJAX DMT implementation (`rheojax/models/dmt/`) correctly implements the
**de Souza Mendes-Thompson** model:

- `_kernels.py`: Implements exponential and HB closures, Maxwell stress evolution,
  structure kinetics with (1-lambda)/t_eq buildup and a*lambda*|gamma_dot|^c/t_eq breakdown
- `_base.py`: Parameters match DSM-T formulation (eta_0, eta_inf, G0, m_G, t_eq, a, c)
- `local.py`: ODE integration for startup, creep, relaxation, SAOS, LAOS
- `nonlocal_model.py`: Adds diffusion term D_lambda * d^2(lambda)/dy^2 for shear banding

**The implementation does NOT contain the Dullaert-Mewis model.** If you need the
Dullaert-Mewis model specifically, it would require a new model family with:

1. Separate elastic/viscous stress tracking (sigma_e, sigma_v)
2. Aggregate strain variable gamma_a with its own evolution equation
3. Free exponents b and d in the structure kinetics
4. Separate k_B and k_D rate constants (not a single t_eq)
5. Optional distribution of relaxation times

---

## 5. Review Papers for Further Reference

- **Larson, R.G. & Wei, Y. (2019)** "A review of thixotropy and its rheological modeling."
  *J. Rheol.* 63(3), 477-501.

- **Mewis, J. & Wagner, N.J. (2009)** "Thixotropy." *Adv. Colloid Interface Sci.* 147-148, 214-227.

- **Mewis, J. & Wagner, N.J. (2012)** *Colloidal Suspension Rheology*, Chapter 7: Thixotropy.
  Cambridge University Press.

- **Armstrong, M.J. et al. (2016)** "Dynamic Shear Rheology of Thixotropic Suspensions:
  Comparison of Structure-Based Models with Large Amplitude Oscillatory Shear Experiments."
  *J. Rheol.* (applied Dullaert-Mewis framework to LAOS).

---

## Sources

- [Dullaert & Mewis 2006 (Semantic Scholar)](https://www.semanticscholar.org/paper/A-structural-kinetics-model-for-thixotropy-Dullaert-Mewis/519930dde56571e90afd65f39184632eadb30cfc)
- [Dullaert & Mewis 2006 (ScienceDirect)](https://www.sciencedirect.com/science/article/abs/pii/S0377025706001431)
- [Dullaert & Mewis 2005 (J. Rheol.)](https://sor.scitation.org/doi/10.1122/1.2039868)
- [Larson & Wei 2019 Review](https://pubs.aip.org/sor/jor/article/63/3/477/241126/A-review-of-thixotropy-and-its-rheological)
- [Armstrong et al. LAOS study (UDel)](https://bpb-us-w2.wpmucdn.com/sites.udel.edu/dist/8/715/files/2014/11/Armstrong-et-al-Thixotropy-of-Transient-and-LAOS-08152015-1zaa32b.pdf)
- [de Souza Mendes & Thompson 2013 (Rheol. Acta)](https://link.springer.com/article/10.1007/s00397-013-0699-1)
- [Gas Bubbles in Dullaert-Mewis Fluid (ResearchGate)](https://www.researchgate.net/publication/270432311_Dynamic_of_Gas_Bubbles_Surrounded_by_a_Dullaert-Mewis_Thixotropic_Fluid)
- [Turbulent Pipe Flow of Thixotropic Fluids (arXiv)](https://arxiv.org/abs/2501.01597)
