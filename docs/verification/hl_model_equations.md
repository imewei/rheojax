# Hebraud-Lequeux (HL) Model: Equations and Physics

**Reference:** P. Hebraud and F. Lequeux, "Mode-Coupling Theory for the Pasty Rheology
of Soft Glassy Materials," Phys. Rev. Lett. **81**, 2934 (1998).
DOI: [10.1103/PhysRevLett.81.2934](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.81.2934)

---

## 1. Physical Picture

The HL model is a **mean-field elastoplastic model** for yield-stress fluids and soft
glassy materials (foams, emulsions, dense suspensions). The material is divided into
mesoscopic blocks, each carrying a local shear stress sigma. The central quantity is
the **probability density** P(sigma, t) of finding a block at stress sigma at time t.

Each block evolves via four competing processes:

1. **Elastic loading**: External shear rate gamma_dot loads stress at rate G_0 * gamma_dot
   (G_0 is the local elastic modulus, often normalized to 1).
2. **Mechanical noise (diffusion)**: Plastic events elsewhere create random stress kicks,
   modeled as Gaussian noise with diffusion coefficient D(t).
3. **Plastic yielding**: When |sigma| > sigma_c, the block yields at rate 1/tau.
4. **Stress renewal**: Yielded blocks are reinjected at sigma = 0 (delta function source).

---

## 2. The Fokker-Planck Equation (Core PDE)

The evolution of P(sigma, t) is governed by:

```
dP/dt = -G_0 * gamma_dot * dP/dsigma          [Elastic advection]
        + D(t) * d^2P/dsigma^2                 [Mechanical noise diffusion]
        - (1/tau) * P * Theta(|sigma| - sigma_c) [Plastic yielding sink]
        + Gamma(t) * delta(sigma)               [Renewal source at sigma=0]
```

where:
- **G_0**: Local elastic modulus (often set to 1 in dimensionless form)
- **gamma_dot**: Applied shear rate
- **sigma_c**: Critical yield stress threshold
- **tau**: Microscopic yielding timescale
- **Theta(x)**: Heaviside step function (1 if x > 0, else 0)
- **delta(sigma)**: Dirac delta function centered at sigma = 0

### Normalization constraint:
```
integral from -inf to +inf of P(sigma, t) dsigma = 1   for all t
```

---

## 3. Self-Consistent Closure Relations

### 3a. Plastic activity (yielding rate):
```
Gamma(t) = (1/tau) * integral_{|sigma| > sigma_c} P(sigma, t) dsigma
```
This is the total rate of plastic rearrangements -- the fraction of blocks with
|sigma| > sigma_c, weighted by the attempt rate 1/tau.

### 3b. Diffusion coefficient (mechanical noise):
```
D(t) = alpha * Gamma(t)
```
where **alpha** is the dimensionless **coupling parameter**. This is the key
self-consistent closure: the diffusion coefficient is proportional to the total
plastic activity. More yielding -> more mechanical noise -> more yielding.

This closure makes the equation **nonlinear** and **self-consistent**: D depends on P
through Gamma, and P depends on D.

### 3c. Conservation (mass balance):
The renewal source exactly balances the yielding sink:
```
Gamma(t) = (1/tau) * integral_{|sigma| > sigma_c} P(sigma, t) dsigma
```
ensuring that integral P dsigma = 1 is maintained.

---

## 4. Macroscopic Stress

The macroscopic shear stress is the first moment of the distribution:
```
sigma_macro(t) = integral from -inf to +inf of sigma * P(sigma, t) dsigma
```

The macroscopic viscosity is:
```
eta = sigma_macro / gamma_dot
```

---

## 5. Phase Behavior and Critical Point

The coupling parameter **alpha** controls the phase state:

| alpha         | Behavior                                    |
|---------------|---------------------------------------------|
| alpha < 1/2   | **Glassy/Jammed**: Finite yield stress      |
| alpha = 1/2   | **Critical point**: Yield stress vanishes    |
| alpha > 1/2   | **Fluid**: No yield stress, Newtonian-like at low rates |

### Physical interpretation:
- **Small alpha**: Weak mechanical noise coupling. Yielding events do not generate
  enough noise to trigger further events. The system jams and has a yield stress.
- **Large alpha**: Strong coupling. Each yielding event efficiently triggers further
  events (positive feedback). The system flows easily.

---

## 6. Steady-State Flow Curve

In steady state (dP/dt = 0), the flow curve sigma(gamma_dot) is obtained by solving:

```
0 = -G_0 * gamma_dot * dP/dsigma + D_ss * d^2P/dsigma^2
    - (1/tau) * P * Theta(|sigma| - sigma_c) + Gamma_ss * delta(sigma)
```

with D_ss = alpha * Gamma_ss determined self-consistently.

### Known analytical results:

#### For alpha < 1/2 (Glassy phase, yield stress):

**Near the yield point (gamma_dot -> 0+):**
```
sigma(gamma_dot) = sigma_y + A * gamma_dot^(1/2) + ...
```
This is a **Herschel-Bulkley** flow curve with exponent **n = 1/2** (equivalently,
beta = 2 in sigma - sigma_y ~ gamma_dot^(1/beta)).

The yield stress sigma_y depends on alpha and sigma_c. It is determined implicitly
by the steady-state equation at gamma_dot = 0+.

**At large shear rates (gamma_dot -> infinity):**
```
sigma(gamma_dot) ~ eta_inf * gamma_dot
```
The material behaves as a **Newtonian fluid** with effective viscosity eta_inf.
This was rigorously proven in Olivier (2012), Sci. China Math. 55, 435-452.

#### For alpha = 1/2 (Critical point):
```
sigma(gamma_dot) ~ gamma_dot^(1/2)    (power-law fluid, no yield stress)
```

#### For alpha > 1/2 (Fluid phase):
```
sigma(gamma_dot) ~ eta_0 * gamma_dot   at low shear rates (Newtonian)
sigma(gamma_dot) ~ eta_inf * gamma_dot  at high shear rates (Newtonian)
```
with possible shear-thinning in between.

---

## 7. Dimensionless Form

Setting G_0 = 1 and sigma_c = 1, the model has only two dimensionless parameters:
- **alpha**: coupling strength
- **gamma_dot * tau**: Weissenberg number (Wi)

The dimensionless PDE becomes:
```
dP/dt = -gamma_dot * dP/dsigma + D * d^2P/dsigma^2
        - P * Theta(|sigma| - 1) + Gamma * delta(sigma)
```
with D = alpha * Gamma and Gamma = integral_{|sigma|>1} P dsigma (using tau = 1).

---

## 8. Aging Dynamics (No Shear, gamma_dot = 0)

In the absence of shear (Sollich, Olivier, Bresch 2016, arXiv:1611.06681):

- The stress diffusion constant decays as D(t) ~ 1/t^2 during aging.
- The cumulative memory integral (integral of D(t) dt) is **finite**.
- The shear stress relaxation function decays only to a **plateau** (not to zero).
- The system becomes progressively more elastic as it ages.
- Frequency-dependent shear modulus: G''(omega) decreases with system age, while
  relaxation times scale linearly with age.

---

## 9. Extensions and Variants

### 9a. Bocquet-Colin-Ajdari (BCA) model
Extension to include spatial heterogeneity and flow coupling. Adds a Stokes-like
equation coupling the stress distribution to the local flow field.

### 9b. Thermal HL (Baron & Biroli, 2023, arXiv:2312.03627)
Generalization to include thermal fluctuations. The yielding rate becomes:
```
nu(sigma, sigma_c) = exp(-(sigma_c - |sigma|)/T)  for |sigma| < sigma_c
nu(sigma, sigma_c) = 1                             for |sigma| >= sigma_c
```
This introduces a temperature T and activated dynamics below sigma_c.
Predicts MCT-like power-law divergence near T_c and Arrhenius below T_c.

### 9c. Multidimensional generalization (Olivier, 2012, HAL hal-01263132)
Extension to tensorial stress in multidimensional flow configurations.

### 9d. Generalized HL with different yielding rules
- Different Herschel-Bulkley exponents beta can arise from non-Gaussian mechanical
  noise or modified yielding criteria (Lin & Wyart, 2018, PRX 8, 011005).
- The standard HL Gaussian noise gives beta = 2 (i.e., n = 1/2).
- Non-Gaussian (e.g., Levy) noise gives different exponents.

---

## 10. Benchmark Results for Numerical Verification

### 10a. Flow curve shape
- For alpha = 0.3, tau = 1, sigma_c = 1:
  - Yield stress sigma_y > 0 (finite, since alpha < 0.5)
  - Near yield: sigma ~ sigma_y + A * sqrt(gamma_dot)
  - High gamma_dot: sigma ~ eta_inf * gamma_dot (Newtonian)

### 10b. Phase transition
- At alpha = 0.5: yield stress vanishes continuously
- For alpha just below 0.5: very small yield stress

### 10c. Probability distribution shape
- Steady state at finite gamma_dot: P(sigma) is asymmetric (skewed in flow direction)
- P(sigma) = 0 is NOT reached; distribution has finite support up to edges
- Center bin (sigma = 0) has a peak from the renewal source

### 10d. Macroscopic stress
- sigma_macro = integral(sigma * P * dsigma) is always positive for gamma_dot > 0
- For gamma_dot = 0 from rest: sigma_macro = 0

### 10e. Conservation check
- integral(P * dsigma) must remain exactly 1 at all times
- Activity Gamma must match the yielded fraction / tau

### 10f. Large shear rate asymptote
- sigma / gamma_dot -> constant (Newtonian viscosity) as gamma_dot -> infinity
- This is proven rigorously in the mathematical literature

---

## 11. Numerical Implementation Notes

The PDE is solved on a finite grid sigma in [-sigma_max, sigma_max] with n_bins points.
The existing RheoJAX implementation uses:

1. **Finite Volume / Finite Difference** discretization
2. **Upwind scheme** for the advection term (elastic loading)
3. **Central differences** for the diffusion term
4. **Explicit Euler** time stepping with CFL-based sub-stepping
5. **Delta function** approximated as mass injection into the center bin
6. **Boundary conditions**: P = 0 at grid edges
7. **Positivity enforcement** and **renormalization** at each step

Key numerical parameters:
- Grid: sigma in [-5*sigma_c, 5*sigma_c], 501 bins (default)
- Sub-stepping: 25 fixed sub-steps per outer time step for CFL stability
- Max scan steps: 20000 (flow curve), 500 (creep, due to XLA compilation cost)

---

## References

1. P. Hebraud, F. Lequeux, "Mode-Coupling Theory for the Pasty Rheology of Soft
   Glassy Materials," Phys. Rev. Lett. 81, 2934 (1998).
   https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.81.2934

2. P. Sollich, "Rheological constitutive equation for a model of soft glassy
   materials," Phys. Rev. E 58, 738 (1998). arXiv:cond-mat/9712001.
   https://arxiv.org/abs/cond-mat/9712001

3. P. Sollich, J. Olivier, D. Bresch, "Aging and linear response in the
   Hebraud-Lequeux model for amorphous rheology," arXiv:1611.06681 (2016).
   https://arxiv.org/abs/1611.06681

4. A. Gimenez et al., "Stability and Numerical Analysis of the Hebraud-Lequeux
   Model for Suspensions," Discrete Dyn. Nat. Soc. 2011, 415921 (2011).
   https://onlinelibrary.wiley.com/doi/10.1155/2011/415921

5. J. Olivier, "Large shear rate behavior for the Hebraud-Lequeux model,"
   Sci. China Math. 55, 435-452 (2012).
   https://link.springer.com/article/10.1007/s11425-011-4350-2

6. Asymptotic analysis in flow curves for a model of soft glassy fluids,
   Z. Angew. Math. Phys. 61, 445-466 (2010).
   https://link.springer.com/article/10.1007/s00033-009-0022-2

7. J. Lin, M. Wyart, "Microscopic processes controlling the Herschel-Bulkley
   exponent," Phys. Rev. X 8, 011005 (2018). arXiv:1708.00516.

8. J.W. Baron, G. Biroli, "Mean-Field Analysis of the Glassy Dynamics of an
   Elastoplastic Model of Super-Cooled Liquids," arXiv:2312.03627 (2023).
   https://arxiv.org/html/2312.03627

9. L. Bocquet, A. Colin, A. Ajdari, "Kinetic Theory of Plastic Flow in Soft
   Glassy Materials," Phys. Rev. Lett. 103, 036001 (2009).
