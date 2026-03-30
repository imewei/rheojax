# Classical Linear Viscoelastic Model Equations -- Verification Reference

Canonical formulations for verifying RheoJAX implementations against standard
textbook references.

## References

- [Ferry1980] Ferry, J. D. (1980). *Viscoelastic Properties of Polymers*, 3rd ed. Wiley.
- [Macosko1994] Macosko, C. W. (1994). *Rheology: Principles, Measurements, and Applications*. Wiley-VCH.
- [BAH1987] Bird, R. B., Armstrong, R. C., Hassager, O. (1987). *Dynamics of Polymeric Liquids*, Vol. 1, 2nd ed. Wiley.
- [Tschoegl1989] Tschoegl, N. W. (1989). *The Phenomenological Theory of Linear Viscoelastic Behavior*. Springer.
- [Findley1989] Findley, W. N., Lai, J. S., Onaran, K. (1989). *Creep and Relaxation of Nonlinear Viscoelastic Materials*. Dover.

---

## 1. Maxwell Model (Spring + Dashpot in Series)

**Parameters:** G_0 (elastic modulus, Pa), eta (viscosity, Pa.s), tau = eta/G_0 (relaxation time, s)

**Constitutive equation** [BAH1987, Eq. 5.2-12]:

    sigma + tau * d(sigma)/dt = eta * d(gamma)/dt

**Relaxation modulus** [Ferry1980, Ch. 3; Macosko1994, Eq. 5.24]:

    G(t) = G_0 * exp(-t/tau)

**Creep compliance** [Tschoegl1989, Ch. 2; Findley1989, Ch. 5]:

    J(t) = 1/G_0 + t/eta

**Storage and loss moduli** [Ferry1980, Eq. 3.16-3.17; Macosko1994, Eq. 5.29-5.30]:

    G'(omega) = G_0 * (omega*tau)^2 / (1 + (omega*tau)^2)
    G''(omega) = G_0 * (omega*tau)   / (1 + (omega*tau)^2)

**Complex modulus:**

    G*(omega) = G'(omega) + i*G''(omega)

**Complex viscosity** [Ferry1980, Eq. 3.19]:

    eta*(omega) = eta / (1 + i*omega*tau)
               = eta*(1 - i*omega*tau) / (1 + (omega*tau)^2)

**Steady shear:** sigma = eta * gamma_dot (Newtonian, no shear thinning)

**Loss tangent:** tan(delta) = G''/G' = 1/(omega*tau)

**RheoJAX file:** `rheojax/models/classical/maxwell.py`
**Status:** All equations match implementation. VERIFIED.

---

## 2. Kelvin-Voigt Model (Spring + Dashpot in Parallel)

**Parameters:** G_0 (elastic modulus, Pa), eta (viscosity, Pa.s), tau_ret = eta/G_0 (retardation time, s)

**Constitutive equation** [BAH1987, Eq. 5.2-6; Macosko1994, Eq. 5.19]:

    sigma(t) = G_0 * gamma(t) + eta * d(gamma)/dt

**Creep compliance** [Ferry1980, Ch. 3; Tschoegl1989, Ch. 2]:

    J(t) = (1/G_0) * (1 - exp(-t/tau_ret))

Note: J(0) = 0 (instantaneous elasticity suppressed), J(inf) = 1/G_0

**Relaxation modulus:** The Kelvin-Voigt element does not have a well-defined
stress relaxation function in the usual sense; it exhibits an instantaneous
stress response:

    G(t) = G_0 + eta * delta(t)

where delta(t) is the Dirac delta function. [Tschoegl1989, Ch. 3]

**Storage and loss moduli** [Ferry1980; Macosko1994, Eq. 5.21]:

    G'(omega) = G_0
    G''(omega) = eta * omega

**Complex modulus:**

    G*(omega) = G_0 + i*eta*omega

**Steady shear:** Not meaningful -- Kelvin-Voigt is a solid model with no flow.

**RheoJAX file:** Not implemented as standalone (appears in fractional variant:
`rheojax/models/fractional/fractional_kelvin_voigt.py`)

---

## 3. Jeffreys Model (Maxwell Element + Solvent Viscosity)

Also known as the **Oldroyd-B** model in the limit of small deformations.

**Parameters:** eta_0 (total zero-shear viscosity), lambda_1 (relaxation time),
lambda_2 (retardation time), with lambda_2 < lambda_1

Alternative parameterization: G_1 (elastic modulus), eta_s (solvent viscosity),
eta_p (polymer viscosity), where:
- eta_0 = eta_s + eta_p
- lambda_1 = eta_p / G_1
- lambda_2 = eta_s * lambda_1 / eta_0

**Constitutive equation** [BAH1987, Eq. 5.2-26; Macosko1994, Eq. 5.37]:

    sigma + lambda_1 * d(sigma)/dt = eta_0 * (d(gamma)/dt + lambda_2 * d^2(gamma)/dt^2)

**Relaxation modulus** [BAH1987, Eq. 5.2-28]:

    G(t) = eta_0 * delta(t) * (1 - lambda_2/lambda_1) + (eta_0/lambda_1) * (lambda_2/lambda_1) * 1  [incorrect form]

More precisely, for the Jeffreys model decomposed as Maxwell + solvent:

    G(t) = G_1 * exp(-t/lambda_1)   +   eta_s * delta(t)

The delta function arises from the pure viscous (solvent) response.

**Storage and loss moduli** [BAH1987, Eq. 5.3-14; Ferry1980]:

    G'(omega) = eta_0 * omega^2 * lambda_1 * (1 - lambda_2/lambda_1) / (1 + omega^2 * lambda_1^2)
              = G_1 * (omega*lambda_1)^2 / (1 + (omega*lambda_1)^2)

    G''(omega) = eta_0 * omega * (1 + omega^2 * lambda_1 * lambda_2) / (1 + omega^2 * lambda_1^2)
              = eta_s * omega + G_1 * omega*lambda_1 / (1 + (omega*lambda_1)^2)

Equivalently using the Maxwell + solvent decomposition:

    G'(omega) = G_1 * (omega*lambda_1)^2 / (1 + (omega*lambda_1)^2)
    G''(omega) = G_1 * (omega*lambda_1) / (1 + (omega*lambda_1)^2) + eta_s * omega

**Complex viscosity:**

    eta*(omega) = eta_0 * (1 + i*omega*lambda_2) / (1 + i*omega*lambda_1)

**RheoJAX file:** Not implemented as standalone classical model. Fractional variant:
`rheojax/models/fractional/fractional_jeffreys.py`

---

## 4. Zener / Standard Linear Solid (SLS) Model

Two equivalent representations:
- **(a) Maxwell + parallel spring:** Maxwell element (G_m, eta) in parallel with spring G_e
- **(b) Voigt + series spring:** Kelvin-Voigt element in series with a spring

**Parameters:** G_e (equilibrium modulus), G_m (Maxwell arm modulus), eta (dashpot viscosity)
- Relaxation time: tau = eta / G_m
- Retardation time: tau_c = eta * (G_e + G_m) / (G_e * G_m)  [always tau_c > tau]
- Instantaneous modulus: G_0 = G_e + G_m

**Constitutive equation** [Tschoegl1989, Ch. 4; Findley1989]:

    sigma + tau * d(sigma)/dt = (G_e + G_m) * gamma + G_e * tau * d(gamma)/dt

Or equivalently:

    sigma + tau * d(sigma)/dt = G_0 * gamma + G_e * tau * d(gamma)/dt

**Relaxation modulus** [Ferry1980; Tschoegl1989; Macosko1994, Eq. 5.34]:

    G(t) = G_e + G_m * exp(-t/tau)

Limits: G(0) = G_e + G_m = G_0,  G(inf) = G_e

**Creep compliance** [Tschoegl1989; Findley1989]:

    J(t) = 1/G_0 + (G_m / (G_e * G_0)) * (1 - exp(-t/tau_c))

where G_0 = G_e + G_m and tau_c = eta * G_0 / (G_e * G_m)

Equivalently:

    J(t) = 1/(G_e + G_m) + (G_m/(G_e*(G_e + G_m))) * (1 - exp(-t/tau_c))

Limits: J(0) = 1/G_0,  J(inf) = 1/G_e

**Storage and loss moduli** [Ferry1980; Macosko1994]:

    G'(omega) = G_e + G_m * (omega*tau)^2 / (1 + (omega*tau)^2)
    G''(omega) = G_m * (omega*tau) / (1 + (omega*tau)^2)

This is just the Maxwell G',G'' superposed with G_e (parallel combination).

**Loss tangent:**

    tan(delta) = G_m * omega*tau / ((G_e + G_m*(omega*tau)^2/(1+(omega*tau)^2)) * (1+(omega*tau)^2))

Simplified:

    tan(delta) = G_m * omega*tau / (G_e * (1 + (omega*tau)^2) + G_m * (omega*tau)^2)

**Steady shear:** sigma = eta * gamma_dot (from the dashpot in the Maxwell arm)

**RheoJAX file:** `rheojax/models/classical/zener.py`
**Status:** All equations match implementation. VERIFIED.

---

## 5. Generalized Maxwell Model (GMM / Prony Series)

**Parameters:** N modes, each with (G_i, tau_i), plus optional G_inf (equilibrium modulus)

**Relaxation modulus** [Ferry1980, Eq. 3.6; Tschoegl1989; Park & Schapery 1999]:

    G(t) = G_inf + SUM_{i=1}^{N} G_i * exp(-t/tau_i)

This is the Prony series representation. G_inf >= 0 (zero for fluids, positive for solids).

**Storage and loss moduli** (by Boltzmann superposition) [Ferry1980, Eq. 3.16-3.17]:

    G'(omega)  = G_inf + SUM_{i=1}^{N} G_i * (omega*tau_i)^2 / (1 + (omega*tau_i)^2)
    G''(omega) = SUM_{i=1}^{N} G_i * (omega*tau_i) / (1 + (omega*tau_i)^2)

**Creep compliance** (via interconversion) [Tschoegl1989; Park & Schapery 1999]:

Exact Prony series form:

    J(t) = J_g + SUM_{j=1}^{M} J_j * (1 - exp(-t/tau_j^ret)) + t/eta_0

where J_g is glassy compliance, tau_j^ret are retardation times, and
eta_0 is the zero-shear viscosity. The {J_j, tau_j^ret} are obtained by
interconversion from {G_i, tau_i}, NOT a simple inversion.

For a fluid (G_inf = 0):

    eta_0 = SUM_{i=1}^{N} G_i * tau_i

**Discrete relaxation spectrum:**

    H(tau) = SUM_{i=1}^{N} G_i * delta(tau - tau_i)

**Complex viscosity:**

    eta*(omega) = G*(omega) / (i*omega)
    eta'(omega) = G''(omega) / omega
    eta''(omega) = G'(omega) / omega

**RheoJAX file:** `rheojax/models/multimode/generalized_maxwell.py`
**Status:** Relaxation and oscillation equations match. VERIFIED.

---

## 6. SpringPot / Scott-Blair Fractional Element

**Parameters:** c_alpha (quasi-property, Pa.s^alpha), alpha (fractional order, 0 <= alpha <= 1)

**Constitutive equation** [Schiessel et al. 1995; Bagley & Torvik 1983]:

    sigma(t) = c_alpha * D^alpha [gamma(t)]

where D^alpha is the Caputo fractional derivative of order alpha.

**Relaxation modulus** [Schiessel et al. 1995]:

    G(t) = c_alpha * t^(-alpha) / Gamma(1 - alpha)

Limits: alpha=0 -> G(t) = c_alpha (pure spring), alpha=1 -> G(t) = c_alpha*delta(t) (pure dashpot)

**Creep compliance:**

    J(t) = (1/c_alpha) * t^alpha / Gamma(1 + alpha)

Limits: alpha=0 -> J(t) = 1/c_alpha (instant elastic), alpha=1 -> J(t) = t/c_alpha (viscous flow)

**Complex modulus:**

    G*(omega) = c_alpha * (i*omega)^alpha
              = c_alpha * omega^alpha * [cos(pi*alpha/2) + i*sin(pi*alpha/2)]

Therefore:

    G'(omega)  = c_alpha * omega^alpha * cos(pi*alpha/2)
    G''(omega) = c_alpha * omega^alpha * sin(pi*alpha/2)

**Loss tangent:** tan(delta) = tan(pi*alpha/2) -- frequency-independent!

**RheoJAX file:** `rheojax/models/classical/springpot.py`
**Status:** All equations match implementation. VERIFIED.

---

## Summary of Verification

| Model                | File                                          | Relaxation | Creep | Oscillation | Flow  |
|---------------------|-----------------------------------------------|:----------:|:-----:|:-----------:|:-----:|
| Maxwell             | `models/classical/maxwell.py`                 | OK         | OK    | OK          | OK    |
| Kelvin-Voigt        | (fractional variant only)                     | N/A        | N/A   | N/A         | N/A   |
| Jeffreys            | (fractional variant only)                     | N/A        | N/A   | N/A         | N/A   |
| Zener (SLS)         | `models/classical/zener.py`                   | OK         | OK    | OK          | OK    |
| Generalized Maxwell | `models/multimode/generalized_maxwell.py`     | OK         | (interconv.) | OK    | OK    |
| SpringPot           | `models/classical/springpot.py`               | OK         | OK    | OK          | N/A   |

All implemented models match their canonical textbook formulations.
