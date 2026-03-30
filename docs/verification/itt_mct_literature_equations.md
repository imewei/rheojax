# ITT-MCT Literature Review: Key Equations

## References

1. **Fuchs & Cates (2002)** "Theory of nonlinear rheology and yielding of dense colloidal suspensions" *Phys. Rev. Lett.* 89, 248304
2. **Fuchs & Cates (2003)** "Schematic models for dynamic yielding of sheared colloidal glasses" *Faraday Discuss.* 123, 267-286
3. **Fuchs & Cates (2009)** "A mode coupling theory for Brownian particles in homogeneous steady shear flow" *J. Rheol.* 53(4), 957-1000
4. **Brader, Voigtmann, Fuchs, Larson, Cates (2009)** "Glass rheology: From mode-coupling theory to a dynamical yield criterion" *PNAS* 106(36), 15186-15191
5. **Voigtmann, Brader, Fuchs, Cates (2012)** "Schematic mode coupling theory of glass rheology: single and double step strains" *Soft Matter* 8, 4244-4253

---

## 1. Equilibrium MCT (No Flow)

### 1.1 Density correlator

$$\Phi(k,t) = \frac{\langle \rho_k(t) \rho_{-k}(0) \rangle}{\langle |\rho_k|^2 \rangle}$$

### 1.2 Full MCT equation of motion (underdamped)

$$\ddot{\Phi}(t) + \Omega^2 \left[ \Phi(t) + \int_0^t m(t-s) \dot{\Phi}(s)\, ds \right] = 0$$

where $\Omega(k) = k^2 k_B T / (m S(k))$.

### 1.3 Full MCT memory kernel (microscopic)

$$m(k,t) = \sum_{\mathbf{q}} V(k, q, |\mathbf{k}-\mathbf{q}|)\; \Phi(q,t)\; \Phi(|\mathbf{k}-\mathbf{q}|, t)$$

with vertex:

$$V(\mathbf{k}, \mathbf{q}) = \frac{(\mathbf{k}\cdot\mathbf{q})\, c(q)}{k} + \frac{\mathbf{k}\cdot(\mathbf{k}-\mathbf{q})\, c(|\mathbf{k}-\mathbf{q}|)}{k}$$

where $c(k)$ is the direct correlation function related to $S(k)$ via $S(k) = 1/(1 - \rho c(k))$.

### 1.4 Overdamped (Brownian/colloidal) limit

$$\partial_t \Phi(t) + \Gamma \left[ \Phi(t) + \int_0^t m(t-s)\, \partial_s \Phi(s)\, ds \right] = 0$$

where $\Gamma = D_0 k^2 / S(k)$ is the initial decay rate.

---

## 2. F12 Schematic Model (Equilibrium)

### 2.1 Schematic equation of motion

$$\partial_t \Phi(t) + \Gamma \left[ \Phi(t) + \int_0^t m(\Phi(s))\, \partial_s \Phi(s)\, ds \right] = 0$$

with initial condition $\Phi(0) = 1$.

### 2.2 Schematic memory kernel

$$m(\Phi) = v_1 \Phi + v_2 \Phi^2$$

- $v_1$: linear vertex (coupling constant)
- $v_2$: quadratic vertex (coupling constant)
- The quadratic term creates the feedback mechanism for glass transition

### 2.3 Glass transition criterion

The non-ergodicity parameter $f = \lim_{t\to\infty} \Phi(t)$ satisfies:

$$\frac{f}{1-f} = m(f) = v_1 f + v_2 f^2$$

Glass transition occurs when this has a non-zero solution, i.e. when:

$$v_2 > v_{2,c} = \frac{4}{(1 - v_1)^2}$$

For $v_1 = 0$: $v_{2,c} = 4$.

### 2.4 Separation parameter

$$\varepsilon = \frac{v_2 - v_{2,c}}{v_{2,c}}$$

- $\varepsilon < 0$: Ergodic fluid ($\Phi \to 0$ at long times)
- $\varepsilon = 0$: Critical point (power-law decay)
- $\varepsilon > 0$: Glass state ($\Phi \to f > 0$)

### 2.5 Non-ergodicity parameter (for $v_1 = 0$)

At the critical point ($v_2 = 4$):

$$f_c = 1/2$$

Above the transition ($\varepsilon > 0$):

$$f = f_c + \sqrt{\varepsilon/(1-f_c)} + O(\varepsilon)$$

### 2.6 MCT exponents and two-step relaxation

Near the glass transition:
- $\beta$-relaxation: $\Phi(t) \approx f_c + h \cdot (t/t_0)^{-a}$
- $\alpha$-relaxation: $\Phi(t) \approx f \cdot \exp[-(t/\tau_\alpha)^b]$

The exponents $a$ and $b$ satisfy:

$$\frac{\Gamma(1-a)^2}{\Gamma(1-2a)} = \frac{\Gamma(1+b)^2}{\Gamma(1+2b)} = \lambda$$

For $F_{12}$ with $v_1 = 0$: $\lambda = 1$.

---

## 3. Integration Through Transients (ITT) -- Extension to Flow

### 3.1 Deformation gradient

For homogeneous flow with velocity gradient $\boldsymbol{\kappa}(t) = \nabla\mathbf{v}(t)$:

$$\mathbf{E}(t,t') = \mathcal{T}\exp\left(\int_{t'}^{t} \boldsymbol{\kappa}(s)\, ds\right)$$

### 3.2 Wavevector advection

Wavevectors are back-advected:

$$\mathbf{q}(t,t') = \mathbf{q} \cdot \mathbf{E}^{-1}(t,t')$$

For simple shear (flow in x, gradient in y):

$$k_x(t,t') = k_x - k_y \gamma(t,t')$$

where $\gamma(t,t') = \int_{t'}^t \dot{\gamma}(s)\, ds$.

### 3.3 Transient density correlator (under shear)

$$\Phi_{\mathbf{q}}(t,t') = \frac{\langle \rho_{\mathbf{q}(t,t')}(t)\, \rho_{-\mathbf{q}}(t') \rangle}{N S(q)}$$

### 3.4 Generalized Green-Kubo relation (ITT stress functional)

$$\boxed{\sigma_{xy}(t) = \int_{-\infty}^{t} dt'\; \dot{\gamma}(t')\, G(t,t')}$$

This is the central result of ITT: stress as a history integral over a generalized modulus.

### 3.5 Microscopic modulus (isotropized MCT)

$$G(t,t') = \frac{k_B T}{60\pi^2} \int_0^{\infty} dk\; k^4 \left[\frac{S'(k)}{S(k)^2}\right]^2 \Phi_k(t,t')^2$$

### 3.6 Schematic modulus

$$G(t,t') = G_\infty\, \Phi(t,t')^2$$

where $G_\infty$ is the high-frequency (instantaneous) modulus.

### 3.7 Full correlator equation under shear

$$\partial_t \Phi_{\mathbf{q}}(t,t') + \Gamma_{\mathbf{q}}(t,t') \left[ \Phi_{\mathbf{q}}(t,t') + \int_{t'}^{t} ds\; m_{\mathbf{q}}(t,s,t')\; \partial_s \Phi_{\mathbf{q}}(s,t') \right] = 0$$

with advected decay rate:

$$\Gamma_{\mathbf{q}}(t,t') = D_0\, \frac{q(t,t')^2}{S(q(t,t'))}$$

### 3.8 Memory kernel under shear (microscopic, bilinear)

$$m_{\mathbf{q}}(t,s,t') = \int \frac{d^3k}{(2\pi)^3}\; V_{\mathbf{q},\mathbf{k},\mathbf{p}}(t,s,t')\; \Phi_{\mathbf{k}}(t,s)\, \Phi_{\mathbf{p}}(t,s)$$

---

## 4. Schematic ITT-MCT Under Shear (F12-dot-gamma model)

### 4.1 Strain decorrelation function

The advected correlator factorizes as:

$$\Phi(t,t') = \Phi_{\text{eq}}(t-t') \cdot h(\gamma(t,t'))$$

**Gaussian form** (Fuchs & Cates 2002, most common):

$$h(\gamma) = \exp\left[-\left(\frac{\gamma}{\gamma_c}\right)^2\right]$$

**Lorentzian form** (Brader et al. 2008):

$$h(\gamma) = \frac{1}{1 + (\gamma/\gamma_c)^2}$$

where $\gamma_c \approx 0.1$ is the critical cage strain.

### 4.2 Memory kernel -- simplified form

$$m(\Phi) = h[\gamma_{\text{acc}}] \times (v_1 \Phi + v_2 \Phi^2)$$

Single decorrelation factor depending on total accumulated strain.

### 4.3 Memory kernel -- full two-time form (Fuchs & Cates 2002)

$$m(t,s,t_0) = h[\gamma(t,t_0)] \times h[\gamma(t,s)] \times (v_1 \Phi + v_2 \Phi^2)$$

Two decorrelation factors:
- $h[\gamma(t,t_0)]$: cage breaking since flow started
- $h[\gamma(t,s)]$: cage breaking during the memory integral window

---

## 5. Protocol-Specific Equations

### 5.1 Steady-state flow curve

At constant $\dot{\gamma}$, $\gamma(t,t') = \dot{\gamma}(t-t')$:

$$\sigma_{ss} = \dot{\gamma} \int_0^\infty G(s) \cdot h(\dot{\gamma}\, s)\, ds$$

with $G(s) = G_\infty \Phi_{\text{eq}}(s)^2$ (schematic, Fuchs & Cates 2002).

**Yield stress** (glass, $\varepsilon > 0$):

$$\sigma_y = \lim_{\dot{\gamma}\to 0} \sigma_{ss} = G_\infty\, \gamma_c\, f^2$$

**Effective viscosity:**

$$\eta_{\text{eff}} = \sigma / \dot{\gamma}$$

### 5.2 Linear viscoelasticity (SAOS)

For $\gamma_0 \ll \gamma_c$, linearize around equilibrium:

$$G^*(\omega) = i\omega \int_0^\infty G_{\text{eq}}(t)\, e^{-i\omega t}\, dt$$

$$G'(\omega) = \omega \int_0^\infty G_{\text{eq}}(t) \sin(\omega t)\, dt$$

$$G''(\omega) = \omega \int_0^\infty G_{\text{eq}}(t) \cos(\omega t)\, dt$$

**Glass plateau:** For $\varepsilon > 0$, $G'(\omega \to 0) \to G_\infty f$ (non-zero plateau).

### 5.3 Startup flow

Starting from rest with constant $\dot{\gamma}$:

$$\sigma(t) = \dot{\gamma} \int_0^t G(t-s) \cdot h(\dot{\gamma}(t-s))\, ds$$

Shows stress overshoot when $\dot{\gamma}\, \tau_\alpha > 1$.

### 5.4 Creep

At constant applied stress $\sigma_0$:

$$\sigma_0 = \int_0^t \dot{\gamma}(t')\, G(t,t')\, dt'$$

- $\sigma_0 < \sigma_y$: bounded deformation (solid-like)
- $\sigma_0 > \sigma_y$: continuous flow (fluidization, viscosity bifurcation)

### 5.5 Stress relaxation (cessation of flow)

After cessation at $t=0$:

$$\sigma(t) = \sigma(0) \cdot \Phi_{\text{relax}}(t)$$

In the glass state: $\lim_{t\to\infty} \sigma(t) = \sigma_{\text{res}} > 0$ (residual stress).

### 5.6 LAOS

For $\gamma(t) = \gamma_0 \sin(\omega t)$, stress decomposes into odd harmonics:

$$\sigma(t) = \sum_{n=1,3,5,...} [\sigma'_n \sin(n\omega t) + \sigma''_n \cos(n\omega t)]$$

---

## 6. Key Physical Parameters

| Parameter | Symbol | Typical Values | Meaning |
|-----------|--------|---------------|---------|
| Linear vertex | $v_1$ | 0 | Linear coupling (often set to 0) |
| Quadratic vertex | $v_2$ | 2-6 | Controls distance from glass transition |
| Critical vertex | $v_{2,c}$ | 4 (for $v_1=0$) | Glass transition point |
| Separation parameter | $\varepsilon$ | -0.1 to 0.5 | Distance from glass transition |
| Bare relaxation rate | $\Gamma$ | 1-1000 s$^{-1}$ | Short-time Brownian rate |
| Critical strain | $\gamma_c$ | 0.05-0.2 | Cage-breaking strain |
| High-freq modulus | $G_\infty$ | $10^3$-$10^7$ Pa | Instantaneous elastic modulus |
| Non-ergodicity param | $f$ | 0-1 | Glass plateau height |
| Dynamic yield stress | $\sigma_y$ | 1-1000 Pa | $G_\infty \gamma_c f$ |

---

## 7. Key Relations Between Parameters

1. **v2 from epsilon:** $v_2 = v_{2,c}(1 + \varepsilon) = 4(1+\varepsilon)$ (for $v_1 = 0$)
2. **f from v2:** Solve $f/(1-f) = v_2 f^2$ giving $f = 1 - 1/\sqrt{v_2}$ (for $v_1 = 0$, $v_2 > 4$)
3. **Yield stress:** $\sigma_y = G_\infty \gamma_c f$
4. **Alpha relaxation time divergence:** $\tau_\alpha \sim |\varepsilon|^{-\gamma}$ with $\gamma = 1/(2a) + 1/(2b)$

---

## 8. Comparison: Full MCT vs Schematic F12

| Aspect | Full MCT | Schematic F12 |
|--------|----------|---------------|
| Correlator | $\Phi(k,t)$ for all $k$ | Single $\Phi(t)$ |
| Memory kernel | $k$-space integral with $V(\mathbf{k},\mathbf{q})$ | $v_1\Phi + v_2\Phi^2$ |
| Stress | $k$-weighted integral with $S(k)$ | $G_\infty \Phi^2$ |
| Parameters | $S(k)$ (from liquid state theory) | $v_1, v_2, \Gamma, \gamma_c, G_\infty$ |
| Glass transition | Volume fraction $\phi_g \approx 0.516$ | $v_{2,c} = 4$ |
| Computational cost | Hours-days | Seconds-minutes |
| Quantitative | Yes (with good $S(k)$) | Qualitative/semi-quantitative |
