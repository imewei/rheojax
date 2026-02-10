# Model Test Mode Compatibility

This table documents the supported test modes for each of the **53 models** in RheoJAX.

**Legend:**
**✓** = Fully Supported |
**✗** = Not Supported

## Protocol Definitions

| Protocol | Description |
|---|---|
| **Relaxation** | Stress relaxation after step strain |
| **Creep** | Creep compliance under step stress |
| **Oscillation** | Small-amplitude oscillatory shear (SAOS) — G', G'' |
| **Flow Curve** | Steady-state viscosity/stress vs shear rate |
| **Startup** | Stress growth upon shear startup |
| **LAOS** | Large-amplitude oscillatory shear — Lissajous figures, harmonics |

---

## Capability Matrix

### Classical Models (3)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Maxwell | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Zener (SLS) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| SpringPot | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |

### Flow Models (6)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Power Law | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Carreau | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Carreau-Yasuda | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Cross | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Bingham | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Herschel-Bulkley | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |

### Fractional Models (11)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fractional Maxwell | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Maxwell Gel | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Maxwell Liquid | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Fractional Kelvin-Voigt | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional KV-Zener | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Poynting-Thomson | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Zener (SS) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Zener (SL) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Zener (LL) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Jeffreys | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
| Fractional Burgers | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |

### Multi-Mode (1)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Generalized Maxwell | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Giesekus (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Giesekus Single-Mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Giesekus Multi-Mode | ✗ | ✗ | ✓ | ✓ | ✓ | ✗ |

### IKH — Isotropic Kinematic Hardening (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| MIKH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| MLIKH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### FIKH — Fractional IKH (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| FIKH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| FMLIKH | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### DMT — de Souza Mendes-Thompson (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| DMT Local | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| DMT Nonlocal | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ |

### Fluidity (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Fluidity Local | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fluidity Nonlocal | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### Saramito EVP (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Saramito Local | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Saramito Nonlocal | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ |

### Hebraud-Lequeux (1)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Hebraud-Lequeux | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### SGR — Soft Glassy Rheology (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SGR Conventional | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| SGR Generic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### STZ — Shear Transformation Zones (1)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| STZ Conventional | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### EPM — Elasto-Plastic Models (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Lattice EPM | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |
| Tensorial EPM | ✓ | ✓ | ✓ | ✓ | ✓ | ✗ |

### ITT-MCT — Mode-Coupling Theory (2)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| ITT-MCT Schematic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| ITT-MCT Isotropic | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### SPP — Sequence of Physical Processes (1)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| SPP Yield Stress | ✗ | ✗ | ✗ | ✓ | ✗ | ✓ |

### TNT — Transient Network Theory (5)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| TNT Single-Mode | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TNT Cates | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TNT Loop-Bridge | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TNT Multi-Species | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| TNT Sticky Rouse | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### VLB — Vasquez-Cook-McKinley (4)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| VLB Local | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VLB Variant | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VLB Multi-Network | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| VLB Nonlocal | ✗ | ✓ | ✗ | ✓ | ✓ | ✗ |

### HVM — Hybrid Vitrimer Model (1)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HVM Local | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

### HVNM — Hybrid Vitrimer Nanocomposite (1)

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| HVNM Local | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

---

## Summary by Protocol Count

| Protocols | Count | Models |
|---|---|---|
| All 6 | 26 | GMM, Giesekus (single), IKH (2), FIKH (2), DMT Local, Fluidity (2), Saramito Local, HL, SGR (2), STZ, ITT-MCT (2), SPP excluded, TNT (5), VLB (3), HVM, HVNM |
| 5 (no LAOS) | 2 | Lattice EPM, Tensorial EPM |
| 4 | 4 | Maxwell, Zener, Fractional Jeffreys, Fractional Maxwell Liquid |
| 3 | 12 | SpringPot, Giesekus Multi-Mode, 9 fractional models, DMT/Saramito/VLB nonlocal variants |
| 2 | 1 | SPP Yield Stress |
| 1 | 6 | All 6 flow models (flow_curve only) |

**All 53 models support Bayesian inference via NumPyro NUTS.**
