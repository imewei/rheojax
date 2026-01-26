# Model Test Mode Compatibility

This table documents the supported test modes for each model in RheoJax.
**✓** = Fully Supported
**✗** = Not Supported
**~** = Partial/Experimental Support

| Model Family | Test Mode Support Breakdown |
|---|---|
| **Relaxation** | Stress relaxation step strain |
| **Creep** | Creep compliance step stress |
| **Oscillation** | Small Amplitude Oscillatory Shear (SAOS) - $G', G''$ |
| **Flow Curve** | Steady state viscosity/stress vs shear rate |
| **Startup** | Stress growth upon shear startup |
| **LAOS** | Large Amplitude Oscillatory Shear (Lissajous figures, harmonics) |

## Capability Matrix

| Model | Relax | Creep | Osc. | Flow | Startup | LAOS |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Classical & Fractional** | | | | | | |
| Maxwell / Kelvin-Voigt | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Zener (SLS) | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Generalized Maxwell | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Fractional Maxwell | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Kelvin-Voigt | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Zener | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Jeffreys | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| Fractional Burgers | ✓ | ✓ | ✓ | ✗ | ✗ | ✗ |
| **Generalized Newtonian** | | | | | | |
| Power Law | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Cross / Carreau / Yasuda | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| Bingham / Herschel-Bulkley | ✗ | ✗ | ✗ | ✓ | ✗ | ✗ |
| **Microstructural / Thixotropic** | | | | | | |
| **SGR** (Conventional / Generic) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **STZ** (Conventional) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **FIKH** (FIKH / FMLIKH) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **IKH** (MIKH / MLIKH) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Hebraud-Lequeux** | ✓ | ✓ | ✗ | ✓ | ✓ | ✓ |
| **DMT** (Local) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **DMT** (Nonlocal) | ✗ | ✗ | ✗ | ✓ | ✓ | ✗ |
| **Fluidity** (Local / Nonlocal) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **Saramito** (Local / Nonlocal) | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| **SPP** (Yield Stress) | ✓ | ✓ | ✓ | ✓ | ✗ | ✗ |
