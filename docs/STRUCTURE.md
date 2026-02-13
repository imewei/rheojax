# RheoJAX Documentation Structure

**Last Updated:** February 9, 2026

## Production Documentation

```
docs/
├── README.md                         # Documentation overview and navigation guide
├── Makefile                          # Sphinx build commands
├── requirements-docs.txt             # Documentation dependencies
├── .gitignore                        # Excluded files/folders
├── model-test-mode-compatibility.md  # Model-test mode matrix (53 models)
├── STRUCTURE.md                      # This file
│
├── source/                           # Sphinx source files
│   ├── conf.py                       # Sphinx configuration (Furo theme)
│   ├── index.rst                     # Documentation home page
│   ├── quickstart.rst                # 5-minute getting started
│   ├── installation.rst              # Installation guide
│   ├── contributing.rst              # Contributing guidelines
│   ├── api_reference.rst             # API reference entry point
│   ├── development_status.rst        # Development phases & benchmarks
│   │
│   ├── _static/                      # Static assets
│   │   └── custom.css                # Custom CSS (table striping, typography)
│   │
│   ├── _includes/                    # Shared RST fragments (8 files)
│   │   ├── bayesian_usage_template.rst
│   │   ├── bayesian_workflow.rst
│   │   ├── fitting_troubleshooting.rst
│   │   ├── fractional_seealso.rst
│   │   ├── glass_transition_physics.rst
│   │   ├── thixotropy_foundations.rst
│   │   ├── transient_network_foundations.rst
│   │   └── validity_linear.rst
│   │
│   ├── _guides/                      # Style guides
│   │   └── model_documentation_style.rst
│   │
│   ├── _templates/                   # Document templates
│   │   └── model_handbook_template.rst
│   │
│   ├── user_guide/                   # Graduate student learning pathway (6 sections)
│   │   ├── index.rst                 # Learning pathway overview
│   │   ├── 01_fundamentals/          # Weeks 1-2: Rheology basics (6 files)
│   │   ├── 02_model_usage/           # Weeks 3-6: Fitting workflows (5 files)
│   │   ├── 03_advanced_topics/       # Weeks 7-12: Bayesian, fractional, networks (13 files)
│   │   ├── 04_practical_guides/      # Weeks 13-16: APIs, I/O, batch (9 files)
│   │   ├── 05_appendices/            # Reference material (5 files)
│   │   └── 06_gui/                   # GUI reference (9 files)
│   │
│   ├── models/                       # Model Handbook — 53 models across 20 families
│   │   ├── index.rst                 # Models overview (grouped toctree)
│   │   ├── summary.rst               # Comparison matrix
│   │   │
│   │   │  # Linear Viscoelastic
│   │   ├── classical/                # 3 models: Maxwell, Zener, SpringPot (4 rst)
│   │   ├── fractional/               # 11 models: FM, FMG, FML, FKV, FKVZ, FPT, FZss, FZsl, FZll, FJ, FB (12 rst)
│   │   ├── multi_mode/               # 1 model: GeneralizedMaxwell (1 rst)
│   │   │
│   │   │  # Nonlinear & Flow
│   │   ├── flow/                     # 6 models: PowerLaw, Bingham, HB, Carreau, Cross, CY (7 rst)
│   │   ├── giesekus/                 # 2 models: SingleMode, MultiMode (2 rst)
│   │   │
│   │   │  # Elasto-Viscoplastic
│   │   ├── ikh/                      # 2 models: MIKH, MLIKH (3 rst)
│   │   ├── fikh/                     # 2 models: FIKH, FMLIKH (3 rst)
│   │   │
│   │   │  # Thixotropic & Yield Stress
│   │   ├── dmt/                      # 2 models: DMTLocal, DMTNonlocal (2 rst)
│   │   ├── fluidity/                 # 4 models: FluidityLocal/Nonlocal + SaramitoLocal/Nonlocal (4 rst)
│   │   ├── hl/                       # 1 model: HebraudLequeux (2 rst)
│   │   ├── stz/                      # 1 model: STZConventional (2 rst)
│   │   ├── epm/                      # 2 models: LatticeEPM, TensorialEPM (3 rst)
│   │   │
│   │   │  # Soft Glassy & Mode-Coupling
│   │   ├── sgr/                      # 2 models: SGRConventional, SGRGeneric (3 rst)
│   │   ├── itt_mct/                  # 2 models: Schematic, Isotropic (4 rst)
│   │   │
│   │   │  # Transient Networks
│   │   ├── tnt/                      # 5 models: SingleMode, Cates, LoopBridge, MultiSpecies, StickyRouse (12 rst)
│   │   ├── vlb/                      # 4 models: Local, Variant, MultiNetwork, Nonlocal (7 rst)
│   │   │
│   │   │  # Vitrimer & Nanocomposite
│   │   ├── hvm/                      # 1 model: HVMLocal (5 rst)
│   │   ├── hvnm/                     # 1 model: HVNMLocal (5 rst)
│   │   │
│   │   │  # LAOS Analysis
│   │   └── spp/                      # 1 model: SPPYieldStress + SPPDecomposer (3 rst)
│   │
│   ├── transforms/                   # Transform Reference — 7 transforms (9 rst)
│   │   ├── index.rst
│   │   ├── summary.rst               # Application guide
│   │   ├── fft.rst                   # FFT analysis
│   │   ├── mastercurve.rst           # Time-temperature superposition
│   │   ├── mutation_number.rst       # Material classification
│   │   ├── owchirp.rst              # Fast rheometry
│   │   ├── smooth_derivative.rst     # Noise-robust differentiation
│   │   ├── spp.rst                   # SPP decomposition transform
│   │   └── srfs.rst                  # Strain-rate frequency superposition
│   │
│   ├── api/                          # API Reference (auto-generated, 9 files)
│   ├── developer/                    # Contributing guides (2 files)
│   └── examples/                     # Example notebooks overview (1 file)
│
├── build/                            # Generated documentation (not in Git)
│   └── html/                         # Built HTML files
│
└── .archive/                         # Working files and reports (not built)
    ├── README.md                     # Archive guide
    └── restructuring_2025_11_13/     # November 2025 restructuring
```

## Documentation Tiers

### Tier 1: User Guide (Conceptual Learning)
- **Purpose:** Teach "why" and "when"
- **Audience:** Graduate students, new users
- **Content:** Zero math derivations, pure concepts + worked examples
- **Sections:** 6 (Fundamentals, Model Usage, Advanced Topics, Practical Guides, Appendices, GUI)
- **Size:** 47 files across 6 sections

### Tier 2: Model Handbook (Technical Reference)
- **Purpose:** Mathematical "what" and "how"
- **Audience:** Researchers, practitioners
- **Content:** Full equations, Quick Reference summaries, boxed governing equations
- **Size:** 86 rst files, 53 models across 20 families

### Tier 3: Transform Reference (Preprocessing Math)
- **Purpose:** Data preprocessing theory
- **Audience:** Advanced practitioners
- **Content:** FFT, WLF/TTS, SRFS, SPP, mutation number, OWChirp, derivatives
- **Size:** 9 rst files covering 7 transforms

## Building Documentation

```bash
cd /Users/b80985/Projects/rheojax

# Build HTML (from project root)
uv run sphinx-build -b html docs/source docs/_build

# Or from docs/ directory
cd docs && make html

# View locally
open docs/_build/index.html
```

## Archive Files

Working files, reports, and analysis documents are preserved in `.archive/` but excluded from Sphinx builds. See `.archive/README.md` for details.

## Key Features

- **53 models** across 20 families with full Bayesian inference support
- **86 rst files** in the Model Handbook (equations, protocols, troubleshooting)
- **7 transforms** with mathematical derivations
- **6-section User Guide** structured as a 16-week graduate course
- **9 learning pathways** for different user backgrounds
- **240+ example notebooks** across all model families
- **GUI reference** for interactive analysis (PyQt/PySide6)
- **Furo theme** with custom CSS, light/dark modes

---

**Documentation Version:** 0.6.0
**Build Status:** Clean (0 errors, 0 warnings)
