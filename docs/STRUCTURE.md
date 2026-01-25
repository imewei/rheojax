# RheoJAX Documentation Structure

**Last Updated:** January 25, 2026

## Production Documentation

```
docs/
├── README.md                      # Documentation overview and navigation guide
├── Makefile                       # Sphinx build commands
├── requirements-docs.txt          # Documentation dependencies
├── .gitignore                     # Excluded files/folders
├── model-test-mode-compatibility.md  # Model-test mode matrix
├── STRUCTURE.md                   # This file
│
├── source/                        # Sphinx source files
│   ├── conf.py                    # Sphinx configuration
│   ├── index.rst                  # Documentation home page
│   │
│   ├── user_guide/                # Graduate student learning pathway
│   │   ├── index.rst              # Learning pathway overview
│   │   ├── 01_fundamentals/       # Weeks 1-2: Rheology basics
│   │   ├── 02_model_usage/        # Weeks 3-6: Fitting workflows
│   │   ├── 03_advanced_topics/    # Weeks 7-12: Bayesian, fractional
│   │   ├── 04_practical_guides/   # Weeks 13-16: APIs, I/O
│   │   └── 05_appendices/         # Reference material
│   │
│   ├── models/                    # Model Handbook (technical reference)
│   │   ├── index.rst              # Models overview
│   │   ├── summary.rst            # Comparison matrix
│   │   ├── classical/             # 3 models (Maxwell, Zener, SpringPot)
│   │   ├── flow/                  # 6 flow models (PowerLaw, Bingham, HerschelBulkley, Carreau, Cross, CarreauYasuda)
│   │   ├── fractional/            # 11 fractional models
│   │   ├── multi_mode/            # 1 model (GeneralizedMaxwell)
│   │   ├── sgr/                   # 2 models (SGRConventional, SGRGeneric)
│   │   ├── spp/                   # 2 docs (SPPDecomposer, SPPYieldStress) — LAOS analysis
│   │   ├── stz/                   # 1 model (STZConventional)
│   │   ├── itt_mct/               # 2 models + protocols (ITTMCTSchematic, ITTMCTIsotropic)
│   │   ├── dmt/                   # 2 models (DMTLocal, DMTNonlocal)
│   │   ├── fluidity/              # 2 models (FluiditySaramitoLocal, FluiditySaramitoNonlocal)
│   │   ├── epm/                   # 2 models (LatticeEPM, TensorialEPM)
│   │   ├── hl/                    # 1 model (HebraudLequeux)
│   │   └── ikh/                   # 2 models (MIKH, MLIKH)
│   │
│   ├── transforms/                # Transform Reference
│   │   ├── index.rst
│   │   ├── summary.rst            # Application guide
│   │   ├── fft.rst                # FFT analysis
│   │   ├── mastercurve.rst        # Time-temperature superposition
│   │   ├── mutation_number.rst    # Material classification
│   │   ├── owchirp.rst            # Fast rheometry
│   │   └── smooth_derivative.rst  # Noise-robust differentiation
│   │
│   ├── api/                       # API Reference (auto-generated)
│   │   ├── core.rst
│   │   ├── models.rst
│   │   ├── transforms.rst
│   │   ├── pipeline.rst
│   │   └── ...
│   │
│   ├── examples/                  # Example notebooks overview
│   ├── developer/                 # Contributing guides
│   └── ...
│
├── build/                         # Generated documentation (not in Git)
│   └── html/                      # Built HTML files
│
└── .archive/                      # Working files and reports (not built)
    ├── README.md                  # Archive guide
    └── restructuring_2025_11_13/  # November 2025 restructuring
        ├── DOCUMENTATION_RESTRUCTURING_COMPLETE.md
        ├── PHASE_2_COMPLETION_REPORT.md
        ├── RHEOLOGICAL_FUNDAMENTALS.md
        └── audits/                # Analysis and implementation guides
```

## Documentation Tiers

### Tier 1: User Guide (Conceptual Learning)
- **Purpose:** Teach "why" and "when"
- **Audience:** Graduate students, new users
- **Content:** Zero math derivations, pure concepts
- **Size:** 35,336 words, 28 files

### Tier 2: Model Handbook (Technical Reference)
- **Purpose:** Mathematical "what" and "how"
- **Audience:** Researchers, practitioners
- **Content:** Full equations, Quick Reference summaries
- **Size:** ~40,000+ words, 38 models across 14 categories

### Tier 3: Transform Reference (Preprocessing Math)
- **Purpose:** Data preprocessing theory
- **Audience:** Advanced practitioners
- **Content:** FFT, WLF, etc. derivations
- **Size:** 6,400+ words, 7 files

## Building Documentation

```bash
cd /Users/b80985/Projects/rheojax/docs

# Clean build
make clean

# Build HTML
make html

# View locally
open build/html/index.html
```

## Archive Files

Working files, reports, and analysis documents are preserved in `.archive/` but excluded from Sphinx builds. See `.archive/README.md` for details.

## Key Features

✅ **Zero duplication** (down from 40%)
✅ **70 learning objectives** across User Guide
✅ **Quick Reference** on all 38 models
✅ **4 learning pathways** (1-16 weeks)
✅ **100+ material database**
✅ **Graduate student ready**
✅ **14 model categories** (Classical, Flow, Fractional, Multi-Mode, SGR, SPP, STZ, ITT-MCT, DMT, Fluidity, EPM, HL, IKH)

---

**Documentation Version:** 0.6.0
**Build Status:** ✅ Clean (0 errors, 0 warnings)
