# RheoJAX Documentation Structure

**Last Updated:** November 14, 2025

## Production Documentation

```
docs/
├── README.md                      # Documentation overview and navigation guide
├── Makefile                       # Sphinx build commands
├── requirements-docs.txt          # Documentation dependencies
├── .gitignore                     # Excluded files/folders
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
│   │   ├── summary.rst            # Comparison matrix (2,569 words)
│   │   ├── classical/             # 3 models (Maxwell, Zener, SpringPot)
│   │   ├── fractional/            # 11 fractional models
│   │   └── flow/                  # 6 flow models
│   │
│   ├── transforms/                # Transform Reference
│   │   ├── index.rst
│   │   ├── summary.rst            # Application guide (3,854 words)
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
- **Size:** 17,350 words, 22 files

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
✅ **Quick Reference** on all 20 models
✅ **4 learning pathways** (1-16 weeks)
✅ **100+ material database**
✅ **Graduate student ready**

---

**Documentation Version:** 0.2.0
**Build Status:** ✅ Clean (0 errors, 3 warnings)
