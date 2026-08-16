# RheoJAX Documentation Structure

**Last Updated:** 2026-08-16 (refreshed against `graphify-out/GRAPH_REPORT.md` — 24531 nodes,
47741 edges, 0 import cycles — and a direct `find docs` sweep)

## Production Documentation

```
docs/
├── Makefile                          # Sphinx build commands (SOURCEDIR=source, BUILDDIR=build)
├── .gitignore                        # Excludes build/, _build/, source/api/generated/
├── model-test-mode-compatibility.md  # Model-test mode matrix (53 models)
├── architecture-overview.md          # Package structure, core abstractions, invariants (hand-maintained)
├── tech-stack.md                     # Dependency/tooling reference table
├── STRUCTURE.md                      # This file
│
├── CODEMAPS/                         # Machine-generated architecture codemaps (ecc:update-codemaps)
│   ├── architecture.md               # Module graph, dependency direction, entry points
│   ├── backend.md                    # core/pipeline/cli — CLI command → core call chains
│   ├── data.md                       # RheoData + io readers/writers, GUI project file, provenance
│   ├── dependencies.md               # External dependency reference (JAX/NumPyro/PySide6/...)
│   └── frontend.md                   # gui/ — WorkspaceWindow shell, state, services, jobs
│
├── agents/                           # AI-agent workflow docs (referenced from root CLAUDE.md)
│   ├── domain.md                     # Domain-docs convention (CONTEXT.md + docs/adr/)
│   ├── issue-tracker.md              # GitHub-issues-as-tracker convention
│   └── triage-labels.md              # 5-role triage label vocabulary
│
├── internal/                         # Working notes, not built into Sphinx
│   ├── spp_parity_reference.md       # MATLAB SPPplus/R oreo/RheoJAX SPP gap matrix
│   └── spp_parity_status.md          # SPP golden-parity harness status
│
├── verification/                     # Literature/equation verification notes (17 files, not built)
│   ├── *_equation*.md, *_verification.md, *_literature*.md  — per-family math/citation audits
│   └── *.rst (2 files: fluidity_saramito_equations.rst, tnt_equations_verification.rst)
│
├── examples/
│   └── README.md                     # Redirect stub — notebooks moved to top-level examples/
│
├── source/                           # Sphinx source (built by `make html` / `sphinx-build`)
│   ├── conf.py                       # Sphinx configuration (Furo theme)
│   ├── index.rst                     # Documentation home page
│   ├── quickstart.rst                # 5-minute getting started
│   ├── installation.rst              # Installation guide
│   ├── api_reference.rst             # API reference entry point
│   ├── development_status.rst        # Development phases & benchmarks
│   │
│   ├── _static/                      # Static assets
│   │   └── custom.css                # Custom CSS (table striping, typography)
│   │
│   ├── _includes/                    # Shared RST fragments, pulled in via `.. include::` (5 files)
│   │   ├── bayesian_workflow.rst
│   │   ├── fractional_seealso.rst
│   │   ├── glass_transition_physics.rst
│   │   ├── thixotropy_foundations.rst
│   │   └── transient_network_foundations.rst
│   │
│   ├── _guides/                      # Style guides
│   │   └── model_documentation_style.rst
│   │
│   ├── _templates/                   # Document templates
│   │   └── model_handbook_template.rst
│   │
│   ├── developer/                    # Contributing guides (2 files)
│   │   ├── contributing.rst          # Contribution guidelines
│   │   └── architecture.rst          # Package design patterns
│   │
│   ├── architecture/                 # 2 files (io_architecture.rst, fitting_transforms_prompt.md)
│   │
│   ├── user_guide/                   # Graduate student learning pathway (6 sections)
│   │   ├── index.rst                 # Learning pathway overview
│   │   ├── 01_fundamentals/          # Rheology basics (6 files)
│   │   ├── 02_model_usage/           # Fitting workflows (5 files)
│   │   ├── 03_advanced_topics/       # Bayesian, fractional, networks (13 files)
│   │   ├── 04_practical_guides/      # APIs, I/O, batch (9 files)
│   │   ├── 05_appendices/            # Reference material (5 files)
│   │   └── 06_gui/                   # GUI reference (11 files)
│   │
│   ├── models/                       # Model Handbook — 53 models across 22 families, 19 directories
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
│   │   ├── fluidity/                 # 4 models: FluidityLocal/Nonlocal + SaramitoLocal/Nonlocal (5 rst)
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
│   ├── transforms/                   # Transform Reference — 11 transforms (13 rst)
│   │   ├── index.rst
│   │   ├── summary.rst               # Application guide
│   │   ├── fft.rst                   # FFT analysis
│   │   ├── mastercurve.rst           # Time-temperature superposition
│   │   ├── mutation_number.rst       # Material classification
│   │   ├── owchirp.rst               # Fast rheometry
│   │   ├── smooth_derivative.rst     # Noise-robust differentiation
│   │   ├── spp.rst                   # SPP decomposition transform
│   │   ├── srfs.rst                  # Strain-rate frequency superposition
│   │   ├── cox_merz.rst              # Cox-Merz rule validation
│   │   ├── prony_conversion.rst      # Prony series time<->frequency conversion
│   │   ├── spectrum_inversion.rst    # Relaxation spectrum H(tau) recovery
│   │   └── lve_envelope.rst          # Linear viscoelastic startup envelope
│   │
│   ├── api/                          # API Reference (auto-generated, 9 files)
│   ├── verification/                 # Sphinx-built verification index (1 file: index.rst)
│   └── examples/                     # Example notebooks overview (1 file)
│
└── build/                            # Generated documentation — gitignored, not present until built
    └── html/                         # `make html` output (docs/build/html/index.html)
```

## Documentation Tiers

### Tier 1: User Guide (Conceptual Learning)
- **Purpose:** Teach "why" and "when"
- **Audience:** Graduate students, new users
- **Content:** Zero math derivations, pure concepts + worked examples
- **Sections:** 6 (Fundamentals, Model Usage, Advanced Topics, Practical Guides, Appendices, GUI)
- **Size:** 50 files across 6 sections

### Tier 2: Model Handbook (Technical Reference)
- **Purpose:** Mathematical "what" and "how"
- **Audience:** Researchers, practitioners
- **Content:** Full equations, Quick Reference summaries, boxed governing equations
- **Size:** 87 rst files, 53 models across 22 families

### Tier 3: Transform Reference (Preprocessing Math)
- **Purpose:** Data preprocessing theory
- **Audience:** Advanced practitioners
- **Content:** FFT, WLF/TTS, SRFS, SPP, mutation number, OWChirp, derivatives, Cox-Merz,
  Prony conversion, spectrum inversion, LVE envelope
- **Size:** 13 rst files covering 11 transforms

## Building Documentation

```bash
# From repo root
uv run sphinx-build -b html docs/source docs/build/html

# Or from docs/ (uses Makefile's SOURCEDIR=source, BUILDDIR=build)
cd docs && make html

# View locally
open docs/build/html/index.html   # macOS; xdg-open on Linux
```

## Key Features

- **53 models** across 22 families with full Bayesian inference support
- **87 rst files** in the Model Handbook (equations, protocols, troubleshooting)
- **11 transforms** with mathematical derivations
- **6-section User Guide** structured as a 16-week graduate course
- **GUI reference** for interactive analysis (PySide6)
- **Furo theme** with custom CSS, light/dark modes

---

**Documentation Version:** 0.7.1
**Build Status:** Clean (0 errors, 0 warnings)
