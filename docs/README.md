# RheoJAX Documentation

**Comprehensive Handbook for JAX-Accelerated Rheological Analysis**

[![Documentation Status](https://readthedocs.org/projects/rheojax/badge/?version=latest)](https://rheojax.readthedocs.io/en/latest/?badge=latest)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 📚 What is RheoJAX?

RheoJAX is a **JAX-accelerated rheological analysis package** providing a unified framework for analyzing experimental rheology data. It integrates classical rheological models with modern data transforms, offering **2-10x performance improvements** through JAX + GPU acceleration and **comprehensive Bayesian inference** capabilities.

This documentation serves as both a **practical user guide** and a **comprehensive rheology handbook**, suitable for:
- 🎓 **Researchers** - Deep theoretical foundations and cutting-edge techniques
- 👨‍🎓 **Students** - Educational content with worked examples
- 🏭 **Industrial Users** - Practical guidance for materials characterization
- 💻 **Software Developers** - API reference and integration guides

---

## 🎯 Quick Navigation

### For New Users
- **[Getting Started](source/user_guide/getting_started.rst)** - Installation and first analysis
- **[Quickstart Guide](source/quickstart.rst)** - 5-minute introduction
- **[Model Selection Guide](source/user_guide/model_selection.rst)** - Choose the right model for your data

### By Analysis Type
- **[Oscillatory (SAOS)](source/models/index.rst)** - Frequency sweeps, G′, G″, tan δ
- **[Stress Relaxation](source/models/index.rst)** - Time-domain G(t) analysis
- **[Creep & Recovery](source/models/index.rst)** - Compliance J(t) measurements
- **[Flow Curves](source/models/flow/power_law.rst)** - Viscosity vs shear rate

### By Material Type
- **[Polymer Melts](source/models/classical/maxwell.rst)** - Linear viscoelastic liquids
- **[Gels & Networks](source/models/fractional/fractional_zener_ss.rst)** - Cross-linked systems
- **[Suspensions & Pastes](source/models/flow/herschel_bulkley.rst)** - Yield stress fluids
- **[Biological Materials](source/models/fractional/fractional_burgers.rst)** - Tissues, cells, biopolymers

### Advanced Topics
- **[Bayesian Inference](source/user_guide/bayesian_inference.rst)** - NUTS sampling with NumPyro
- **[Time-Temperature Superposition](source/transforms/mastercurve.rst)** - WLF master curves
- **[Multi-Technique Fitting](source/user_guide/multi_technique_fitting.rst)** - Combined data analysis

---

## 📖 Documentation Structure

### **1. Models Handbook** (20 Models Documented)

Comprehensive documentation for all rheological models with handbook-quality depth:

#### **Classical Models** (3 models)
Fundamental viscoelastic models with spring-dashpot analogues:
- **[Maxwell](source/models/classical/maxwell.rst)** - Single relaxation time, exponential decay
- **[Zener](source/models/classical/zener.rst)** - Standard linear solid, creep recovery
- **[SpringPot](source/models/classical/springpot.rst)** - Fractional element, power-law behavior

**Key Features:**
- Physical foundations (microstructural interpretation)
- Mathematical derivations (Laplace/Fourier transforms)
- Material examples (polymer melts, elastomers)
- Experimental protocols (SAOS, relaxation, creep)
- Fitting strategies with troubleshooting

#### **Fractional Models** (11 models)
Advanced models using fractional calculus for broad relaxation spectra:

- **[Fractional Zener Solid-Solid (FZSS)](source/models/fractional/fractional_zener_ss.rst)** - Two plateaus, solid behavior
- **[Fractional Maxwell Liquid (FML)](source/models/fractional/fractional_maxwell_liquid.rst)** - Polymer melts, liquid behavior
- **[Fractional Maxwell Gel (FMG)](source/models/fractional/fractional_maxwell_gel.rst)** - Critical gels, power-law modulus
- **[Fractional Burgers](source/models/fractional/fractional_burgers.rst)** - Creep compliance, long-term deformation
- **[Fractional Kelvin-Voigt](source/models/fractional/fractional_kelvin_voigt.rst)** - Viscoelastic solids
- Plus 6 additional fractional models

**Fractional Calculus Coverage:**
- Mittag-Leffler functions (1-parameter and 2-parameter)
- Physical meaning of fractional order α
- SpringPot constitutive equations
- Applications to soft matter (gels, tissues, colloids)
- Smart initialization for oscillation data (v0.2.0)

#### **Flow & Viscoplastic Models** (6 models)
Non-Newtonian fluid models for industrial applications:

- **[Power-Law](source/models/flow/power_law.rst)** - Shear-thinning/thickening
- **[Carreau](source/models/flow/carreau.rst)** - Full flow curves with plateaus
- **[Cross](source/models/flow/cross.rst)** - Alternative plateau model
- **[Bingham](source/models/flow/bingham.rst)** - Yield stress fluids
- **[Herschel-Bulkley](source/models/flow/herschel_bulkley.rst)** - Yield + power-law
- **[Carreau-Yasuda](source/models/flow/carreau_yasuda.rst)** - Enhanced flexibility

**Industrial Applications:**
- Polymer processing (extrusion, injection molding)
- Coatings & paints (application rheology)
- Food products (texture, processing)
- Cosmetics & pharmaceuticals
- Oil & gas (drilling fluids)
- Construction materials (cement, asphalt)

**📊 [Models Summary](source/models/summary.rst)** - Comparison matrix, decision flowcharts, quick selection guide

---

### **2. Transforms** (5 Transforms Documented)

Data preprocessing and analysis tools with mathematical foundations:

- **[FFT Analysis](source/transforms/fft.rst)** - Time-to-frequency conversion, Kramers-Kronig relations, LAOS harmonics
- **[Mastercurve (TTS)](source/transforms/mastercurve.rst)** - Time-temperature superposition, WLF/Arrhenius shifting
- **[Mutation Number](source/transforms/mutation_number.rst)** - Material classification (tan δ), gel point detection
- **[OWChirp](source/transforms/owchirp.rst)** - Fast time-resolved rheometry (10-100x speedup)
- **[Smooth Derivative](source/transforms/smooth_derivative.rst)** - Noise-robust differentiation (Savitzky-Golay)

**Workflow Integration:**
- 5 complete pipelines with code examples
- Transform chaining compatibility matrix
- Quality checkpoints between stages
- Parameter selection guidelines

**📊 [Transforms Summary](source/transforms/summary.rst)** - Application guide, workflow pipelines, parameter tuning

---

### **3. User Guides**

Comprehensive tutorials and conceptual explanations:

- **[Core Concepts](source/user_guide/core_concepts.rst)** - Rheological fundamentals
- **[Pipeline API](source/user_guide/pipeline_api.rst)** - Fluent interface for workflows
- **[Bayesian Inference](source/user_guide/bayesian_inference.rst)** - NLSQ → NUTS workflow
- **[Model Selection](source/user_guide/model_selection.rst)** - Decision trees and diagnostics
- **[Visualization](source/user_guide/visualization_guide.rst)** - Publication-quality plots
- **[I/O Guide](source/user_guide/io_guide.rst)** - TRIOS, Excel, HDF5 formats

---

### **4. API Reference**

Complete API documentation with autodoc:

- **[Core](source/api/core.rst)** - BaseModel, RheoData, ParameterSet, BayesianMixin
- **[Models](source/api/models.rst)** - All 20 rheological models
- **[Transforms](source/api/transforms.rst)** - All 5 data transforms
- **[Pipeline](source/api/pipeline.rst)** - Pipeline, BayesianPipeline, BatchPipeline
- **[Visualization](source/api/visualization.rst)** - Plotter, BayesianPlotter, templates
- **[Utilities](source/api/utils.rst)** - Optimization, initialization, compatibility

---

### **5. Examples** (24 Jupyter Notebooks)

Executable tutorials covering all major features:

- **Basic** (5 notebooks) - Maxwell, Zener, SpringPot, Bingham, PowerLaw fitting
- **Transforms** (6 notebooks) - FFT, Mastercurve, Mutation Number, OWChirp, Derivatives
- **Bayesian** (6 notebooks) - Basics, Priors, Diagnostics, Model Comparison, Uncertainty, Workflow
- **Advanced** (7 notebooks) - Multi-technique, Batch, Custom Models, Fractional Deep-Dive, Performance

---

## 🔬 Rheological Fundamentals

### Material Classification

#### **Viscoelastic Liquids**
- **Behavior:** G″ > G′ at low ω, equilibrium modulus Ge = 0
- **Examples:** Polymer melts, concentrated solutions, wormlike micelles
- **Models:** Maxwell, FML, Carreau
- **Applications:** Polymer processing, flow simulations

#### **Viscoelastic Solids**
- **Behavior:** G′ > G″ at all ω, finite equilibrium modulus Ge > 0
- **Examples:** Cross-linked polymers, hydrogels, biological tissues
- **Models:** Zener, FZSS, Burgers
- **Applications:** Soft robotics, tissue engineering, material design

#### **Critical Gels**
- **Behavior:** G′ ~ G″ ~ ω^0.5, tan δ = constant (frequency-independent)
- **Examples:** Gels at gel point, colloidal gels
- **Models:** FMG, SpringPot (α ≈ 0.5)
- **Applications:** Gelation kinetics, percolation

#### **Yield Stress Fluids**
- **Behavior:** Solid when τ < τ₀, liquid when τ > τ₀
- **Examples:** Toothpaste, mayonnaise, drilling muds, blood
- **Models:** Bingham, Herschel-Bulkley
- **Applications:** Formulation, process design

#### **Shear-Thinning/Thickening**
- **Thinning (η decreases):** Polymer melts, paints, blood, ketchup
- **Thickening (η increases):** Corn starch suspensions, dense colloids
- **Models:** Power-Law, Carreau, Cross
- **Applications:** Coating, mixing, pumping

### Test Mode Guide

| Test Mode | Measures | Best For | Typical Models |
|-----------|----------|----------|----------------|
| **SAOS** | G′(ω), G″(ω), tan δ | Broad frequency range, non-destructive | Maxwell, Zener, FZSS, FML |
| **Relaxation** | G(t) | Single-step deformation, model validation | Maxwell, FMG, Fractional models |
| **Creep** | J(t) | Long-term deformation, recovery | Burgers, FKV, Zener |
| **Flow** | η(γ̇), σ(γ̇) | Non-Newtonian behavior, yield stress | Power-Law, Carreau, Bingham, HB |

### Parameter Interpretation

#### **Fractional Order α (0 < α < 1)**

The fractional order is a fundamental parameter in fractional models with deep physical meaning:

1. **Relaxation Spectrum Width:**
   - α = 1: Single relaxation time (exponential, classical Maxwell)
   - 0 < α < 1: Broad power-law distribution of relaxation times
   - α → 0: Very broad spectrum (many time scales)

2. **Material Character:**
   - α < 0.5: Solid-like (elastic forces dominate)
   - α = 0.5: Critical gel (balanced elastic/viscous)
   - α > 0.5: Liquid-like (viscous forces dominate)

3. **Microstructural Heterogeneity:**
   - Lower α: Higher disorder, heterogeneity, polydispersity
   - Higher α: More uniform microstructure

4. **Typical Ranges by Material:**
   - Polymer gels: α ≈ 0.3-0.7
   - Biological tissues: α ≈ 0.1-0.5
   - Soft glasses: α ≈ 0.2-0.4
   - Colloidal gels: α ≈ 0.4-0.6

### Model Selection Flowchart

```
START: What data do you have?

┌─ Oscillation (G′, G″ from frequency sweep)
│  │
│  ├─ Two plateaus (low and high ω)?
│  │  └─► FZSS (Fractional Zener Solid-Solid)
│  │
│  ├─ One plateau (high ω only)?
│  │  └─► FML (Fractional Maxwell Liquid)
│  │
│  ├─ No plateaus (power-law)?
│  │  └─► FMG (Fractional Maxwell Gel)
│  │
│  └─ Narrow spectrum (single peak in G″)?
│     └─► Maxwell or Zener (classical)
│
├─ Relaxation (G(t) from step strain)
│  │
│  ├─ Exponential decay?
│  │  └─► Maxwell or Zener
│  │
│  ├─ Power-law decay (t^-α)?
│  │  └─► FMG or SpringPot
│  │
│  └─ Mittag-Leffler decay?
│     └─► Fractional Maxwell/Zener family
│
├─ Creep (J(t) from constant stress)
│  │
│  ├─ Bounded compliance (recovery)?
│  │  └─► Zener, FZSS, Burgers
│  │
│  └─ Unbounded (terminal flow)?
│     └─► Maxwell, FML
│
└─ Flow (η vs γ̇ from steady shear)
   │
   ├─ Yield stress present?
   │  ├─ Constant η after yield → Bingham
   │  └─ Thinning after yield → Herschel-Bulkley
   │
   ├─ Plateaus at low/high γ̇?
   │  └─► Carreau or Cross
   │
   └─ No plateaus (log-log linear)?
      └─► Power-Law
```

---

## 🚀 Key Features

### **Performance**
- **2-10x faster** than scipy-based optimization (CPU)
- **20-100x faster** with GPU acceleration (CUDA 12+)
- NLSQ optimization: 5-270x speedup for rheological models
- Automatic JIT compilation via JAX

### **Bayesian Inference** (NEW in v0.2.0)
- **All 20 models** support complete Bayesian workflows
- NLSQ → NUTS warm-start for 2-5x faster convergence
- NumPyro integration with comprehensive diagnostics
- ArviZ visualization (6 MCMC diagnostic plots)
- BayesianPipeline with fluent API

### **Smart Initialization** (NEW in v0.2.0)
- Automatic parameter initialization for fractional models
- Oscillation data: extracts features from G′, G″
- Template Method pattern (11 model initializers)
- Resolves Issue #9 (oscillation mode instability)

### **Model-Data Compatibility** (NEW in v0.2.0)
- Intelligent detection of physics mismatches
- Decay type classification (exponential, power-law, Mittag-Leffler)
- Material type detection (solid, liquid, gel)
- Enhanced error messages with recommendations

### **Data I/O**
- TRIOS (.xlsx) - TA Instruments rheometer data
- CSV/Excel - generic formats with auto-detection
- Anton Paar (.xlsx) - rheometer data
- HDF5 - full-fidelity storage with metadata
- Chunked reading for large files (>1GB)

### **Visualization**
- Publication-quality plots (3 styles: default, publication, presentation)
- Automatic plot selection by test mode
- BayesianPlotter: 6 MCMC diagnostics (pair, forest, energy, autocorr, rank, ESS)
- Template-based consistency

---

## 💡 Quick Examples

### Basic Model Fitting

```python
from rheojax.models import Maxwell
import numpy as np

# Oscillatory data
omega = np.logspace(-2, 2, 50)
G_star = ...  # Complex modulus [G′, G″]

# Fit model
model = Maxwell()
model.fit(omega, G_star, test_mode='oscillation')

# Get parameters
G0 = model.parameters.get_value('G0')
eta = model.parameters.get_value('eta')
tau = eta / G0

# Predict
G_pred = model.predict(omega)
```

### Bayesian Inference Workflow

```python
from rheojax.models import FractionalZenerSolidSolid

# 1. NLSQ point estimation (fast)
model = FractionalZenerSolidSolid()
model.fit(omega, G_star)

# 2. Bayesian inference with warm-start
result = model.fit_bayesian(
    omega, G_star,
    num_warmup=1000,
    num_samples=2000
)

# 3. Analyze results
print(f"Posterior mean α: {result.summary['alpha']['mean']:.4f}")
print(f"R-hat: {result.diagnostics['r_hat']['alpha']:.4f}")

intervals = model.get_credible_intervals(result.posterior_samples)
print(f"α 95% CI: {intervals['alpha']}")
```

### Fluent Pipeline API

```python
from rheojax.pipeline import BayesianPipeline

# Complete workflow
pipeline = BayesianPipeline()

(pipeline
    .load('data.csv', x_col='frequency', y_col='modulus')
    .fit_nlsq('fractional_zener_ss')
    .fit_bayesian(num_samples=2000)
    .plot_posterior()
    .plot_trace()
    .plot_pair(divergences=True)
    .save('results.hdf5'))

# Access diagnostics
diagnostics = pipeline.get_diagnostics()
summary = pipeline.get_posterior_summary()
```

### Time-Temperature Superposition

```python
from rheojax.transforms import Mastercurve

datasets = [data_25C, data_40C, data_55C]
temps = [298.15, 313.15, 328.15]  # K

mc = Mastercurve(
    reference_temp=313.15,
    shift_model='wlf',
    bounds={'C1': (8.0, 20.0), 'C2': (30.0, 150.0)}
)

master = mc.create_mastercurve(datasets, temps)
C1, C2 = mc.get_wlf_parameters()
```

---

## 📊 Documentation Statistics

### **Coverage**
- **Models:** 20/20 (100%) - 3 classical, 11 fractional, 6 flow
- **Transforms:** 5/5 (100%) - FFT, Mastercurve, MutationNumber, OWChirp, SmoothDerivative
- **Test Modes:** 4/4 (100%) - SAOS, Relaxation, Creep, Flow
- **Material Types:** 10+ categories documented
- **Examples:** 24 Jupyter notebooks

### **Quality Metrics**
- **Total documentation:** 12,000+ words
- **Average page length:** 600-900 words (3-6x original)
- **Material examples:** 100+ with fitted parameters
- **Cross-references:** 20-30 per page
- **References:** 200+ citations (foundational + modern 2020-2024)
- **Code examples:** 50+ complete workflows

### **Research Foundation**
- **Sources analyzed:** 100+ (textbooks, journals, reviews)
- **Time span:** 1867-2024 (Maxwell to present)
- **Key references:** 25 carefully selected
- **Rheological fundamentals:** 63-page comprehensive synthesis

---

## 🎓 Learning Paths

### **Beginner Path** (New to Rheology)
1. Start: [Core Concepts](source/user_guide/core_concepts.rst) - Learn rheological fundamentals
2. Practice: [Quickstart](source/quickstart.rst) - Fit your first model
3. Explore: [Model Selection](source/user_guide/model_selection.rst) - Understand when to use each model
4. Examples: Basic notebooks (Maxwell, Zener, PowerLaw)

### **Intermediate Path** (Experienced with Rheology)
1. Start: [Getting Started](source/user_guide/getting_started.rst) - RheoJAX workflow
2. Models: [Models Summary](source/models/summary.rst) - Quick reference
3. Transforms: [Transforms Summary](source/transforms/summary.rst) - Data preprocessing
4. Examples: Transform notebooks (FFT, Mastercurve)

### **Advanced Path** (Bayesian Inference & Fractional Models)
1. Start: [Bayesian Inference](source/user_guide/bayesian_inference.rst) - NLSQ → NUTS workflow
2. Theory: [Fractional Models](source/models/fractional/fractional_zener_ss.rst) - Mittag-Leffler functions
3. Practice: Bayesian notebooks (Priors, Diagnostics, Model Comparison)
4. Applications: Advanced notebooks (Multi-technique, Batch)

### **Industrial Path** (Practical Applications)
1. Materials: [Flow models by industry](source/models/summary.rst#industrial-applications)
2. Protocols: Experimental design sections in model docs
3. Troubleshooting: Fitting guidance tables
4. QC: [Multi-technique fitting](source/user_guide/multi_technique_fitting.rst)

---

## 🛠️ Building the Documentation Locally

### Prerequisites
```bash
pip install sphinx furo sphinx-autodoc-typehints
```

### Build HTML
```bash
cd docs
make clean
make html
```

The built documentation will be in `docs/build/html/index.html`.

### Build PDF (Optional)
```bash
make latexpdf
```

---

## 📝 Contributing to Documentation

We welcome contributions! Please see:
- **[Developer Guide](source/developer/contributing.rst)** - Contribution guidelines
- **[Architecture](source/developer/architecture.rst)** - Package design patterns

### Documentation Standards
- **Template:** All model/transform pages follow consistent structure
- **Examples:** Include worked examples with actual data
- **References:** Cite foundational and modern sources
- **Cross-links:** Connect related models, transforms, examples
- **Code:** Test all code snippets before committing

---

## 📚 Key References

### **Foundational Textbooks**
1. Ferry, J.D. (1980). *Viscoelastic Properties of Polymers* (3rd Ed.). Wiley.
2. Macosko, C.W. (1994). *Rheology: Principles, Measurements, and Applications*. Wiley-VCH.
3. Barnes, H.A., Hutton, J.F., Walters, K. (1989). *An Introduction to Rheology*. Elsevier.
4. Doi, M., Edwards, S.F. (1986). *The Theory of Polymer Dynamics*. Oxford.

### **Fractional Calculus**
5. Mainardi, F. (2010). *Fractional Calculus and Waves in Linear Viscoelasticity*. Imperial College Press.
6. Gorenflo, R., Kilbas, A.A., Mainardi, F., Rogosin, S.V. (2014). *Mittag-Leffler Functions*. Springer.

### **Modern Techniques**
7. Martinetti, L. et al. (2018). "Optimally Windowed Chirp Rheometry." *Physical Review X*, 8(4), 041042.
8. Winter, H.H., Chambon, F. (1986). "Analysis of Linear Viscoelasticity of a Crosslinking Polymer." *J. Rheology*, 30(2), 367-382.

### **Recent Reviews (2020-2024)**
9. Tutorial reviews on fractional models in *Polymer Chemistry* (RSC, 2024)
10. Non-Maxwellian relaxations in *Soft Matter* (RSC, 2023)

See individual model/transform pages for complete bibliographies with 200+ citations.

---

## 🔗 Useful Links

- **[RheoJAX GitHub](https://github.com/username/rheojax)** - Source code and issue tracker
- **[PyPI Package](https://pypi.org/project/rheojax/)** - Installation via pip
- **[Example Notebooks](source/examples/index.rst)** - 24 tutorials
- **[Migration Guide](source/migration_guide.rst)** - Upgrade from v0.1.x

---

## 📄 License

RheoJAX is released under the MIT License. See LICENSE file for details.

---

## 🙏 Acknowledgments

This documentation builds upon decades of rheological research by:
- James Clerk Maxwell (1867) - Viscoelastic theory foundations
- John D. Ferry (1980) - Polymer viscoelasticity
- Christopher W. Macosko (1994) - Rheological measurements
- Francesco Mainardi (2010) - Fractional calculus applications

Modern enhancements incorporate techniques from:
- NumPyro team - Bayesian inference with NUTS
- JAX team - Automatic differentiation and JIT compilation
- ArviZ team - MCMC diagnostics and visualization

---

## 📮 Support

- **Documentation Issues:** [GitHub Issues](https://github.com/username/rheojax/issues)
- **Questions:** [Discussions](https://github.com/username/rheojax/discussions)
- **Email:** support@rheojax.org (for private inquiries)

---

**Last Updated:** November 2025
**Documentation Version:** 0.2.0
**Build Status:** ✅ Clean (0 errors, 0 warnings)

*Transforming rheological analysis through JAX acceleration and comprehensive Bayesian inference.*
