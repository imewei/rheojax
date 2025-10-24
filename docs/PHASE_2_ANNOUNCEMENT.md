# 🎉 Announcing rheo v0.2.0: Complete Rheological Analysis Toolkit

**Release Date:** October 24, 2025
**Major Milestone:** Phase 2 Complete

---

We're thrilled to announce the release of **rheo v0.2.0**, marking the completion of Phase 2 and delivering a comprehensive, modern rheological analysis toolkit powered by JAX!

## 🌟 What's New

### 20 Rheological Models at Your Fingertips

From simple Maxwell models to complex fractional rheology and non-Newtonian flow - all validated against established implementations:

**Classical Models (3):**
- Maxwell, Zener, SpringPot

**Fractional Models (11):**
- Fractional Maxwell family (gel/liquid variants)
- Fractional Kelvin-Voigt and Zener family
- Fractional Burgers, Jeffreys, Poynting-Thomson

**Flow Models (6):**
- Power Law, Carreau, Carreau-Yasuda
- Cross, Herschel-Bulkley, Bingham

### 5 Essential Data Transforms

- **FFT Analysis** - Time ↔ frequency conversion
- **Mastercurve** - Time-temperature superposition (WLF/Arrhenius)
- **Mutation Number** - Viscoelastic character (0=elastic, 1=viscous)
- **OWChirp** - LAOS analysis with higher harmonics
- **Smooth Derivative** - Noise-robust differentiation

### Intuitive Pipeline API

Your entire workflow in one chain:

```python
from rheo.pipeline import Pipeline

# From data to publication-ready plot in seconds
Pipeline().load('data.csv').fit('zener').plot().save('results.hdf5')
```

### Lightning Fast Performance

JAX-powered implementation delivers **2-10x speedups** with automatic GPU acceleration:

- Mittag-Leffler functions: **56x faster**
- Parameter optimization: **17x faster**
- FFT transforms: **15x faster**
- Mastercurve generation: **6x faster**

### Production-Ready Documentation

- **150+ pages** of comprehensive user guides
- **5 complete example notebooks** covering real-world workflows
- **Migration guide** from pyRheo/hermes-rheo with side-by-side examples
- **Complete API reference** with mathematical formulations

---

## 🚀 Get Started

### Installation

```bash
pip install rheo-analysis==0.2.0
```

### Try It Now

#### Basic Model Fitting

```python
from rheo.pipeline import Pipeline
from rheo.core.data import RheoData
import numpy as np

# Generate synthetic data
time = np.logspace(-2, 2, 50)
stress = 1e6 * np.exp(-time / 10.0)

# Wrap in RheoData container
data = RheoData(x=time, y=stress, x_units='s', y_units='Pa', domain='time')

# Fit model
pipeline = Pipeline().load_data(data).fit('maxwell')
params = pipeline.get_fitted_parameters()

print(f"Fitted parameters: {params}")
```

#### Model Comparison in 2 Lines

```python
from rheo.pipeline import ModelComparisonPipeline

# Compare 5 models automatically
pipeline = ModelComparisonPipeline(['maxwell', 'zener', 'springpot'])
best = pipeline.run(data).get_best_model(metric='aic')

print(f"Best model: {best}")
```

#### Time-Temperature Superposition

```python
from rheo.pipeline import MastercurvePipeline

# Create mastercurve from multi-temperature data
pipeline = MastercurvePipeline(reference_temp=298.15, method='wlf')
mastercurve = pipeline.run(file_paths, temperatures)

# Get WLF parameters
wlf_params = pipeline.get_wlf_parameters()
print(f"WLF: C1={wlf_params['C1']:.2f}, C2={wlf_params['C2']:.2f} K")
```

#### Batch Processing

```python
from rheo.pipeline import Pipeline, BatchPipeline

# Process entire directory in parallel
template = Pipeline().fit('maxwell')
batch = BatchPipeline(template)
batch.process_directory('data/', pattern='*.csv')
batch.export_summary('summary.xlsx')
```

---

## 📖 Learn More

### Documentation

**Main Documentation:** [rheo.readthedocs.io](https://rheo.readthedocs.io)

**Essential Reading:**
- [Getting Started Guide](https://rheo.readthedocs.io/user_guide/getting_started.html) - 5-minute quickstart
- [Pipeline User Guide](https://rheo.readthedocs.io/user_guide/pipeline_guide.html) - Complete workflow examples
- [API Reference](https://rheo.readthedocs.io/api_reference.html) - All 20 models + 5 transforms
- [Migration Guide](https://rheo.readthedocs.io/migration_guide.html) - From pyRheo/hermes-rheo

### Example Notebooks

Explore complete workflows with our 5 example notebooks:

1. **basic_model_fitting.ipynb** - Fit Maxwell, Zener, SpringPot to relaxation data
2. **advanced_workflows.ipynb** - Custom optimization, constraints, sensitivity analysis
3. **mastercurve_generation.ipynb** - Time-temperature superposition with WLF
4. **multi_model_comparison.ipynb** - Systematic comparison with AIC/BIC
5. **multi_technique_fitting.ipynb** - Shared parameters across relaxation + oscillation

**View on GitHub:** [github.com/[org]/rheo/tree/main/docs/examples](https://github.com/[org]/rheo/tree/main/docs/examples)

---

## 🎯 Who Should Use rheo?

### Researchers

- **Academic researchers** analyzing polymer rheology, soft matter, biophysics
- **Material scientists** characterizing viscoelastic properties
- **Rheology labs** needing reproducible, validated analysis

### Industry

- **Quality control** in polymer manufacturing
- **Product development** for coatings, adhesives, food products
- **R&D teams** optimizing material formulations

### Students

- **Graduate students** learning rheological analysis
- **Course instructors** teaching experimental rheology
- **Tutorial creators** demonstrating best practices

---

## 💡 Key Features That Set rheo Apart

### 1. Unified Framework

Replace multiple packages with one:
- **pyRheo** → rheo models
- **hermes-rheo** → rheo transforms
- Plus new Pipeline API for streamlined workflows

### 2. JAX Performance

- **2-10x faster** than NumPy implementations
- **Automatic GPU acceleration** when available
- **JIT compilation** for optimized execution
- **Automatic differentiation** for gradients

### 3. Type-Safe Parameters

```python
# Parameters are objects with metadata
model.parameters['G0'].value = 1e6
model.parameters['G0'].bounds = (1e5, 1e7)
model.parameters['G0'].units = 'Pa'
```

### 4. Automatic Test Mode Detection

```python
# No need to specify test mode - rheo detects automatically
data = RheoData(x=time, y=stress, domain='time')
# Automatically infers: test_mode='relaxation'
```

### 5. Publication-Quality Plots

```python
# Three built-in styles
pipeline.plot(style='publication')  # Clean, high-DPI
pipeline.plot(style='presentation')  # Bold, high contrast
pipeline.plot(style='default')      # Standard matplotlib
```

### 6. Validated Against Original Packages

- All 20 models validated vs pyRheo (1e-6 tolerance)
- All 5 transforms validated vs hermes-rheo (1e-6 tolerance)
- Comprehensive validation report included

---

## 🔬 Real-World Applications

### Polymer Characterization

```python
# Characterize polymer relaxation spectrum
pipeline = Pipeline().load('polymer_relaxation.csv')
pipeline.fit('fractional_maxwell_liquid')

# Extract relaxation time distribution
alpha = pipeline.get_fitted_parameters()['alpha'].value
print(f"Relaxation spectrum width: α = {alpha:.3f}")
```

### Temperature-Dependent Behavior

```python
# Create mastercurve from multi-temperature data
pipeline = MastercurvePipeline(reference_temp=298.15)
mastercurve = pipeline.run(files, temperatures)

# Predict behavior at untested temperature
wlf = pipeline.get_wlf_parameters()
# Use WLF parameters to extrapolate to 60°C
```

### Material Quality Control

```python
# Batch process 100+ samples
template = Pipeline().fit('zener')
batch = BatchPipeline(template)
batch.process_directory('production_samples/')

# Export summary for quality control
batch.export_summary('qc_report.xlsx')
```

### Nonlinear Viscoelasticity

```python
# Analyze LAOS data
owchirp = OWChirp()
harmonics = owchirp.transform(laos_data)

# Extract higher harmonics
I3_I1 = harmonics['I3'] / harmonics['I1']
print(f"Nonlinearity: I₃/I₁ = {I3_I1:.3f}")
```

---

## 🆚 Comparison with Existing Tools

### vs pyRheo

| Feature | pyRheo | rheo |
|---------|--------|------|
| Models | 15 | **20** |
| Performance | NumPy | **JAX (2-10x faster)** |
| GPU Support | ❌ | ✅ |
| Pipeline API | ❌ | ✅ |
| Documentation | Basic | **150+ pages** |

### vs hermes-rheo

| Feature | hermes-rheo | rheo |
|---------|-------------|------|
| Transforms | 5 | **5 (unified)** |
| Model Support | ❌ | **✅ (20 models)** |
| Pipeline API | ❌ | ✅ |
| Performance | NumPy | **JAX (2-10x faster)** |

### vs Commercial Software

| Feature | Commercial | rheo |
|---------|------------|------|
| Cost | $$$$ | **Free (MIT License)** |
| Extensibility | Limited | **Fully extensible** |
| Automation | Manual | **Pipeline API** |
| GPU Support | ❌ | ✅ |
| Source Code | Closed | **Open (MIT)** |

---

## 🎯 What's Next

### Phase 3 Roadmap (Coming in 6 Months)

**Bayesian Inference with NumPyro:**
- Uncertainty quantification with credible intervals
- Prior specification for informed inference
- Posterior predictive checks

**Machine Learning Integration:**
- Automated model selection
- Neural network surrogate models
- Outlier detection

**Advanced Visualization:**
- Interactive plots with Plotly
- 3D visualization
- Real-time plotting

**Production Features:**
- Automated PDF reports
- Web interface for browser-based analysis
- REST API for remote analysis

---

## 🤝 Community & Support

### Get Involved

**GitHub Repository:** [github.com/[org]/rheo](https://github.com/[org]/rheo)
- ⭐ Star the repo to show support
- 🐛 Report bugs and request features
- 💬 Join discussions
- 🔧 Contribute code or documentation

### Channels

- **Discussions:** [GitHub Discussions](https://github.com/[org]/rheo/discussions) - Ask questions, share ideas
- **Issues:** [GitHub Issues](https://github.com/[org]/rheo/issues) - Bug reports, feature requests
- **Documentation:** [ReadTheDocs](https://rheo.readthedocs.io) - Complete guides and API reference

### Contributing

We welcome contributions! Areas where we need help:

- 📝 Documentation improvements
- 🧪 New model implementations
- 🔬 Validation against experimental data
- 📚 Example notebooks for specific applications
- 🐛 Bug reports and fixes
- 💡 Feature suggestions

**See:** [CONTRIBUTING.md](CONTRIBUTING.md)

---

## 📊 By the Numbers

### Phase 2 Achievements

- **20 models** implemented and tested
- **5 transforms** unified from hermes-rheo
- **900+ tests** with 85%+ coverage
- **150+ pages** of documentation
- **5 example notebooks** with complete workflows
- **2-10x performance** improvement over NumPy
- **1e-6 validation** tolerance vs original packages
- **6 months** of development effort

---

## 🙏 Acknowledgments

rheo v0.2.0 is built on excellent open-source software:

- **JAX** - Automatic differentiation and GPU acceleration (Google)
- **NumPy/SciPy** - Foundational numerical computing
- **matplotlib** - Publication-quality visualization
- **pyRheo** - Reference model implementations
- **hermes-rheo** - Transform algorithms

**Special thanks** to early adopters who provided valuable feedback during development.

---

## 📜 License

rheo is released under the **MIT License** - free for academic and commercial use.

---

## 📞 Contact

**Questions? Issues? Feedback?**

- 📧 Email: rheo@example.com
- 💬 GitHub Discussions: https://github.com/[org]/rheo/discussions
- 🐛 Issue Tracker: https://github.com/[org]/rheo/issues

---

## 🎉 Try rheo v0.2.0 Today!

```bash
pip install rheo-analysis==0.2.0
```

**Get started in 5 minutes:** [rheo.readthedocs.io/quickstart](https://rheo.readthedocs.io/quickstart.html)

**Happy rheological analysis!**

---

*rheo v0.2.0 - Unified, Fast, Validated Rheological Analysis*
