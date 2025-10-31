# Tutorial Notebooks Have Moved!

**The example notebooks have been reorganized and moved to a new location.**

## New Location

All tutorial notebooks are now in the `examples/` directory at the project root:

```
Rheo/
└── examples/
    ├── README.md              # Complete tutorial guide
    ├── basic/                 # Basic model fitting (5 notebooks)
    ├── transforms/            # Data analysis workflows (6 notebooks)
    ├── bayesian/              # Bayesian inference (5 notebooks)
    └── advanced/              # Advanced patterns (6 notebooks)
```

## Migration Guide

### For Users with Bookmarks

**Update your bookmarks from:**
```
docs/examples/
```

**To:**
```
examples/
```

### Notebook Mapping

Legacy notebooks have been replaced with improved versions:

| Legacy (`docs/examples/`) | New Location (`examples/`) | Status |
|---------------------------|----------------------------|--------|
| `basic_model_fitting.ipynb` | `basic/01-maxwell-fitting.ipynb`<br>`basic/02-zener-fitting.ipynb`<br>`basic/03-springpot-fitting.ipynb` | ✅ Superseded by 3 focused notebooks |
| `mastercurve_generation.ipynb` | `transforms/02b-mastercurve-wlf-validation.ipynb` | ✅ Migrated with improvements |
| `multi_model_comparison.ipynb` | `advanced/06-frequentist-model-selection.ipynb` | ✅ Migrated (AIC/BIC model selection) |
| `multi_technique_fitting.ipynb` | `advanced/01-multi-technique-fitting.ipynb` | ✅ Covered in new comprehensive version |
| `advanced_workflows.ipynb` | `advanced/03-custom-models.ipynb`<br>`advanced/05-performance-optimization.ipynb` | ✅ Core concepts integrated |

### For Scripts and Automation

If you have scripts referencing the old location, update your paths:

```python
# Old
notebook_path = "docs/examples/basic_model_fitting.ipynb"

# New - See examples/README.md for complete listing
notebook_path = "examples/basic/01-maxwell-fitting.ipynb"
```

### What's New?

The new tutorial structure provides:

✅ **22 comprehensive notebooks** (vs 5 legacy notebooks)
✅ **4 clear learning paths**: Basic → Transforms → Bayesian → Advanced
✅ **13-16 hours of content** with estimated completion times
✅ **Better organization** by topic and difficulty
✅ **More depth** with focused notebooks per model/technique
✅ **Improved documentation** with learning objectives and prerequisites

## Get Started

**📚 Read the complete tutorial guide:**
```bash
cat examples/README.md
```

**🚀 Start with the basics:**
```bash
jupyter notebook examples/basic/01-maxwell-fitting.ipynb
```

**📖 Browse online documentation:**
- [Tutorial Index](https://rheojax.readthedocs.io/en/latest/examples/index.html)
- [User Guide](https://rheojax.readthedocs.io/en/latest/user_guide.html)

## Questions?

- **Issues**: [GitHub Issues](https://github.com/imewei/Rheo/issues)
- **Documentation**: [Read the Docs](https://rheojax.readthedocs.io/)
- **Examples**: `examples/README.md`

---

**Last Updated:** 2025-10-31
**Migration Date:** Commit f284b8a
