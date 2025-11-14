# Notebook Validation Tests - Tiered Testing Strategy

## Overview

The notebook validation tests in `test_migrated_notebooks.py` validate all 24 example notebooks across 4 categories (basic, transforms, bayesian, advanced). To optimize CI/CD pipeline performance, tests are organized into 3 tiers.

## Test Tiers

### Tier 1: Structural Tests (Fast - ~5 seconds)
**What:** Verify notebook files exist and directory structure is correct
**When to run:** Every CI build
**Command:**
```bash
pytest tests/validation/test_migrated_notebooks.py -m "not slow"
```

**Tests:**
- `test_*_structure` - Check directory structure
- `test_*_exists` - Verify notebook files exist
- `test_*_all_present` - Check all notebooks are present

**Time savings:** ~99% (5s vs 60+ minutes)

---

### Tier 2: Execution Tests (Medium - ~15-20 minutes)
**What:** Execute notebooks to completion, verify no runtime errors
**When to run:** Standard CI builds (on PR, push to main)
**Command:**
```bash
pytest tests/validation/test_migrated_notebooks.py -m "slow and not notebook_comprehensive"
```

**Tests:**
- `test_*_execution` - Execute notebook end-to-end
- `test_*_executes` - Verify successful execution
- Basic output checks (no deep numerical validation)

**Time savings:** ~60-70% (~20min vs 60+ minutes)

---

### Tier 3: Comprehensive Validation (Slow - ~60+ minutes)
**What:** Full numerical accuracy and Bayesian convergence validation
**When to run:** Weekly/nightly CI, manual runs, release validation
**Command:**
```bash
pytest tests/validation/test_migrated_notebooks.py
# OR explicitly:
pytest tests/validation/test_migrated_notebooks.py -m "notebook_comprehensive or not slow"
```

**Tests marked with `@pytest.mark.notebook_comprehensive`:**
- `test_*_parameters` - Validate fitted parameter accuracy (relative error < 1e-6)
- `test_*_convergence` - Check Bayesian MCMC convergence (R-hat < 1.01, ESS > 400)
- `test_*_validation` - Numerical accuracy checks
- `test_*_detection` - Specific feature detection validation
- All other detailed validation tests

**Time:** Full test suite time (~60+ minutes on single core)

---

## CI Configuration Examples

### GitHub Actions `.github/workflows/test.yml`

```yaml
# Fast CI (on every push/PR)
- name: Run fast tests
  run: pytest -m "not slow" --cov

# Standard CI (on PR to main)
- name: Run standard tests (skip comprehensive)
  run: pytest -m "not notebook_comprehensive" --cov

# Weekly/nightly full validation
- name: Run full test suite
  run: pytest --cov
  if: github.event_name == 'schedule'
```

### Local Development

```bash
# Fast feedback during development
make test-fast  # Equivalent to: pytest -m "not slow"

# Standard test run (skip comprehensive validation)
pytest -m "not notebook_comprehensive"

# Full validation before release
make test  # Runs all tests including comprehensive
```

---

## Performance Comparison

| Tier | Tests Run | Approximate Time | Use Case |
|------|-----------|------------------|----------|
| Tier 1 (Structural) | ~12 | ~5 seconds | Every commit |
| Tier 2 (Execution) | ~48 | ~15-20 minutes | PRs, main branch |
| Tier 3 (Full) | ~88 | ~60+ minutes | Weekly, releases |

**Note:** Times are approximate and depend on hardware (CPU, GPU availability).

---

## Future Optimizations

1. **Parallel Execution** (Phase 3): Use `pytest-xdist` for 2-4x speedup
   ```bash
   pytest -n auto tests/validation/test_migrated_notebooks.py
   ```

2. **Notebook Caching**: Cache notebook execution results for unchanged notebooks

3. **Sampling Strategy**: Test representative subset of notebooks per run, rotate weekly

4. **GPU Acceleration**: Run GPU-dependent tests only on GPU-enabled runners

---

## Adding New Tests

When adding new validation tests to `test_migrated_notebooks.py`:

1. **Structural checks** - No markers needed (fast by default)
2. **Execution tests** - Add `@pytest.mark.slow` and `@pytest.mark.validation`
3. **Comprehensive validation** - Add `@pytest.mark.slow` + `@pytest.mark.notebook_comprehensive`

Example:
```python
class TestNewNotebook:
    # Tier 1: Structural (no markers)
    def test_new_notebook_exists(self, examples_dir):
        """Verify notebook file exists."""
        assert (examples_dir / "category" / "notebook.ipynb").exists()

    # Tier 2: Execution
    @pytest.mark.validation
    @pytest.mark.slow
    def test_new_notebook_executes(self, examples_dir):
        """Execute notebook and verify no errors."""
        nb = _execute_notebook(examples_dir / "category" / "notebook.ipynb")
        assert nb is not None

    # Tier 3: Comprehensive
    @pytest.mark.validation
    @pytest.mark.slow
    @pytest.mark.notebook_comprehensive
    def test_new_notebook_parameter_accuracy(self, examples_dir):
        """Validate fitted parameters match expected values."""
        # Detailed numerical validation...
```

---

## Markers Reference

- `@pytest.mark.slow` - Test takes >5 seconds (notebook execution)
- `@pytest.mark.validation` - Validation against expected behavior
- `@pytest.mark.notebook_comprehensive` - Comprehensive numerical/convergence validation
- `@pytest.mark.notebook_smoke` - Basic framework smoke test
- `@pytest.mark.gpu` - Requires GPU hardware

---

## Troubleshooting

**Q: How do I run only Tier 2 tests?**
A: `pytest tests/validation -m "slow and not notebook_comprehensive"`

**Q: How do I run a specific notebook's tests?**
A: `pytest tests/validation -k "test_fft_" -v`

**Q: Tests are still slow in CI. What's wrong?**
A: Check that CI is using the tier-appropriate marker expression. For standard CI, use `-m "not notebook_comprehensive"`.

**Q: How do I add parallel execution?**
A: Install `pytest-xdist` and use `pytest -n auto` (see Phase 3 optimizations).

---

**Last Updated:** 2025-11-14
**Version:** 0.2.0 (Tiered testing strategy implemented)
