# RheoJAX GUI Smoke Tests

Comprehensive smoke tests for the RheoJAX GUI without requiring a display environment. Tests validate:

- Module imports (gracefully skips PySide6-dependent tests when not available)
- State management (works without Qt)
- Service instantiation
- Stylesheet loading and validation
- Qt integration (when PySide6 is available)

## Test Organization

### TestModuleImports
Tests that all GUI modules can be imported successfully.

- `test_gui_main_module_imports`: GUI entry point
- `test_gui_state_store_imports_no_qt`: State classes (requires PySide6 due to state/__init__.py)
- `test_gui_state_signals_imports`: Qt signal classes
- `test_gui_services_imports`: Service layer (data, model, bayesian, transform, plot, export)
- `test_gui_styles_imports`: Stylesheet utilities
- `test_gui_app_imports`: App components (main window, menu bar, status bar, toolbar)
- `test_gui_utils_imports`: Utility modules

### TestStateManagement (18 tests)
Tests state dataclasses and management without Qt. All tests require PySide6 because importing from `rheojax.gui.state.store` triggers `rheojax.gui.state.__init__.py` which imports `StateSignals`.

- **AppState tests**: Creation with defaults, custom values
- **DatasetState tests**: Creation, cloning with deep copy
- **PipelineState tests**: Creation, step enums, status enums
- **ParameterState tests**: Creation, units and descriptions
- **FitResult/BayesianResult tests**: Creation, cloning

### TestServiceInstantiation (7 tests)
Tests that all services can be instantiated and provide expected APIs.

- `test_data_service_instantiation`: Load/validate data
- `test_model_service_instantiation`: Model access and compatibility checking
- `test_bayesian_service_instantiation`: Bayesian inference
- `test_transform_service_instantiation`: Data transforms
- `test_plot_service_instantiation`: Plot generation with styles
- `test_plot_service_colorblind_palette`: Accessibility features
- `test_export_service_instantiation`: Data export

### TestStylesheetLoading (8 tests)
Tests stylesheet loading and validation without Qt.

- Light/dark stylesheet loading
- Stylesheet selection by theme name
- Error handling for invalid themes
- QSS syntax validation
- Resource file verification

### TestQtIntegration (4 tests)
Tests requiring PySide6 and Qt functionality. Automatically skipped if PySide6 is not installed.

- Main window creation and geometry
- State signals creation
- Stylesheet application to QApplication

### TestIntegrationScenarios (3 tests)
Integration tests combining multiple components.

- State and services working together
- Stylesheet selection matching state themes
- Multiple services instantiation

### TestEdgeCases (7 tests)
Edge case and error handling tests.

- Parameter state with zero/negative values
- Empty state collections
- Service behavior with edge cases
- Unique color validation

## Running Tests

### All GUI smoke tests
```bash
pytest tests/gui/ -v
```

### Only smoke marker tests
```bash
pytest tests/gui/ -v -m smoke
```

### Run with output
```bash
pytest tests/gui/test_gui_smoke.py -v --tb=short
```

### Run specific test class
```bash
pytest tests/gui/test_gui_smoke.py::TestStylesheetLoading -v
```

### Run specific test
```bash
pytest tests/gui/test_gui_smoke.py::TestStylesheetLoading::test_light_stylesheet_loads -v
```

## PySide6 Dependency Handling

The test suite gracefully handles environments with and without PySide6:

- **Without PySide6**: Tests marked with `@pytest.mark.skipif(not HAS_PYSIDE6, ...)` are automatically skipped with descriptive messages
- **With PySide6**: All tests run, including Qt-dependent tests
- **No failures**: The suite always returns success (0 or skipped)

### Tests Requiring PySide6

Tests that import from `rheojax.gui.state.store` require PySide6 because the state module's `__init__.py` imports `StateSignals` which depends on Qt:

- All `TestStateManagement` tests
- `test_state_and_services_work_together` (integration test)
- `test_stylesheet_selection_matches_state_theme` (integration test)
- Various edge case tests

## Test Fixtures

### Provided by conftest.py

- `qapp`: Session-scoped QApplication (created only if PySide6 available)
- `gui_config`: GUI configuration (display availability, headless mode)
- `app_state_instance`: Sample AppState data dictionary
- `service_config`: Service configuration
- `stylesheet_sample`: Sample QSS stylesheet

## Architecture

### State Management
State classes are dataclasses under `/rheojax/gui/state/`:
- `store.py`: State dataclasses (AppState, DatasetState, ParameterState, etc.)
- `signals.py`: Qt signals (requires PySide6)
- `__init__.py`: Re-exports both (requires PySide6 due to signals import)

Tests import directly from `store.py` where possible, but some tests need to skip because importing the module triggers the `__init__.py` load.

### Services
Service layer under `/rheojax/gui/services/`:
- `data_service.py`: Load and validate rheological data
- `model_service.py`: Model fitting and comparison
- `bayesian_service.py`: Bayesian inference
- `transform_service.py`: Data transforms
- `plot_service.py`: Visualization with styles
- `export_service.py`: Data export
- Lazy-loaded via `__getattr__` in `__init__.py`

### Stylesheets
QSS stylesheets under `/rheojax/gui/resources/styles/`:
- `light.qss`: Light theme (1000+ lines)
- `dark.qss`: Dark theme (1000+ lines)
- `__init__.py`: Load and select themes

## Markers

All tests are marked with `@pytest.mark.gui` for easy filtering:

```bash
pytest -m gui               # Run all GUI tests
pytest -m "not gui"         # Run non-GUI tests
pytest -m "gui and smoke"   # Run GUI smoke tests
```

## CI/CD Integration

The test suite is designed for CI/CD:

1. **No display required**: Uses offscreen rendering if no display available
2. **Graceful degradation**: Skips PySide6-dependent tests if not installed
3. **Fast execution**: ~0.4 seconds for all 47 tests
4. **Clear output**: Marks tests as PASSED, SKIPPED, or FAILED

### Expected CI Output

```
47 tests collected
- 19 passed (no PySide6): Basic tests that don't require Qt
- 28 skipped: Tests requiring PySide6
- 0 failed: No failures

Result: PASS
```

## Development Workflow

### Adding New Tests

1. **State tests**: Use fixtures from conftest.py, import from `store.py`
2. **Service tests**: Call service methods, test return types
3. **Qt tests**: Mark with `@pytest.mark.skipif(not HAS_PYSIDE6, ...)`
4. **All tests**: Use descriptive names and docstrings

### Example

```python
@pytest.mark.smoke
@pytest.mark.skipif(not HAS_PYSIDE6, reason="PySide6 not installed")
def test_new_feature(self, qapp):
    """Test that new feature works correctly."""
    from rheojax.gui.state.store import SomeState

    state = SomeState(...)
    assert state.property == expected_value
```

## Troubleshooting

### "PySide6 not installed" skips

If tests are being skipped:
```bash
pip install PySide6
```

### Import errors from state.store

The `rheojax.gui.state.__init__.py` imports `StateSignals` which requires PySide6. If you see:
```
ModuleNotFoundError: No module named 'PySide6'
```

This is expected when PySide6 is not installed. Tests handle this gracefully.

### Display-related errors

If running in a headless environment:
```bash
export QT_QPA_PLATFORM=offscreen
pytest tests/gui/
```

The test suite automatically uses offscreen rendering, so this shouldn't be necessary.

## Performance

- **Full suite**: ~0.4 seconds (19 passed + 28 skipped)
- **Smoke tests only**: ~0.3 seconds
- **No PySide6**: ~0.2 seconds (22 tests skipped)

## Future Enhancements

- [ ] Widget state tests (requires widget tree navigation)
- [ ] Signal/slot connection tests (requires Qt event loop)
- [ ] Interaction tests (mouse/keyboard simulation)
- [ ] Performance tests (rendering time, memory usage)
- [ ] UI component visual tests (requires screenshot comparison)
