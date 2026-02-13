Contributing to RheoJAX
====================

We welcome contributions! This guide will help you get started.

Getting Started
---------------

Development Setup
~~~~~~~~~~~~~~~~~

1. **Fork and Clone**

   .. code-block:: bash

       # Fork on GitHub, then clone your fork
       git clone https://github.com/imewei/rheojax.git
       cd rheojax

2. **Create Virtual Environment**

   .. code-block:: bash

       # Using venv
       python3.12 -m venv .venv
       source .venv/bin/activate  # On Windows: .venv\Scripts\activate

       # Or using conda
       conda create -n rheojax python=3.12
       conda activate rheojax

3. **Install in Development Mode**

   .. code-block:: bash

       # Install with uv (preferred — manages venv + dependencies)
       uv sync

       # Or with pip (legacy)
       pip install -e ".[dev]"

4. **Install Pre-commit Hooks**

   .. code-block:: bash

       # Install pre-commit hooks
       pre-commit install

       # Test hooks (optional)
       pre-commit run --all-files

5. **Verify Installation**

   .. code-block:: bash

       # Run smoke tests (1077 tests, ~30s-2min)
       uv run pytest -n auto -m "smoke"

       # Run standard tests (3349 tests, ~5-10 min)
       uv run pytest -n auto -m "not slow"

       # Run full suite (3576 tests)
       uv run pytest -n auto

       # Quick format + lint + smoke
       make format && make quick

       # Check imports
       python -c "import rheojax; print(rheojax.__version__)"

Development Workflow
--------------------

Branch Strategy
~~~~~~~~~~~~~~~

.. code-block:: bash

    # Create feature branch
    git checkout -b feature/your-feature-name

    # Or bug fix branch
    git checkout -b fix/bug-description

    # Make changes, commit, push
    git add .
    git commit -m "Add feature X"
    git push origin feature/your-feature-name

Commit Messages
~~~~~~~~~~~~~~~

Follow conventional commits format:

.. code-block:: text

    <type>(<scope>): <subject>

    <body>

    <footer>

**Types:**
- ``feat``: New feature
- ``fix``: Bug fix
- ``docs``: Documentation changes
- ``style``: Code style changes (formatting, no logic change)
- ``refactor``: Code refactoring
- ``test``: Adding or updating tests
- ``chore``: Build process, dependencies

**Examples:**

.. code-block:: text

    feat(models): add Carreau-Yasuda flow model

    Implement Carreau-Yasuda model with JAX support and
    automatic parameter bounds.

    Closes #123

.. code-block:: text

    fix(io): handle missing units in TRIOS files

    TRIOS files sometimes omit units in column headers.
    Now defaults to standard rheological units.

    Fixes #456

Code Standards
--------------

Style Guide
~~~~~~~~~~~

We follow PEP 8 with some modifications:

- Line length: 100 characters (not 79)
- Use double quotes for strings
- Use trailing commas in multi-line structures

.. code-block:: python

    # Good
    def function_name(
        param1: str,
        param2: int,
        param3: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Function with good style."""
        result = {
            "key1": value1,
            "key2": value2,
        }
        return result

    # Bad
    def function_name(param1,param2,param3=None):
        result={'key1':value1,'key2':value2}
        return result

Type Hints
~~~~~~~~~~

Use type hints for all public functions:

.. code-block:: python

    from typing import Optional, Union, List, Dict, Any
    import numpy as np
    import jax.numpy as jnp

    ArrayLike = Union[np.ndarray, jnp.ndarray, List]

    def process_data(
        data: RheoData,
        method: str = "default",
        parameters: Optional[Dict[str, Any]] = None
    ) -> RheoData:
        """Process rheological data."""
        pass

Docstrings
~~~~~~~~~~

Use NumPy-style docstrings:

.. code-block:: python

    def fit_model(
        data: RheoData,
        model_name: str,
        initial_params: Optional[Dict[str, float]] = None
    ) -> BaseModel:
        """Fit rheological model to data.

        Parameters
        ----------
        data : RheoData
            Rheological data to fit
        model_name : str
            Name of model to fit (e.g., "Maxwell", "Zener")
        initial_params : dict, optional
            Initial parameter values, by default None

        Returns
        -------
        BaseModel
            Fitted model instance

        Raises
        ------
        ValueError
            If model_name is not recognized
        RuntimeError
            If fitting fails to converge

        Examples
        --------
        >>> data = RheoData(x=time, y=stress)
        >>> model = fit_model(data, "Maxwell")
        >>> predictions = model.predict(time)

        Notes
        -----
        Uses JAX automatic differentiation for gradient-based optimization.

        See Also
        --------
        BaseModel : Base class for all models
        nlsq_optimize : Optimization function

        References
        ----------
        .. [1] Maxwell, J.C. "On the dynamical theory of gases",
               Phil. Trans. R. Soc., 1867.
        """
        pass

Imports
~~~~~~~

Organize imports in this order:

.. code-block:: python

    # Standard library
    import os
    import sys
    from pathlib import Path
    from typing import Optional, Union

    # Third-party
    import numpy as np
    import jax
    import jax.numpy as jnp
    from scipy.optimize import minimize

    # Local imports
    from rheojax.core import RheoData, BaseModel
    from rheojax.utils import nlsq_optimize

Testing
-------

Writing Tests
~~~~~~~~~~~~~

Every new feature needs tests:

.. code-block:: python

    # tests/test_new_feature.py
    import pytest
    import numpy as np
    from rheojax.core import RheoData

    def test_rheodata_creation():
        """Test RheoData initialization."""
        x = np.array([1, 2, 3])
        y = np.array([10, 20, 30])
        data = RheoData(x=x, y=y)

        assert len(data.x) == 3
        assert data.shape == (3,)
        assert data.x_units is None

    def test_rheodata_validation():
        """Test RheoData validates input."""
        with pytest.raises(ValueError, match="x and y must have the same shape"):
            RheoData(x=np.array([1, 2]), y=np.array([10, 20, 30]))

    @pytest.mark.parametrize("domain,expected", [
        ("time", "time"),
        ("frequency", "frequency"),
    ])
    def test_rheodata_domain(domain, expected):
        """Test RheoData domain handling."""
        data = RheoData(x=np.array([1]), y=np.array([10]), domain=domain)
        assert data.domain == expected

Running Tests
~~~~~~~~~~~~~

RheoJAX uses a **tiered testing strategy** with pytest markers:

.. code-block:: bash

    # Tier 1: Smoke tests (~1077 tests, ~30s-2min) — used in CI
    uv run pytest -n auto -m "smoke"

    # Tier 2: Standard tests (~3349 tests, ~5-10 min)
    uv run pytest -n auto -m "not slow"

    # Tier 3: Full suite (~3576 tests, includes Bayesian slow tests)
    uv run pytest -n auto

    # Quick format + lint + smoke
    make format && make quick

    # Run specific test file
    uv run pytest tests/core/test_data.py

    # Run specific test
    uv run pytest tests/core/test_data.py::test_rheodata_creation

    # Run with coverage
    uv run pytest --cov=rheojax --cov-report=html

    # Run notebooks (set FAST_MODE=1 for CI-friendly execution)
    FAST_MODE=1 uv run python scripts/run_notebooks.py --subdir examples/hvm

Test Markers
~~~~~~~~~~~~

Use markers for different test categories:

.. code-block:: python

    import pytest

    @pytest.mark.smoke
    def test_model_creation():
        """Smoke test — runs in CI tier 1."""
        model = Maxwell()
        assert model is not None

    @pytest.mark.slow
    def test_bayesian_inference():
        """Slow test — skipped in quick runs, only in tier 3."""
        pass

    @pytest.mark.unit
    def test_single_function():
        """Unit test."""
        pass

    @pytest.mark.integration
    def test_complete_workflow():
        """Integration test."""
        pass

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

    # Build HTML documentation
    cd docs
    make html

    # View documentation
    open build/html/index.html  # macOS
    # or
    xdg-open build/html/index.html  # Linux

    # Clean build
    make clean
    make html

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

Add documentation for new features:

1. **Update API Reference**

   .. code-block:: rst

       # docs/source/api/module_name.rst

       New Function
       ~~~~~~~~~~~~

       .. autofunction:: rheojax.module.new_function

2. **Add User Guide Section**

   .. code-block:: rst

       # docs/source/user_guide/guide_name.rst

       Using New Feature
       ~~~~~~~~~~~~~~~~~

       Description of the new feature...

       .. code-block:: python

           from rheojax import new_feature
           result = new_feature(data)

3. **Include Examples**

   Every new feature should have runnable examples:

   .. code-block:: python

       def new_function(data: RheoData) -> RheoData:
           """Process data with new method.

           Examples
           --------
           >>> import numpy as np
           >>> from rheojax.core import RheoData
           >>> data = RheoData(x=np.array([1, 2, 3]), y=np.array([10, 20, 30]))
           >>> result = new_function(data)
           """
           pass

Adding New Features
-------------------

Adding a Model
~~~~~~~~~~~~~~

1. **Create Model Class**

   .. code-block:: python

       # rheojax/models/new_model.py
       from rheojax.core import BaseModel, ParameterSet
       import jax.numpy as jnp

       class NewModel(BaseModel):
           """Description of new model.

           Mathematical formulation:
           G(t) = E * exp(-t/tau)

           Parameters
           ----------
           E : float
               Elastic modulus (Pa)
           tau : float
               Relaxation time (s)
           """

           def __init__(self, E=1000.0, tau=1.0):
               super().__init__()
               self.parameters = ParameterSet()
               self.parameters.add("E", value=E, bounds=(1, 1e6), units="Pa")
               self.parameters.add("tau", value=tau, bounds=(0.01, 1000), units="s")

           def _fit(self, X, y, **kwargs):
               """Implement fitting."""
               from rheojax.utils.optimization import nlsq_optimize

               def objective(params):
                   predictions = self._predict(X)
                   return jnp.sum((predictions - y)**2)

               nlsq_optimize(objective, self.parameters, use_jax=True)
               return self

           def _predict(self, X):
               """Implement prediction."""
               E = self.parameters.get_value("E")
               tau = self.parameters.get_value("tau")
               return E * jnp.exp(-X / tau)

2. **Add Tests**

   .. code-block:: python

       # tests/models/test_new_model.py
       import pytest
       import numpy as np
       from rheojax.models import NewModel

       def test_new_model_creation():
           """Test model instantiation."""
           model = NewModel(E=1000, tau=1.0)
           assert model.parameters.get_value("E") == 1000

       def test_new_model_fitting():
           """Test model fitting."""
           time = np.logspace(-1, 2, 50)
           stress = 1000 * np.exp(-time / 1.5)

           model = NewModel()
           model.fit(time, stress)

           # Check fitted parameters are reasonable
           assert 900 < model.parameters.get_value("E") < 1100
           assert 1.3 < model.parameters.get_value("tau") < 1.7

3. **Document Model**

   .. code-block:: rst

       # docs/source/api/models.rst

       NewModel
       ~~~~~~~~

       .. autoclass:: rheojax.models.NewModel
          :members:
          :inherited-members:

4. **Register Model**

   .. code-block:: python

       # rheojax/models/__init__.py
       from .new_model import NewModel

       __all__ = [..., "NewModel"]

Adding a Transform
~~~~~~~~~~~~~~~~~~

1. **Create Transform Class**

   .. code-block:: python

       # rheojax/transforms/new_transform.py
       from rheojax.core import BaseTransform, RheoData
       import jax.numpy as jnp

       class NewTransform(BaseTransform):
           """Description of transform."""

           def __init__(self, param=1.0):
               super().__init__()
               self.param = param

           def _transform(self, data):
               """Forward transform."""
               y_transformed = data.y * self.param
               return RheoData(
                   x=data.x,
                   y=y_transformed,
                   x_units=data.x_units,
                   y_units=data.y_units,
                   domain=data.domain,
                   metadata=data.metadata.copy()
               )

           def _inverse_transform(self, data):
               """Inverse transform."""
               y_original = data.y / self.param
               return RheoData(
                   x=data.x,
                   y=y_original,
                   x_units=data.x_units,
                   y_units=data.y_units,
                   domain=data.domain,
                   metadata=data.metadata.copy()
               )

2. **Add Tests and Documentation** (similar to models)

Adding a Reader
~~~~~~~~~~~~~~~

.. code-block:: python

    # rheojax/io/readers/new_reader.py
    import numpy as np
    from rheojax.core import RheoData

    def read_new_format(filepath, **kwargs):
        """Read new file format.

        Parameters
        ----------
        filepath : str or Path
            Path to input file
        **kwargs
            Additional reader options

        Returns
        -------
        RheoData
            Parsed data
        """
        # Parse file
        with open(filepath, 'r') as f:
            # ... parsing logic

        return RheoData(
            x=x_data,
            y=y_data,
            x_units=x_units,
            y_units=y_units,
            domain=domain
        )

Pull Request Process
--------------------

1. **Create Pull Request**

   - Push your branch to GitHub
   - Open pull request against ``main`` branch
   - Fill out PR template with:
     - Description of changes
     - Related issue numbers
     - Testing performed
     - Documentation updates

2. **PR Checklist**

   - [ ] Tests pass locally
   - [ ] New tests added for new features
   - [ ] Documentation updated
   - [ ] Code follows style guide
   - [ ] Docstrings added/updated
   - [ ] No breaking changes (or clearly documented)
   - [ ] CHANGELOG.md updated

3. **Code Review**

   - Address reviewer feedback
   - Make requested changes
   - Push updates to same branch
   - PR will be merged when approved

Example PR Description
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: markdown

    ## Description
    Adds support for Carreau-Yasuda flow model with JAX implementation
    and automatic parameter optimization.

    ## Related Issues
    Closes #123

    ## Changes
    - Added `CarreauYasuda` model in `rheojax/models/carreau_yasuda.py`
    - Implemented JAX-compatible prediction and fitting
    - Added comprehensive unit tests
    - Updated model documentation

    ## Testing
    - All existing tests pass
    - New tests for CarreauYasuda model pass
    - Tested on example flow curve data

    ## Documentation
    - Added API reference documentation
    - Added usage examples in docstrings
    - Updated user guide with flow model section

    ## Checklist
    - [x] Tests added and passing
    - [x] Documentation updated
    - [x] Code follows style guide
    - [x] No breaking changes

Code Review Guidelines
----------------------

For Reviewers
~~~~~~~~~~~~~

When reviewing code, check for:

- **Correctness**: Does it work as intended?
- **Tests**: Are there adequate tests?
- **Documentation**: Is it well-documented?
- **Style**: Does it follow coding standards?
- **Performance**: Any obvious inefficiencies?
- **API Design**: Is the API intuitive?

For Contributors
~~~~~~~~~~~~~~~~

When receiving feedback:

- Be open to suggestions
- Ask questions if unclear
- Make requested changes promptly
- Thank reviewers for their time

Release Process
---------------

(For maintainers)

1. **Version Bump**

   .. code-block:: bash

       # Update version in pyproject.toml
       # Update CHANGELOG.md

2. **Tag Release**

   .. code-block:: bash

       git tag v0.1.0
       git push origin v0.1.0

3. **Build and Publish**

   .. code-block:: bash

       # Build package
       python -m build

       # Upload to PyPI
       python -m twine upload dist/*

Getting Help
------------

If you need help:

- Check existing documentation
- Search GitHub issues
- Ask in GitHub Discussions
- Contact maintainers

Community Guidelines
--------------------

- Be respectful and inclusive
- Help others learn
- Give constructive feedback
- Celebrate contributions

Thank you for contributing to rheojax!

See Also
--------

- :doc:`architecture` - Architecture overview
- :doc:`../user_guide/getting_started` - User guide
- `GitHub repository <https://github.com/imewei/rheojax>`_
