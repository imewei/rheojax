Contributing
============

We welcome contributions to rheo! This guide will help you get started.

Development Setup
-----------------

1. Fork the repository on GitHub
2. Clone your fork:

   .. code-block:: bash

      git clone https://github.com/your-username/rheo.git
      cd rheo

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"
      pre-commit install

4. Create a branch for your changes:

   .. code-block:: bash

      git checkout -b feature/your-feature-name

Code Style
----------

We use several tools to maintain code quality:

* **black**: Code formatting
* **ruff**: Linting and import sorting
* **mypy**: Static type checking

Run all checks:

.. code-block:: bash

   black rheo tests
   ruff check rheo tests
   mypy rheo

Pre-commit hooks will automatically run these checks before each commit.

Testing
-------

All new code should include tests. We use pytest for testing:

.. code-block:: bash

   # Run all tests
   pytest

   # Run with coverage
   pytest --cov=rheo --cov-report=html

   # Run specific test
   pytest tests/core/test_base.py::test_base_model

Test Guidelines
~~~~~~~~~~~~~~~

* Write 2-8 focused tests per component
* Aim for >85% coverage (>90% for models and transforms)
* Use hypothesis for property-based testing where appropriate
* Follow existing test patterns

Documentation
-------------

Documentation is built with Sphinx:

.. code-block:: bash

   cd docs
   make html

View the generated docs at ``docs/build/html/index.html``.

All public APIs must have docstrings following Google or NumPy style.

Submitting Changes
------------------

1. Ensure all tests pass
2. Update documentation as needed
3. Add entry to CHANGELOG.md (if applicable)
4. Push to your fork
5. Submit a pull request

Pull Request Guidelines
~~~~~~~~~~~~~~~~~~~~~~~

* Provide a clear description of changes
* Reference any related issues
* Ensure CI passes
* Request review from maintainers

Code Review Process
-------------------

All contributions go through code review:

1. Automated checks (CI) must pass
2. At least one maintainer must approve
3. Documentation must be complete
4. Test coverage must meet requirements

Questions?
----------

* Open an issue on GitHub
* Join discussions in GitHub Discussions
* Contact maintainers

Thank you for contributing to rheo!
