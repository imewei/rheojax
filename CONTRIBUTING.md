# Contributing to RheoJAX

Thank you for your interest in contributing to RheoJAX! We welcome contributions from the community.

## Development Setup

### Prerequisites

- Python 3.12 or later
- Git
- A GitHub account

### Setting Up Your Development Environment

1. **Fork the repository** on GitHub

2. **Clone your fork**:
```bash
git clone https://github.com/yourusername/rheojax.git
cd rheojax
```

3. **Create a virtual environment**:
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

4. **Install in development mode**:
```bash
pip install -e ".[dev]"
```

5. **Set up pre-commit hooks**:
```bash
pre-commit install
```

## Development Workflow

### 1. Create a feature branch

```bash
git checkout -b feature/your-feature-name
```

### 2. Make your changes

- Write your code following our style guidelines
- Add tests for new functionality
- Update documentation as needed

### 3. Run tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=rheojax

# Run specific test file
pytest tests/test_specific.py
```

### 4. Code Quality Checks

```bash
# Format code with Black
black rheojax tests

# Check with Ruff
ruff check rheojax tests

# Type checking with MyPy
mypy rheojax
```

### 5. Commit your changes

```bash
git add .
git commit -m "Add feature: description of your changes"
```

Pre-commit hooks will automatically run formatting and linting checks.

### 6. Push and create a pull request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

- We use [Black](https://github.com/psf/black) for code formatting
- We use [Ruff](https://github.com/charliermarsh/ruff) for linting
- Maximum line length is 88 characters
- Use type hints where possible
- Write descriptive docstrings for all public functions and classes

## Testing Guidelines

- All new features must include tests
- Maintain or improve code coverage
- Use pytest for testing
- Mock external dependencies in unit tests
- Include both unit and integration tests where appropriate

## Documentation

- Update docstrings for any modified functions
- Update README.md if adding new features
- Add examples for new functionality
- Keep documentation clear and concise

## Pull Request Process

1. Ensure all tests pass
2. Update the README.md with details of changes if applicable
3. Increase version numbers if appropriate (following semantic versioning)
4. The PR will be merged after review and approval

## Reporting Issues

- Use GitHub Issues to report bugs
- Include a minimal reproducible example
- Describe expected vs actual behavior
- Include system information (OS, Python version, package versions)

## Code of Conduct

Please note we have a code of conduct. Please follow it in all your interactions with the project.

## Questions?

Feel free to open an issue for any questions about contributing.

Thank you for contributing to RheoJAX!