# RheoJAX Package Makefile
# ========================
# GPU Acceleration Support and Development Tools

.PHONY: help install install-dev install-jax-gpu install-jax-gpu-cuda12 install-jax-gpu-cuda13 gpu-check gpu-diagnose env-info \
        test test-smoke test-fast test-ci test-ci-full test-coverage test-integration test-validation \
        test-parallel test-all-parallel test-parallel-fast test-coverage-parallel \
        clean clean-all clean-pyc clean-build clean-test clean-venv \
        format lint type-check check quick docs build publish info version \
        verify verify-fast install-hooks check-equations

# Configuration
PYTHON := python
PYTEST := pytest
PACKAGE_NAME := rheojax
SRC_DIR := rheojax
TEST_DIR := tests
DOCS_DIR := docs
VENV := .venv

# Parallel test workers: each xdist worker imports JAX (~1-2 GB).
# Cap at 4 workers to stay within 16 GB RAM (4 x 2 GB = 8 GB headroom).
# Override: XDIST_WORKERS=8 make test-parallel
XDIST_WORKERS ?= 4

# Platform detection
UNAME_S := $(shell uname -s 2>/dev/null || echo "Windows")
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
else ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
else
    PLATFORM := windows
endif

# Package manager detection (prioritize uv > conda/mamba > pip)
UV_AVAILABLE := $(shell command -v uv 2>/dev/null)
CONDA_PREFIX := $(shell echo $$CONDA_PREFIX)
MAMBA_AVAILABLE := $(shell command -v mamba 2>/dev/null)

# Determine package manager and commands
ifdef UV_AVAILABLE
    PKG_MANAGER := uv
    PIP := uv pip
    UNINSTALL_CMD := uv pip uninstall -y
    INSTALL_CMD := uv pip install
    RUN_CMD := uv run
else ifdef CONDA_PREFIX
    # In conda environment - use pip within conda
    ifdef MAMBA_AVAILABLE
        PKG_MANAGER := mamba (using pip for JAX)
    else
        PKG_MANAGER := conda (using pip for JAX)
    endif
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
else
    PKG_MANAGER := pip
    PIP := pip
    UNINSTALL_CMD := pip uninstall -y
    INSTALL_CMD := pip install
    RUN_CMD :=
endif

# GPU installation packages (system CUDA - uses -local suffix)
ifeq ($(PLATFORM),linux)
    JAX_GPU_CUDA13_PKG := "jax[cuda13-local]"
    JAX_GPU_CUDA12_PKG := "jax[cuda12-local]"
else
    JAX_GPU_CUDA13_PKG :=
    JAX_GPU_CUDA12_PKG :=
endif

# All JAX/CUDA packages that must be removed before a clean install
JAX_ALL_PKGS := jax jaxlib jax-cuda12-plugin jax-cuda12-pjrt jax-cuda13-plugin jax-cuda13-pjrt

# ─── Shared validation helpers ──────────────────────────────────────────────
define VALIDATE_CUDA
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found — CUDA toolkit not installed or not in PATH$(RESET)"; \
		echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"; \
		echo "  Or download from: https://developer.nvidia.com/cuda-downloads"; \
		echo "  Then: export PATH=/usr/local/cuda/bin:\$$PATH"; \
		exit 1; \
	elif [ "$$CUDA_VERSION" != "$(1)" ]; then \
		echo "$(RED)Error: System CUDA $$CUDA_FULL detected, but CUDA $(1).x required$(RESET)"; \
		echo "  Use: make install-jax-gpu-cuda$(2) (if you have CUDA $(2).x)"; \
		exit 1; \
	fi; \
	echo "System CUDA: $$CUDA_FULL"
endef

define VALIDATE_GPU_SM
	@SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	if [ -z "$$SM_VERSION" ]; then \
		echo "$(RED)Error: Could not detect GPU (nvidia-smi failed)$(RESET)"; \
		exit 1; \
	elif [ "$$SM_VERSION" -lt $(1) ]; then \
		echo "$(RED)Error: GPU SM $$SM_DISPLAY too old (requires SM >= $(2))$(RESET)"; \
		exit 1; \
	fi; \
	echo "GPU SM version: $$SM_DISPLAY (compatible)"
endef

define CLEAN_AND_INSTALL
	@echo ""
	@echo "Step 1/2: Removing all existing JAX/CUDA packages..."
	@$(UNINSTALL_CMD) $(JAX_ALL_PKGS) 2>/dev/null || true
	@echo ""
	@echo "Step 2/2: Installing $(1)..."
	@$(INSTALL_CMD) $(1)
	@echo ""
	@$(MAKE) gpu-check
endef

# Colors for output
BOLD := \033[1m
RESET := \033[0m
BLUE := \033[34m
GREEN := \033[32m
YELLOW := \033[33m
RED := \033[31m
CYAN := \033[36m

# Default target
.DEFAULT_GOAL := help

# ===================
# Help target
# ===================
help:
	@echo "$(BOLD)$(BLUE)RheoJAX Development Commands$(RESET)"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET) make $(CYAN)<target>$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)ENVIRONMENT$(RESET)"
	@echo "  $(CYAN)env-info$(RESET)         Show detailed environment information"
	@echo "  $(CYAN)info$(RESET)             Show project and environment info"
	@echo "  $(CYAN)version$(RESET)          Show package version"
	@echo ""
	@echo "$(BOLD)$(GREEN)INSTALLATION$(RESET)"
	@echo "  $(CYAN)install$(RESET)               Install package with all runtime deps (CPU-only JAX)"
	@echo "  $(CYAN)install-dev$(RESET)           Install with development dependencies (testing, docs, linting)"
	@echo ""
	@echo "$(BOLD)$(GREEN)GPU COMMANDS (System CUDA)$(RESET)"
	@echo "  $(CYAN)install-jax-gpu$(RESET)       Auto-detect system CUDA and install JAX (Linux only)"
	@echo "  $(CYAN)install-jax-gpu-cuda13$(RESET) Install JAX with system CUDA 13 (requires CUDA 13.x)"
	@echo "  $(CYAN)install-jax-gpu-cuda12$(RESET) Install JAX with system CUDA 12 (requires CUDA 12.x)"
	@echo "  $(CYAN)gpu-check$(RESET)             Verify GPU backend, devices, SVD"
	@echo "  $(CYAN)gpu-diagnose$(RESET)          Check for common GPU issues"
	@echo "  $(CYAN)env-info$(RESET)              Show full environment information"
	@echo ""
	@echo "$(BOLD)$(GREEN)TESTING$(RESET)"
	@echo "  $(CYAN)test$(RESET)                   Run all tests (full suite with fast MCMC, ~50-60min)"
	@echo "  $(CYAN)test-smoke$(RESET)             Run smoke tests (105 critical tests, ~30s-2min)"
	@echo "  $(CYAN)test-fast$(RESET)              Run tests excluding slow Bayesian tests (~15-25min)"
	@echo "  $(CYAN)test-ci$(RESET)                Run CI test suite (matches GitHub Actions, 105 smoke tests)"
	@echo "  $(CYAN)test-ci-full$(RESET)           Run full CI suite (1069 tests, pre-v0.2.1 behavior)"
	@echo "  $(CYAN)test-validation$(RESET)        Run with production MCMC (PYTEST_FULL_VALIDATION=1, ~90min)"
	@echo "  $(CYAN)test-parallel$(RESET)          Run all tests in parallel (2-4x faster)"
	@echo "  $(CYAN)test-all-parallel$(RESET)      Run full test suite in parallel (~1245 tests)"
	@echo "  $(CYAN)test-parallel-fast$(RESET)     Run fast tests in parallel"
	@echo "  $(CYAN)test-coverage$(RESET)          Run tests with coverage report"
	@echo "  $(CYAN)test-coverage-parallel$(RESET) Run coverage with parallel execution"
	@echo "  $(CYAN)test-integration$(RESET)       Run integration tests only"
	@echo ""
	@echo "$(BOLD)MCMC Configuration:$(RESET)"
	@echo "  CI=1                   Force fast MCMC (200/200 samples, auto-detected in CI)"
	@echo "  PYTEST_FAST_MCMC=1     Force fast MCMC for local testing"
	@echo "  PYTEST_FULL_VALIDATION=1   Force production MCMC (2000/1000 samples)"
	@echo ""
	@echo "$(BOLD)$(GREEN)CODE QUALITY$(RESET)"
	@echo "  $(CYAN)format$(RESET)           Format code with black and ruff"
	@echo "  $(CYAN)lint$(RESET)             Run linting checks (ruff)"
	@echo "  $(CYAN)type-check$(RESET)       Run type checking (mypy)"
	@echo "  $(CYAN)check$(RESET)            Run all checks (format + lint + type)"
	@echo "  $(CYAN)quick$(RESET)            Fast iteration: format + smoke tests (~30s-2min)"
	@echo "  $(CYAN)check-equations$(RESET)  Verify all kernel files cite equation references"
	@echo ""
	@echo "$(BOLD)$(GREEN)PRE-PUSH VERIFICATION$(RESET)"
	@echo "  $(CYAN)verify$(RESET)           Run FULL local CI (lint + type + smoke tests) - use before push"
	@echo "  $(CYAN)verify-fast$(RESET)      Quick verification (lint + type only, no tests)"
	@echo "  $(CYAN)install-hooks$(RESET)    Install pre-commit hooks"
	@echo ""
	@echo "$(BOLD)$(GREEN)DOCUMENTATION$(RESET)"
	@echo "  $(CYAN)docs$(RESET)             Build documentation with Sphinx"
	@echo ""
	@echo "$(BOLD)$(GREEN)BUILD & PUBLISH$(RESET)"
	@echo "  $(CYAN)build$(RESET)            Build distribution packages"
	@echo "  $(CYAN)publish$(RESET)          Publish to PyPI (requires credentials)"
	@echo ""
	@echo "$(BOLD)$(GREEN)CLEANUP$(RESET)"
	@echo "  $(CYAN)clean$(RESET)            Remove build artifacts and caches (preserves venv, .claude, .specify, agent-os)"
	@echo "  $(CYAN)clean-all$(RESET)        Deep clean of all caches (preserves venv, .claude, .specify, agent-os)"
	@echo "  $(CYAN)clean-pyc$(RESET)        Remove Python file artifacts"
	@echo "  $(CYAN)clean-build$(RESET)      Remove build artifacts"
	@echo "  $(CYAN)clean-test$(RESET)       Remove test and coverage artifacts"
	@echo "  $(CYAN)clean-venv$(RESET)       Remove virtual environment (use with caution)"
	@echo ""
	@echo "$(BOLD)Environment Detection:$(RESET)"
	@echo "  Platform: $(PLATFORM)"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo ""

# ===================
# Installation targets
# ===================
install:
	@echo "$(BOLD)$(BLUE)Installing $(PACKAGE_NAME) in editable mode...$(RESET)"
	@$(INSTALL_CMD) -e .
	@echo "$(BOLD)$(GREEN)✓ Package installed!$(RESET)"

install-dev: install
	@echo "$(BOLD)$(BLUE)Installing development dependencies...$(RESET)"
	@$(INSTALL_CMD) -e ".[dev]"
	@echo "$(BOLD)$(GREEN)✓ Dev dependencies installed!$(RESET)"

# Auto-detect system CUDA version and install matching JAX package
install-jax-gpu:
	@echo "$(BOLD)$(BLUE)Installing JAX with GPU support (system CUDA auto-detect)...$(RESET)"
	@echo "============================================================"
	@echo "$(BOLD)Platform:$(RESET) $(PLATFORM)"
	@echo "$(BOLD)Package manager:$(RESET) $(PKG_MANAGER)"
	@echo ""
ifeq ($(PLATFORM),linux)
	@# Step 1: Detect system CUDA version
	@CUDA_VERSION=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+' | head -1); \
	if [ -z "$$CUDA_VERSION" ]; then \
		echo "$(RED)Error: nvcc not found - CUDA toolkit not installed or not in PATH$(RESET)"; \
		echo ""; \
		echo "Please install CUDA toolkit:"; \
		echo "  Ubuntu/Debian: sudo apt install nvidia-cuda-toolkit"; \
		echo "  Or download from: https://developer.nvidia.com/cuda-downloads"; \
		echo ""; \
		echo "After installation, ensure nvcc is in PATH:"; \
		echo "  export PATH=/usr/local/cuda/bin:\$$PATH"; \
		exit 1; \
	fi; \
	CUDA_FULL=$$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+'); \
	echo "Detected system CUDA version: $$CUDA_FULL (major: $$CUDA_VERSION)"; \
	echo ""; \
	\
	# Step 2: Detect GPU SM version \
	SM_VERSION=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.'); \
	SM_DISPLAY=$$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1); \
	GPU_NAME=$$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1); \
	if [ -z "$$SM_VERSION" ]; then \
		echo "$(RED)Error: Could not detect GPU (nvidia-smi failed)$(RESET)"; \
		exit 1; \
	fi; \
	echo "Detected GPU: $$GPU_NAME (SM $$SM_DISPLAY)"; \
	echo ""; \
	\
	# Step 3: Validate compatibility and install \
	if [ "$$CUDA_VERSION" = "13" ]; then \
		if [ "$$SM_VERSION" -ge 75 ]; then \
			echo "Compatibility: System CUDA 13 + GPU SM $$SM_DISPLAY = Compatible"; \
			echo "Installing: $(JAX_GPU_CUDA13_PKG)"; \
			$(MAKE) install-jax-gpu-cuda13; \
		else \
			echo "$(RED)Error: GPU SM $$SM_DISPLAY does not support CUDA 13 (requires SM >= 7.5)$(RESET)"; \
			echo "Your GPU requires CUDA 12. Please install CUDA 12.x toolkit."; \
			exit 1; \
		fi; \
	elif [ "$$CUDA_VERSION" = "12" ]; then \
		if [ "$$SM_VERSION" -ge 52 ]; then \
			echo "Compatibility: System CUDA 12 + GPU SM $$SM_DISPLAY = Compatible"; \
			echo "Installing: $(JAX_GPU_CUDA12_PKG)"; \
			$(MAKE) install-jax-gpu-cuda12; \
		else \
			echo "$(RED)Error: GPU SM $$SM_DISPLAY too old (requires SM >= 5.2)$(RESET)"; \
			echo "Kepler and older GPUs are not supported by JAX 0.8+"; \
			exit 1; \
		fi; \
	else \
		echo "$(RED)Error: CUDA $$CUDA_VERSION not supported by JAX 0.8+$(RESET)"; \
		echo "JAX requires CUDA 12.x or 13.x"; \
		echo "Please upgrade your CUDA installation."; \
		exit 1; \
	fi
else
	@echo "$(YELLOW)Error: GPU acceleration only available on Linux$(RESET)"
	@echo "  Current platform: $(PLATFORM)"
	@echo "  Keeping CPU-only installation"
	@echo ""
	@echo "Platform support:"
	@echo "  - Linux + NVIDIA GPU + System CUDA: Full GPU acceleration"
	@echo "  - Windows WSL2: Experimental (use Linux wheels)"
	@echo "  - macOS: CPU-only (no NVIDIA GPU support)"
	@echo "  - Windows native: CPU-only (no pre-built wheels)"
endif

# CUDA 13 installation (requires system CUDA 13.x)
install-jax-gpu-cuda13:
	@echo "$(BOLD)$(BLUE)Installing JAX with system CUDA 13...$(RESET)"
ifeq ($(PLATFORM),linux)
	$(call VALIDATE_CUDA,13,12)
	$(call VALIDATE_GPU_SM,75,7.5)
	$(call CLEAN_AND_INSTALL,$(JAX_GPU_CUDA13_PKG))
	@echo "Done — using system CUDA 13.x"
else
	@echo "$(RED)Error: CUDA 13 GPU acceleration requires Linux$(RESET)"
endif

# CUDA 12 installation (requires system CUDA 12.x)
install-jax-gpu-cuda12:
	@echo "$(BOLD)$(BLUE)Installing JAX with system CUDA 12...$(RESET)"
ifeq ($(PLATFORM),linux)
	$(call VALIDATE_CUDA,12,13)
	$(call VALIDATE_GPU_SM,52,5.2)
	$(call CLEAN_AND_INSTALL,$(JAX_GPU_CUDA12_PKG))
	@echo "Done — using system CUDA 12.x"
else
	@echo "$(RED)Error: CUDA 12 GPU acceleration requires Linux$(RESET)"
endif

# ===================
# GPU verification
# ===================
gpu-check:
	@echo "$(BOLD)$(BLUE)GPU Verification$(RESET)"
	@echo "================"
	@$(PYTHON) -c "\
import sys; \
import jax; \
v = jax.__version__; \
b = jax.default_backend(); \
d = jax.devices(); \
gpu = sum(1 for x in d if 'cuda' in str(x).lower()); \
print(f'JAX {v}  backend={b}  devices={gpu} GPU'); \
if b != 'gpu': \
    print('WARNING: Not using GPU'); sys.exit(1); \
import jax.numpy as jnp; \
s = jnp.linalg.svd(jnp.eye(3))[1]; \
print(f'SVD check: {s}'); \
print('All checks passed')"

gpu-diagnose:
	@echo "$(BOLD)$(BLUE)GPU Diagnostics$(RESET)"
	@echo "==============="
	@echo ""
	@echo "1. Installed JAX/CUDA packages:"
	@$(PYTHON) -m pip list 2>/dev/null | grep -iE "^(jax|cuda)" || echo "  (none found)"
	@echo ""
	@echo "2. Plugin conflict check:"
	@HAS12=$$($(PYTHON) -m pip list 2>/dev/null | grep -c "jax-cuda12-plugin"); \
	HAS13=$$($(PYTHON) -m pip list 2>/dev/null | grep -c "jax-cuda13-plugin"); \
	if [ "$$HAS12" -gt 0 ] && [ "$$HAS13" -gt 0 ]; then \
		echo "  $(RED)CONFLICT: Both cuda12 and cuda13 plugins installed!$(RESET)"; \
		echo "  Fix: make install-jax-gpu (will clean and reinstall)"; \
	elif [ "$$HAS12" -gt 0 ] || [ "$$HAS13" -gt 0 ]; then \
		echo "  OK: Single plugin set installed"; \
	else \
		echo "  No CUDA plugins installed (CPU-only mode)"; \
	fi
	@echo ""
	@echo "3. Version match check:"
	@$(PYTHON) -c "\
import importlib.metadata as md; \
jaxlib_v = md.version('jaxlib'); \
print(f'  jaxlib: {jaxlib_v}'); \
for pkg in ['jax-cuda12-plugin','jax-cuda13-plugin','jax-cuda12-pjrt','jax-cuda13-pjrt']: \
    try: \
        v = md.version(pkg); \
        match = 'OK' if v == jaxlib_v else 'MISMATCH'; \
        print(f'  {pkg}: {v} [{match}]'); \
    except md.PackageNotFoundError: pass" 2>/dev/null || echo "  (could not check)"
	@echo ""
	@echo "4. System CUDA:"
	@nvcc --version 2>/dev/null | grep "release" || echo "  nvcc not found"
	@echo ""
	@echo "5. GPU hardware:"
	@nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null \
		|| echo "  nvidia-smi not found"

# Environment info target
env-info:
	@echo "$(BOLD)$(BLUE)Environment Information$(RESET)"
	@echo "======================"
	@echo "  Platform: $(PLATFORM) ($(UNAME_S))"
	@echo "  Package manager: $(PKG_MANAGER)"
	@echo "  Python: $(shell $(PYTHON) --version 2>&1)"
	@echo ""
	@$(MAKE) gpu-diagnose

# ===================
# Testing targets
# ===================
test:
	@echo "$(BOLD)$(BLUE)Running all tests...$(RESET)"
	$(RUN_CMD) $(PYTEST)

test-smoke:
	@echo "$(BOLD)$(BLUE)Running smoke tests (105 critical tests, ~30s-2min)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) -m "smoke"
	@echo "$(BOLD)$(GREEN)✓ Smoke tests passed!$(RESET)"

test-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests (excluding slow Bayesian tests)...$(RESET)"
	@echo "$(BOLD)Note:$(RESET) Excludes 34 slow Bayesian tests (60-90 min), ~15-25 min runtime"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) -m "not slow"

test-parallel:
	@echo "$(BOLD)$(BLUE)Running tests in parallel (2-4x speedup)...$(RESET)"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS)

test-all-parallel:
	@echo "$(BOLD)$(BLUE)Running full test suite in parallel (~1245 tests)...$(RESET)"
	@echo "$(BOLD)Note:$(RESET) Includes all tests (slow Bayesian, integration, etc.)"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS)
	@echo "$(BOLD)$(GREEN)✓ Full test suite passed!$(RESET)"

test-parallel-fast:
	@echo "$(BOLD)$(BLUE)Running fast tests in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) -m "not slow"

test-ci:
	@echo "$(BOLD)$(BLUE)Running CI test suite (matches GitHub Actions)...$(RESET)"
	@echo "$(BOLD)Tests:$(RESET) 105 smoke tests, ~30s-2min"
	@echo "$(BOLD)Note:$(RESET) GitHub CI now runs smoke tests only for fast feedback"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) -m "smoke"
	@echo "$(BOLD)$(GREEN)✓ CI test suite passed!$(RESET)"

test-ci-full:
	@echo "$(BOLD)$(BLUE)Running full CI test suite (pre-v0.2.1 behavior)...$(RESET)"
	@echo "$(BOLD)Excludes:$(RESET) slow, validation, benchmark, notebook_comprehensive"
	@echo "$(BOLD)Tests:$(RESET) ~1069/1154 tests, ~5-10 minutes"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) -m "not slow and not validation and not benchmark and not notebook_comprehensive"
	@echo "$(BOLD)$(GREEN)✓ Full CI test suite passed!$(RESET)"

test-coverage:
	@echo "$(BOLD)$(BLUE)Running tests with coverage report...$(RESET)"
	$(RUN_CMD) $(PYTEST) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

test-coverage-parallel:
	@echo "$(BOLD)$(BLUE)Running tests with coverage in parallel...$(RESET)"
	$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) --cov=$(PACKAGE_NAME) --cov-report=term-missing --cov-report=html --cov-report=xml
	@echo "$(BOLD)$(GREEN)✓ Coverage report generated!$(RESET)"
	@echo "View HTML report: open htmlcov/index.html"

test-integration:
	@echo "$(BOLD)$(BLUE)Running integration tests...$(RESET)"
	$(RUN_CMD) $(PYTEST) -m integration

test-validation:
	@echo "$(BOLD)$(BLUE)Running validation tests with production-quality MCMC...$(RESET)"
	@echo "$(BOLD)Configuration:$(RESET) PYTEST_FULL_VALIDATION=1 (num_warmup=2000, num_samples=1000)"
	@echo "$(BOLD)Runtime:$(RESET) ~90 minutes (for weekly validation and releases)"
	PYTEST_FULL_VALIDATION=1 $(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS)

# ===================
# Code quality targets
# ===================
format:
	@echo "$(BOLD)$(BLUE)Formatting code with black and ruff...$(RESET)"
	$(RUN_CMD) black $(PACKAGE_NAME) tests
	$(RUN_CMD) ruff check --fix $(PACKAGE_NAME) tests
	@echo "$(BOLD)$(GREEN)✓ Code formatted!$(RESET)"

lint:
	@echo "$(BOLD)$(BLUE)Running linting checks...$(RESET)"
	$(RUN_CMD) ruff check $(PACKAGE_NAME) tests
	@echo "$(BOLD)$(GREEN)✓ No linting errors!$(RESET)"

type-check:
	@echo "$(BOLD)$(BLUE)Running type checks...$(RESET)"
	$(RUN_CMD) mypy $(PACKAGE_NAME)
	@echo "$(BOLD)$(GREEN)✓ Type checking passed!$(RESET)"

check: lint type-check
	@echo "$(BOLD)$(GREEN)✓ All checks passed!$(RESET)"

quick: format test-smoke
	@echo "$(BOLD)$(GREEN)✓ Quick iteration complete!$(RESET)"

# ===================
# Documentation targets
# ===================
docs:
	@echo "$(BOLD)$(BLUE)Building documentation...$(RESET)"
	cd docs && $(MAKE) html
	@echo "$(BOLD)$(GREEN)✓ Documentation built!$(RESET)"
	@echo "Open: docs/_build/html/index.html"

# ===================
# Build and publish targets
# ===================
build: clean
	@echo "$(BOLD)$(BLUE)Building distribution packages...$(RESET)"
	$(PYTHON) -m build
	@echo "$(BOLD)$(GREEN)✓ Build complete!$(RESET)"
	@echo "Distributions in dist/"

publish: build
	@echo "$(BOLD)$(YELLOW)This will publish $(PACKAGE_NAME) to PyPI!$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Publishing to PyPI...$(RESET)"; \
		$(PYTHON) -m twine upload dist/*; \
		echo "$(BOLD)$(GREEN)✓ Published to PyPI!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Cleanup targets
# ===================
clean-build:
	@echo "$(BOLD)$(BLUE)Removing build artifacts...$(RESET)"
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .benchmarks/
	find . -type d -name "*.egg-info" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg" \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true

clean-pyc:
	@echo "$(BOLD)$(BLUE)Removing Python file artifacts...$(RESET)"
	find . -type d -name __pycache__ \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type f \( -name "*.pyc" -o -name "*.pyo" \) \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-delete 2>/dev/null || true

clean-test:
	@echo "$(BOLD)$(BLUE)Removing test and coverage artifacts...$(RESET)"
	find . -type d -name .pytest_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .nlsq_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .ruff_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name htmlcov \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .hypothesis \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	find . -type d -name _run_logs \
		-not -path "./.venv/*" \
		-not -path "./venv/*" \
		-not -path "./.claude/*" \
		-not -path "./.specify/*" \
		-not -path "./agent-os/*" \
		-exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage*
	rm -rf coverage.xml

clean: clean-build clean-pyc clean-test
	@echo "$(BOLD)$(BLUE)Removing temporary work directories...$(RESET)"
	rm -rf fix-imports/ .nlsq_cache/ 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Cleaned!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, .specify/, agent-os/"

clean-all: clean
	@echo "$(BOLD)$(BLUE)Performing deep clean of additional caches...$(RESET)"
	rm -rf .tox/ 2>/dev/null || true
	rm -rf .nox/ 2>/dev/null || true
	rm -rf .eggs/ 2>/dev/null || true
	rm -rf .cache/ 2>/dev/null || true
	rm -rf .benchmarks/ 2>/dev/null || true
	@echo "$(BOLD)$(GREEN)✓ Deep clean complete!$(RESET)"
	@echo "$(BOLD)Protected directories preserved:$(RESET) .venv/, venv/, .claude/, .specify/, agent-os/"

clean-venv:
	@echo "$(BOLD)$(YELLOW)WARNING: This will remove the virtual environment!$(RESET)"
	@echo "$(BOLD)You will need to recreate it manually.$(RESET)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		echo "$(BOLD)$(BLUE)Removing virtual environment...$(RESET)"; \
		rm -rf $(VENV) venv; \
		echo "$(BOLD)$(GREEN)✓ Virtual environment removed!$(RESET)"; \
	else \
		echo "Cancelled."; \
	fi

# ===================
# Utility targets
# ===================
info:
	@echo "$(BOLD)$(BLUE)Project Information$(RESET)"
	@echo "===================="
	@echo "Project: $(PACKAGE_NAME)"
	@echo "Python: $(shell $(PYTHON) --version 2>&1)"
	@echo "Platform: $(PLATFORM)"
	@echo "Package manager: $(PKG_MANAGER)"
	@echo ""
	@echo "$(BOLD)$(BLUE)Directory Structure$(RESET)"
	@echo "===================="
	@echo "Source: $(SRC_DIR)/"
	@echo "Tests: $(TEST_DIR)/"
	@echo "Docs: $(DOCS_DIR)/"
	@echo ""
	@echo "$(BOLD)$(BLUE)JAX Configuration$(RESET)"
	@echo "=================="
	@$(PYTHON) -c "import jax; print('JAX version:', jax.__version__); print('Default backend:', jax.default_backend())" 2>/dev/null || echo "JAX not installed"

version:
	@$(PYTHON) -c "import $(PACKAGE_NAME); print($(PACKAGE_NAME).__version__)" 2>/dev/null || \
		echo "$(BOLD)$(RED)Error: Package not installed. Run 'make install' first.$(RESET)"

# ===================
# Pre-push verification (run before pushing to ensure CI will pass)
# ===================
verify:
	@echo "$(BOLD)$(BLUE)======================================$(RESET)"
	@echo "$(BOLD)$(BLUE)  FULL LOCAL CI VERIFICATION$(RESET)"
	@echo "$(BOLD)$(BLUE)======================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 1/3: Linting$(RESET)"
	@$(RUN_CMD) ruff check $(PACKAGE_NAME) tests || (echo "$(RED)Lint check failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)Step 2/3: Type checking (advisory)$(RESET)"
	@$(RUN_CMD) mypy $(PACKAGE_NAME) --no-error-summary 2>&1 | tail -1 || true
	@echo "$(YELLOW)Note: Type checking is advisory. See 'make type-check' for full report.$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 3/3: Smoke tests$(RESET)"
	@$(RUN_CMD) $(PYTEST) -n $(XDIST_WORKERS) -m "smoke" || (echo "$(RED)Smoke tests failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)$(GREEN)======================================$(RESET)"
	@echo "$(BOLD)$(GREEN)  ALL CHECKS PASSED - SAFE TO PUSH$(RESET)"
	@echo "$(BOLD)$(GREEN)======================================$(RESET)"

verify-fast:
	@echo "$(BOLD)$(BLUE)======================================$(RESET)"
	@echo "$(BOLD)$(BLUE)  QUICK LOCAL CI VERIFICATION$(RESET)"
	@echo "$(BOLD)$(BLUE)======================================$(RESET)"
	@echo ""
	@echo "$(BOLD)Step 1/2: Linting$(RESET)"
	@$(RUN_CMD) ruff check $(PACKAGE_NAME) tests || (echo "$(RED)Lint check failed!$(RESET)" && exit 1)
	@echo ""
	@echo "$(BOLD)Step 2/2: Type checking (advisory)$(RESET)"
	@$(RUN_CMD) mypy $(PACKAGE_NAME) --no-error-summary 2>&1 | tail -1 || true
	@echo "$(YELLOW)Note: Type checking is advisory. See 'make type-check' for full report.$(RESET)"
	@echo ""
	@echo "$(BOLD)$(GREEN)======================================$(RESET)"
	@echo "$(BOLD)$(GREEN)  QUICK CHECKS PASSED$(RESET)"
	@echo "$(BOLD)$(GREEN)======================================$(RESET)"

install-hooks:
	@echo "$(BOLD)$(BLUE)Installing git hooks...$(RESET)"
	@pre-commit install 2>/dev/null || echo "$(YELLOW)pre-commit not installed, skipping$(RESET)"
	@echo "$(BOLD)$(GREEN)✓ Git hooks installed!$(RESET)"
	@echo ""
	@echo "$(BOLD)Hooks installed:$(RESET)"
	@echo "  - pre-commit: lint, format, type checks"
	@echo ""
	@echo "$(BOLD)Usage:$(RESET)"
	@echo "  git commit -m 'msg'  → runs pre-commit hooks"
	@echo "  git push             → triggers GitHub Actions CI"
	@echo "  make verify          → full local verification (recommended before push)"

# ===================
# Physics equation reference gate
# ===================
check-equations:
	@echo "$(BOLD)$(BLUE)Checking kernel files for equation references...$(RESET)"
	@failed=0; \
	for f in rheojax/utils/*_kernels*.py; do \
		if ! grep -qiE '(Eq\.|equation|Eqn\.)' "$$f" 2>/dev/null; then \
			echo "$(RED)MISSING: $$f$(RESET)"; \
			failed=1; \
		fi; \
	done; \
	for f in $$(find rheojax/models -name '*_kernels*.py'); do \
		if ! grep -qiE '(Eq\.|equation|Eqn\.)' "$$f" 2>/dev/null; then \
			echo "$(RED)MISSING: $$f$(RESET)"; \
			failed=1; \
		fi; \
	done; \
	if [ "$$failed" = "1" ]; then \
		echo "$(RED)ERROR: Some kernel files lack equation references$(RESET)"; \
		echo "Every kernel function should cite its source paper and equation number."; \
		exit 1; \
	fi; \
	echo "$(BOLD)$(GREEN)✓ All kernel files have equation references$(RESET)"
