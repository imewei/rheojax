Installation
============

Requirements
------------

* Python 3.12 or higher (3.8-3.11 NOT supported)
* JAX >= 0.8.3
* jaxlib >= 0.8.3 (must be compatible with JAX version)
* NumPy >= 2.3.5
* SciPy >= 1.17.0
* NLSQ >= 0.6.10 (GPU-accelerated optimization)
* NumPyro >= 0.20.0 (Bayesian inference)
* ArviZ >= 0.23.4 (Bayesian visualization)

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install rheojax

Development Installation
------------------------

For development, clone the repository and install with uv:

.. code-block:: bash

   git clone https://github.com/imewei/rheojax.git
   cd rheojax
   uv sync              # Install all dependencies from uv.lock
   pre-commit install   # Set up pre-commit hooks

Optional Dependencies
---------------------

GPU Support (Linux + CUDA 12+ or 13+)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU acceleration provides 20-100x speedup for large datasets.

**Platform Requirements:**

* ✅ Linux + NVIDIA GPU + CUDA 12.1+: Full GPU acceleration
* ✅ Linux + NVIDIA GPU + CUDA 13.x: Full GPU acceleration with latest toolkit
* ❌ macOS: CPU-only (Apple Silicon/Intel, no NVIDIA GPU support)
* ❌ Windows: CPU-only (CUDA support experimental/unstable)

**Quick Install (from repository):**

.. code-block:: bash

   make install-jax-gpu  # Auto-detects CUDA 12 or 13

**Manual Install:**

.. code-block:: bash

   # Remove ALL existing JAX/CUDA packages first (prevents plugin conflicts)
   pip uninstall -y jax jaxlib \
       jax-cuda13-plugin jax-cuda13-pjrt \
       jax-cuda12-plugin jax-cuda12-pjrt

   # For CUDA 13.x (SM >= 7.5):
   pip install "jax[cuda13-local]"

   # For CUDA 12.x (SM >= 5.2):
   pip install "jax[cuda12-local]"

**Or with project extras:**

.. code-block:: bash

   pip install "rheojax[gpu_cuda13]"   # CUDA 13
   pip install "rheojax[gpu_cuda12]"   # CUDA 12

.. important::

   Never install both cuda12 and cuda13 plugins simultaneously.
   Only ONE CUDA plugin set can be active — having both causes PJRT registration conflicts.

**Requirements:**

* System CUDA 12.1+ or 13.x pre-installed (not bundled)
* NVIDIA driver >= 525 (CUDA 12) or >= 560 (CUDA 13)
* Linux x86_64 or aarch64

**Verify GPU Detection:**

.. code-block:: python

   import jax
   print(jax.devices())  # Should show [cuda(id=0)] if GPU detected

**Diagnostics:**

.. code-block:: bash

   make gpu-check      # Verify GPU backend, devices, SVD computation
   make gpu-diagnose   # Check for plugin conflicts, version mismatches

Verifying Installation
----------------------

Verify the installation:

.. code-block:: python

   import rheojax
   print(f"RheoJAX version: {rheojax.__version__}")

   import jax
   print(f"JAX version: {jax.__version__}")
   print(f"JAX devices: {jax.devices()}")

This should display the version numbers and available devices (CPU or GPU) without errors.

Troubleshooting
---------------

GPU Issues
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 30 30 40

   * - Symptom
     - Cause
     - Fix
   * - ``plugin version X is not compatible with jaxlib Y``
     - Plugin/jaxlib version mismatch
     - Uninstall all, reinstall: ``make install-jax-gpu``
   * - ``PJRT_Api already exists for device type cuda``
     - Both cuda12 and cuda13 plugins installed
     - Uninstall all, reinstall only ONE
   * - ``nvcc not found``
     - CUDA toolkit missing or not in PATH
     - ``sudo apt install nvidia-cuda-toolkit`` or ``export PATH=/usr/local/cuda/bin:$PATH``
   * - ``Backend: cpu`` (GPU exists)
     - GPU JAX packages not installed
     - ``make install-jax-gpu``
   * - ``libcuda.so not found``
     - CUDA libs not in LD_LIBRARY_PATH
     - ``export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH``
   * - ``JaxRuntimeError: NOT_FOUND: cusolver_*``
     - Plugin/jaxlib version mismatch
     - Same as first row

Use ``make gpu-diagnose`` to automatically detect plugin conflicts and version mismatches.

JAX Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues with JAX installation, refer to the
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.

* GPU acceleration requires Linux + CUDA 12.1+ or 13.x + NVIDIA driver >= 525 (CUDA 12) or >= 560 (CUDA 13)
* macOS and Windows only support CPU mode

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure all dependencies are installed:

.. code-block:: bash

   uv sync
