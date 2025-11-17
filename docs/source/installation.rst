Installation
============

Requirements
------------

* Python 3.12 or higher (3.8-3.11 NOT supported)
* JAX 0.8.0 (exact version required)
* jaxlib 0.8.0 (must match JAX version exactly)
* NumPy 2.0.0 or higher
* SciPy 1.16.0 or higher
* NLSQ 0.2.1 or higher (GPU-accelerated optimization)
* NumPyro 0.19.0 or higher (Bayesian inference)
* ArviZ 0.22.0 or higher (Bayesian visualization)

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install rheojax

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/imewei/rheojax.git
   cd rheojax
   pip install -e ".[dev]"

Optional Dependencies
---------------------

GPU Support (Linux + CUDA 12.1-12.9 Only)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

GPU acceleration provides 20-100x speedup for large datasets.

**Platform Requirements:**

* ✅ Linux + NVIDIA GPU + CUDA 12.1-12.9: Full GPU acceleration
* ❌ macOS: CPU-only (Apple Silicon/Intel, no NVIDIA GPU support)
* ❌ Windows: CPU-only (CUDA support experimental/unstable)

**Quick Install (from repository):**

.. code-block:: bash

   make install-jax-gpu

**Manual Install:**

.. code-block:: bash

   pip uninstall -y jax jaxlib
   pip install jax[cuda12-local]==0.8.0 jaxlib==0.8.0

**Requirements:**

* System CUDA 12.1-12.9 pre-installed (not bundled)
* NVIDIA driver >= 525
* Linux x86_64 or aarch64

**Verify GPU Detection:**

.. code-block:: python

   import jax
   print(jax.devices())  # Should show [cuda(id=0)] if GPU detected

**Note:** There is no ``[gpu]`` pip extra. GPU installation must be done manually to avoid platform conflicts.

I/O Support
~~~~~~~~~~~

For additional file format support (HDF5, Excel):

.. code-block:: bash

   pip install "rheojax[io]"

Machine Learning
~~~~~~~~~~~~~~~~

For ML-based transforms:

.. code-block:: bash

   pip install "rheojax[ml]"

All Dependencies
~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

   pip install "rheojax[all]"

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

JAX Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues with JAX installation, refer to the
`JAX installation guide <https://jax.readthedocs.io/en/latest/installation.html>`_.

**GPU Support:**

* GPU acceleration requires Linux + CUDA 12.1-12.9 + NVIDIA driver >= 525
* macOS and Windows only support CPU mode
* JAX and jaxlib versions must match exactly (both 0.8.0)

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure all dependencies are installed:

.. code-block:: bash

   pip install -r requirements.txt
