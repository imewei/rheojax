Installation
============

Requirements
------------

* Python 3.9 or higher
* JAX 0.4.20 or higher
* NumPy 1.24.0 or higher
* SciPy 1.11.0 or higher

Basic Installation
------------------

Install from PyPI:

.. code-block:: bash

   pip install rheo

Development Installation
------------------------

For development, clone the repository and install in editable mode:

.. code-block:: bash

   git clone https://github.com/username/rheo.git
   cd rheo
   pip install -e ".[dev]"

Optional Dependencies
---------------------

GPU Support
~~~~~~~~~~~

For GPU acceleration:

.. code-block:: bash

   pip install "rheo[gpu]"

I/O Support
~~~~~~~~~~~

For additional file format support (HDF5, Excel):

.. code-block:: bash

   pip install "rheo[io]"

Machine Learning
~~~~~~~~~~~~~~~~

For ML-based transforms:

.. code-block:: bash

   pip install "rheo[ml]"

All Dependencies
~~~~~~~~~~~~~~~~

To install all optional dependencies:

.. code-block:: bash

   pip install "rheo[all]"

Verifying Installation
----------------------

Verify the installation:

.. code-block:: python

   import rheo
   print(rheo.__version__)
   print(rheo.__jax_version__)

This should display the version numbers without errors.

Troubleshooting
---------------

JAX Installation Issues
~~~~~~~~~~~~~~~~~~~~~~~

If you encounter issues with JAX installation, particularly on Windows, refer to the
`JAX installation guide <https://github.com/google/jax#installation>`_.

For GPU support, ensure you have the appropriate CUDA version installed.

Import Errors
~~~~~~~~~~~~~

If you encounter import errors, ensure all dependencies are installed:

.. code-block:: bash

   pip install -r requirements.txt
