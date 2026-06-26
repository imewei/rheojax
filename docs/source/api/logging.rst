Logging (rheojax.logging)
=========================

Structured logging system for monitoring and debugging RheoJAX operations.

Configuration
-------------

configure_logging
~~~~~~~~~~~~~~~~~

.. autofunction:: rheojax.logging.configure_logging
   :noindex:

   Configure the RheoJAX logging system.

   **Quick Start:**

   .. code-block:: python

      from rheojax.logging import configure_logging

      # Basic configuration
      configure_logging(level="INFO")

      # Verbose debugging
      configure_logging(level="DEBUG")

      # With file output
      configure_logging(level="INFO", log_file="rheojax.log")

get_logger
~~~~~~~~~~

.. autofunction:: rheojax.logging.get_logger
   :noindex:

   Get a logger instance for the specified name.

   .. code-block:: python

      from rheojax.logging import get_logger

      logger = get_logger(__name__)
      logger.info("Starting model fitting", model="Maxwell")
      logger.debug("Iteration 100", cost=1e-5)

LogConfig
~~~~~~~~~

.. autoclass:: rheojax.logging.LogConfig
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

LogFormat
~~~~~~~~~

.. autoclass:: rheojax.logging.LogFormat
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

The logging system respects these environment variables:

.. list-table:: Environment Variables
   :header-rows: 1
   :widths: 30 50 20

   * - Variable
     - Description
     - Default
   * - ``RHEOJAX_LOG_LEVEL``
     - Global log level (DEBUG, INFO, WARNING, ERROR)
     - INFO
   * - ``RHEOJAX_LOG_FILE``
     - Path to log file (enables file logging)
     - None
   * - ``RHEOJAX_LOG_FORMAT``
     - Output format (standard, detailed, json)
     - standard

Context Managers
----------------

Operation Logging
~~~~~~~~~~~~~~~~~

Context managers for automatic timing and context tracking:

log_fit
^^^^^^^

.. autofunction:: rheojax.logging.log_fit
   :noindex:

   Log model fitting operations with timing.

   .. code-block:: python

      from rheojax.logging import log_fit, get_logger

      logger = get_logger(__name__)

      with log_fit(logger, model="Maxwell", data_shape=(100,)) as ctx:
          result = model.fit(x, y)
          ctx["R2"] = result.r_squared  # Add to completion log

log_bayesian
^^^^^^^^^^^^

.. autofunction:: rheojax.logging.log_bayesian
   :noindex:

   Log Bayesian inference operations.

   .. code-block:: python

      from rheojax.logging import log_bayesian, get_logger

      logger = get_logger(__name__)

      with log_bayesian(logger, "Maxwell", num_warmup=1000, num_samples=2000) as ctx:
          result = model.fit_bayesian(x, y)
          ctx["r_hat"] = compute_rhat(result)
          ctx["divergences"] = result.divergences

log_transform
^^^^^^^^^^^^^

.. autofunction:: rheojax.logging.log_transform
   :noindex:

   Log data transformation operations.

log_io
^^^^^^

.. autofunction:: rheojax.logging.log_io
   :noindex:

   Log I/O operations (file reading/writing).

log_pipeline_stage
^^^^^^^^^^^^^^^^^^

.. autofunction:: rheojax.logging.log_pipeline_stage
   :noindex:

   Log pipeline stage execution.

log_operation
^^^^^^^^^^^^^

.. autofunction:: rheojax.logging.log_operation
   :noindex:

   Generic operation logging context manager.

Formatters
----------

StandardFormatter
~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.StandardFormatter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Standard log format: ``LEVEL - message [key=value ...]``

DetailedFormatter
~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.DetailedFormatter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Detailed format with timestamp, logger name, file location.

JSONFormatter
~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.JSONFormatter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   JSON output for machine parsing and log aggregation.

ScientificFormatter
~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.ScientificFormatter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Scientific notation for numerical values.

Handlers
--------

RheoJAXStreamHandler
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.RheoJAXStreamHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Console output handler with optional color support.

RheoJAXRotatingFileHandler
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.RheoJAXRotatingFileHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Rotating file handler for log rotation.

RheoJAXMemoryHandler
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.RheoJAXMemoryHandler
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   In-memory buffer for log capture and testing.

Examples
--------

Basic Setup
~~~~~~~~~~~

.. code-block:: python

   from rheojax.logging import configure_logging, get_logger

   # Configure once at startup
   configure_logging(level="INFO")

   # Get logger in each module
   logger = get_logger(__name__)

   # Log with structured data
   logger.info("Model fitted", model="Maxwell", R2=0.9987, time=1.23)
   logger.debug("Parameter values", G0=1e5, eta=1000)

Production Configuration
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.logging import configure_logging

   configure_logging(
       level="INFO",
       log_file="/var/log/rheojax/app.log",
       format="json",  # Machine-readable
       max_bytes=10_000_000,  # 10 MB rotation
       backup_count=5
   )

Debugging Workflow
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import os

   # Enable debug logging via environment
   os.environ["RHEOJAX_LOG_LEVEL"] = "DEBUG"

   from rheojax.logging import configure_logging, get_logger, log_fit

   configure_logging()  # Uses environment variable
   logger = get_logger(__name__)

   # All operations now logged at debug level
   with log_fit(logger, "FractionalMaxwell", data_shape=(500,)) as ctx:
       result = model.fit(x, y)
       ctx["iterations"] = result.nit
       ctx["final_cost"] = result.fun

Integration with Model Fitting
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from rheojax.logging import (
       configure_logging,
       get_logger,
       log_fit,
       log_bayesian,
   )
   from rheojax.models import Maxwell

   configure_logging(level="INFO")
   logger = get_logger(__name__)

   model = Maxwell()

   # NLSQ fitting with logging
   with log_fit(logger, "Maxwell", data_shape=x.shape) as ctx:
       model.fit(x, y)
       ctx["R2"] = model.score(x, y)

   # Bayesian inference with logging
   with log_bayesian(logger, "Maxwell", num_samples=2000) as ctx:
       result = model.fit_bayesian(x, y, num_samples=2000)

       ctx["r_hat"] = max(result.diagnostics["r_hat"].values())
       ctx["divergences"] = result.diagnostics.get("divergences", 0)

See Also
--------

- :doc:`../user_guide/05_appendices/troubleshooting` - Debugging with logs
- :doc:`core` - Core module with BaseModel and BayesianMixin
- :doc:`pipeline` - Pipeline API with built-in logging
