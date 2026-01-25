Logging (rheojax.logging)
=========================

Comprehensive logging system for monitoring and debugging RheoJAX operations.
Provides structured logging, JAX-safe utilities, and performance metrics.

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

JAX-Safe Utilities
------------------

Utilities for logging JAX arrays without triggering device transfers:

log_array_info
~~~~~~~~~~~~~~

.. autofunction:: rheojax.logging.log_array_info
   :noindex:

   Get array metadata without device transfer (shape, dtype only).

   .. code-block:: python

      from rheojax.logging import log_array_info

      # Safe at INFO level - no device transfer
      info = log_array_info(jax_array, "residuals")
      print(info)  # {"name": "residuals", "shape": (100,), "dtype": "float64"}

log_array_stats
~~~~~~~~~~~~~~~

.. autofunction:: rheojax.logging.log_array_stats
   :noindex:

   Get array statistics (triggers device transfer - use at DEBUG level).

log_numerical_issue
~~~~~~~~~~~~~~~~~~~

.. autofunction:: rheojax.logging.log_numerical_issue
   :noindex:

   Check for and log NaN/Inf values.

   .. code-block:: python

      from rheojax.logging import log_numerical_issue, get_logger

      logger = get_logger(__name__)

      if log_numerical_issue(logger, residuals, "residuals", "during fitting"):
          raise ValueError("Numerical instability detected")

log_jax_config
~~~~~~~~~~~~~~

.. autofunction:: rheojax.logging.log_jax_config
   :noindex:

   Log JAX configuration (version, devices, float64 status).

   .. code-block:: python

      from rheojax.logging import log_jax_config, get_logger

      logger = get_logger(__name__)
      log_jax_config(logger)  # Logs JAX version, devices, precision

jax_safe_log
~~~~~~~~~~~~

.. autofunction:: rheojax.logging.jax_safe_log
   :noindex:

   Safely log a value that may be a JAX array.

jax_debug_log
~~~~~~~~~~~~~

.. autofunction:: rheojax.logging.jax_debug_log
   :noindex:

   Log JAX values only at DEBUG level (with device transfer).

Performance Tracking
--------------------

timed
~~~~~

.. autofunction:: rheojax.logging.timed
   :noindex:

   Decorator for timing function execution.

   .. code-block:: python

      from rheojax.logging import timed
      import logging

      @timed(level=logging.INFO)
      def expensive_operation():
          # ... computation ...
          pass

log_memory
~~~~~~~~~~

.. autofunction:: rheojax.logging.log_memory
   :noindex:

   Log current memory usage.

IterationLogger
~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.IterationLogger
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Log optimization iterations at configurable frequency.

   .. code-block:: python

      from rheojax.logging import IterationLogger, get_logger

      logger = get_logger(__name__)
      iter_logger = IterationLogger(logger, log_every=100)

      for i in range(1000):
          cost = optimizer.step()
          iter_logger.log(cost=cost)

      iter_logger.log_final()

ConvergenceTracker
~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.ConvergenceTracker
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Track convergence metrics over iterations.

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

Exporters (OpenTelemetry)
-------------------------

LogExporter
~~~~~~~~~~~

.. autoclass:: rheojax.logging.LogExporter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Base class for log exporters.

OpenTelemetryLogExporter
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.OpenTelemetryLogExporter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Export logs to OpenTelemetry-compatible backends.

ConsoleExporter
~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.ConsoleExporter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Export logs to console (for debugging).

BatchingExporter
~~~~~~~~~~~~~~~~

.. autoclass:: rheojax.logging.BatchingExporter
   :members:
   :undoc-members:
   :show-inheritance:
   :noindex:

   Batch logs before exporting for efficiency.

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
       log_numerical_issue
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

       # Check for numerical issues
       if log_numerical_issue(logger, result.posterior_samples["G0"], "G0", "posterior"):
           logger.warning("Posterior samples contain numerical issues")

       ctx["r_hat"] = max(result.diagnostics["r_hat"].values())
       ctx["divergences"] = result.diagnostics.get("divergences", 0)

See Also
--------

- :doc:`../user_guide/05_appendices/troubleshooting` - Debugging with logs
- :doc:`core` - Core module with BaseModel and BayesianMixin
- :doc:`pipeline` - Pipeline API with built-in logging
