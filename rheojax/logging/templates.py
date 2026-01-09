"""
RheoJAX Logging Message Templates.

Standard message templates for consistent logging across all modules.
Use these templates to ensure grep-able, parseable log messages.

Usage:
    from rheojax.logging.templates import MSG

    logger.debug(MSG.ENTERING, function="fit", params={"model": "Maxwell"})
    logger.info(MSG.OPERATION_START, operation="model_fit", model="Maxwell")
    logger.error(MSG.OPERATION_FAILED, operation="fit", error=str(e), exc_info=True)
"""


class MessageTemplates:
    """Standard log message templates for consistency.

    These templates ensure all log messages follow a consistent format
    that can be easily searched, filtered, and parsed.
    """

    # Function entry/exit (DEBUG level)
    ENTERING = "Entering {function}"
    EXITING = "Exiting {function}"
    INITIALIZING = "Initializing {class_name}"
    INITIALIZED = "{class_name} initialized"

    # Operation lifecycle (INFO level)
    OPERATION_START = "Starting {operation}"
    OPERATION_COMPLETE = "{operation} complete"
    OPERATION_FAILED = "{operation} failed"

    # GUI interactions (DEBUG level)
    BUTTON_CLICKED = "Button clicked"
    MENU_SELECTED = "Menu item selected"
    PAGE_NAVIGATED = "Page navigation"
    DIALOG_OPENED = "Dialog opened"
    DIALOG_CLOSED = "Dialog closed"
    SELECTION_CHANGED = "Selection changed"
    VALUE_CHANGED = "Value changed"
    WIDGET_CREATED = "Widget created"
    WIDGET_DESTROYED = "Widget destroyed"

    # State management (DEBUG level)
    ACTION_DISPATCHED = "Action dispatched"
    STATE_UPDATED = "State updated"
    SIGNAL_EMITTED = "Signal emitted"
    SIGNAL_CONNECTED = "Signal connected"
    SELECTOR_CALLED = "Selector called"
    CACHE_HIT = "Cache hit"
    CACHE_MISS = "Cache miss"

    # Data flow (DEBUG level)
    DATA_RECEIVED = "Data received"
    DATA_TRANSFORMED = "Data transformed"
    DATA_VALIDATED = "Data validated"
    DATA_EXPORTED = "Data exported"

    # Service operations (INFO level)
    FIT_START = "Starting model fit"
    FIT_COMPLETE = "Model fit complete"
    LOAD_START = "Loading data"
    LOAD_COMPLETE = "Data loaded"
    INFERENCE_START = "Starting Bayesian inference"
    INFERENCE_COMPLETE = "Bayesian inference complete"
    TRANSFORM_START = "Starting transform"
    TRANSFORM_COMPLETE = "Transform complete"
    EXPORT_START = "Starting export"
    EXPORT_COMPLETE = "Export complete"
    PLOT_START = "Generating plot"
    PLOT_COMPLETE = "Plot generated"

    # Performance (INFO level)
    TIMING_ITERATION = "Iteration complete"
    TIMING_BATCH = "Batch complete"
    TIMING_TOTAL = "Total elapsed"

    # Errors (ERROR level)
    ERROR_GENERAL = "{operation} failed"
    ERROR_VALIDATION = "Validation failed"
    ERROR_IO = "I/O operation failed"
    ERROR_CONVERGENCE = "Convergence failed"

    # Warnings (WARNING level)
    WARNING_SLOW = "Operation slower than expected"
    WARNING_MEMORY = "High memory usage detected"
    WARNING_FALLBACK = "Using fallback method"
    WARNING_DEPRECATED = "Deprecated feature used"


# Singleton instance for easy import
MSG = MessageTemplates()


# Common field names for structured logging
class Fields:
    """Standard field names for structured logging context."""

    # Identifiers
    MODEL = "model"
    TRANSFORM = "transform"
    OPERATION = "operation"
    FUNCTION = "function"
    CLASS_NAME = "class_name"
    WIDGET = "widget"
    PAGE = "page"
    DIALOG = "dialog"
    BUTTON_ID = "button_id"
    ACTION_TYPE = "action_type"

    # Data shapes
    DATA_SHAPE = "data_shape"
    INPUT_SHAPE = "input_shape"
    OUTPUT_SHAPE = "output_shape"
    N_POINTS = "n_points"
    N_RECORDS = "n_records"
    N_DATASETS = "n_datasets"

    # Timing
    ELAPSED_SECONDS = "elapsed_seconds"
    ITERATION = "iteration"
    TOTAL_ITERATIONS = "total_iterations"

    # Results
    SUCCESS = "success"
    STATUS = "status"
    R_SQUARED = "r_squared"
    RMSE = "rmse"
    R_HAT = "r_hat"
    ESS = "ess"
    DIVERGENCES = "divergences"

    # Errors
    ERROR = "error"
    ERROR_TYPE = "error_type"
    ERROR_MESSAGE = "error_message"

    # File operations
    FILEPATH = "filepath"
    FILE_SIZE = "file_size"
    FORMAT = "format"

    # State
    BEFORE = "before"
    AFTER = "after"
    CHANGED_KEYS = "changed_keys"
    PAYLOAD_KEYS = "payload_keys"


# Singleton for fields
F = Fields()


__all__ = ["MSG", "MessageTemplates", "F", "Fields"]
