"""
Status Bar
==========

Application status bar with progress indicators, JAX device status, and memory monitoring.
"""

from PySide6.QtWidgets import QLabel, QProgressBar, QStatusBar, QWidget


class StatusBar(QStatusBar):
    """Application status bar with progress tracking and system indicators.

    Layout:
        - Left: Status message area
        - Center: Progress bar (hidden when not in use)
        - Right sections:
          - JAX device indicator (e.g., "cuda:0")
          - Memory usage (e.g., "245/8192 MB")
          - Float64 indicator (checkmark if enabled)

    Example
    -------
    >>> status_bar = StatusBar()  # doctest: +SKIP
    >>> status_bar.show_message("Ready", 3000)  # doctest: +SKIP
    >>> status_bar.show_progress(50, 100, "Processing...")  # doctest: +SKIP
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize status bar.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)

        # Left: Status message (permanent)
        self.message_label = QLabel("Ready")
        self.addWidget(self.message_label, 1)  # Stretch factor 1

        # Center: Progress bar (initially hidden)
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setMaximumWidth(300)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setVisible(False)
        self.addWidget(self.progress_bar)

        # Right: Permanent widgets
        # JAX device indicator
        self.jax_label = QLabel("JAX: cpu")
        self.jax_label.setStyleSheet("QLabel { padding: 0 10px; }")
        self.addPermanentWidget(self.jax_label)

        # Memory usage indicator
        self.memory_label = QLabel("Memory: 0/0 MB")
        self.memory_label.setStyleSheet("QLabel { padding: 0 10px; }")
        self.addPermanentWidget(self.memory_label)

        # Float64 indicator
        # Note: Using ASCII characters to avoid potential CoreText/ImageIO
        # crashes on macOS when rendering Unicode symbols in Qt widgets
        self.float64_label = QLabel("Float64: [X]")
        self.float64_label.setStyleSheet("QLabel { padding: 0 10px; }")
        self.addPermanentWidget(self.float64_label)

    def show_message(self, message: str, timeout: int = 0) -> None:
        """Display temporary status message.

        Parameters
        ----------
        message : str
            Status message
        timeout : int, default=0
            Timeout in milliseconds (0 = permanent)
        """
        self.message_label.setText(message)
        if timeout > 0:
            # Also show in Qt status bar for timeout support
            super().showMessage(message, timeout)

    def show_progress(self, value: int, maximum: int, text: str = "") -> None:
        """Update and show progress bar.

        Parameters
        ----------
        value : int
            Current progress value
        maximum : int
            Maximum progress value
        text : str, optional
            Progress text to display
        """
        self.progress_bar.setMaximum(maximum)
        self.progress_bar.setValue(value)
        if text:
            self.progress_bar.setFormat(f"{text} (%p%)")
        else:
            self.progress_bar.setFormat("%p%")
        self.progress_bar.setVisible(True)

    def hide_progress(self) -> None:
        """Hide progress bar."""
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)

    def update_jax_status(
        self,
        device: str,
        memory_used: float,
        memory_total: float,
        float64_enabled: bool,
    ) -> None:
        """Update JAX device and memory status.

        Parameters
        ----------
        device : str
            JAX device name (e.g., "cpu", "cuda:0", "gpu")
        memory_used : float
            Used memory in MB
        memory_total : float
            Total memory in MB
        float64_enabled : bool
            Whether float64 is enabled
        """
        # Update JAX device
        self.jax_label.setText(f"JAX: {device}")

        # Update memory usage
        self.memory_label.setText(f"Memory: {int(memory_used)}/{int(memory_total)} MB")

        # Update float64 indicator (ASCII to avoid macOS rendering issues)
        if float64_enabled:
            self.float64_label.setText("Float64: [OK]")
            self.float64_label.setStyleSheet(
                "QLabel { padding: 0 10px; color: green; }"
            )
        else:
            self.float64_label.setText("Float64: [X]")
            self.float64_label.setStyleSheet(
                "QLabel { padding: 0 10px; color: orange; }"
            )

    def update_memory(self, used_mb: float, total_mb: float) -> None:
        """Update memory usage indicator.

        Parameters
        ----------
        used_mb : float
            Used memory in megabytes
        total_mb : float
            Total available memory in megabytes
        """
        self.memory_label.setText(f"Memory: {int(used_mb)}/{int(total_mb)} MB")

    def set_jax_device(self, device: str) -> None:
        """Update JAX device indicator.

        Parameters
        ----------
        device : str
            Device name (e.g., "cpu", "cuda:0", "gpu")
        """
        self.jax_label.setText(f"JAX: {device}")

    def set_float64_status(self, enabled: bool) -> None:
        """Update float64 status indicator.

        Parameters
        ----------
        enabled : bool
            Whether float64 is enabled
        """
        # ASCII characters to avoid macOS CoreText rendering issues
        if enabled:
            self.float64_label.setText("Float64: [OK]")
            self.float64_label.setStyleSheet(
                "QLabel { padding: 0 10px; color: green; }"
            )
        else:
            self.float64_label.setText("Float64: [X]")
            self.float64_label.setStyleSheet(
                "QLabel { padding: 0 10px; color: orange; }"
            )
