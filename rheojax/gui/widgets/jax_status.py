"""
JAX Status Widget
================

GPU/device status indicator with memory monitoring.
"""

from rheojax.gui.compat import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


class JAXStatusWidget(QWidget):
    """Compact widget for JAX runtime information and device control.

    Features:
        - Device dropdown (cpu, cuda:0, cuda:1, etc.)
        - Memory bar (used/total)
        - Float64 indicator
        - JIT cache count
        - Compact horizontal layout

    Signals
    -------
    device_change_requested : Signal(str)
        Emitted when user selects a different device

    Example
    -------
    >>> status = JAXStatusWidget()  # doctest: +SKIP
    >>> status.update_device_list(['cpu', 'cuda:0'])  # doctest: +SKIP
    >>> status.set_current_device('cuda:0')  # doctest: +SKIP
    >>> status.update_memory(4096, 8192)  # doctest: +SKIP
    """

    device_change_requested = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize JAX status widget.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Main horizontal layout
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(15)

        # Device selector
        device_layout = QVBoxLayout()
        device_layout.setSpacing(2)
        device_label = QLabel("Device:")
        device_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(100)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self._device_combo)
        main_layout.addLayout(device_layout)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep1)

        # Memory usage
        memory_layout = QVBoxLayout()
        memory_layout.setSpacing(2)
        memory_label = QLabel("Memory:")
        memory_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        self._memory_bar = QProgressBar()
        self._memory_bar.setMinimumWidth(120)
        self._memory_bar.setMaximumHeight(18)
        self._memory_bar.setTextVisible(True)
        self._memory_bar.setFormat("%v MB / %m MB")
        memory_layout.addWidget(memory_label)
        memory_layout.addWidget(self._memory_bar)
        main_layout.addLayout(memory_layout)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep2)

        # Float64 indicator
        float64_layout = QVBoxLayout()
        float64_layout.setSpacing(2)
        float64_label = QLabel("Float64:")
        float64_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        self._float64_indicator = QLabel("Unknown")
        self._float64_indicator.setAlignment(Qt.AlignCenter)
        self._float64_indicator.setStyleSheet(
            "padding: 3px 10px; border-radius: 3px; background-color: #e0e0e0;"
        )
        float64_layout.addWidget(float64_label)
        float64_layout.addWidget(self._float64_indicator)
        main_layout.addLayout(float64_layout)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setFrameShadow(QFrame.Sunken)
        main_layout.addWidget(sep3)

        # JIT cache count
        jit_layout = QVBoxLayout()
        jit_layout.setSpacing(2)
        jit_label = QLabel("JIT Cache:")
        jit_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        self._jit_count_label = QLabel("0")
        self._jit_count_label.setAlignment(Qt.AlignCenter)
        self._jit_count_label.setStyleSheet("font-size: 10pt;")
        jit_layout.addWidget(jit_label)
        jit_layout.addWidget(self._jit_count_label)
        main_layout.addLayout(jit_layout)

        # Compiling indicator
        compile_layout = QVBoxLayout()
        compile_layout.setSpacing(2)
        compile_label = QLabel("Compiling:")
        compile_label.setStyleSheet("font-weight: bold; font-size: 9pt;")
        self._compile_indicator = QLabel("Idle")
        self._compile_indicator.setAlignment(Qt.AlignCenter)
        self._compile_indicator.setStyleSheet(
            "padding: 3px 10px; border-radius: 3px; background-color: #e0e0e0;"
        )
        compile_layout.addWidget(compile_label)
        compile_layout.addWidget(self._compile_indicator)
        main_layout.addLayout(compile_layout)

        main_layout.addStretch()

        # Initialize state
        self._current_device = "cpu"
        self.update_jit_cache(0)
        self.set_compiling(False)
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def update_device_list(self, devices: list[str]) -> None:
        """Update the list of available devices.

        Parameters
        ----------
        devices : list[str]
            List of device names (e.g., ['cpu', 'cuda:0', 'cuda:1'])
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="update_device_list",
            devices=devices,
        )
        # Disconnect signal temporarily
        self._device_combo.currentTextChanged.disconnect(self._on_device_changed)

        # Save current selection
        current = self._device_combo.currentText()

        # Update list
        self._device_combo.clear()
        self._device_combo.addItems(devices)

        # Restore selection if still available
        if current in devices:
            self._device_combo.setCurrentText(current)
        elif devices:
            self._device_combo.setCurrentIndex(0)

        # Reconnect signal
        self._device_combo.currentTextChanged.connect(self._on_device_changed)

    def set_current_device(self, device: str) -> None:
        """Set the currently active device.

        Parameters
        ----------
        device : str
            Device name (e.g., 'cpu', 'cuda:0')
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_current_device",
            device=device,
        )
        self._current_device = device

        # Update combo box if device is in list
        index = self._device_combo.findText(device)
        if index >= 0:
            self._device_combo.blockSignals(True)
            self._device_combo.setCurrentIndex(index)
            self._device_combo.blockSignals(False)

    def update_memory(self, used: int, total: int) -> None:
        """Update memory usage display.

        Parameters
        ----------
        used : int
            Used memory in MB
        total : int
            Total memory in MB
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="update_memory",
            used_mb=used,
            total_mb=total,
        )
        self._memory_bar.setMaximum(total)
        self._memory_bar.setValue(used)

        # Color based on usage
        if used / total > 0.9:
            # Red for >90% usage
            self._memory_bar.setStyleSheet(
                """
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #F44336;
                }
            """
            )
        elif used / total > 0.7:
            # Orange for >70% usage
            self._memory_bar.setStyleSheet(
                """
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #FF9800;
                }
            """
            )
        else:
            # Green for normal usage
            self._memory_bar.setStyleSheet(
                """
                QProgressBar {
                    border: 1px solid #ccc;
                    border-radius: 3px;
                    text-align: center;
                }
                QProgressBar::chunk {
                    background-color: #4CAF50;
                }
            """
            )

    def set_float64_enabled(self, enabled: bool) -> None:
        """Set float64 enabled indicator.

        Parameters
        ----------
        enabled : bool
            Whether float64 is enabled
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_float64_enabled",
            enabled=enabled,
        )
        if enabled:
            self._float64_indicator.setText("Enabled")
            self._float64_indicator.setStyleSheet(
                "padding: 3px 10px; border-radius: 3px; "
                "background-color: #4CAF50; color: white; font-weight: bold;"
            )
        else:
            self._float64_indicator.setText("Disabled")
            self._float64_indicator.setStyleSheet(
                "padding: 3px 10px; border-radius: 3px; "
                "background-color: #F44336; color: white; font-weight: bold;"
            )

    def update_jit_cache(self, count: int) -> None:
        """Update JIT cache count.

        Parameters
        ----------
        count : int
            Number of cached JIT compilations
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="update_jit_cache",
            count=count,
        )
        self._jit_count_label.setText(str(count))

        # Color based on count
        if count > 100:
            color = "#FF9800"  # Orange for many cached functions
        elif count > 0:
            color = "#4CAF50"  # Green for normal
        else:
            color = "#666666"  # Gray for zero

        self._jit_count_label.setStyleSheet(
            f"font-size: 10pt; color: {color}; font-weight: bold;"
        )

    def set_compiling(self, compiling: bool) -> None:
        """Show compiling indicator."""
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_compiling",
            compiling=compiling,
        )
        if compiling:
            self._compile_indicator.setText("Compiling...")
            self._compile_indicator.setStyleSheet(
                "padding:3px 10px; border-radius:3px; background-color:#FFB74D; color:#000;"
            )
        else:
            self._compile_indicator.setText("Idle")
            self._compile_indicator.setStyleSheet(
                "padding:3px 10px; border-radius:3px; background-color:#C8E6C9; color:#1B5E20;"
            )

    def _on_device_changed(self, device: str) -> None:
        """Handle device selection change.

        Parameters
        ----------
        device : str
            Selected device name
        """
        if device and device != self._current_device:
            logger.debug(
                "User interaction",
                widget=self.__class__.__name__,
                action="device_changed",
                old_device=self._current_device,
                new_device=device,
            )
            self.device_change_requested.emit(device)
