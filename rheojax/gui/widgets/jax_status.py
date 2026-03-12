"""
JAX Status Widget
================

GPU/device status indicator with memory monitoring.
Uses the RheoJAX design token system for consistent theming.
"""

from rheojax.gui.compat import (
    QComboBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    Qt,
    QVBoxLayout,
    QWidget,
    Signal,
)
from rheojax.gui.resources.styles.tokens import (
    BorderRadius,
    Spacing,
    Typography,
    themed,
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
        - Compact horizontal layout with themed status badges

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

        # Main horizontal layout — no separators, just spacing
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(Spacing.MD, Spacing.SM, Spacing.MD, Spacing.SM)
        main_layout.setSpacing(Spacing.XL)

        # Shared label style
        label_style = f"""
            font-family: {Typography.FONT_FAMILY};
            font-weight: {Typography.WEIGHT_SEMIBOLD};
            font-size: {Typography.SIZE_XS}pt;
            color: {themed('TEXT_MUTED')};
            text-transform: uppercase;
            letter-spacing: 0.5px;
        """

        # --- Device selector ---
        device_layout = QVBoxLayout()
        device_layout.setSpacing(Spacing.XXS)
        device_label = QLabel("Device")
        device_label.setStyleSheet(label_style)
        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(110)
        self._device_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {themed('BG_SURFACE')};
                border: 1px solid {themed('BORDER_SUBTLE')};
                border-radius: {BorderRadius.MD}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-family: {Typography.FONT_FAMILY_MONO};
                font-size: {Typography.SIZE_SM}pt;
                color: {themed('TEXT_PRIMARY')};
            }}
            QComboBox:hover {{
                border-color: {themed('BORDER_DEFAULT')};
            }}
            QComboBox:focus {{
                border-color: {themed('BORDER_FOCUS')};
            }}
            QComboBox::drop-down {{
                border: none;
                width: 20px;
            }}
        """)
        self._device_combo.currentTextChanged.connect(self._on_device_changed)
        device_layout.addWidget(device_label)
        device_layout.addWidget(self._device_combo)
        main_layout.addLayout(device_layout)

        # --- Memory usage ---
        memory_layout = QVBoxLayout()
        memory_layout.setSpacing(Spacing.XXS)
        memory_label = QLabel("Memory")
        memory_label.setStyleSheet(label_style)
        self._memory_bar = QProgressBar()
        self._memory_bar.setMinimumWidth(130)
        self._memory_bar.setMaximumHeight(20)
        self._memory_bar.setTextVisible(True)
        self._memory_bar.setFormat("%v MB / %m MB")
        self._set_memory_bar_style("normal")
        memory_layout.addWidget(memory_label)
        memory_layout.addWidget(self._memory_bar)
        main_layout.addLayout(memory_layout)

        # --- Float64 indicator ---
        float64_layout = QVBoxLayout()
        float64_layout.setSpacing(Spacing.XXS)
        float64_label = QLabel("Float64")
        float64_label.setStyleSheet(label_style)
        self._float64_indicator = QLabel("Unknown")
        self._float64_indicator.setAlignment(Qt.AlignCenter)
        self._float64_indicator.setMinimumWidth(72)
        self._set_badge_style(self._float64_indicator, "pending")
        float64_layout.addWidget(float64_label)
        float64_layout.addWidget(self._float64_indicator)
        main_layout.addLayout(float64_layout)

        # --- JIT cache count ---
        jit_layout = QVBoxLayout()
        jit_layout.setSpacing(Spacing.XXS)
        jit_label = QLabel("JIT Cache")
        jit_label.setStyleSheet(label_style)
        self._jit_count_label = QLabel("0")
        self._jit_count_label.setAlignment(Qt.AlignCenter)
        self._jit_count_label.setMinimumWidth(48)
        jit_layout.addWidget(jit_label)
        jit_layout.addWidget(self._jit_count_label)
        main_layout.addLayout(jit_layout)

        # --- Compiling indicator ---
        compile_layout = QVBoxLayout()
        compile_layout.setSpacing(Spacing.XXS)
        compile_label = QLabel("Status")
        compile_label.setStyleSheet(label_style)
        self._compile_indicator = QLabel("Idle")
        self._compile_indicator.setAlignment(Qt.AlignCenter)
        self._compile_indicator.setMinimumWidth(80)
        compile_layout.addWidget(compile_label)
        compile_layout.addWidget(self._compile_indicator)
        main_layout.addLayout(compile_layout)

        main_layout.addStretch()

        # Initialize state
        self._current_device = "cpu"
        self.update_jit_cache(0)
        self.set_compiling(False)
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    # ------------------------------------------------------------------
    # Styling helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _set_badge_style(label: QLabel, status: str) -> None:
        """Apply themed badge styling to a QLabel.

        Parameters
        ----------
        label : QLabel
            Target label widget
        status : str
            One of: success, warning, error, pending
        """
        status_map = {
            "success": (themed("SUCCESS"), themed("SUCCESS_LIGHT")),
            "warning": (themed("WARNING"), themed("WARNING_LIGHT")),
            "error": (themed("ERROR"), themed("ERROR_LIGHT")),
            "pending": (themed("TEXT_MUTED"), themed("BG_HOVER")),
        }
        fg, bg = status_map.get(status, status_map["pending"])
        label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border-radius: {BorderRadius.SM}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-family: {Typography.FONT_FAMILY};
                font-size: {Typography.SIZE_SM}pt;
                font-weight: {Typography.WEIGHT_SEMIBOLD};
            }}
        """)

    def _set_memory_bar_style(self, level: str) -> None:
        """Apply themed styling to the memory progress bar.

        Parameters
        ----------
        level : str
            One of: normal, warning, critical
        """
        chunk_colors = {
            "normal": themed("SUCCESS"),
            "warning": themed("WARNING"),
            "critical": themed("ERROR"),
        }
        chunk_color = chunk_colors.get(level, themed("SUCCESS"))

        self._memory_bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: {themed('BG_HOVER')};
                border: 1px solid {themed('BORDER_SUBTLE')};
                border-radius: {BorderRadius.SM}px;
                text-align: center;
                font-family: {Typography.FONT_FAMILY_MONO};
                font-size: {Typography.SIZE_XS}pt;
                color: {themed('TEXT_SECONDARY')};
            }}
            QProgressBar::chunk {{
                background-color: {chunk_color};
                border-radius: {BorderRadius.SM - 1}px;
            }}
        """)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

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

        # Color based on usage level
        if total > 0 and used / total > 0.9:
            self._set_memory_bar_style("critical")
        elif total > 0 and used / total > 0.7:
            self._set_memory_bar_style("warning")
        else:
            # Covers both normal usage (<70%) and total == 0 (unknown)
            self._set_memory_bar_style("normal")

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
            self._set_badge_style(self._float64_indicator, "success")
        else:
            self._float64_indicator.setText("Disabled")
            self._set_badge_style(self._float64_indicator, "error")

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

        # Style based on count
        if count > 100:
            fg = themed("WARNING")
            bg = themed("WARNING_LIGHT")
        elif count > 0:
            fg = themed("PRIMARY")
            bg = themed("PRIMARY_SUBTLE")
        else:
            fg = themed("TEXT_MUTED")
            bg = themed("BG_HOVER")

        self._jit_count_label.setStyleSheet(f"""
            QLabel {{
                background-color: {bg};
                color: {fg};
                border-radius: {BorderRadius.SM}px;
                padding: {Spacing.XXS}px {Spacing.SM}px;
                font-family: {Typography.FONT_FAMILY_MONO};
                font-size: {Typography.SIZE_SM}pt;
                font-weight: {Typography.WEIGHT_BOLD};
            }}
        """)

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
            self._set_badge_style(self._compile_indicator, "warning")
        else:
            self._compile_indicator.setText("Idle")
            self._set_badge_style(self._compile_indicator, "success")

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
