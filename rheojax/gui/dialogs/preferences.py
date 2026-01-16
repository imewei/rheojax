"""
Preferences Dialog
=================

Application settings and preferences.
"""

from typing import Any

from rheojax.gui.compat import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    Qt,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from rheojax.logging import get_logger

logger = get_logger(__name__)


class PreferencesDialog(QDialog):
    """Application preferences dialog.

    Settings organized in tabs:
        - General: Theme, auto-save, recent projects
        - JAX: Device selection, float64 warnings, memory
        - Visualization: Plot style, colors, font size

    Example
    -------
    >>> dialog = PreferencesDialog()  # doctest: +SKIP
    >>> if dialog.exec() == QDialog.DialogCode.Accepted:  # doctest: +SKIP
    ...     prefs = dialog.get_preferences()  # doctest: +SKIP
    """

    def __init__(
        self,
        current_preferences: dict[str, Any] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Initialize preferences dialog.

        Parameters
        ----------
        current_preferences : dict[str, Any], optional
            Current preferences
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        self.current_preferences = current_preferences or {}

        self.setWindowTitle("Preferences")
        self.setMinimumSize(600, 550)

        self._setup_ui()
        self._load_current_preferences()

        logger.debug(
            "Dialog initialized",
            dialog=self.__class__.__name__,
            has_current_preferences=bool(current_preferences),
        )

    def _setup_ui(self) -> None:
        """Set up user interface."""
        layout = QVBoxLayout()

        # Tab widget
        self.tab_widget = QTabWidget()

        # General tab
        self.general_tab = self._create_general_tab()
        self.tab_widget.addTab(self.general_tab, "General")

        # JAX tab
        self.jax_tab = self._create_jax_tab()
        self.tab_widget.addTab(self.jax_tab, "JAX")

        # Visualization tab
        self.viz_tab = self._create_visualization_tab()
        self.tab_widget.addTab(self.viz_tab, "Visualization")

        layout.addWidget(self.tab_widget)

        # Restore defaults button
        restore_layout = QHBoxLayout()
        restore_button = QPushButton("Restore Defaults")
        restore_button.clicked.connect(self._restore_defaults)
        restore_layout.addStretch()
        restore_layout.addWidget(restore_button)
        layout.addLayout(restore_layout)

        # Button box
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Apply
            | QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel
        )
        button_box.button(QDialogButtonBox.StandardButton.Apply).clicked.connect(
            self._on_apply
        )
        button_box.accepted.connect(self._on_accept)
        button_box.rejected.connect(self._on_reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

    def _on_accept(self) -> None:
        """Handle dialog accept."""
        logger.debug("Dialog closed", dialog=self.__class__.__name__, result="accepted")
        self.accept()

    def _on_reject(self) -> None:
        """Handle dialog reject."""
        logger.debug("Dialog closed", dialog=self.__class__.__name__, result="rejected")
        self.reject()

    def showEvent(self, event) -> None:
        """Handle dialog show event."""
        super().showEvent(event)
        logger.debug("Dialog opened", dialog=self.__class__.__name__)

    def _create_general_tab(self) -> QWidget:
        """Create general settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Theme selection
        theme_group = QGroupBox("Appearance")
        theme_layout = QFormLayout()

        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Light", "Dark", "System"])
        self.theme_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="theme",
                value=value,
            )
        )
        theme_layout.addRow("Theme:", self.theme_combo)

        theme_group.setLayout(theme_layout)
        layout.addWidget(theme_group)

        # Auto-save settings
        autosave_group = QGroupBox("Auto-Save")
        autosave_layout = QVBoxLayout()

        self.autosave_check = QCheckBox("Enable auto-save")
        self.autosave_check.setChecked(True)
        self.autosave_check.stateChanged.connect(self._on_autosave_changed)
        autosave_layout.addWidget(self.autosave_check)

        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Auto-save interval:"))
        self.autosave_spin = QSpinBox()
        self.autosave_spin.setRange(1, 60)
        self.autosave_spin.setValue(5)
        self.autosave_spin.setSuffix(" minutes")
        self.autosave_spin.valueChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="autosave_interval",
                value=value,
            )
        )
        interval_layout.addWidget(self.autosave_spin)
        interval_layout.addStretch()
        autosave_layout.addLayout(interval_layout)

        autosave_group.setLayout(autosave_layout)
        layout.addWidget(autosave_group)

        # Recent projects
        recent_group = QGroupBox("Recent Projects")
        recent_layout = QFormLayout()

        self.recent_count_spin = QSpinBox()
        self.recent_count_spin.setRange(5, 50)
        self.recent_count_spin.setValue(10)
        self.recent_count_spin.valueChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="recent_count",
                value=value,
            )
        )
        recent_layout.addRow("Number of recent projects:", self.recent_count_spin)

        recent_group.setLayout(recent_layout)
        layout.addWidget(recent_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_jax_tab(self) -> QWidget:
        """Create JAX settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Device selection
        device_group = QGroupBox("Device Configuration")
        device_layout = QFormLayout()

        self.device_combo = QComboBox()
        self.device_combo.addItems(["cpu", "gpu", "tpu"])
        self.device_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="device",
                value=value,
            )
        )
        device_layout.addRow("Default Device:", self.device_combo)

        device_group.setLayout(device_layout)
        layout.addWidget(device_group)

        # Float64 settings
        float64_group = QGroupBox("Precision")
        float64_layout = QVBoxLayout()

        self.float64_warning_check = QCheckBox(
            "Warn when float64 is not enabled (recommended)"
        )
        self.float64_warning_check.setChecked(True)
        self.float64_warning_check.stateChanged.connect(
            lambda state: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="float64_warning",
                value=state == Qt.CheckState.Checked.value,
            )
        )
        float64_layout.addWidget(self.float64_warning_check)

        self.auto_enable_float64_check = QCheckBox(
            "Automatically enable float64 on startup"
        )
        self.auto_enable_float64_check.setChecked(True)
        self.auto_enable_float64_check.stateChanged.connect(
            lambda state: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="auto_enable_float64",
                value=state == Qt.CheckState.Checked.value,
            )
        )
        float64_layout.addWidget(self.auto_enable_float64_check)

        float64_group.setLayout(float64_layout)
        layout.addWidget(float64_group)

        # Memory management
        memory_group = QGroupBox("Memory Management")
        memory_layout = QVBoxLayout()

        memory_limit_layout = QHBoxLayout()
        memory_limit_layout.addWidget(QLabel("GPU Memory Limit:"))
        self.memory_limit_spin = QSpinBox()
        self.memory_limit_spin.setRange(512, 32768)
        self.memory_limit_spin.setValue(8192)
        self.memory_limit_spin.setSingleStep(512)
        self.memory_limit_spin.setSuffix(" MB")
        self.memory_limit_spin.valueChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="memory_limit",
                value=value,
            )
        )
        memory_limit_layout.addWidget(self.memory_limit_spin)
        memory_limit_layout.addStretch()
        memory_layout.addLayout(memory_limit_layout)

        self.preallocate_check = QCheckBox("Pre-allocate GPU memory")
        self.preallocate_check.setChecked(False)
        self.preallocate_check.stateChanged.connect(
            lambda state: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="preallocate_memory",
                value=state == Qt.CheckState.Checked.value,
            )
        )
        memory_layout.addWidget(self.preallocate_check)

        memory_group.setLayout(memory_layout)
        layout.addWidget(memory_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _create_visualization_tab(self) -> QWidget:
        """Create visualization settings tab."""
        tab = QWidget()
        layout = QVBoxLayout()

        # Plot style
        style_group = QGroupBox("Plot Style")
        style_layout = QFormLayout()

        self.plot_style_combo = QComboBox()
        self.plot_style_combo.addItems(
            [
                "default",
                "publication",
                "presentation",
                "poster",
                "seaborn",
                "ggplot",
                "bmh",
                "fivethirtyeight",
            ]
        )
        self.plot_style_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="plot_style",
                value=value,
            )
        )
        style_layout.addRow("Default Plot Style:", self.plot_style_combo)

        self.color_palette_combo = QComboBox()
        self.color_palette_combo.addItems(
            [
                "tab10",
                "Set1",
                "Set2",
                "Paired",
                "viridis",
                "plasma",
                "inferno",
                "magma",
            ]
        )
        self.color_palette_combo.currentTextChanged.connect(
            lambda value: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="color_palette",
                value=value,
            )
        )
        style_layout.addRow("Default Color Palette:", self.color_palette_combo)

        style_group.setLayout(style_layout)
        layout.addWidget(style_group)

        # Font settings
        font_group = QGroupBox("Font Settings")
        font_layout = QVBoxLayout()

        # Font size slider
        font_size_layout = QVBoxLayout()
        font_size_label_layout = QHBoxLayout()
        font_size_label_layout.addWidget(QLabel("Font Size:"))
        self.font_size_label = QLabel("12 pt")
        font_size_label_layout.addWidget(self.font_size_label)
        font_size_label_layout.addStretch()
        font_size_layout.addLayout(font_size_label_layout)

        self.font_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.font_size_slider.setRange(8, 24)
        self.font_size_slider.setValue(12)
        self.font_size_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.font_size_slider.setTickInterval(2)
        self.font_size_slider.valueChanged.connect(self._on_font_size_changed)
        font_size_layout.addWidget(self.font_size_slider)

        font_layout.addLayout(font_size_layout)
        font_group.setLayout(font_layout)
        layout.addWidget(font_group)

        # Grid and legend
        display_group = QGroupBox("Display Options")
        display_layout = QVBoxLayout()

        self.show_grid_check = QCheckBox("Show grid by default")
        self.show_grid_check.setChecked(True)
        self.show_grid_check.stateChanged.connect(
            lambda state: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="show_grid",
                value=state == Qt.CheckState.Checked.value,
            )
        )
        display_layout.addWidget(self.show_grid_check)

        self.show_legend_check = QCheckBox("Show legend by default")
        self.show_legend_check.setChecked(True)
        self.show_legend_check.stateChanged.connect(
            lambda state: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="show_legend",
                value=state == Qt.CheckState.Checked.value,
            )
        )
        display_layout.addWidget(self.show_legend_check)

        self.tight_layout_check = QCheckBox("Use tight layout")
        self.tight_layout_check.setChecked(True)
        self.tight_layout_check.stateChanged.connect(
            lambda state: logger.debug(
                "Value changed",
                dialog=self.__class__.__name__,
                field="tight_layout",
                value=state == Qt.CheckState.Checked.value,
            )
        )
        display_layout.addWidget(self.tight_layout_check)

        display_group.setLayout(display_layout)
        layout.addWidget(display_group)

        layout.addStretch()
        tab.setLayout(layout)
        return tab

    def _load_current_preferences(self) -> None:
        """Load current preferences into UI."""
        if not self.current_preferences:
            return

        logger.debug(
            "Loading current preferences",
            dialog=self.__class__.__name__,
            preference_keys=list(self.current_preferences.keys()),
        )

        # General tab
        if "theme" in self.current_preferences:
            theme = self.current_preferences["theme"]
            idx = self.theme_combo.findText(theme, Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.theme_combo.setCurrentIndex(idx)

        if "autosave_enabled" in self.current_preferences:
            self.autosave_check.setChecked(self.current_preferences["autosave_enabled"])

        if "autosave_interval" in self.current_preferences:
            self.autosave_spin.setValue(self.current_preferences["autosave_interval"])

        if "recent_count" in self.current_preferences:
            self.recent_count_spin.setValue(self.current_preferences["recent_count"])

        # JAX tab
        if "device" in self.current_preferences:
            device = self.current_preferences["device"]
            idx = self.device_combo.findText(device, Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.device_combo.setCurrentIndex(idx)

        if "float64_warning" in self.current_preferences:
            self.float64_warning_check.setChecked(
                self.current_preferences["float64_warning"]
            )

        if "auto_enable_float64" in self.current_preferences:
            self.auto_enable_float64_check.setChecked(
                self.current_preferences["auto_enable_float64"]
            )

        if "memory_limit" in self.current_preferences:
            self.memory_limit_spin.setValue(self.current_preferences["memory_limit"])

        if "preallocate_memory" in self.current_preferences:
            self.preallocate_check.setChecked(
                self.current_preferences["preallocate_memory"]
            )

        # Visualization tab
        if "plot_style" in self.current_preferences:
            style = self.current_preferences["plot_style"]
            idx = self.plot_style_combo.findText(style, Qt.MatchFlag.MatchFixedString)
            if idx >= 0:
                self.plot_style_combo.setCurrentIndex(idx)

        if "color_palette" in self.current_preferences:
            palette = self.current_preferences["color_palette"]
            idx = self.color_palette_combo.findText(
                palette, Qt.MatchFlag.MatchFixedString
            )
            if idx >= 0:
                self.color_palette_combo.setCurrentIndex(idx)

        if "font_size" in self.current_preferences:
            self.font_size_slider.setValue(self.current_preferences["font_size"])

        if "show_grid" in self.current_preferences:
            self.show_grid_check.setChecked(self.current_preferences["show_grid"])

        if "show_legend" in self.current_preferences:
            self.show_legend_check.setChecked(self.current_preferences["show_legend"])

        if "tight_layout" in self.current_preferences:
            self.tight_layout_check.setChecked(self.current_preferences["tight_layout"])

    def _on_autosave_changed(self, state: int) -> None:
        """Handle auto-save checkbox change."""
        enabled = state == Qt.CheckState.Checked.value
        self.autosave_spin.setEnabled(enabled)
        logger.debug(
            "Value changed",
            dialog=self.__class__.__name__,
            field="autosave_enabled",
            value=enabled,
        )

    def _on_font_size_changed(self, value: int) -> None:
        """Handle font size slider change."""
        self.font_size_label.setText(f"{value} pt")
        logger.debug(
            "Value changed",
            dialog=self.__class__.__name__,
            field="font_size",
            value=value,
        )

    def _on_apply(self) -> None:
        """Handle Apply button click.

        Applies current preferences without closing the dialog,
        allowing users to see changes before committing.
        """
        logger.debug("Apply button clicked", dialog=self.__class__.__name__)

        # Get current preferences and update internal state
        self.current_preferences = self.get_preferences()

        # Emit custom signal if parent is listening for preference changes
        # This allows the main window to apply preferences immediately
        parent = self.parent()
        if parent is not None and hasattr(parent, "apply_preferences"):
            parent.apply_preferences(self.current_preferences)  # type: ignore[attr-defined]

    def _restore_defaults(self) -> None:
        """Restore all preferences to defaults."""
        logger.debug("Restoring defaults", dialog=self.__class__.__name__)

        # General
        self.theme_combo.setCurrentText("Light")
        self.autosave_check.setChecked(True)
        self.autosave_spin.setValue(5)
        self.recent_count_spin.setValue(10)

        # JAX
        self.device_combo.setCurrentText("cpu")
        self.float64_warning_check.setChecked(True)
        self.auto_enable_float64_check.setChecked(True)
        self.memory_limit_spin.setValue(8192)
        self.preallocate_check.setChecked(False)

        # Visualization
        self.plot_style_combo.setCurrentText("default")
        self.color_palette_combo.setCurrentText("tab10")
        self.font_size_slider.setValue(12)
        self.show_grid_check.setChecked(True)
        self.show_legend_check.setChecked(True)
        self.tight_layout_check.setChecked(True)

    def get_preferences(self) -> dict[str, Any]:
        """Get all preferences.

        Returns
        -------
        dict[str, Any]
            Preferences dictionary with keys:
            - theme: UI theme (Light/Dark/System)
            - autosave_enabled: Whether auto-save is enabled
            - autosave_interval: Auto-save interval in minutes
            - recent_count: Number of recent projects to remember
            - device: Default JAX device
            - float64_warning: Whether to warn about float64
            - auto_enable_float64: Whether to enable float64 automatically
            - memory_limit: GPU memory limit in MB
            - preallocate_memory: Whether to pre-allocate GPU memory
            - plot_style: Default plot style
            - color_palette: Default color palette
            - font_size: Font size in points
            - show_grid: Whether to show grid by default
            - show_legend: Whether to show legend by default
            - tight_layout: Whether to use tight layout
        """
        preferences = {
            # General
            "theme": self.theme_combo.currentText(),
            "autosave_enabled": self.autosave_check.isChecked(),
            "autosave_interval": self.autosave_spin.value(),
            "recent_count": self.recent_count_spin.value(),
            # JAX
            "device": self.device_combo.currentText(),
            "float64_warning": self.float64_warning_check.isChecked(),
            "auto_enable_float64": self.auto_enable_float64_check.isChecked(),
            "memory_limit": self.memory_limit_spin.value(),
            "preallocate_memory": self.preallocate_check.isChecked(),
            # Visualization
            "plot_style": self.plot_style_combo.currentText(),
            "color_palette": self.color_palette_combo.currentText(),
            "font_size": self.font_size_slider.value(),
            "show_grid": self.show_grid_check.isChecked(),
            "show_legend": self.show_legend_check.isChecked(),
            "tight_layout": self.tight_layout_check.isChecked(),
        }

        logger.debug(
            "Getting preferences",
            dialog=self.__class__.__name__,
            preference_keys=list(preferences.keys()),
        )

        return preferences


# Alias for backward compatibility
Preferences = PreferencesDialog
