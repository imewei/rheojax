"""
Parameter Table Widget
=====================

Interactive table for model parameter editing.
"""

from rheojax.gui.compat import (
    Qt,
    Signal,
    QBrush,
    QColor,
    QCheckBox,
    QHBoxLayout,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
)
from rheojax.gui.state.store import ParameterState
from rheojax.logging import get_logger

logger = get_logger(__name__)


class ParameterTable(QTableWidget):
    """Editable table widget for model parameters with validation.

    Features:
        - Editable Value, Min, Max cells
        - Fixed column with checkbox
        - Color coding for fixed parameters, out-of-bounds, modified values
        - Real-time validation
        - Bounds checking

    Columns:
        - Parameter: Parameter name (read-only)
        - Value: Current parameter value (editable)
        - Min: Minimum bound (editable)
        - Max: Maximum bound (editable)
        - Fixed: Checkbox to fix parameter during fitting

    Signals
    -------
    parameter_changed : Signal(str, float)
        Emitted when parameter value changes
    bounds_changed : Signal(str, float, float)
        Emitted when bounds change (param_name, min_val, max_val)
    fixed_toggled : Signal(str, bool)
        Emitted when fixed checkbox is toggled

    Example
    -------
    >>> table = ParameterTable()  # doctest: +SKIP
    >>> table.set_parameters({'G0': param_state})  # doctest: +SKIP
    >>> table.parameter_changed.connect(on_param_changed)  # doctest: +SKIP
    """

    parameter_changed = Signal(str, float)
    bounds_changed = Signal(str, float, float)
    fixed_toggled = Signal(str, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        """Initialize parameter table.

        Parameters
        ----------
        parent : QWidget, optional
            Parent widget
        """
        super().__init__(parent)
        logger.debug("Initializing", class_name=self.__class__.__name__)

        # Configure table
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Fixed"])
        self.setStyleSheet(
            "QTableWidget { font-size: 11pt; } "
            "QHeaderView::section { font-size: 11pt; }"
        )

        # Column widths
        self.setColumnWidth(0, 120)
        self.setColumnWidth(1, 100)
        self.setColumnWidth(2, 80)
        self.setColumnWidth(3, 80)
        self.setColumnWidth(4, 60)

        # Enable sorting
        self.setSortingEnabled(False)

        # Track original values for modification detection
        self._original_values: dict[str, float] = {}
        self._parameter_states: dict[str, ParameterState] = {}

        # Connect signals
        self.itemChanged.connect(self._on_item_changed)
        logger.debug("Initialization complete", class_name=self.__class__.__name__)

    def set_parameters(self, parameters: dict[str, ParameterState]) -> None:
        """Update displayed parameters.

        Parameters
        ----------
        parameters : dict[str, ParameterState]
            Dictionary mapping parameter names to ParameterState objects
        """
        logger.debug(
            "State updated",
            widget=self.__class__.__name__,
            action="set_parameters",
            num_parameters=len(parameters),
        )
        # Disconnect signal temporarily to avoid triggering during setup
        # Use try/finally to ensure signal is reconnected even if exception occurs
        try:
            self.itemChanged.disconnect(self._on_item_changed)
        except (TypeError, RuntimeError):
            # Signal may not be connected yet
            logger.debug("itemChanged disconnect skipped; not connected")

        try:
            # Clear existing rows
            self.setRowCount(0)
            self._original_values.clear()
            self._parameter_states = parameters.copy()

            # Add parameters
            for row, (param_name, param_state) in enumerate(parameters.items()):
                self.insertRow(row)

                # Column 0: Parameter name (read-only)
                name_item = QTableWidgetItem(param_name)
                name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                if param_state.description:
                    name_item.setToolTip(param_state.description)
                self.setItem(row, 0, name_item)

                # Column 1: Value (editable)
                value_item = QTableWidgetItem(f"{param_state.value:.6g}")
                self.setItem(row, 1, value_item)

                # Column 2: Min bound (editable)
                min_item = QTableWidgetItem(f"{param_state.min_bound:.6g}")
                self.setItem(row, 2, min_item)

                # Column 3: Max bound (editable)
                max_item = QTableWidgetItem(f"{param_state.max_bound:.6g}")
                self.setItem(row, 3, max_item)

                # Column 4: Fixed checkbox
                checkbox_widget = self._create_checkbox_widget(
                    param_state.fixed, param_name
                )
                self.setCellWidget(row, 4, checkbox_widget)

                # Store original value
                self._original_values[param_name] = param_state.value

                # Apply styling based on state
                self._update_row_styling(row, param_state)
        finally:
            # Always reconnect signal, even if exception occurred
            try:
                self.itemChanged.connect(self._on_item_changed)
            except (TypeError, RuntimeError):
                logger.debug("itemChanged reconnect skipped; already connected")
        logger.debug(
            "Parameters loaded",
            widget=self.__class__.__name__,
            param_names=list(parameters.keys()),
        )

    def get_parameters(self) -> dict[str, ParameterState]:
        """Get current parameter values and states.

        Returns
        -------
        dict[str, ParameterState]
            Dictionary of current parameter states
        """
        parameters = {}

        for row in range(self.rowCount()):
            param_name = self.item(row, 0).text()

            # Get values from cells
            value = float(self.item(row, 1).text())
            min_bound = float(self.item(row, 2).text())
            max_bound = float(self.item(row, 3).text())

            # Get fixed state from checkbox
            checkbox_widget = self.cellWidget(row, 4)
            checkbox = checkbox_widget.findChild(QCheckBox)
            fixed = checkbox.isChecked() if checkbox else False

            # Get original parameter state
            original = self._parameter_states.get(param_name)

            # Create updated parameter state
            parameters[param_name] = ParameterState(
                name=param_name,
                value=value,
                min_bound=min_bound,
                max_bound=max_bound,
                fixed=fixed,
                unit=original.unit if original else "",
                description=original.description if original else "",
            )

        logger.debug(
            "Parameters retrieved",
            widget=self.__class__.__name__,
            num_parameters=len(parameters),
        )
        return parameters

    def reset_to_defaults(self) -> None:
        """Reset all parameters to their original values."""
        logger.info(
            "State changed",
            widget=self.__class__.__name__,
            action="reset_to_defaults",
        )
        try:
            self.itemChanged.disconnect(self._on_item_changed)
        except (TypeError, RuntimeError):
            logger.debug("itemChanged disconnect skipped during reset")

        try:
            for row in range(self.rowCount()):
                param_name = self.item(row, 0).text()
                if param_name in self._original_values:
                    original_value = self._original_values[param_name]
                    self.item(row, 1).setText(f"{original_value:.6g}")

                    # Reset styling
                    if param_name in self._parameter_states:
                        self._update_row_styling(
                            row, self._parameter_states[param_name]
                        )
        finally:
            try:
                self.itemChanged.connect(self._on_item_changed)
            except (TypeError, RuntimeError):
                logger.debug("itemChanged reconnect skipped during reset")
        logger.debug("Parameters reset complete", widget=self.__class__.__name__)

    def _create_checkbox_widget(self, checked: bool, param_name: str) -> QWidget:
        """Create centered checkbox widget for Fixed column.

        Parameters
        ----------
        checked : bool
            Initial checked state
        param_name : str
            Parameter name for signal emission

        Returns
        -------
        QWidget
            Widget containing centered checkbox
        """
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setAlignment(Qt.AlignCenter)

        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        checkbox.stateChanged.connect(
            lambda state: self._on_fixed_toggled(param_name, state == Qt.Checked)
        )

        layout.addWidget(checkbox)
        return widget

    def _on_item_changed(self, item: QTableWidgetItem) -> None:
        """Handle item value changes.

        Parameters
        ----------
        item : QTableWidgetItem
            Changed item
        """
        row = item.row()
        col = item.column()
        param_name = self.item(row, 0).text()

        try:
            # Validate numeric input
            value = float(item.text())

            if col == 1:  # Value column
                logger.debug(
                    "User interaction",
                    widget=self.__class__.__name__,
                    action="value_changed",
                    param_name=param_name,
                    value=value,
                )
                # Check bounds
                min_val = float(self.item(row, 2).text())
                max_val = float(self.item(row, 3).text())

                if value < min_val or value > max_val:
                    # Out of bounds - red text
                    item.setForeground(QBrush(QColor(255, 0, 0)))
                else:
                    # In bounds - check if modified
                    if param_name in self._original_values:
                        is_modified = (
                            abs(value - self._original_values[param_name]) > 1e-10
                        )
                        if is_modified:
                            # Modified - bold
                            font = item.font()
                            font.setBold(True)
                            item.setFont(font)
                        else:
                            # Unmodified - normal
                            font = item.font()
                            font.setBold(False)
                            item.setFont(font)

                    item.setForeground(QBrush(QColor(0, 0, 0)))

                # Emit signal
                self.parameter_changed.emit(param_name, value)

            elif col in (2, 3):  # Min/Max columns
                min_val = float(self.item(row, 2).text())
                max_val = float(self.item(row, 3).text())
                logger.debug(
                    "User interaction",
                    widget=self.__class__.__name__,
                    action="bounds_changed",
                    param_name=param_name,
                    min_val=min_val,
                    max_val=max_val,
                )

                # Emit bounds changed
                self.bounds_changed.emit(param_name, min_val, max_val)

                # Revalidate value
                value_item = self.item(row, 1)
                value = float(value_item.text())
                if value < min_val or value > max_val:
                    value_item.setForeground(QBrush(QColor(255, 0, 0)))
                else:
                    value_item.setForeground(QBrush(QColor(0, 0, 0)))

        except ValueError:
            # Invalid number - reset to previous value without cascading signals
            logger.error(
                "Invalid numeric value entered",
                widget=self.__class__.__name__,
                param_name=param_name,
                column=col,
                invalid_value=item.text(),
                exc_info=True,
            )
            try:
                self.itemChanged.disconnect(self._on_item_changed)
            except (TypeError, RuntimeError):
                pass

            if param_name in self._parameter_states:
                param_state = self._parameter_states[param_name]
                if col == 1:
                    item.setText(f"{param_state.value:.6g}")
                elif col == 2:
                    item.setText(f"{param_state.min_bound:.6g}")
                elif col == 3:
                    item.setText(f"{param_state.max_bound:.6g}")

                # Restore default styling
                font = item.font()
                font.setBold(False)
                item.setFont(font)
                item.setForeground(QBrush(QColor(0, 0, 0)))

            try:
                self.itemChanged.connect(self._on_item_changed)
            except (TypeError, RuntimeError):
                pass

    def _on_fixed_toggled(self, param_name: str, is_fixed: bool) -> None:
        """Handle fixed checkbox toggle.

        Parameters
        ----------
        param_name : str
            Parameter name
        is_fixed : bool
            New fixed state
        """
        logger.debug(
            "User interaction",
            widget=self.__class__.__name__,
            action="fixed_toggled",
            param_name=param_name,
            is_fixed=is_fixed,
        )
        # Find row
        for row in range(self.rowCount()):
            if self.item(row, 0).text() == param_name:
                # Update styling
                if param_name in self._parameter_states:
                    param_state = self._parameter_states[param_name]
                    param_state.fixed = is_fixed
                    self._update_row_styling(row, param_state)
                break

        # Emit signal
        self.fixed_toggled.emit(param_name, is_fixed)

    def _update_row_styling(self, row: int, param_state: ParameterState) -> None:
        """Update row styling based on parameter state.

        Parameters
        ----------
        row : int
            Row index
        param_state : ParameterState
            Parameter state
        """
        if param_state.fixed:
            # Gray background for fixed parameters
            gray_brush = QBrush(QColor(240, 240, 240))
            for col in range(4):  # All columns except checkbox
                item = self.item(row, col)
                if item:
                    item.setBackground(gray_brush)
        else:
            # White background for active parameters
            white_brush = QBrush(QColor(255, 255, 255))
            for col in range(4):
                item = self.item(row, col)
                if item:
                    item.setBackground(white_brush)
