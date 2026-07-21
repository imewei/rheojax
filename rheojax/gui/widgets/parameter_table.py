"""
Parameter Table Widget
=====================

Interactive table for model parameter editing.
"""

import math
from dataclasses import replace

from rheojax.gui.compat import (
    QBrush,
    QCheckBox,
    QColor,
    QHBoxLayout,
    QHeaderView,
    QPalette,
    Qt,
    QTableWidget,
    QTableWidgetItem,
    QWidget,
    Signal,
)
from rheojax.gui.foundation.state import ParameterState
from rheojax.gui.resources.styles.tokens import Typography, themed
from rheojax.gui.utils.layout_helpers import set_zero_margins
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
        self.setAccessibleName("Parameter table")
        self.setAccessibleDescription(
            "Model parameters: value, min bound, max bound, and whether "
            "each is fixed during fitting."
        )

        # Configure table
        self.setColumnCount(5)
        self.setHorizontalHeaderLabels(["Parameter", "Value", "Min", "Max", "Fixed"])
        # Bumped from SIZE_MD_SM to the next type-scale step: this is a dense
        # numeric-entry grid, borderline small at typical desk-viewing distance.
        _table_font = f"{Typography.SIZE_BASE}pt"
        self.setStyleSheet(
            f"QTableWidget {{ font-size: {_table_font}; }} "
            f"QHeaderView::section {{ font-size: {_table_font}; }}"
        )

        # Column sizing: stretch to fill available width
        header = self.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)

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
        # Block signals during bulk update to avoid triggering per-cell callbacks
        self.blockSignals(True)
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
            self.blockSignals(False)
        logger.debug(
            "Parameters loaded",
            widget=self.__class__.__name__,
            param_names=list(parameters.keys()),
        )

    @staticmethod
    def _bounds_valid(min_val: float, max_val: float) -> bool:
        return math.isfinite(min_val) and math.isfinite(max_val) and min_val <= max_val

    @staticmethod
    def _value_valid(value: float, min_val: float, max_val: float) -> bool:
        return math.isfinite(value) and min_val <= value <= max_val

    def has_invalid_rows(self) -> bool:
        """Return True if any row currently holds a non-numeric, non-finite,
        out-of-range, or inverted-bounds entry.

        get_parameters() silently skips such rows (a caller trusting its
        returned dict alone would launch a fit with those parameters
        defaulted with no indication anything was wrong) -- callers that
        need to warn the user before launching should check this first.
        """
        for row in range(self.rowCount()):
            val_item = self.item(row, 1)
            min_item = self.item(row, 2)
            max_item = self.item(row, 3)
            if val_item is None or min_item is None or max_item is None:
                continue
            try:
                value = float(val_item.text())
                min_bound = float(min_item.text())
                max_bound = float(max_item.text())
            except ValueError:
                return True
            if not self._bounds_valid(min_bound, max_bound) or not self._value_valid(
                value, min_bound, max_bound
            ):
                return True
        return False

    def get_parameters(self) -> dict[str, ParameterState]:
        """Get current parameter values and states.

        Returns
        -------
        dict[str, ParameterState]
            Dictionary of current parameter states
        """
        parameters = {}

        for row in range(self.rowCount()):
            # PTBL-001: item() returns None if the row was removed while we
            # are iterating (e.g. a concurrent clear triggered by a state
            # update).  Guard every cell access to avoid AttributeError.
            name_item = self.item(row, 0)
            val_item = self.item(row, 1)
            min_item = self.item(row, 2)
            max_item = self.item(row, 3)
            if (
                name_item is None
                or val_item is None
                or min_item is None
                or max_item is None
            ):
                logger.warning(
                    "Skipping incomplete table row",
                    widget=self.__class__.__name__,
                    row=row,
                )
                continue
            param_name = name_item.text()
            if not param_name:
                continue

            # Get values from cells
            try:
                value = float(val_item.text())
                min_bound = float(min_item.text())
                max_bound = float(max_item.text())
            except ValueError:
                logger.warning(
                    "Skipping row with non-numeric cell — parameter will use "
                    "model default; check the table for typos",
                    widget=self.__class__.__name__,
                    row=row,
                    param_name=param_name,
                )
                continue

            # PTBL-002: same invariant _on_item_changed enforces on live edits
            # ("invalid values must never reach the state store") also applies
            # here, since a fit can be launched without ever triggering that
            # per-cell signal (e.g. paste, or a row never focused after edit).
            if not self._bounds_valid(min_bound, max_bound) or not self._value_valid(
                value, min_bound, max_bound
            ):
                logger.warning(
                    "Skipping row with invalid value/bounds — parameter will "
                    "use model default; check the table for out-of-range or "
                    "non-finite entries",
                    widget=self.__class__.__name__,
                    row=row,
                    param_name=param_name,
                    value=value,
                    min_bound=min_bound,
                    max_bound=max_bound,
                )
                continue

            # Get fixed state from checkbox
            checkbox_widget = self.cellWidget(row, 4)
            checkbox = checkbox_widget.findChild(QCheckBox) if checkbox_widget else None
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
        set_zero_margins(layout)
        layout.setAlignment(Qt.AlignCenter)

        checkbox = QCheckBox()
        checkbox.setChecked(checked)
        checkbox.setAccessibleName(f"Fix {param_name}")
        checkbox.setAccessibleDescription(
            f"Hold {param_name} constant during fitting instead of optimizing it."
        )
        checkbox.stateChanged.connect(
            lambda state: self._on_fixed_toggled(
                param_name, state == Qt.CheckState.Checked.value
            )
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

                if not self._value_valid(value, min_val, max_val):
                    # Out of bounds (or nan/inf, which NaN comparisons would
                    # otherwise silently pass as "in bounds") - red text.
                    # Do not emit: invalid values must never reach the state store.
                    item.setForeground(QBrush(QColor(themed("ERROR"))))
                    # WCAG 1.4.1: don't rely on color alone -- a tooltip
                    # explaining the problem is the minimal non-color cue.
                    item.setToolTip(
                        f"Value must be finite and within [{min_val:.6g}, "
                        f"{max_val:.6g}]"
                    )
                    return

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

                item.setForeground(
                    QBrush(self.palette().color(QPalette.ColorRole.Text))
                )
                item.setToolTip("")

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

                if not self._bounds_valid(min_val, max_val):
                    # Invalid bounds (non-finite or inverted) - red text.
                    # Do not emit: invalid bounds must never reach the state store.
                    item.setForeground(QBrush(QColor(themed("ERROR"))))
                    # WCAG 1.4.1: don't rely on color alone.
                    item.setToolTip("Bounds must be finite with min <= max")
                    return

                item.setForeground(
                    QBrush(self.palette().color(QPalette.ColorRole.Text))
                )
                item.setToolTip("")

                # Emit bounds changed
                self.bounds_changed.emit(param_name, min_val, max_val)

                # Revalidate value
                value_item = self.item(row, 1)
                value = float(value_item.text())
                if not self._value_valid(value, min_val, max_val):
                    value_item.setForeground(QBrush(QColor(themed("ERROR"))))
                    value_item.setToolTip(
                        f"Value must be finite and within [{min_val:.6g}, "
                        f"{max_val:.6g}]"
                    )
                else:
                    value_item.setForeground(
                        QBrush(self.palette().color(QPalette.ColorRole.Text))
                    )
                    value_item.setToolTip("")

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
                item.setForeground(
                    QBrush(self.palette().color(QPalette.ColorRole.Text))
                )

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
                # GUI-010 fix: Use replace() to create a new ParameterState
                # instead of mutating fixed in-place, keeping cached state
                # consistent with immutable-by-convention dataclass pattern.
                if param_name in self._parameter_states:
                    param_state = replace(
                        self._parameter_states[param_name], fixed=is_fixed
                    )
                    self._parameter_states[param_name] = param_state
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
            # Muted surface for fixed (locked) parameters
            gray_brush = QBrush(QColor(themed("BG_HOVER")))
            for col in range(4):  # All columns except checkbox
                item = self.item(row, col)
                if item:
                    item.setBackground(gray_brush)
        else:
            # Normal surface for active (editable) parameters
            white_brush = QBrush(QColor(themed("BG_SURFACE")))
            for col in range(4):
                item = self.item(row, col)
                if item:
                    item.setBackground(white_brush)
